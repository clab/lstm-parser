#include "lstm-parser.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "cnn/model.h"
#include "cnn/tensor.h"


using namespace cnn::expr;
using namespace cnn;
using namespace std;

namespace lstm_parser {


const string ParseTree::NO_LABEL("ERROR");


void LSTMParser::LoadPretrainedWords(const string& words_path) {
  cerr << "Loading word vectors from " << words_path;
  ifstream in;
  in.open(words_path);
  if (!in) {
    cout << "..." << endl;
    cerr << "ERROR: failed to open word vectors file " << words_path
         << "; exiting" << endl;
    abort();
  }
  // Read header
  string line;
  getline(in, line);
  istringstream first_line(line);
  unsigned num_words;
  first_line >> num_words;
  unsigned pretrained_dim;
  first_line >> pretrained_dim;
  cerr << " with " << pretrained_dim << " dimensions..." << endl;

  // Read vectors
  pretrained[vocab.words_to_int[vocab.UNK]] = vector<float>(pretrained_dim, 0);
  vector<float> v(pretrained_dim, 0);
  string word;
  while (getline(in, line)) {
    istringstream lin(line);
    lin >> word;
    for (unsigned i = 0; i < pretrained_dim; ++i)
      lin >> v[i];
    // We DON'T yet know this word is present in training data.
    unsigned id = vocab.GetOrAddWord(word, false);
    pretrained[id] = v;
  }
  assert(num_words == pretrained.size() - 1); // -1 for UNK
  cerr << "Loaded " << pretrained.size() - 1 << " words" << endl;
}


void LSTMParser::InitializeNetworkParameters() {
  // Now that the vocab is ready to be finalized, we can set all the network
  // parameters.
  unsigned action_size = vocab.CountActions() + 1;
  unsigned pos_size = vocab.CountPOS() + 10; // bad way of dealing with new POS
  unsigned vocab_size = vocab.CountWords() + 1;
  n_possible_actions = vocab.CountActions();

  if (!pretrained.empty()) {
    unsigned pretrained_dim = pretrained.begin()->second.size();
    p_t = model.add_lookup_parameters(vocab_size, {pretrained_dim});
    for (const auto& it : pretrained)
      p_t->Initialize(it.first, it.second);
    p_t2l = model.add_parameters({options.lstm_input_dim, pretrained_dim});
  } else {
    p_t = nullptr;
    p_t2l = nullptr;
  }

  p_w = model.add_lookup_parameters(vocab_size, {options.input_dim});
  p_a = model.add_lookup_parameters(action_size, {options.action_dim});
  p_r = model.add_lookup_parameters(action_size, {options.rel_dim});
  p_pbias = model.add_parameters({options.hidden_dim});
  p_A = model.add_parameters({options.hidden_dim, options.hidden_dim});
  p_B = model.add_parameters({options.hidden_dim, options.hidden_dim});
  p_S = model.add_parameters({options.hidden_dim, options.hidden_dim});
  p_H = model.add_parameters({options.lstm_input_dim, options.lstm_input_dim});
  p_D = model.add_parameters({options.lstm_input_dim, options.lstm_input_dim});
  p_R = model.add_parameters({options.lstm_input_dim, options.rel_dim});
  p_w2l = model.add_parameters({options.lstm_input_dim, options.input_dim});
  p_ib = model.add_parameters({options.lstm_input_dim});
  p_cbias = model.add_parameters({options.lstm_input_dim});
  p_p2a = model.add_parameters({action_size, options.hidden_dim});
  p_action_start = model.add_parameters({options.action_dim});
  p_abias = model.add_parameters({action_size});
  p_buffer_guard = model.add_parameters({options.lstm_input_dim});
  p_stack_guard = model.add_parameters({options.lstm_input_dim});

  if (options.use_pos) {
    p_p = model.add_lookup_parameters(pos_size, {options.pos_dim});
    p_p2l = model.add_parameters({options.lstm_input_dim, options.pos_dim});
  } else {
    p_p = nullptr;
    p_p2l = nullptr;
  }
}


LSTMParser::LSTMParser(const ParserOptions& poptions,
                       const string& pretrained_words_path, bool finalize) :
      options(poptions),
      kROOT_SYMBOL(vocab.GetOrAddWord(vocab.ROOT)),
      stack_lstm(options.layers, options.lstm_input_dim, options.hidden_dim,
                 &model),
      buffer_lstm(options.layers, options.lstm_input_dim, options.hidden_dim,
                  &model),
      action_lstm(options.layers, options.action_dim, options.hidden_dim,
                  &model) {
  // First load words if needed before creating network parameters.
  // That will ensure that the vocab has the final number of words.
  if (!pretrained_words_path.empty()) {
    LoadPretrainedWords(pretrained_words_path);
  }

  if (finalize) {
    FinalizeVocab(); // sets finalized
  } else {
    finalized = false;
  }
}


bool LSTMParser::IsActionForbidden(const unsigned action,
                                   const TaggerState& state) const {
  const string& action_name = vocab.action_names[action];
  const ParserState& real_state = static_cast<const ParserState&>(state);
  unsigned ssize = real_state.stack.size();
  unsigned bsize = real_state.buffer.size();

  if (action_name[1] == 'W' && ssize < 3)
    return true;
  if (action_name[1] == 'W') {
    int top = real_state.stacki[real_state.stacki.size() - 1];
    int sec = real_state.stacki[real_state.stacki.size() - 2];
    if (sec > top)
      return true;
  }

  bool is_shift = (action_name[0] == 'S' && action_name[1] == 'H');
  bool is_reduce = !is_shift;
  if (is_shift && bsize == 1)
    return true;
  if (is_reduce && ssize < 3)
    return true;
  if (bsize == 2 && // ROOT is the only thing remaining on buffer
      ssize > 2 && // there is more than a single element on the stack
      is_shift)
    return true;
  // only attach left to ROOT
  if (bsize == 1 && ssize == 3 && action_name[0] == 'R')
    return true;
  return false;
}


ParseTree LSTMParser::RecoverParseTree(
    const Sentence& sentence, const vector<unsigned>& actions, double logprob,
    bool labeled) const {
  ParseTree tree(sentence, labeled);
  vector<int> bufferi(sentence.Size() + 1);
  bufferi[0] = -999;
  vector<int> stacki(1, -999);
  unsigned added_to_buffer = 0;
  for (const auto& index_and_word_id : sentence.words) {
    // ROOT is set to -1, so it'll come last in a sequence of unsigned ints.
    bufferi[sentence.Size() - added_to_buffer++] =
        index_and_word_id.first;
  }
  for (auto action : actions) { // loop over transitions for sentence
    const string& action_string = vocab.action_names[action];
    const char ac = action_string[0];
    const char ac2 = action_string[1];
    if (ac == 'S' && ac2 == 'H') {  // SHIFT
      assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
      stacki.push_back(bufferi.back());
      bufferi.pop_back();
    } else if (ac == 'S' && ac2 == 'W') { // SWAP
      assert(stacki.size() > 2);
      unsigned ii;
      unsigned jj;
      jj = stacki.back();
      stacki.pop_back();
      ii = stacki.back();
      stacki.pop_back();
      bufferi.push_back(ii);
      stacki.push_back(jj);
    } else { // LEFT or RIGHT
      assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
      assert(ac == 'L' || ac == 'R');
      unsigned depi;
      unsigned headi;
      (ac == 'R' ? depi : headi) = stacki.back();
      stacki.pop_back();
      (ac == 'R' ? headi : depi) = stacki.back();
      stacki.pop_back();
      stacki.push_back(headi);
      tree.SetParent(depi, headi, vocab.actions_to_arc_labels[action]);
    }
  }
  assert(bufferi.size() == 1);
  //assert(stacki.size() == 2);

  tree.logprob = logprob;
  return tree;
}


Expression LSTMParser::GetActionProbabilities(const TaggerState& state) {
  // p_t = pbias + S * slstm + B * blstm + A * alstm
  Expression p_t = affine_transform(
      {GetParamExpr(p_pbias), GetParamExpr(p_S), stack_lstm.back(),
          GetParamExpr(p_B), buffer_lstm.back(), GetParamExpr(p_A),
          action_lstm.back()});
  Expression nlp_t = rectify(p_t);
  // r_t = abias + p2a * nlp
  Expression r_t = affine_transform(
      {GetParamExpr(p_abias), GetParamExpr(p_p2a), nlp_t});
  return r_t;
}


void LSTMParser::DoAction(unsigned action, TaggerState* state,
                          ComputationGraph* cg) {
  ParserState* real_state = static_cast<ParserState*>(state);
  // add current action to action LSTM
  Expression action_e = lookup(*cg, p_a, action);
  action_lstm.add_input(action_e);

  // get relation embedding from action (TODO: convert to rel from action?)
  Expression relation = lookup(*cg, p_r, action);

  // do action
  const string& action_string = vocab.action_names[action];
  const char ac = action_string[0];
  const char ac2 = action_string[1];

  if (ac == 'S' && ac2 == 'H') {  // SHIFT
    assert(real_state->buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
    real_state->stack.push_back(real_state->buffer.back());
    stack_lstm.add_input(real_state->buffer.back());
    real_state->buffer.pop_back();
    buffer_lstm.rewind_one_step();
    real_state->stacki.push_back(real_state->bufferi.back());
    real_state->bufferi.pop_back();
  } else if (ac == 'S' && ac2 == 'W') { //SWAP --- Miguel
    assert(real_state->stack.size() > 2); // dummy symbol means > 2 (not >= 2)

    Expression toki, tokj;
    unsigned ii = 0, jj = 0;
    tokj = real_state->stack.back();
    jj = real_state->stacki.back();
    real_state->stack.pop_back();
    real_state->stacki.pop_back();

    toki = real_state->stack.back();
    ii = real_state->stacki.back();
    real_state->stack.pop_back();
    real_state->stacki.pop_back();

    real_state->buffer.push_back(toki);
    real_state->bufferi.push_back(ii);

    stack_lstm.rewind_one_step();
    stack_lstm.rewind_one_step();

    buffer_lstm.add_input(real_state->buffer.back());

    real_state->stack.push_back(tokj);
    real_state->stacki.push_back(jj);

    stack_lstm.add_input(real_state->stack.back());
  } else { // LEFT or RIGHT
    assert(real_state->stack.size() > 2); // dummy symbol means > 2 (not >= 2)
    assert(ac == 'L' || ac == 'R');
    Expression dep, head;
    unsigned depi = 0, headi = 0;
    (ac == 'R' ? dep : head) = real_state->stack.back();
    (ac == 'R' ? depi : headi) = real_state->stacki.back();
    real_state->stack.pop_back();
    real_state->stacki.pop_back();
    (ac == 'R' ? head : dep) = real_state->stack.back();
    (ac == 'R' ? headi : depi) = real_state->stacki.back();
    real_state->stack.pop_back();
    real_state->stacki.pop_back();
    // composed = cbias + H * head + D * dep + R * relation
    Expression composed = affine_transform({GetParamExpr(p_cbias),
        GetParamExpr(p_H), head, GetParamExpr(p_D), dep, GetParamExpr(p_R),
        relation});
    Expression nlcomposed = tanh(composed);
    stack_lstm.rewind_one_step();
    stack_lstm.rewind_one_step();
    stack_lstm.add_input(nlcomposed);
    real_state->stack.push_back(nlcomposed);
    real_state->stacki.push_back(headi);
  }
}


NeuralTransitionTagger::TaggerState* LSTMParser::InitializeParserState(
    ComputationGraph* cg,
    const Sentence& raw_sent,
    const Sentence::SentenceMap& sent,  // sentence with OOVs replaced
    const vector<unsigned>& correct_actions) {
  stack_lstm.new_graph(*cg);
  buffer_lstm.new_graph(*cg);
  action_lstm.new_graph(*cg);
  stack_lstm.start_new_sequence();
  buffer_lstm.start_new_sequence();
  action_lstm.start_new_sequence();

  Expression stack_guard = GetParamExpr(p_stack_guard);
  ParserState* state = new ParserState(raw_sent, sent, stack_guard);
  action_lstm.add_input(GetParamExpr(p_action_start));
  stack_lstm.add_input(stack_guard);

  // precompute buffer representation from left to right
  unsigned added_to_buffer = 0;
  for (const auto& index_and_word_id : sent) {
    unsigned token_index = index_and_word_id.first;
    unsigned word_id = index_and_word_id.second;

    assert(word_id < vocab.CountWords());
    Expression w = lookup(*cg, p_w, word_id);

    vector<Expression> args = {GetParamExpr(p_ib), GetParamExpr(p_w2l),
                               w};  // learn embeddings
    if (options.use_pos) {  // learn POS tag?
      unsigned pos_id = raw_sent.poses.at(token_index);
      Expression p = lookup(*cg, p_p, pos_id);
      args.push_back(GetParamExpr(p_p2l));
      args.push_back(p);
    }
    unsigned raw_word_id = raw_sent.words.at(token_index);
    if (p_t && pretrained.count(raw_word_id)) {  // include pretrained vectors?
      Expression t = const_lookup(*cg, p_t, raw_word_id);
      args.push_back(GetParamExpr(p_t2l));
      args.push_back(t);
    }
    state->buffer[sent.size() - added_to_buffer] = rectify(affine_transform(args));
    state->bufferi[sent.size() - added_to_buffer] = token_index;
    added_to_buffer++;
  }
  // dummy symbol to represent the empty buffer
  state->buffer[0] = parameter(*cg, p_buffer_guard);
  state->bufferi[0] = -999;
  for (auto& b : state->buffer)
    buffer_lstm.add_input(b);

  return state;
}


void LSTMParser::Train(const ParserTrainingCorpus& corpus,
                       const ParserTrainingCorpus& dev_corpus,
                       const double unk_prob, const string& model_fname,
                       const volatile bool* requested_stop) {
  bool softlink_created = false;
  int best_correct_heads = 0;
  unsigned status_every_i_iterations = 100;
  SimpleSGDTrainer sgd(&model);
  //MomentumSGDTrainer sgd(model);
  sgd.eta_decay = 0.08;
  //sgd.eta_decay = 0.05;
  unsigned num_sentences = corpus.sentences.size();
  vector<unsigned> order(corpus.sentences.size());
  for (unsigned i = 0; i < corpus.sentences.size(); ++i)
    order[i] = i;
  double tot_seen = 0;
  status_every_i_iterations = min(status_every_i_iterations, num_sentences);
  cerr << "NUMBER OF TRAINING SENTENCES: " << num_sentences << endl;
  unsigned trs = 0;
  double correct = 0;
  double llh = 0;
  bool first = true;
  int iter = -1;
  time_t time_start = chrono::system_clock::to_time_t(
      chrono::system_clock::now());
  cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z")
       << endl;

  unsigned si = num_sentences;
  while (!requested_stop || !(*requested_stop)) {
    ++iter;
    for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
      if (si == num_sentences) {
        si = 0;
        if (first) {
          first = false;
        } else {
          sgd.update_epoch();
        }
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }
      tot_seen += 1;
      const Sentence& sentence = corpus.sentences[order[si]];
      Sentence::SentenceMap tsentence(sentence.words);
      if (options.unk_strategy == 1) {
        for (auto& index_and_id : tsentence) { // use reference to overwrite
          if (corpus.singletons.count(index_and_id.second)
              && cnn::rand01() < unk_prob) {
            index_and_id.second = vocab.kUNK;
          }
        }
      }
      const vector<unsigned>& actions = corpus.correct_act_sent[order[si]];
      ComputationGraph hg;
      LogProbTagger(&hg, sentence, tsentence, actions, &correct);
      double lp = as_scalar(hg.incremental_forward());
      if (lp < 0) {
        cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp
             << endl;
        assert(lp >= 0.0);
      }
      hg.backward();
      sgd.update(1.0);
      llh += lp;
      ++si;
      trs += actions.size();
    }
    sgd.status();
    time_t time_now = chrono::system_clock::to_time_t(
        chrono::system_clock::now());
    cerr << "update #" << iter << " (epoch " << (tot_seen / num_sentences)
         << " |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "
         << llh << " ppl: " << exp(llh / trs) << " err: "
         << (trs - correct) / trs << endl;
    llh = trs = correct = 0;
    static int logc = 0;
    ++logc;
    if (logc % 25 == 1) {
      // report on dev set
      unsigned dev_size = dev_corpus.sentences.size();
      // dev_size = 100;
      double llh = 0;
      double trs = 0;
      double correct_heads = 0;
      double total_heads = 0;
      auto t_start = chrono::high_resolution_clock::now();
      for (unsigned sii = 0; sii < dev_size; ++sii) {
        const Sentence& sentence = dev_corpus.sentences[sii];

        ParseTree hyp = Parse(sentence, vocab, false);
        llh += hyp.logprob;

        const vector<unsigned>& actions = dev_corpus.correct_act_sent[sii];
        ParseTree ref = RecoverParseTree(sentence, actions);

        trs += actions.size();
        correct_heads += ComputeCorrect(ref, hyp);
        total_heads += sentence.Size() - 1; // -1 to account for ROOT
      }

      auto t_end = chrono::high_resolution_clock::now();
      auto ms = chrono::duration<double, milli>(t_end - t_start).count();
      cerr << "  **dev (iter=" << iter << " epoch="
           << (tot_seen / num_sentences) << ")\tllh=" << llh
           << " ppl: " << exp(llh / trs)
           << " uas: " << (correct_heads / total_heads)
           << "\t[" << dev_size << " sents in " << ms << " ms]" << endl;

      if (correct_heads > best_correct_heads) {
        best_correct_heads = correct_heads;
        SaveModel(model_fname, softlink_created);
        softlink_created = true;
      }
    }
  }
}


ParseTree LSTMParser::Parse(const Sentence& sentence,
                            const CorpusVocabulary& vocab, bool labeled) {
  ComputationGraph cg;
  vector<unsigned> pred = LogProbTagger(sentence, &cg);
  double lp = as_scalar(cg.incremental_forward());
  return RecoverParseTree(sentence, pred, labeled, lp);
}


void LSTMParser::DoTest(const Corpus& corpus, bool evaluate,
                        bool output_parses) {
  if (!output_parses) {
    // Show a message so they know something's happening.
    cerr << "Parsing sentences..." << endl;
  }
  double llh = 0;
  double trs = 0;
  double correct_heads = 0;
  double total_heads = 0;
  auto t_start = chrono::high_resolution_clock::now();
  unsigned corpus_size = corpus.sentences.size();
  for (unsigned sii = 0; sii < corpus_size; ++sii) {
    const Sentence& sentence = corpus.sentences[sii];
    ParseTree hyp = Parse(sentence, vocab, true);
    if (output_parses) {
      OutputConll(sentence, corpus.vocab->int_to_words,
                  corpus.vocab->int_to_pos, corpus.vocab->words_to_int, hyp);
    }

    if (evaluate) {
      // Downcast to ParserTrainingCorpus to get gold-standard data. We can only
      // get here if this function was called by Evaluate, which statically
      // checks that the corpus is in fact a ParserTrainingCorpus, so this cast
      // is safe.
      const ParserTrainingCorpus& training_corpus =
          static_cast<const ParserTrainingCorpus&>(corpus);
      const vector<unsigned>& actions = training_corpus.correct_act_sent[sii];
      ParseTree ref = RecoverParseTree(sentence, actions, true);
      trs += actions.size();
      llh += hyp.logprob;
      correct_heads += ComputeCorrect(ref, hyp);
      total_heads += sentence.Size() - 1; // -1 to account for ROOT
    }
  }

  auto t_end = chrono::high_resolution_clock::now();
  if (evaluate) {
    cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs)
         << " uas: " << (correct_heads / total_heads)
         << "\t[" << corpus_size << " sents in "
         << chrono::duration<double, milli>(t_end - t_start).count() << " ms]"
         << endl;
  } else {
    cerr << "Parsed " << corpus_size << " sentences in "
         << chrono::duration<double, milli>(t_end - t_start).count()
         << " milliseconds." << endl;
  }
}


void LSTMParser::OutputConll(const Sentence& sentence,
                             const vector<string>& int_to_words,
                             const vector<string>& int_to_pos,
                             const map<string, unsigned>& words_to_int,
                             const ParseTree& tree) {
  const unsigned int unk_word = words_to_int.at(CorpusVocabulary::UNK);
  for (const auto& token_index_and_word : sentence.words) {
    unsigned token_index = token_index_and_word.first;
    unsigned word_id = token_index_and_word.second;
    if (token_index == Corpus::ROOT_TOKEN_ID) // don't output anything for ROOT
      continue;

    auto unk_strs_iter = sentence.unk_surface_forms.find(token_index);
    assert(unk_strs_iter != sentence.unk_surface_forms.end() &&
           ((word_id == unk_word && unk_strs_iter->second.size() > 0) ||
            (word_id != unk_word && unk_strs_iter->second.size() == 0 &&
             int_to_words.size() > word_id)));
    string wit = (unk_strs_iter->second.size() > 0) ?
                  unk_strs_iter->second : int_to_words[word_id];
    const string& pos_tag = int_to_pos[sentence.poses.at(token_index)];
    unsigned parent = tree.GetParent(token_index);
    if (parent == Corpus::ROOT_TOKEN_ID)
      parent = 0;
    const string& deprel = tree.GetArcLabel(token_index);
    cout << token_index << '\t' //  1. ID
         << wit << '\t'         //  2. FORM
         << "_" << '\t'         //  3. LEMMA
         << "_" << '\t'         //  4. CPOSTAG
         << pos_tag << '\t'     //  5. POSTAG
         << "_" << '\t'         //  6. FEATS
         << parent << '\t'      //  7. HEAD
         << deprel << '\t'      //  8. DEPREL
         << "_" << '\t'         //  9. PHEAD
         << "_" << endl;        // 10. PDEPREL
  }
  cout << endl;
}


} // namespace lstm_parser
