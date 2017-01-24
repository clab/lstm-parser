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
#include "eos/portable_archive.hpp"


using namespace cnn::expr;
using namespace cnn;
using namespace std;

namespace lstm_parser {


string ParseTree::NO_LABEL = "ERROR";


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


void LSTMParser::FinalizeVocab() {
  if (finalized)
    return;

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

  finalized = true;
}


LSTMParser::LSTMParser(const ParserOptions& poptions,
                       const string& pretrained_words_path, bool finalize) :
      options(poptions),
      kUNK(vocab.GetOrAddWord(vocab.UNK)),
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


bool LSTMParser::IsActionForbidden(const string& a, unsigned bsize,
                                   unsigned ssize, const vector<int>& stacki) {
  if (a[1] == 'W' && ssize < 3)
    return true;
  if (a[1] == 'W') {
    int top = stacki[stacki.size() - 1];
    int sec = stacki[stacki.size() - 2];
    if (sec > top)
      return true;
  }

  bool is_shift = (a[0] == 'S' && a[1] == 'H');
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
  if (bsize == 1 && ssize == 3 && a[0] == 'R')
    return true;
  return false;
}


ParseTree LSTMParser::RecoverParseTree(
    const map<unsigned, unsigned>& sentence, const vector<unsigned>& actions,
    const vector<string>& action_names,
    const vector<string>& actions_to_arc_labels, bool labeled) {
  ParseTree tree(sentence, labeled);
  vector<int> bufferi(sentence.size() + 1);
  bufferi[0] = -999;
  vector<int> stacki(1, -999);
  unsigned added_to_buffer = 0;
  for (const auto& index_and_word_id : sentence) {
    // ROOT is set to -1, so it'll come last in a sequence of unsigned ints.
    bufferi[sentence.size() - added_to_buffer++] = index_and_word_id.first;
  }
  for (auto action : actions) { // loop over transitions for sentence
    const string& action_string = action_names[action];
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
      tree.SetParent(depi, headi, actions_to_arc_labels[action]);
    }
  }
  assert(bufferi.size() == 1);
  //assert(stacki.size() == 2);
  return tree;
}


vector<unsigned> LSTMParser::LogProbParser(
    ComputationGraph* hg,
    const map<unsigned, unsigned>& raw_sent,  // raw sentence
    const map<unsigned, unsigned>& sent,  // sentence with OOVs replaced
    const map<unsigned, unsigned>& sent_pos,
    const vector<unsigned>& correct_actions, const vector<string>& action_names,
    const vector<string>& int_to_words, double* correct) {
  // TODO: break up this function?
  assert(finalized);
  vector<unsigned> results;
  const bool build_training_graph = correct_actions.size() > 0;

  stack_lstm.new_graph(*hg);
  buffer_lstm.new_graph(*hg);
  action_lstm.new_graph(*hg);
  stack_lstm.start_new_sequence();
  buffer_lstm.start_new_sequence();
  action_lstm.start_new_sequence();
  // variables in the computation graph representing the parameters
  Expression pbias = parameter(*hg, p_pbias);
  Expression H = parameter(*hg, p_H);
  Expression D = parameter(*hg, p_D);
  Expression R = parameter(*hg, p_R);
  Expression cbias = parameter(*hg, p_cbias);
  Expression S = parameter(*hg, p_S);
  Expression B = parameter(*hg, p_B);
  Expression A = parameter(*hg, p_A);
  Expression ib = parameter(*hg, p_ib);
  Expression w2l = parameter(*hg, p_w2l);
  Expression p2l;
  if (options.use_pos)
    p2l = parameter(*hg, p_p2l);
  Expression t2l;
  if (p_t2l)
    t2l = parameter(*hg, p_t2l);
  Expression p2a = parameter(*hg, p_p2a);
  Expression abias = parameter(*hg, p_abias);
  Expression action_start = parameter(*hg, p_action_start);

  action_lstm.add_input(action_start);

  // variables representing word embeddings (possibly including POS info)
  vector<Expression> buffer(sent.size() + 1);
  vector<int> bufferi(sent.size() + 1); // position of the words in the sentence
  // precompute buffer representation from left to right

  unsigned added_to_buffer = 0;
  for (const auto& index_and_word_id : sent) {
    unsigned token_index = index_and_word_id.first;
    unsigned word_id = index_and_word_id.second;

    assert(word_id < vocab.CountWords());
    Expression w = lookup(*hg, p_w, word_id);

    vector<Expression> args = {ib, w2l, w}; // learn embeddings
    if (options.use_pos) { // learn POS tag?
      unsigned pos_id = sent_pos.find(token_index)->second;
      Expression p = lookup(*hg, p_p, pos_id);
      args.push_back(p2l);
      args.push_back(p);
    }
    unsigned raw_word_id = raw_sent.find(token_index)->second;
    if (p_t && pretrained.count(raw_word_id)) { // include pretrained vectors?
      Expression t = const_lookup(*hg, p_t, raw_word_id);
      args.push_back(t2l);
      args.push_back(t);
    }
    buffer[sent.size() - added_to_buffer] = rectify(affine_transform(args));
    bufferi[sent.size() - added_to_buffer] = token_index;
    added_to_buffer++;
  }
  // dummy symbol to represent the empty buffer
  buffer[0] = parameter(*hg, p_buffer_guard);
  bufferi[0] = -999;
  for (auto& b : buffer)
    buffer_lstm.add_input(b);

  vector<Expression> stack;  // variables representing subtree embeddings
  vector<int> stacki; // position of words in the sentence of head of subtree
  stack.push_back(parameter(*hg, p_stack_guard));
  stacki.push_back(-999); // not used for anything
  // drive dummy symbol on stack through LSTM
  stack_lstm.add_input(stack.back());
  vector<Expression> log_probs;
  unsigned action_count = 0;  // incremented at each prediction
  while (stack.size() > 2 || buffer.size() > 1) {
    // get list of possible actions for the current parser state
    vector<unsigned> current_valid_actions;
    for (unsigned action = 0; action < n_possible_actions; ++action) {
      if (IsActionForbidden(action_names[action], buffer.size(), stack.size(),
                            stacki))
        continue;
      current_valid_actions.push_back(action);
    }

    // p_t = pbias + S * slstm + B * blstm + A * almst
    Expression p_t = affine_transform(
        {pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A,
         action_lstm.back()});
    Expression nlp_t = rectify(p_t);
    // r_t = abias + p2a * nlp
    Expression r_t = affine_transform({abias, p2a, nlp_t});

    // adist = log_softmax(r_t, current_valid_actions)
    Expression adiste = log_softmax(r_t, current_valid_actions);
    vector<float> adist = as_vector(hg->incremental_forward());
    double best_score = adist[current_valid_actions[0]];
    unsigned best_a = current_valid_actions[0];
    for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
      if (adist[current_valid_actions[i]] > best_score) {
        best_score = adist[current_valid_actions[i]];
        best_a = current_valid_actions[i];
      }
    }
    unsigned action = best_a;
    // If we have reference actions (for training), use the reference action.
    if (build_training_graph) {
      action = correct_actions[action_count];
      if (correct && best_a == action) {
        (*correct)++;
      }
    }
    ++action_count;
    log_probs.push_back(pick(adiste, action));
    results.push_back(action);

    // add current action to action LSTM
    Expression action_e = lookup(*hg, p_a, action);
    action_lstm.add_input(action_e);

    // get relation embedding from action (TODO: convert to rel from action?)
    Expression relation = lookup(*hg, p_r, action);

    // do action
    const string& action_string = action_names[action];
    const char ac = action_string[0];
    const char ac2 = action_string[1];

    if (ac == 'S' && ac2 == 'H') {  // SHIFT
      assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
      stack.push_back(buffer.back());
      stack_lstm.add_input(buffer.back());
      buffer.pop_back();
      buffer_lstm.rewind_one_step();
      stacki.push_back(bufferi.back());
      bufferi.pop_back();
    } else if (ac == 'S' && ac2 == 'W') { //SWAP --- Miguel
      assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)

      Expression toki, tokj;
      unsigned ii = 0, jj = 0;
      tokj = stack.back();
      jj = stacki.back();
      stack.pop_back();
      stacki.pop_back();

      toki = stack.back();
      ii = stacki.back();
      stack.pop_back();
      stacki.pop_back();

      buffer.push_back(toki);
      bufferi.push_back(ii);

      stack_lstm.rewind_one_step();
      stack_lstm.rewind_one_step();

      buffer_lstm.add_input(buffer.back());

      stack.push_back(tokj);
      stacki.push_back(jj);

      stack_lstm.add_input(stack.back());
    } else { // LEFT or RIGHT
      assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
      assert(ac == 'L' || ac == 'R');
      Expression dep, head;
      unsigned depi = 0, headi = 0;
      (ac == 'R' ? dep : head) = stack.back();
      (ac == 'R' ? depi : headi) = stacki.back();
      stack.pop_back();
      stacki.pop_back();
      (ac == 'R' ? head : dep) = stack.back();
      (ac == 'R' ? headi : depi) = stacki.back();
      stack.pop_back();
      stacki.pop_back();
      // composed = cbias + H * head + D * dep + R * relation
      Expression composed = affine_transform({cbias, H, head, D, dep, R,
                                              relation});
      Expression nlcomposed = tanh(composed);
      stack_lstm.rewind_one_step();
      stack_lstm.rewind_one_step();
      stack_lstm.add_input(nlcomposed);
      stack.push_back(nlcomposed);
      stacki.push_back(headi);
    }
  }
  assert(stack.size() == 2); // guard symbol, root
  assert(stacki.size() == 2);
  assert(buffer.size() == 1); // guard symbol
  assert(bufferi.size() == 1);
  Expression tot_neglogprob = -sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
  return results;
}


void LSTMParser::SaveModel(const string& model_fname, bool softlink_created) {
  ofstream out_file(model_fname);
  eos::portable_oarchive archive(out_file);
  archive << *this;
  cerr << "Model saved." << endl;
  // Create a soft link to the most recent model in order to make it
  // easier to refer to it in a shell script.
  if (!softlink_created) {
    string softlink = "latest_model.params";

    if (system((string("rm -f ") + softlink).c_str()) == 0
        && system(("ln -s " + model_fname + " " + softlink).c_str()) == 0) {
      cerr << "Created " << softlink << " as a soft link to " << model_fname
           << " for convenience." << endl;
    }
  }
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
      const map<unsigned, unsigned>& sentence = corpus.sentences[order[si]];
      map<unsigned, unsigned> tsentence(sentence);
      if (options.unk_strategy == 1) {
        for (auto& index_and_id : tsentence) { // use reference to overwrite
          if (corpus.singletons.count(index_and_id.second)
              && cnn::rand01() < unk_prob) {
            index_and_id.second = kUNK;
          }
        }
      }
      const map<unsigned, unsigned>& sentence_pos =
          corpus.sentences_pos[order[si]];
      const vector<unsigned>& actions = corpus.correct_act_sent[order[si]];
      ComputationGraph hg;
      LogProbParser(&hg, sentence, tsentence, sentence_pos, actions,
                    corpus.vocab->actions, corpus.vocab->int_to_words,
                    &correct);
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
      double correct = 0;
      double correct_heads = 0;
      double total_heads = 0;
      auto t_start = chrono::high_resolution_clock::now();
      for (unsigned sii = 0; sii < dev_size; ++sii) {
        const map<unsigned, unsigned>& sentence = dev_corpus.sentences[sii];
        const map<unsigned, unsigned>& sentence_pos =
            dev_corpus.sentences_pos[sii];
        ParseTree hyp = Parse(sentence, sentence_pos, vocab, false, &correct);

        double lp = 0;
        llh -= lp;
        const vector<unsigned>& actions = dev_corpus.correct_act_sent[sii];
        ParseTree ref = RecoverParseTree(
            sentence, actions, dev_corpus.vocab->actions,
            dev_corpus.vocab->actions_to_arc_labels);

        trs += actions.size();
        correct_heads += ComputeCorrect(ref, hyp);
        total_heads += sentence.size() - 1; // -1 to account for ROOT
      }
      auto t_end = chrono::high_resolution_clock::now();
      cerr << "  **dev (iter=" << iter << " epoch="
           << (tot_seen / num_sentences) << ")\tllh=" << llh << " ppl: "
           << exp(llh / trs) << " err: " << (trs - correct) / trs << " uas: "
           << (correct_heads / total_heads) << "\t[" << dev_size << " sents in "
           << chrono::duration<double, milli>(t_end - t_start).count() << " ms]"
           << endl;
      if (correct_heads > best_correct_heads) {
        best_correct_heads = correct_heads;
        SaveModel(model_fname, softlink_created);
        softlink_created = true;
      }
    }
  }
}


vector<unsigned> LSTMParser::LogProbParser(
    const map<unsigned, unsigned>& sentence,
    const map<unsigned, unsigned>& sentence_pos, const CorpusVocabulary& vocab,
    ComputationGraph *cg, double* correct) {
  map<unsigned, unsigned> tsentence(sentence); // sentence with OOVs replaced
  for (auto& index_and_id : tsentence) { // use reference to overwrite
    if (!vocab.int_to_training_word[index_and_id.second]) {
      index_and_id.second = kUNK;
    }
  }
  return LogProbParser(cg, sentence, tsentence, sentence_pos,
                       vector<unsigned>(), vocab.actions,
                       vocab.int_to_words, correct);
}


ParseTree LSTMParser::Parse(const map<unsigned, unsigned>& sentence,
                const map<unsigned, unsigned>& sentence_pos,
                const CorpusVocabulary& vocab,
                bool labeled, double* correct) {
  ComputationGraph cg;
  vector<unsigned> pred = LogProbParser(sentence, sentence_pos, vocab, &cg,
                                        correct);
  return RecoverParseTree(sentence, pred, vocab.actions,
                          vocab.actions_to_arc_labels, labeled);
}


void LSTMParser::DoTest(const Corpus& corpus, bool evaluate,
                        bool output_parses) {
  if (!output_parses) {
    // Show a message so they know something's happening.
    cerr << "Parsing sentences..." << endl;
  }
  double llh = 0;
  double trs = 0;
  double correct = 0;
  double correct_heads = 0;
  double total_heads = 0;
  auto t_start = chrono::high_resolution_clock::now();
  unsigned corpus_size = corpus.sentences.size();
  for (unsigned sii = 0; sii < corpus_size; ++sii) {
    const map<unsigned, unsigned>& sentence = corpus.sentences[sii];
    const map<unsigned, unsigned>& sentence_pos = corpus.sentences_pos[sii];
    const map<unsigned, string>& sentence_unk_str =
        corpus.sentences_unk_surface_forms[sii];
    ParseTree hyp = Parse(sentence, sentence_pos, vocab, true, &correct);
    if (output_parses) {
      OutputConll(sentence, sentence_pos, sentence_unk_str,
                  corpus.vocab->int_to_words, corpus.vocab->int_to_pos,
                  corpus.vocab->words_to_int, hyp);
    }

    if (evaluate) {
      // Downcast to ParserTrainingCorpus to get gold-standard data. We can only
      // get here if this function was called by Evaluate, which statically
      // checks that the corpus is in fact a ParserTrainingCorpus, so this cast
      // is safe.
      const ParserTrainingCorpus& training_corpus =
          static_cast<const ParserTrainingCorpus&>(corpus);
      const vector<unsigned>& actions = training_corpus.correct_act_sent[sii];
      ParseTree ref = RecoverParseTree(sentence, actions, corpus.vocab->actions,
                                       corpus.vocab->actions_to_arc_labels,
                                       true);
      trs += actions.size();
      correct_heads += ComputeCorrect(ref, hyp);
      total_heads += sentence.size() - 1; // -1 to account for ROOT
    }
  }
  auto t_end = chrono::high_resolution_clock::now();
  if (evaluate) {
    cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs) << " err: "
         << (trs - correct) / trs << " uas: " << (correct_heads / total_heads)
         << "\t[" << corpus_size << " sents in "
         << chrono::duration<double, milli>(t_end - t_start).count() << " ms]"
         << endl;
  } else {
    cerr << "Parsed " << corpus_size << " sentences in "
         << chrono::duration<double, milli>(t_end - t_start).count()
         << " milliseconds." << endl;
  }
}


void LSTMParser::OutputConll(const map<unsigned, unsigned>& sentence,
                             const map<unsigned, unsigned>& pos,
                             const map<unsigned, string>& sentence_unk_strings,
                             const vector<string>& int_to_words,
                             const vector<string>& int_to_pos,
                             const map<string, unsigned>& words_to_int,
                             const ParseTree& tree) {
  const unsigned int unk_word =
      words_to_int.find(CorpusVocabulary::UNK)->second;
  for (const auto& token_index_and_word : sentence) {
    unsigned token_index = token_index_and_word.first;
    unsigned word_id = token_index_and_word.second;
    if (token_index == Corpus::ROOT_TOKEN_ID) // don't output anything for ROOT
      continue;

    auto unk_strs_iter = sentence_unk_strings.find(token_index);
    assert(unk_strs_iter != sentence_unk_strings.end() &&
           ((word_id == unk_word && unk_strs_iter->second.size() > 0) ||
            (word_id != unk_word && unk_strs_iter->second.size() == 0 &&
             int_to_words.size() > word_id)));
    string wit = (unk_strs_iter->second.size() > 0) ?
                  unk_strs_iter->second : int_to_words[word_id];
    const string& pos_tag = int_to_pos[pos.find(token_index)->second];
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
