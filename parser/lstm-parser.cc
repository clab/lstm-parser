#include "lstm-parser.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <cassert>
#include <chrono>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cnn/model.h"
#include "cnn/tensor.h"

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace io = boost::iostreams;

namespace lstm_parser {


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
    for (auto it : pretrained)
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
    FinalizeVocab();
  }
  finalized = finalize;
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
    const vector<unsigned>& sentence, const vector<unsigned>& actions,
    const vector<string>& action_names,
    const vector<string>& actions_to_arc_labels, bool labeled) {
  ParseTree tree(sentence, labeled);
  vector<int> bufferi(sentence.size() + 1, 0), stacki(1, -999);
  for (unsigned i = 0; i < sentence.size(); ++i)
    bufferi[sentence.size() - i] = i;
  bufferi[0] = -999;
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
      unsigned ii = 0, jj = 0;
      jj = stacki.back();
      stacki.pop_back();
      ii = stacki.back();
      stacki.pop_back();
      bufferi.push_back(ii);
      stacki.push_back(jj);
    } else { // LEFT or RIGHT
      assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
      assert(ac == 'L' || ac == 'R');
      unsigned depi = 0, headi = 0;
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
    const vector<unsigned>& raw_sent,  // raw sentence
    const vector<unsigned>& sent,  // sentence with OOVs replaced
    const vector<unsigned>& sent_pos, const vector<unsigned>& correct_actions,
    const vector<string>& action_names, const vector<string>& int_to_words,
    double* correct) {
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

  for (unsigned i = 0; i < sent.size(); ++i) {
    assert(sent[i] < vocab.CountWords());
    Expression w = lookup(*hg, p_w, sent[i]);

    vector<Expression> args = {ib, w2l, w}; // learn embeddings
    if (options.use_pos) { // learn POS tag?
      Expression p = lookup(*hg, p_p, sent_pos[i]);
      args.push_back(p2l);
      args.push_back(p);
    }
    if (p_t && pretrained.count(raw_sent[i])) { // include pretrained vectors?
      Expression t = const_lookup(*hg, p_t, raw_sent[i]);
      args.push_back(t2l);
      args.push_back(t);
    }
    buffer[sent.size() - i] = rectify(affine_transform(args));
    bufferi[sent.size() - i] = i;
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
  string rootword;
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
      if (best_a == action) {
        (*correct)++;
      }
    }
    ++action_count;
    log_probs.push_back(pick(adiste, action));
    results.push_back(action);

    // add current action to action LSTM
    Expression actione = lookup(*hg, p_a, action);
    action_lstm.add_input(actione);

    // get relation embedding from action (TODO: convert to rel from action?)
    Expression relation = lookup(*hg, p_r, action);

    // do action
    const string& actionString = action_names[action];
    const char ac = actionString[0];
    const char ac2 = actionString[1];

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
      if (headi == sent.size() - 1)
        rootword = int_to_words[sent[depi]];
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


void LSTMParser::SaveModel(const string& model_fname, bool compress,
                           bool softlink_created) {
  ofstream out_file(model_fname);
  if (compress) {
    io::filtering_streambuf<io::output> filter;
    filter.push(io::gzip_compressor());
    filter.push(out_file);
    boost::archive::binary_oarchive oa(filter);
    oa << *this;
  } else {
    boost::archive::text_oarchive oa(out_file);
    oa << *this;
  }
  cerr << "Model saved." << endl;
  // Create a soft link to the most recent model in order to make it
  // easier to refer to it in a shell script.
  if (!softlink_created) {
    string softlink = "latest_model.params";
    if (compress)
      softlink += ".gz";

    if (system((string("rm -f ") + softlink).c_str()) == 0
        && system(("ln -s " + model_fname + " " + softlink).c_str()) == 0) {
      cerr << "Created " << softlink << " as a soft link to " << model_fname
           << " for convenience." << endl;
    }
  }
}


void LSTMParser::Train(const TrainingCorpus& corpus,
                       const TrainingCorpus& dev_corpus, const double unk_prob,
                       const string& model_fname, bool compress,
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
      const vector<unsigned>& sentence = corpus.sentences[order[si]];
      vector<unsigned> tsentence(sentence);
      if (options.unk_strategy == 1) {
        for (auto& w : tsentence) { // use reference to overwrite
          if (corpus.singletons.count(w) && cnn::rand01() < unk_prob) {
            w = kUNK;
          }
        }
      }
      const vector<unsigned>& sentence_pos = corpus.sentences_pos[order[si]];
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
        const vector<unsigned>& sentence = dev_corpus.sentences[sii];
        const vector<unsigned>& sentence_pos = dev_corpus.sentences_pos[sii];
        const vector<unsigned>& actions = dev_corpus.correct_act_sent[sii];
        ParseTree hyp = Parse(sentence, sentence_pos, vocab, false, &correct);

        double lp = 0;
        llh -= lp;
        trs += actions.size();
        ParseTree ref = RecoverParseTree(
            sentence, actions, dev_corpus.vocab->actions,
            dev_corpus.vocab->actions_to_arc_labels);
        correct_heads += ComputeCorrect(ref, hyp);
        total_heads += sentence.size() - 1;
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
        SaveModel(model_fname, compress, softlink_created);
        softlink_created = true;
      }
    }
  }
}


vector<unsigned> LSTMParser::LogProbParser(
    const vector<unsigned>& sentence, const vector<unsigned>& sentence_pos,
    const CorpusVocabulary& vocab, ComputationGraph *cg,
    double* correct) {
  vector<unsigned> tsentence(sentence); // sentence with OOVs replaced
  for (unsigned& word_id : tsentence) { // use reference to overwrite
    if (vocab.int_to_training_word[word_id]) {
      word_id = kUNK;
    }
  }
  return LogProbParser(cg, sentence, tsentence, sentence_pos,
                       vector<unsigned>(), vocab.actions,
                       vocab.int_to_words, correct);
}


ParseTree LSTMParser::Parse(const vector<unsigned>& sentence,
                const vector<unsigned>& sentence_pos,
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
    const vector<unsigned>& sentence = corpus.sentences[sii];
    const vector<unsigned>& sentence_pos = corpus.sentences_pos[sii];
    const vector<string>& sentence_unk_str =
        corpus.sentences_unk_surface_forms[sii];
    ParseTree hyp = Parse(sentence, sentence_pos, vocab, true, &correct);
    if (output_parses) {
      OutputConll(sentence, sentence_pos, sentence_unk_str,
                  corpus.vocab->int_to_words, corpus.vocab->int_to_pos,
                  corpus.vocab->words_to_int, hyp);
    }

    if (evaluate) {
      // Downcast to TrainingCorpus to get gold-standard data. We can only get
      // here if this function was called by Evaluate, which statically checks
      // that the corpus is in fact a TrainingCorpus, so this cast is safe.
      const TrainingCorpus& training_corpus =
          static_cast<const TrainingCorpus&>(corpus);
      const vector<unsigned>& actions = training_corpus.correct_act_sent[sii];
      ParseTree ref = RecoverParseTree(sentence, actions, corpus.vocab->actions,
                                       corpus.vocab->actions_to_arc_labels,
                                       true);
      trs += actions.size();
      correct_heads += ComputeCorrect(ref, hyp);
      total_heads += sentence.size() - 1;
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


void LSTMParser::OutputConll(const vector<unsigned>& sentence,
                             const vector<unsigned>& pos,
                             const vector<string>& sentence_unk_strings,
                             const vector<string>& int_to_words,
                             const vector<string>& int_to_pos,
                             const map<string, unsigned>& words_to_int,
                             const ParseTree& tree) {
  const unsigned int unk_word =
      words_to_int.find(CorpusVocabulary::UNK)->second;
  for (unsigned i = 0; i < (sentence.size() - 1); ++i) {
    auto index = i + 1;
    assert(i < sentence_unk_strings.size() &&
           ((sentence[i] == unk_word && sentence_unk_strings[i].size() > 0) ||
            (sentence[i] != unk_word && sentence_unk_strings[i].size() == 0 &&
             int_to_words.size() > sentence[i])));
    string wit = (sentence_unk_strings[i].size() > 0) ?
                  sentence_unk_strings[i] : int_to_words[sentence[i]];
    const string& pos_tag = int_to_pos[pos[i]];
    int parent = tree.GetParents()[i] + 1;
    if (parent == (int) sentence.size())
      parent = 0;
    const string& deprel = tree.GetArcLabels()[i];
    cout << index << '\t'       // 1. ID
         << wit << '\t'         // 2. FORM
         << "_" << '\t'         // 3. LEMMA
         << "_" << '\t'         // 4. CPOSTAG
         << pos_tag << '\t'     // 5. POSTAG
         << "_" << '\t'         // 6. FEATS
         << parent << '\t'    // 7. HEAD
         << deprel << '\t'     // 8. DEPREL
         << "_" << '\t'         // 9. PHEAD
         << "_" << endl;        // 10. PDEPREL
  }
  cout << endl;
}


} // namespace lstm_parser
