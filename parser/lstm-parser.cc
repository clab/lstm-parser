#include "lstm-parser.h"

#include <cassert>
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

namespace lstm_parser {

constexpr const char* ParserBuilder::ROOT_SYMBOL;

void ParserBuilder::FinalizeVocab() {
  // assert (!finalized);
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

ParserBuilder::ParserBuilder(const string& pretrained_words_path,
                             const ParserOptions& poptions, bool finalize) :
      options(poptions),
      kUNK(vocab.GetOrAddWord(vocab.UNK)),
      kROOT_SYMBOL(vocab.GetOrAddWord(ROOT_SYMBOL)),
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


bool ParserBuilder::IsActionForbidden(const string& a, unsigned bsize,
                                      unsigned ssize,
                                      const vector<int>& stacki) {
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


map<int, int> ParserBuilder::ComputeHeads(unsigned sent_len,
                                          const vector<unsigned>& actions,
                                          const vector<string>& setOfActions,
                                          map<int, string>* pr) {
  map<int, int> heads;
  map<int, string> r;
  map<int, string>& rels = (pr ? *pr : r);
  for (unsigned i = 0; i < sent_len; i++) {
    heads[i] = -1;
    rels[i] = "ERROR";
  }
  vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
  for (unsigned i = 0; i < sent_len; ++i)
    bufferi[sent_len - i] = i;
  bufferi[0] = -999;
  for (auto action : actions) { // loop over transitions for sentence
    const string& actionString = setOfActions[action];
    const char ac = actionString[0];
    const char ac2 = actionString[1];
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
      heads[depi] = headi;
      rels[depi] = actionString;
    }
  }
  assert(bufferi.size() == 1);
  //assert(stacki.size() == 2);
  return heads;
}


vector<unsigned> ParserBuilder::LogProbParser(
    ComputationGraph* hg,
    const vector<unsigned>& raw_sent,  // raw sentence
    const vector<unsigned>& sent,  // sentence with OOVs replaced
    const vector<unsigned>& sentPos, const vector<unsigned>& correct_actions,
    const vector<string>& setOfActions, const vector<std::string>& intToWords,
    double* right) {
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
      Expression p = lookup(*hg, p_p, sentPos[i]);
      args.push_back(p2l);
      args.push_back(p);
    }
    if (p_t && pretrained.count(raw_sent[i])) { // include fixed pretrained vectors?
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
      if (IsActionForbidden(setOfActions[action], buffer.size(), stack.size(),
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
    if (build_training_graph) { // if we have reference actions (for training) use the reference action
      action = correct_actions[action_count];
      if (best_a == action) {
        (*right)++;
      }
    }
    ++action_count;
    log_probs.push_back(pick(adiste, action));
    results.push_back(action);

    // add current action to action LSTM
    Expression actione = lookup(*hg, p_a, action);
    action_lstm.add_input(actione);

    // get relation embedding from action (TODO: convert to relation from action?)
    Expression relation = lookup(*hg, p_r, action);

    // do action
    const string& actionString = setOfActions[action];
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
        rootword = intToWords[sent[depi]];
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

} // namespace lstm_parser
