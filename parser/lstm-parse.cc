#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "c2.h"

cpyp::Corpus corpus;
volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;


bool USE_POS = false;

constexpr const char* ROOT_SYMBOL = "ROOT";
unsigned kROOT_SYMBOL = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned POS_SIZE = 0;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {

  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder buffer_lstm;
  LSTMBuilder action_lstm;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_r; // relation embeddings
  LookupParameters* p_p; // pos tag embeddings
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_H; // head matrix for composition function
  Parameters* p_D; // dependency matrix for composition function
  Parameters* p_R; // relation matrix for composition function
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_p2l; // POS to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      buffer_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_r(model->add_lookup_parameters(ACTION_SIZE, {REL_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_H(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_D(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_R(model->add_parameters({LSTM_INPUT_DIM, REL_DIM})),
      p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model->add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),
      p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})) {
    if (USE_POS) {
      p_p = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
      p_p2l = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
    }
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
      for (auto it : pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
    }
  }

static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, const vector<int>& stacki) {
  if (a[1]=='W' && ssize<3) return true;
  if (a[1]=='W') {
        int top=stacki[stacki.size()-1];
        int sec=stacki[stacki.size()-2];
        if (sec>top) return true;
  }

  bool is_shift = (a[0] == 'S' && a[1]=='H');
  bool is_reduce = !is_shift;
  if (is_shift && bsize == 1) return true;
  if (is_reduce && ssize < 3) return true;
  if (bsize == 2 && // ROOT is the only thing remaining on buffer
      ssize > 2 && // there is more than a single element on the stack
      is_shift) return true;
  // only attach left to ROOT
  if (bsize == 1 && ssize == 3 && a[0] == 'R') return true;
  return false;
}

// take a vector of actions and return a parse tree (labeling of every
// word position with its head's position)
static map<int,int> compute_heads(unsigned sent_len, const vector<unsigned>& actions, const vector<string>& setOfActions, map<int,string>* pr = nullptr) {
  map<int,int> heads;
  map<int,string> r;
  map<int,string>& rels = (pr ? *pr : r);
  for(unsigned i=0;i<sent_len;i++) { heads[i]=-1; rels[i]="ERROR"; }
  vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
  for (unsigned i = 0; i < sent_len; ++i)
    bufferi[sent_len - i] = i;
  bufferi[0] = -999;
  for (auto action: actions) { // loop over transitions for sentence
    const string& actionString=setOfActions[action];
    const char ac = actionString[0];
    const char ac2 = actionString[1];
    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
      stacki.push_back(bufferi.back());
      bufferi.pop_back();
    } else if (ac=='S' && ac2=='W') { // SWAP
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

// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const vector<unsigned>& raw_sent,  // raw sentence
                     const vector<unsigned>& sent,  // sent with oovs replaced
                     const vector<unsigned>& sentPos,
                     const vector<unsigned>& correct_actions,
                     const vector<string>& setOfActions,
                     const map<unsigned, std::string>& intToWords,
                     double *right) {
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
    if (USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < VOCAB_SIZE);
      Expression w =lookup(*hg, p_w, sent[i]);

      vector<Expression> args = {ib, w2l, w}; // learn embeddings
      if (USE_POS) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sentPos[i]);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (p_t && pretrained.count(raw_sent[i])) {  // include fixed pretrained vectors?
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
    while(stack.size() > 2 || buffer.size() > 1) {
      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a: possible_actions) {
        if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki))
          continue;
        current_valid_actions.push_back(a);
      }

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
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
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        action = correct_actions[action_count];
        if (best_a == action) { (*right)++; }
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
      const string& actionString=setOfActions[action];
      const char ac = actionString[0];
      const char ac2 = actionString[1];


      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
        stack.push_back(buffer.back());
        stack_lstm.add_input(buffer.back());
        buffer.pop_back();
        buffer_lstm.rewind_one_step();
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
      } else if (ac=='S' && ac2=='W'){ //SWAP --- Miguel
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)

        Expression toki, tokj;
        unsigned ii = 0, jj = 0;
        tokj=stack.back();
        jj=stacki.back();
        stack.pop_back();
        stacki.pop_back();

        toki=stack.back();
        ii=stacki.back();
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
        if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
        // composed = cbias + H * head + D * dep + R * relation
        Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
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
};

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

unsigned compute_correct(const map<int,int>& ref, const map<int,int>& hyp, unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second) ++res;
  }
  return res;
}

void output_conll(const vector<unsigned>& sentence, const vector<unsigned>& pos,
                  const vector<string>& sentenceUnkStrings, 
                  const map<unsigned, string>& intToWords, 
                  const map<unsigned, string>& intToPos, 
                  const map<int,int>& hyp, const map<int,string>& rel_hyp) {
  for (unsigned i = 0; i < (sentence.size()-1); ++i) {
    auto index = i + 1;
    assert(i < sentenceUnkStrings.size() && 
           ((sentence[i] == corpus.get_or_add_word(cpyp::Corpus::UNK) &&
             sentenceUnkStrings[i].size() > 0) ||
            (sentence[i] != corpus.get_or_add_word(cpyp::Corpus::UNK) &&
             sentenceUnkStrings[i].size() == 0 &&
             intToWords.find(sentence[i]) != intToWords.end())));
    string wit = (sentenceUnkStrings[i].size() > 0)? 
      sentenceUnkStrings[i] : intToWords.find(sentence[i])->second;
    auto pit = intToPos.find(pos[i]);
    assert(hyp.find(i) != hyp.end());
    auto hyp_head = hyp.find(i)->second + 1;
    if (hyp_head == (int)sentence.size()) hyp_head = 0;
    auto hyp_rel_it = rel_hyp.find(i);
    assert(hyp_rel_it != rel_hyp.end());
    auto hyp_rel = hyp_rel_it->second;
    size_t first_char_in_rel = hyp_rel.find('(') + 1;
    size_t last_char_in_rel = hyp_rel.rfind(')') - 1;
    hyp_rel = hyp_rel.substr(first_char_in_rel, last_char_in_rel - first_char_in_rel + 1);
    cout << index << '\t'       // 1. ID 
         << wit << '\t'         // 2. FORM
         << "_" << '\t'         // 3. LEMMA 
         << "_" << '\t'         // 4. CPOSTAG 
         << pit->second << '\t' // 5. POSTAG
         << "_" << '\t'         // 6. FEATS
         << hyp_head << '\t'    // 7. HEAD
         << hyp_rel << '\t'     // 8. DEPREL
         << "_" << '\t'         // 9. PHEAD
         << "_" << endl;        // 10. PDEPREL
  }
  cout << endl;
}


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  USE_POS = conf.count("use_pos_tags");

  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  REL_DIM = conf["rel_dim"].as<unsigned>();
  const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  cerr << "Unknown word strategy: ";
  if (unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  const double unk_prob = conf["unk_prob"].as<double>();
  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  ostringstream os;
  os << "parser_" << (USE_POS ? "pos" : "nopos")
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << '_' << POS_DIM
     << '_' << REL_DIM
     << "-pid" << getpid() << ".params";
  int best_correct_heads = 0;
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;
  bool softlinkCreated = false;
  corpus.load_correct_actions(conf["training_data"].as<string>());	
  const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
  kROOT_SYMBOL = corpus.get_or_add_word(ROOT_SYMBOL);

  if (conf.count("words")) {
    pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
    cerr << "Loading from " << conf["words"].as<string>() << " with" << PRETRAINED_DIM << " dimensions\n";
    ifstream in(conf["words"].as<string>().c_str());
    string line;
    getline(in, line);
    vector<float> v(PRETRAINED_DIM, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
      unsigned id = corpus.get_or_add_word(word);
      pretrained[id] = v;
    }
  }

  set<unsigned> training_vocab; // words available in the training corpus
  set<unsigned> singletons;
  {  // compute the singletons in the parser's training data
    map<unsigned, unsigned> counts;
    for (auto sent : corpus.sentences)
      for (auto word : sent.second) { training_vocab.insert(word); counts[word]++; }
    for (auto wc : counts)
      if (wc.second == 1) singletons.insert(wc.first);
  }

  cerr << "Number of words: " << corpus.nwords << endl;
  VOCAB_SIZE = corpus.nwords + 1;
  ACTION_SIZE = corpus.nactions + 1;
  POS_SIZE = corpus.npos + 10;  // bad way of dealing with the fact that we may see new POS tags in the test set
  possible_actions.resize(corpus.nactions);
  for (unsigned i = 0; i < corpus.nactions; ++i)
    possible_actions[i] = i;

  Model model;
  ParserBuilder parser(&model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  // OOV words will be replaced by UNK tokens
  corpus.load_correct_actionsDev(conf["dev_data"].as<string>());
  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.nsentences);
    for (unsigned i = 0; i < corpus.nsentences; ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.nsentences);
    unsigned si = corpus.nsentences;
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.nsentences << endl;
    unsigned trs = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    time_t time_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.nsentences) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           const vector<unsigned>& sentence=corpus.sentences[order[si]];
           vector<unsigned> tsentence=sentence;
           if (unk_strategy == 1) {
             for (auto& w : tsentence)
               if (singletons.count(w) && cnn::rand01() < unk_prob) w = kUNK;
           }
	   const vector<unsigned>& sentencePos=corpus.sentencesPos[order[si]]; 
	   const vector<unsigned>& actions=corpus.correct_act_sent[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,actions,corpus.actions,corpus.intToWords,&right);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
      }
      sgd.status();
      time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.nsentences) << " |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< llh<<" ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << endl;
      llh = trs = right = 0;

      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = corpus.nsentencesDev;
        // dev_size = 100;
        double llh = 0;
        double trs = 0;
        double right = 0;
        double correct_heads = 0;
        double total_heads = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const vector<unsigned>& sentence=corpus.sentencesDev[sii];
	   const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
	   const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
           vector<unsigned> tsentence=sentence;
           for (auto& w : tsentence)
             if (training_vocab.count(w) == 0) w = kUNK;

           ComputationGraph hg;
	   vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,vector<unsigned>(),corpus.actions,corpus.intToWords,&right);
	   double lp = 0;
           llh -= lp;
           trs += actions.size();
           map<int,int> ref = parser.compute_heads(sentence.size(), actions, corpus.actions);
           map<int,int> hyp = parser.compute_heads(sentence.size(), pred, corpus.actions);
           //output_conll(sentence, corpus.intToWords, ref, hyp);
           correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
           total_heads += sentence.size() - 1;
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ")\tllh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " uas: " << (correct_heads / total_heads) << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
        if (correct_heads > best_correct_heads) {
          best_correct_heads = correct_heads;
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }
      }
    }
  } // should do training?
  if (true) { // do test evaluation
    double llh = 0;
    double trs = 0;
    double right = 0;
    double correct_heads = 0;
    double total_heads = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned corpus_size = corpus.nsentencesDev;
    for (unsigned sii = 0; sii < corpus_size; ++sii) {
      const vector<unsigned>& sentence=corpus.sentencesDev[sii];
      const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii]; 
      const vector<string>& sentenceUnkStr=corpus.sentencesStrDev[sii]; 
      const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
      vector<unsigned> tsentence=sentence;
      for (auto& w : tsentence)
        if (training_vocab.count(w) == 0) w = kUNK;
      ComputationGraph cg;
      double lp = 0;
      vector<unsigned> pred;
      pred = parser.log_prob_parser(&cg,sentence,tsentence,sentencePos,vector<unsigned>(),corpus.actions,corpus.intToWords,&right);
      llh -= lp;
      trs += actions.size();
      map<int, string> rel_ref, rel_hyp;
      map<int,int> ref = parser.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
      map<int,int> hyp = parser.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
      output_conll(sentence, sentencePos, sentenceUnkStr, corpus.intToWords, corpus.intToPos, hyp, rel_hyp);
      correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
      total_heads += sentence.size() - 1;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " uas: " << (correct_heads / total_heads) << "\t[" << corpus_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
  }
  for (unsigned i = 0; i < corpus.actions.size(); ++i) {
    //cerr << corpus.actions[i] << '\t' << parser.p_r->values[i].transpose() << endl;
    //cerr << corpus.actions[i] << '\t' << parser.p_p2a->values.col(i).transpose() << endl;
  }
}
