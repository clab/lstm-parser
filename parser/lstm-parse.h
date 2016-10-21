#ifndef LSTM_PARSE_H
#define LSTM_PARSE_H

#include <unordered_map>
#include <vector>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "c2.h"

using namespace cnn;
using namespace std;


class ParserBuilder {
public:
  static constexpr const char* ROOT_SYMBOL = "ROOT";

  cpyp::Corpus corpus;
  bool use_pos;
  unsigned vocab_size;
  unsigned action_size;
  unsigned pos_size;
  unordered_map<unsigned, vector<float>> pretrained;
  size_t n_possible_actions;
  const unsigned kUNK;
  const unsigned kROOT_SYMBOL;

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

  explicit ParserBuilder(Model* model, const string& training_path,
                         const string& pretrained_words_path, bool use_pos,
                         unsigned lstm_input_dim, unsigned hidden_dim,
                         unsigned pretrained_dim, unsigned rel_dim,
                         unsigned action_dim, unsigned pos_dim,
                         unsigned input_dim, unsigned layers);

  static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize,
                                const vector<int>& stacki);

  // take a vector of actions and return a parse tree (labeling of every
  // word position with its head's position)
  static map<int, int> ComputeHeads(unsigned sent_len,
                                     const vector<unsigned>& actions,
                                     const vector<string>& setOfActions,
                                     map<int, string>* pr = nullptr);

  // *** if correct_actions is empty, this runs greedy decoding ***
  // returns parse actions for input sentence (in training just returns the
  // reference)
  // OOV handling: raw_sent will have the actual words
  //               sent will have words replaced by appropriate UNK tokens
  // this lets us use pretrained embeddings, when available, for words that were
  // OOV in the parser training data.
  vector<unsigned> LogProbParser(ComputationGraph* hg,
                       const vector<unsigned>& raw_sent,  // raw sentence
                       const vector<unsigned>& sent,  // sent with oovs replaced
                       const vector<unsigned>& sentPos,
                       const vector<unsigned>& correct_actions,
                       const vector<string>& setOfActions,
                       const map<unsigned, std::string>& intToWords,
                       double *right);

  void LoadPretrainedWords(const string& words_path, unsigned pretrained_dim) {
    // TODO: make it load word vector dimension automatically
    pretrained[kUNK] = vector<float>(pretrained_dim, 0);
    cerr << "Loading from " << words_path << " with " << pretrained_dim
         << " dimensions\n";
    ifstream in(words_path);
    string line;
    getline(in, line);
    vector<float> v(pretrained_dim, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < pretrained_dim; ++i) lin >> v[i];
      unsigned id = corpus.get_or_add_word(word);
      pretrained[id] = v;
    }
    cerr << "Loaded " << pretrained.size() << " words" << endl;
  }
};

#endif // #ifndef LSTM_PARSE_H
