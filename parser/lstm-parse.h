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


class ParserBuilder {
public:
  static constexpr const char* ROOT_SYMBOL = "ROOT";

  cpyp::Corpus corpus;
  bool use_pos;
  unsigned vocab_size;
  unsigned action_size;
  unsigned pos_size;
  std::unordered_map<unsigned, std::vector<float>> pretrained;
  size_t n_possible_actions;
  const unsigned kUNK;
  const unsigned kROOT_SYMBOL;

  cnn::LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  cnn::LSTMBuilder buffer_lstm;
  cnn::LSTMBuilder action_lstm;
  cnn::LookupParameters* p_w; // word embeddings
  cnn::LookupParameters* p_t; // pretrained word embeddings (not updated)
  cnn::LookupParameters* p_a; // input action embeddings
  cnn::LookupParameters* p_r; // relation embeddings
  cnn::LookupParameters* p_p; // pos tag embeddings
  cnn::Parameters* p_pbias; // parser state bias
  cnn::Parameters* p_A; // action lstm to parser state
  cnn::Parameters* p_B; // buffer lstm to parser state
  cnn::Parameters* p_S; // stack lstm to parser state
  cnn::Parameters* p_H; // head matrix for composition function
  cnn::Parameters* p_D; // dependency matrix for composition function
  cnn::Parameters* p_R; // relation matrix for composition function
  cnn::Parameters* p_w2l; // word to LSTM input
  cnn::Parameters* p_p2l; // POS to LSTM input
  cnn::Parameters* p_t2l; // pretrained word embeddings to LSTM input
  cnn::Parameters* p_ib; // LSTM input bias
  cnn::Parameters* p_cbias; // composition function bias
  cnn::Parameters* p_p2a;   // parser state to action
  cnn::Parameters* p_action_start;  // action bias
  cnn::Parameters* p_abias;  // action bias
  cnn::Parameters* p_buffer_guard;  // end of buffer
  cnn::Parameters* p_stack_guard;  // end of stack

  explicit ParserBuilder(cnn::Model* model, const std::string& training_path,
                         const std::string& pretrained_words_path, bool use_pos,
                         unsigned lstm_input_dim, unsigned hidden_dim,
                         unsigned rel_dim, unsigned action_dim,
                         unsigned pos_dim, unsigned input_dim, unsigned layers);

  static bool IsActionForbidden(const std::string& a, unsigned bsize,
                                unsigned ssize, const std::vector<int>& stacki);

  // take a std::vector of actions and return a parse tree (labeling of every
  // word position with its head's position)
  static std::map<int, int> ComputeHeads(
      unsigned sent_len, const std::vector<unsigned>& actions,
      const std::vector<std::string>& setOfActions,
      std::map<int, std::string>* pr = nullptr);

  // *** if correct_actions is empty, this runs greedy decoding ***
  // returns parse actions for input sentence (in training just returns the
  // reference)
  // OOV handling: raw_sent will have the actual words
  //               sent will have words replaced by appropriate UNK tokens
  // this lets us use pretrained embeddings, when available, for words that were
  // OOV in the parser training data.
  std::vector<unsigned> LogProbParser(cnn::ComputationGraph* hg,
                       const std::vector<unsigned>& raw_sent,  // raw sentence
                       const std::vector<unsigned>& sent,  // w/ oovs replaced
                       const std::vector<unsigned>& sentPos,
                       const std::vector<unsigned>& correct_actions,
                       const std::vector<std::string>& setOfActions,
                       const std::map<unsigned, std::string>& intToWords,
                       double *right);

  void LoadPretrainedWords(const std::string& words_path) {
    std::cerr << "Loading word std::vectors from " << words_path;
    std::ifstream in(words_path);

    // Read header
    std::string line;
    std::getline(in, line);
    std::istringstream first_line(line);
    unsigned num_words;
    first_line >> num_words;
    unsigned pretrained_dim;
    first_line >> pretrained_dim;
    std::cerr << " with " << pretrained_dim << " dimensions..." << std::endl;

    // Read std::vectors
    pretrained[kUNK] = std::vector<float>(pretrained_dim, 0);
    std::vector<float> v(pretrained_dim, 0);
    std::string word;
    while (getline(in, line)) {
      std::istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < pretrained_dim; ++i) lin >> v[i];
      unsigned id = corpus.get_or_add_word(word);
      pretrained[id] = v;
    }
    assert(num_words == pretrained.size() - 1); // -1 for UNK
    std::cerr << "Loaded " << pretrained.size() - 1 << " words" << std::endl;
  }
};

#endif // #ifndef LSTM_PARSE_H
