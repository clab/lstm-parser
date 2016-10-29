#ifndef LSTM_PARSER_H
#define LSTM_PARSER_H

#include <boost/serialization/unordered_map.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "corpus.h"


namespace lstm_parser {

struct ParserOptions {
  bool use_pos;
  unsigned layers;
  unsigned input_dim;
  unsigned hidden_dim;
  unsigned action_dim;
  unsigned lstm_input_dim;
  unsigned pos_dim;
  unsigned rel_dim;
  unsigned unk_strategy;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & use_pos;
    ar & layers;
    ar & input_dim;
    ar & hidden_dim;
    ar & action_dim;
    ar & lstm_input_dim;
    ar & pos_dim;
    ar & rel_dim;
    ar & unk_strategy;
  }

  inline bool operator==(const ParserOptions& other) const {
    return use_pos == other.use_pos && layers == other.layers
        && input_dim == other.input_dim && hidden_dim == other.hidden_dim
        && action_dim == other.action_dim
        && lstm_input_dim == other.lstm_input_dim && pos_dim == other.pos_dim
        && rel_dim == other.rel_dim && unk_strategy == other.unk_strategy;
  }

  inline bool operator!=(const ParserOptions& other) const {
    return !operator==(other);
  }
};


class ParserBuilder { // TODO: rename
public:
  static constexpr const char* ROOT_SYMBOL = "ROOT";

  ParserOptions options;
  CorpusVocabulary vocab;
  cnn::Model model;

  bool finalized;
  std::unordered_map<unsigned, std::vector<float>> pretrained;
  unsigned n_possible_actions;
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

  explicit ParserBuilder(const ParserOptions& options,
                         const std::string& pretrained_words_path,
                         bool finalize=true);

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
  std::vector<unsigned> LogProbParser(
      cnn::ComputationGraph* hg,
      const std::vector<unsigned>& raw_sent,  // raw sentence
      const std::vector<unsigned>& sent,  // sentence with OOVs replaced
      const std::vector<unsigned>& sentPos,
      const std::vector<unsigned>& correct_actions,
      const std::vector<std::string>& setOfActions,
      const std::vector<std::string>& intToWords, double* right);

  void LoadPretrainedWords(const std::string& words_path) {
    std::cerr << "Loading word vectors from " << words_path;
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
    pretrained[vocab.wordsToInt[vocab.UNK]] = std::vector<float>(pretrained_dim,
                                                                 0);
    std::vector<float> v(pretrained_dim, 0);
    std::string word;
    while (getline(in, line)) {
      std::istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < pretrained_dim; ++i)
        lin >> v[i];
      // We DON'T yet know this word is present in training data.
      unsigned id = vocab.GetOrAddWord(word, false);
      pretrained[id] = v;
    }
    assert(num_words == pretrained.size() - 1); // -1 for UNK
    std::cerr << "Loaded " << pretrained.size() - 1 << " words" << std::endl;
  }

  void FinalizeVocab();

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & options;
    ar & vocab;
    ar & pretrained;
    FinalizeVocab(); // finalize *after* vocab & pretrained to make load work
    ar & model;
  }
};

} // namespace lstm_parser

#endif // #ifndef LSTM_PARSER_H
