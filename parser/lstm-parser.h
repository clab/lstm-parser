#ifndef LSTM_PARSER_H
#define LSTM_PARSER_H

#include <boost/algorithm/string/predicate.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
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


class ParseTree {
public:
  // Barebones representation of a parse tree.
  const std::vector<unsigned>& sentence;

  ParseTree(const std::vector<unsigned>& sentence, bool labeled = true) :
      sentence(sentence),
      parents(sentence.size(), -1),
      arc_labels( labeled ? new std::vector<std::string>(sentence.size(),
                                                         "ERROR") : nullptr) {
  }

  inline void SetParent(unsigned index, unsigned parent_index,
                      const std::string& arc_label="") {
    parents[index] = parent_index;
    if (arc_labels) {
      (*arc_labels)[index] = arc_label;
    }
  }

  const inline std::vector<int>& GetParents() const { return parents; }
  const inline std::vector<std::string>& GetArcLabels() const {
    return *arc_labels;
  }

private:
  std::vector<int> parents;
  std::unique_ptr<std::vector<std::string>> arc_labels;
};


class LSTMParser {
public:
  // TODO: make some of these members non-public
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

  explicit LSTMParser(const ParserOptions& options,
                         const std::string& pretrained_words_path,
                         bool finalize=true);

  explicit LSTMParser(const std::string& model_path) :
          kUNK(vocab.GetOrAddWord(vocab.UNK)),
          kROOT_SYMBOL(vocab.GetOrAddWord(vocab.ROOT)) {
    std::cerr << "Loading model from " << model_path << "...";
    std::ifstream model_file(model_path.c_str());
    if (boost::algorithm::ends_with(model_path, ".gz")) {
      std::ifstream model_stream(model_path.c_str());
      boost::iostreams::filtering_streambuf<boost::iostreams::input> filter;
      filter.push(boost::iostreams::gzip_decompressor());
      filter.push(model_stream);
      boost::archive::binary_iarchive archive(filter);
      archive >> *this;
    } else {
      boost::archive::text_iarchive archive(model_file);
      archive >> *this;
    }
    std::cerr << "done." << std::endl;
  }

  template <class Archive>
  explicit LSTMParser(Archive* archive) :
      kUNK(vocab.GetOrAddWord(vocab.UNK)),
      kROOT_SYMBOL(vocab.GetOrAddWord(vocab.ROOT)) {
    archive >> *this;
  }

  static bool IsActionForbidden(const std::string& a, unsigned bsize,
                                unsigned ssize, const std::vector<int>& stacki);

  ParseTree Parse(const std::vector<unsigned>& sentence,
                  const std::vector<unsigned>& sentence_pos,
                  const CorpusVocabulary& vocab, bool labeled, double* correct);

  // take a vector of actions and return a parse tree
  static ParseTree RecoverParseTree(
      const std::vector<unsigned>& sentence,
      const std::vector<unsigned>& actions,
      const std::vector<std::string>& action_names,
      const std::vector<std::string>& actions_to_arc_labels,
      bool labeled = false);

  void Train(const TrainingCorpus& corpus, const TrainingCorpus& dev_corpus,
             const double unk_prob, const std::string& model_fname,
             bool compress, const volatile bool* requested_stop = nullptr);

  void Test(const Corpus& corpus) {
    DoTest(corpus, false, true);
  }

  void Evaluate(const TrainingCorpus& corpus, bool output_parses=false) {
    DoTest(corpus, true, output_parses);
  }

  // Used for testing. Replaces OOV with UNK.
  std::vector<unsigned> LogProbParser(
      const std::vector<unsigned>& sentence,
      const std::vector<unsigned>& sentence_pos, const CorpusVocabulary& vocab,
      cnn::ComputationGraph *cg, double* correct);

  void LoadPretrainedWords(const std::string& words_path);

  void FinalizeVocab();

protected:
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
      const std::vector<std::string>& action_names,
      const std::vector<std::string>& int_to_words, double* right);

  void SaveModel(const std::string& model_fname, bool compress,
                 bool softlink_created);

  inline unsigned ComputeCorrect(const ParseTree& ref,
                                 const ParseTree& hyp) const {
    assert(ref.sentence.size() == hyp.sentence.size());
    unsigned correct_count = 0;
    // Ignore last element of sentence, which is always ROOT.
    for (unsigned i = 0; i < ref.sentence.size() - 1; ++i) {
      if (ref.GetParents()[i] == hyp.GetParents()[i])
        ++correct_count;
    }
    return correct_count;
  }

private:
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar & options;
    ar & vocab;
    ar & pretrained;
    ar & model;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar & options;
    ar & vocab;
    // Don't finalize yet...we might get more words from pretrained vectors.
    ar & pretrained;
    // Don't finalize yet...we want to finalize once our model is initialized.

    model = cnn::Model();
    // Reset the LSTMs *before* reading in the network model, to make sure the
    // model knows how big it's supposed to be.
    stack_lstm = cnn::LSTMBuilder(options.layers, options.lstm_input_dim,
                                  options.hidden_dim, &model);
    buffer_lstm = cnn::LSTMBuilder(options.layers, options.lstm_input_dim,
                                   options.hidden_dim, &model);
    action_lstm = cnn::LSTMBuilder(options.layers, options.action_dim,
                                   options.hidden_dim, &model);

    FinalizeVocab(); // OK, now finalize. :)

    ar & model;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  void DoTest(const Corpus& corpus, bool evaluate, bool output_parses);

  static void OutputConll(const std::vector<unsigned>& sentence,
                          const std::vector<unsigned>& pos,
                          const std::vector<std::string>& sentence_unk_strings,
                          const std::vector<std::string>& int_to_words,
                          const std::vector<std::string>& int_to_pos,
                          const std::map<std::string, unsigned>& words_to_int,
                          const ParseTree& tree);
};

} // namespace lstm_parser

#endif // #ifndef LSTM_PARSER_H
