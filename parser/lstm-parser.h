#ifndef LSTM_PARSER_H
#define LSTM_PARSER_H

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <chrono>
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
#include "eos/portable_archive.hpp"
#include "lstm-transition-tagger.h"


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
  double unk_prob;

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
    ar & unk_prob;
  }

  inline bool operator==(const ParserOptions& other) const {
    // Lordy, I can't wait for default comparison operators.
    return use_pos == other.use_pos && layers == other.layers
        && input_dim == other.input_dim && hidden_dim == other.hidden_dim
        && action_dim == other.action_dim
        && lstm_input_dim == other.lstm_input_dim && pos_dim == other.pos_dim
        && rel_dim == other.rel_dim && unk_strategy == other.unk_strategy
        && unk_prob == other.unk_prob;
  }

  inline bool operator!=(const ParserOptions& other) const {
    return !operator==(other);
  }
};


class ParseTree {
public:
  static std::string NO_LABEL;
  // Barebones representation of a parse tree.
  const Sentence& sentence;
  double logprob;

  ParseTree(const Sentence& sentence, bool labeled = true) :
      sentence(sentence),
      logprob(0),
      arc_labels( labeled ? new std::map<unsigned, std::string> : nullptr) {}

  inline void SetParent(unsigned child_index, unsigned parent_index,
                      const std::string& arc_label="") {
    parents[child_index] = parent_index;
    if (arc_labels) {
      (*arc_labels)[child_index] = arc_label;
    }
  }

  const inline unsigned GetParent(unsigned child) const {
    auto parent_iter = parents.find(child);
    if (parent_iter == parents.end()) {
      return Corpus::ROOT_TOKEN_ID; // This is the best guess we've got.
    } else {
      return parent_iter->second;
    }
  }

  const inline std::string& GetArcLabel(unsigned child) const {
    if (!arc_labels)
      return NO_LABEL;
    auto arc_label_iter = arc_labels->find(child);
    if (arc_label_iter == arc_labels->end()) {
      return NO_LABEL;
    } else {
      return arc_label_iter->second;
    }
  }

private:
  std::map<unsigned, unsigned> parents;
  std::unique_ptr<std::map<unsigned, std::string>> arc_labels;
};


class LSTMParser : LSTMTransitionTagger {
public:
  // TODO: make some of these members non-public
  ParserOptions options;
  CorpusVocabulary vocab;
  cnn::Model model;

  std::unordered_map<unsigned, std::vector<float>> pretrained;
  unsigned n_possible_actions;
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
      kROOT_SYMBOL(vocab.GetOrAddWord(vocab.ROOT)) {
    std::cerr << "Loading model from " << model_path << "...";
    auto t_start = std::chrono::high_resolution_clock::now();
    std::ifstream model_file(model_path.c_str(), std::ios::binary);
    if (!model_file) {
      std::cerr << "Unable to open model file; aborting" << std::endl;
      abort();
    }
    eos::portable_iarchive archive(model_file);
    archive >> *this;
    auto t_end = std::chrono::high_resolution_clock::now();
    auto ms_passed =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cerr << "done. (Loading took " << ms_passed << " milliseconds.)" << std::endl;
  }


  template <class Archive>
  explicit LSTMParser(Archive* archive) :
      kROOT_SYMBOL(vocab.GetOrAddWord(vocab.ROOT)) {
    *archive >> *this;
  }

  virtual bool IsActionForbidden(const std::string& a,
                                 const TaggerState& state) override;

  virtual cnn::expr::Expression GetActionProbabilities(const TaggerState& state)
      override;

  virtual void DoAction(unsigned action,
                        const std::vector<std::string>& action_names,
                        TaggerState* state, cnn::ComputationGraph* cg) override;

  ParseTree Parse(const Sentence& sentence,
                  const CorpusVocabulary& vocab, bool labeled);

  // take a vector of actions and return a parse tree
  ParseTree RecoverParseTree(
      const Sentence& sentence,
      const std::vector<unsigned>& actions,
      const std::vector<std::string>& action_names,
      const std::vector<std::string>& actions_to_arc_labels, double logprob = 0,
      bool labeled = false);

  void Train(const ParserTrainingCorpus& corpus,
             const ParserTrainingCorpus& dev_corpus, const double unk_prob,
             const std::string& model_fname,
             const volatile bool* requested_stop = nullptr);

  void Test(const Corpus& corpus) {
    DoTest(corpus, false, true);
  }

  void Evaluate(const ParserTrainingCorpus& corpus, bool output_parses=false) {
    DoTest(corpus, true, output_parses);
  }

  void LoadPretrainedWords(const std::string& words_path);

  void FinalizeVocab();

protected:
  struct ParserState : public TaggerState {
    std::vector<cnn::expr::Expression> buffer;
    std::vector<int> bufferi; // position of the words in the sentence
    std::vector<cnn::expr::Expression> stack;  // subtree embeddings
    std::vector<int> stacki; // word position in sentence of head of subtree

    ~ParserState() {
      assert(stack.size() == 2); // guard symbol, root
      assert(stacki.size() == 2);
      assert(buffer.size() == 1); // guard symbol
      assert(bufferi.size() == 1);
    }
  };

  virtual std::vector<cnn::Parameters*> GetParameters() override {
    std::vector<cnn::Parameters*> all_params {p_pbias, p_H, p_D, p_R, p_cbias,
        p_S, p_B, p_A, p_ib, p_w2l, p_p2a, p_abias, p_action_start};
    if (options.use_pos)
      all_params.push_back(p_p2l);
    if (p_t2l)
      all_params.push_back(p_t2l);
    return all_params;
  }

  virtual TaggerState* InitializeParserState(
      cnn::ComputationGraph* cg, const Sentence& raw_sent,
      const Sentence::SentenceMap& sent,  // sentence with OOVs replaced
      const std::vector<unsigned>& correct_actions,
      const std::vector<std::string>& action_names) override;

  virtual bool ShouldTerminate(const TaggerState& state) override {
    const ParserState& real_state = static_cast<const ParserState&>(state);
    return real_state.stack.size() <= 2 && real_state.buffer.size() <= 1;
  }

  inline unsigned ComputeCorrect(const ParseTree& ref,
                                 const ParseTree& hyp) const {
    assert(ref.sentence.Size() == hyp.sentence.Size());
    unsigned correct_count = 0;
    for (const auto& token_index_and_word : ref.sentence.words) {
      unsigned i = token_index_and_word.first;
      if (i != Corpus::ROOT_TOKEN_ID && ref.GetParent(i) == hyp.GetParent(i))
        ++correct_count;
    }
    return correct_count;
  }

  virtual void DoSave(eos::portable_oarchive& archive) override {
    archive << *this;
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
    finalized = false; // we'll need to re-finalize after resetting the network.

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

  static void OutputConll(const Sentence& sentence,
      const std::vector<std::string>& int_to_words,
      const std::vector<std::string>& int_to_pos,
      const std::map<std::string, unsigned>& words_to_int,
      const ParseTree& tree);
};

} // namespace lstm_parser

#endif // #ifndef LSTM_PARSER_H
