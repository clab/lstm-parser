#ifndef LSTM_PARSER_PARSER_NEURAL_TRANSITION_TAGGER_H_
#define LSTM_PARSER_PARSER_NEURAL_TRANSITION_TAGGER_H_

#include <map>
#include <string>
#include <vector>

#include "cnn/expr.h"
#include "cnn/model.h"
#include "corpus.h"

namespace eos {
class portable_oarchive;
}

namespace lstm_parser {

class NeuralTransitionTagger {
public:
  NeuralTransitionTagger() : finalized(false), model(new cnn::Model) {}
  virtual ~NeuralTransitionTagger() {}

  void FinalizeVocab();

  // Used for testing. Replaces OOV with UNK.
  std::vector<unsigned> LogProbTagger(
      const Sentence& sentence, cnn::ComputationGraph *cg,
      bool replace_unknowns = true,
      cnn::expr::Expression* final_parser_state = nullptr);

  const CorpusVocabulary& GetVocab() const { return vocab; }

  // TODO: arrange things such that we don't need to expose this?
  CorpusVocabulary* GetVocab() { return &vocab; }

protected:
  struct TaggerState {
    TaggerState(const Sentence& raw_sentence,
                const Sentence::SentenceMap& sentence)
        : raw_sentence(raw_sentence), sentence(sentence) {}
    const Sentence& raw_sentence;
    const Sentence::SentenceMap& sentence;
    virtual ~TaggerState() {}
  };

  bool finalized;
  std::map<cnn::Parameters*, cnn::expr::Expression> param_expressions;

  // Store the model as a smart ptr so we can call its destructor when needed.
  std::unique_ptr<cnn::Model> model;
  CorpusVocabulary vocab;

  inline cnn::expr::Expression GetParamExpr(cnn::Parameters* params) {
    return param_expressions.at(params);
  }

  virtual std::vector<cnn::Parameters*> GetParameters() = 0;

  virtual TaggerState* InitializeParserState(
      cnn::ComputationGraph* hg, const Sentence& raw_sent,
      const Sentence::SentenceMap& sent,  // sentence with OOVs replaced
      const std::vector<unsigned>& correct_actions) = 0;

  virtual cnn::expr::Expression GetActionProbabilities(
      const TaggerState& state) = 0;

  virtual bool ShouldTerminate(const TaggerState& state) const = 0;

  virtual bool IsActionForbidden(const unsigned action,
                                 const TaggerState& state) const = 0;

  virtual void DoAction(unsigned action, TaggerState* state,
                        cnn::ComputationGraph* cg) = 0;

  virtual void DoSave(eos::portable_oarchive& archive) = 0;

  virtual void InitializeNetworkParameters() = 0;

  void SaveModel(const std::string& model_fname, bool softlink_created);

  // *** if correct_actions is empty, this runs greedy decoding ***
  // returns actions for input sentence (in training just returns the reference)
  // OOV handling: raw_sent will have the actual words
  //               sent will have words replaced by appropriate UNK tokens
  // this lets us use pretrained embeddings, when available, for words that were
  // OOV in the training data.
  std::vector<unsigned> LogProbTagger(
      cnn::ComputationGraph* hg,
      const Sentence& sentence, // raw sentence
      const Sentence::SentenceMap& sent,  // sentence with OOVs replaced
      const std::vector<unsigned>& correct_actions,
      double* correct, cnn::expr::Expression* final_parser_state = nullptr);

  Sentence::SentenceMap ReplaceUnknowns(const Sentence& sentence);
};

} /* namespace lstm_parser */

#endif /* LSTM_PARSER_PARSER_NEURAL_TRANSITION_TAGGER_H_ */
