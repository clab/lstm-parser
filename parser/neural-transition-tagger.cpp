#include "neural-transition-tagger.h"

#include <fstream>
#include <string>
#include <memory>

#include "cnn/expr.h"
#include "cnn/model.h"
#include "eos/portable_archive.hpp"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

namespace lstm_parser {


void NeuralTransitionTagger::SaveModel(const string& model_fname,
                                     bool softlink_created) {
  ofstream out_file(model_fname);
  eos::portable_oarchive archive(out_file);
  DoSave(archive);
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


void NeuralTransitionTagger::FinalizeVocab() {
  if (finalized)
    return;
  if (!model.get())
    model.reset(new Model);
  InitializeNetworkParameters();
  // Give up memory we don't need.
  vocab.action_names.shrink_to_fit();
  vocab.actions_to_arc_labels.shrink_to_fit();
  vocab.int_to_chars.shrink_to_fit();
  vocab.int_to_pos.shrink_to_fit();
  vocab.int_to_training_word.shrink_to_fit();
  vocab.int_to_words.shrink_to_fit();
  finalized = true;
}

Sentence::SentenceMap NeuralTransitionTagger::ReplaceUnknowns(
    const Sentence& sentence) {
  Sentence::SentenceMap tsentence(sentence.words);  // sentence w/ OOVs replaced
  for (auto& index_and_id : tsentence) {
    // use reference to overwrite
    if (index_and_id.second >= vocab.int_to_training_word.size()
        || !vocab.int_to_training_word[index_and_id.second]) {
      index_and_id.second = vocab.kUNK;
    }
  }
  return tsentence;
}


vector<unsigned> NeuralTransitionTagger::LogProbTagger(
    ComputationGraph* cg,
    const Sentence& raw_sent,  // raw sentence
    const Sentence::SentenceMap& sent,  // sentence with OOVs replaced
    bool training,
    const vector<unsigned>& correct_actions, double* correct,
    map<string, Expression>* states_to_expose) {
  in_training = training;
  if (training)
    assert(!correct_actions.empty());
  assert(finalized);
  vector<unsigned> results;

  // variables in the computation graph representing the parameters
  for (Parameters *params : GetParameters()) {
    param_expressions[params] = parameter(*cg, params);
  }

  unique_ptr<TaggerState> state(
      InitializeParserState(cg, raw_sent, sent, correct_actions));

  vector<Expression> log_probs;
  unsigned action_count = 0;  // incremented at each prediction
  Expression p_t; // declared outside to allow access later
  while (!ShouldTerminate(*state)) {
    // Get list of possible actions for the current parser state.
    vector<unsigned> current_valid_actions;
    for (unsigned action = 0; action < vocab.action_names.size(); ++action) {
      if (IsActionForbidden(action, *state))
        continue;
      current_valid_actions.push_back(action);
    }

    Expression r_t = GetActionProbabilities(*state);
    // adist = log_softmax(r_t, current_valid_actions)
    Expression adiste = log_softmax(r_t, current_valid_actions);
    vector<float> adist = as_vector(cg->incremental_forward());
    double best_score = adist[current_valid_actions[0]];
    unsigned best_a = current_valid_actions[0];
    for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
      if (adist[current_valid_actions[i]] > best_score) {
        best_score = adist[current_valid_actions[i]];
        best_a = current_valid_actions[i];
      }
    }
    unsigned action = best_a;

    if (!correct_actions.empty()) {
      assert(action_count < correct_actions.size());
      unsigned correct_action = correct_actions[action_count];
      if (correct && best_a == correct_action) {
        (*correct)++;
      }
      // If we're training, use the reference action.
      if (training)
        action = correct_action;
    }
    ++action_count;
    log_probs.push_back(pick(adiste, action));
    results.push_back(action);

    DoAction(action, state.get(), cg, states_to_expose);
  }

  Expression tot_neglogprob = -sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);

  param_expressions.clear();
  return results;
}



} /* namespace lstm_parser */
