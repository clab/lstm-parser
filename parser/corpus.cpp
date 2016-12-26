#include "corpus.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace lstm_parser {

constexpr unsigned Corpus::ROOT_TOKEN_ID;

const string CorpusVocabulary::BAD0 = "<BAD0>";
const string CorpusVocabulary::UNK = "<UNK>";
const string CorpusVocabulary::ROOT = "<ROOT>";
const string ORACLE_ROOT_POS = "ROOT";


void ConllUCorpusReader::ReadSentences(const string& file,
                                       Corpus* corpus) const {
  string next_line;
  map<unsigned, string> current_sentence_unk_surface_forms;
  map<unsigned, unsigned> current_sentence;
  map<unsigned, unsigned> current_sentence_pos;

  ifstream conll_file(file);
  unsigned unk_word_symbol = corpus->vocab->GetWord(CorpusVocabulary::UNK);
  unsigned root_symbol = corpus->vocab->GetWord(CorpusVocabulary::ROOT);
  unsigned root_pos_symbol = corpus->vocab->GetPOS(CorpusVocabulary::ROOT);
  while(conll_file) {
    getline(conll_file, next_line);
    if (next_line.empty()) {
      if (!current_sentence.empty()) { // just in case we get 2 blank lines
        current_sentence[Corpus::ROOT_TOKEN_ID] = root_symbol;
        current_sentence_pos[Corpus::ROOT_TOKEN_ID] = root_pos_symbol;
        current_sentence_unk_surface_forms[Corpus::ROOT_TOKEN_ID] = "";

        corpus->sentences.push_back(move(current_sentence));
        current_sentence.clear();

        corpus->sentences_pos.push_back(move(current_sentence_pos));
        current_sentence_pos.clear();

        corpus->sentences_unk_surface_forms.push_back(
            move(current_sentence_unk_surface_forms));
        current_sentence_unk_surface_forms.clear();
      }
      continue;
    } else if (next_line[0] == '#') {
      // TODO: carry over comment lines, as required by CoNLL-U format spec?
      continue;
    }

    istringstream line_stream(next_line);
    unsigned token_index;
    string surface_form;
    string pos;
    string dummy;
    line_stream >> token_index;
    if (token_index < current_sentence.size() + 1) {
      throw ConllFormatException(
          "Format error in file " + file + ": expected token ID at least "
          + to_string(current_sentence.size() + 1) + "; got "
          + to_string(token_index));
    }
    line_stream >> surface_form >> dummy >> dummy // skip lemma and xposttag
                >> pos; // ignore the rest of the line

    unsigned word_id = corpus->vocab->GetWord(surface_form);
    current_sentence_unk_surface_forms[token_index] =
        (word_id == unk_word_symbol ? surface_form : "");
    current_sentence[token_index] = word_id;
    current_sentence_pos[token_index] = corpus->vocab->GetPOS(pos);
  }
}



void TrainingCorpus::CountSingletons() {
  // compute the singletons in the parser's training data
  map<unsigned, unsigned> counts;
  for (const auto& sent : sentences) {
    for (const auto& index_and_word_id : sent) {
      counts[index_and_word_id.second]++;
    }
  }
  for (const auto wc : counts) {
    if (wc.second == 1)
      singletons.insert(wc.first);
  }
}


void TrainingCorpus::OracleTransitionsCorpusReader::LoadCorrectActions(
    const string& file, TrainingCorpus* corpus) const {
  // TODO: break up this function?
  cerr << "Loading " << (is_training ? "training" : "dev")
       << " corpus from " << file << "..." << endl;
  ifstream actionsFile(file);
  string lineS;
  CorpusVocabulary* vocab = corpus->vocab;

  bool next_is_action_line = false;
  bool start_of_sentence = false;
  bool first = true;

  map<unsigned, unsigned> sentence;
  map<unsigned, unsigned> sentence_pos;
  map<unsigned, string> sentence_unk_surface_forms;

  // We'll need to make sure ROOT token has a consistent ID.
  // (Should get inlined; defined here for DRY purposes.)
  auto FixRootID =
      [&sentence, &sentence_pos, &sentence_unk_surface_forms]() {
    // ROOT is always added as the last token in the sentence. Since IDs are
    // 1-indexed, the last element in an n-word sentence will have an ID of n.
    unsigned original_root_id = sentence.size();
    auto root_iter = sentence.find(original_root_id);
    sentence[Corpus::ROOT_TOKEN_ID] = root_iter->second;
    sentence.erase(root_iter);

    auto root_pos_iter = sentence_pos.find(original_root_id);
    sentence_pos[Corpus::ROOT_TOKEN_ID] = root_pos_iter->second;
    sentence_pos.erase(root_pos_iter);

    if (!sentence_unk_surface_forms.empty()) {
      auto root_str_iter = sentence_unk_surface_forms.find(original_root_id);
      sentence_unk_surface_forms[Corpus::ROOT_TOKEN_ID] = "";
      sentence_unk_surface_forms.erase(root_str_iter);
    }
  };

  while (getline(actionsFile, lineS)) {
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    // An empty line marks the end of a sentence.
    if (lineS.empty()) {
      next_is_action_line = false;
      if (!first) { // if first, first line is blank, but no sentence yet
        FixRootID();
        // Store the sentence variables and clear them for the next sentence.
        corpus->sentences.push_back({});
        corpus->sentences.back().swap(sentence);
        corpus->sentences_pos.push_back({});
        corpus->sentences_pos.back().swap(sentence_pos);
        if (!is_training) {
          corpus->sentences_unk_surface_forms.push_back({});
          corpus->sentences_unk_surface_forms.back().swap(
              sentence_unk_surface_forms);
        }
      }
      start_of_sentence = true;
      continue; // don't update next_is_action_line
    }

    if (!next_is_action_line) { // it's a state line
      first = false;
      //stack and buffer, for now, leave it like this.
      if (start_of_sentence) {
        // the initial line in each sentence should look like:
        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        lineS = lineS.substr(3, lineS.size() - 4);
        // read the initial line, token by token "the-det," "cat-noun," ...
        istringstream iss(lineS);
        do {
          string word;
          iss >> word;
          if (word.size() == 0) {
            continue;
          }
          // remove the trailing comma if need be.
          if (word[word.size() - 1] == ',') {
            word = word.substr(0, word.size() - 1);
          }
          // split the string (at '-') into word and POS tag.
          size_t posIndex = word.rfind('-');
          if (posIndex == string::npos) {
            cerr << "can't find the dash in '" << word << "'"
                 << endl;
          }
          assert(posIndex != string::npos);
          string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);

          if (pos == ORACLE_ROOT_POS) {
            // Prevent any confusion with the actual word "ROOT".
            word = CorpusVocabulary::ROOT;
            pos = CorpusVocabulary::ROOT;
          }

          // We assume that we'll have seen all POS tags in training, so don't
          // worry about OOV tags.
          unsigned pos_id = vocab->GetOrAddEntry(pos, &vocab->pos_to_int,
                                                 &vocab->int_to_pos);
          // Use 1-indexed token IDs to leave room for ROOT in position 0.
          unsigned next_token_index = sentence.size() + 1;
          unsigned word_id;
          if (is_training) {
            unsigned num_words = vocab->CountWords(); // store for later check
            word_id = vocab->GetOrAddWord(word, true);
            if (vocab->CountWords() > num_words) {
              // A new word was added; add its chars, too.
              unsigned j = 0;
              while (j < word.length()) {
                unsigned char_utf8_len = UTF8Len(word[j]);
                string next_utf8_char = word.substr(j, char_utf8_len);
                vocab->GetOrAddEntry(next_utf8_char, &vocab->chars_to_int,
                                     &vocab->int_to_chars);
                j += char_utf8_len;
              }
            } else {
              // It's an old word. Make sure it's marked as present in training.
              vocab->int_to_training_word[word_id] = true;
            }
          } else {
            // add an empty string for any token except OOVs (it is easy to
            // recover the surface form of non-OOV using intToWords(id)).
            // OOV word
            if (corpus->USE_SPELLING) {
              word_id = vocab->GetOrAddWord(word); // don't record as training
              sentence_unk_surface_forms[next_token_index] = "";
            } else {
              auto word_iter = vocab->words_to_int.find(word);
              if (word_iter == vocab->words_to_int.end()) {
                // Save the surface form of this OOV.
                sentence_unk_surface_forms[next_token_index] = word;
                word_id = vocab->words_to_int[vocab->UNK];
              } else {
                sentence_unk_surface_forms[next_token_index] = "";
                word_id = word_iter->second;
              }
            }
          }

          sentence[next_token_index] = word_id;
          sentence_pos[next_token_index] = pos_id;
        } while (iss);
      }
    } else if (next_is_action_line) {
      auto action_iter = find(vocab->actions.begin(), vocab->actions.end(),
                              lineS);
      if (action_iter != vocab->actions.end()) {
        unsigned action_index = distance(vocab->actions.begin(), action_iter);
        if (start_of_sentence)
          corpus->correct_act_sent.push_back({action_index});
        else
          corpus->correct_act_sent.back().push_back(action_index);
      } else { // A not-previously-seen action
        if (is_training) {
          vocab->actions.push_back(lineS);
          vocab->actions_to_arc_labels.push_back(
              vocab->GetLabelForAction(lineS));

          unsigned action_index = vocab->actions.size() - 1;
          if (start_of_sentence)
            corpus->correct_act_sent.push_back({action_index});
          else
            corpus->correct_act_sent.back().push_back(action_index);
        } else {
          // TODO: right now, new actions which haven't been observed in
          // training are not added to correct_act_sent. In dev, this may
          // be a problem if there is little training data.
          cerr << "WARNING: encountered unknown transition in dev corpus: "
               << lineS << endl;
          if (start_of_sentence)
            corpus->correct_act_sent.push_back({});
        }
      }
      start_of_sentence = false;
    }

    next_is_action_line = !next_is_action_line;
  }

  // Add the last sentence.
  if (sentence.size() > 0) {
    FixRootID();
    corpus->sentences.push_back(move(sentence));
    corpus->sentences_pos.push_back(move(sentence_pos));
    if (!is_training) {
      corpus->sentences_unk_surface_forms.push_back(
          move(sentence_unk_surface_forms));
    }
  }

  actionsFile.close();

  cerr << "done." << "\n";
  if (is_training) {
    for (auto a : vocab->actions) {
      cerr << a << "\n";
    }
  }
  cerr << "# of actions: " << vocab->CountActions() << "\n";
  cerr << "# of words: " << vocab->CountWords() << "\n";
  if (is_training) {
    for (unsigned i = 0; i < vocab->int_to_pos.size(); i++) {
      cerr << i << ": " << vocab->int_to_pos[i] << "\n";
    }
  } else {
    cerr << "# of POS tags: " << vocab->CountPOS() << "\n";
  }

  if (is_training) {  // compute the singletons in the parser's training data
    corpus->CountSingletons();
  }
}

} // namespace lstm_parser
