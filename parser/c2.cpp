#include "c2.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

using namespace std;

namespace cpyp {

const string ParserVocabulary::UNK = "<UNK>";
const string ParserVocabulary::BAD0 = "<BAD0>";

void Corpus::CountSingletons() {
  // compute the singletons in the parser's training data
  map<unsigned, unsigned> counts;
  for (const auto& sent : sentences) {
    for (const unsigned word : sent) {
      counts[word]++;
    }
  }
  for (const auto wc : counts) {
    if (wc.second == 1)
      singletons.insert(wc.first);
  }
}

void Corpus::load_correct_actions(const string& file, bool is_training) {
  // TODO: break up this function?
  cerr << "Loading " << (is_training ? "training" : "dev")
       << " corpus from " << file << "..." << endl;
  ifstream actionsFile(file);
  string lineS;

  bool next_is_action_line = false;
  bool start_of_sentence = false;
  bool first = true;

  vector<unsigned> current_sent;
  vector<unsigned> current_sent_pos;
  vector<string> current_sent_surface_strs;
  while (getline(actionsFile, lineS)) {
    //istringstream iss(line);
    //string lineS;
    //iss >> lineS;
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    // An empty line marks the end of a sentence.
    if (lineS.empty()) {
      next_is_action_line = false;
      if (!first) { // first line is blank, but no sentence yet
        // Store the sentence variables and clear them for the next sentence.
        sentences.push_back({});
        sentences.back().swap(current_sent);
        sentencesPos.push_back({});
        sentencesPos.back().swap(current_sent_pos);
        if (!is_training) {
          sentencesSurfaceForms.push_back({});
          sentencesSurfaceForms.back().swap(current_sent_surface_strs);
        }
      }
      start_of_sentence = true;
      continue; // don't update next_is_action_line
    }

    if (!next_is_action_line) { // it's a state line
      first = false;
      //stack and buffer, for now, leave it like this.
      if (start_of_sentence) {
        // the initial line in each sentence may look like:
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

          // We assume that we'll have seen all POS tags in training, so don't
          // worry about OOV tags.
          unsigned pos_id = vocab->GetOrAddEntry(pos, &vocab->posToInt,
                                                 &vocab->intToPos);
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
                vocab->GetOrAddEntry(next_utf8_char, &vocab->charsToInt,
                                     &vocab->intToChars);
                j += char_utf8_len;
              }
            } else {
              // It's an old word. Make sure it's marked as present in training.
              vocab->intToTrainingWord[word_id] = true;
            }
          } else {
            // add an empty string for any token except OOVs (it is easy to
            // recover the surface form of non-OOV using intToWords(id)).
            // OOV word
            if (USE_SPELLING) {
              word_id = vocab->GetOrAddWord(word); // don't record as training
              current_sent_surface_strs.push_back("");
            } else {
              auto word_iter = vocab->wordsToInt.find(word);
              if (word_iter == vocab->wordsToInt.end()) {
                // Save the surface form of this OOV.
                current_sent_surface_strs.push_back(word);
                word_id = vocab->wordsToInt[vocab->UNK];
              } else {
                current_sent_surface_strs.push_back("");
                word_id = word_iter->second;
              }
            }
          }

          current_sent.push_back(word_id);
          current_sent_pos.push_back(pos_id);
        } while (iss);
      }
    } else if (next_is_action_line) {
      auto action_iter = find(vocab->actions.begin(), vocab->actions.end(),
                              lineS);
      if (action_iter != vocab->actions.end()) {
        unsigned action_index = distance(vocab->actions.begin(), action_iter);
        if (start_of_sentence)
          correct_act_sent.push_back({action_index});
        else
          correct_act_sent.back().push_back(action_index);
      } else { // A not-previously-seen action
        if (is_training) {
          vocab->actions.push_back(lineS);
          unsigned action_index = vocab->actions.size() - 1;
          if (start_of_sentence)
            correct_act_sent.push_back({action_index});
          else
            correct_act_sent.back().push_back(action_index);
        } else {
          // TODO: right now, new actions which haven't been observed in
          // training are not added to correct_act_sent. In dev/test, this may
          // be a problem if there is little training data.
          cerr << "WARNING: encountered unknown transition in dev/test: "
               << lineS << endl;
          if (start_of_sentence)
            correct_act_sent.push_back({});
        }
      }
      start_of_sentence = false;
    }

    next_is_action_line = !next_is_action_line;
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentences.push_back(move(current_sent));
    sentencesPos.push_back(move(current_sent_pos));
    if (!is_training) {
      sentencesSurfaceForms.push_back(move(current_sent_surface_strs));
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
    for (unsigned i = 0; i < vocab->intToPos.size(); i++) {
      cerr << i << ":" << vocab->intToPos[i] << "\n";
    }
  } else {
    cerr << "# of POS tags: " << vocab->CountPOS() << "\n";
  }

  if (is_training) {  // compute the singletons in the parser's training data
    CountSingletons();
  }

}

} // namespace cpyp
