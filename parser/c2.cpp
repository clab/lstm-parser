#include "c2.h"

using namespace std;

namespace cpyp {

constexpr const char* ParserVocabulary::UNK;
constexpr const char* ParserVocabulary::BAD0;


void Corpus::load_correct_actions(const string& file) {
  cerr << "Loading corpus from " << file << "...";
  ifstream actionsFile(file);
  //correct_act_sent=new vector<vector<unsigned>>();
  string lineS;

  int count = -1;
  int sentence = -1;
  bool initial = false;
  bool first = true;

  vector<unsigned> current_sent;
  vector<unsigned> current_sent_pos;
  while (getline(actionsFile, lineS)) {
    //istringstream iss(line);
    //string lineS;
    //iss >> lineS;
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      count = 0;
      if (!first) {
        sentences[sentence] = current_sent;
        sentencesPos[sentence] = current_sent_pos;
      }

      sentence++;

      initial = true;
      current_sent.clear();
      current_sent_pos.clear();
    } else if (count == 0) {
      first = false;
      //stack and buffer, for now, leave it like this.
      count = 1;
      if (initial) {
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
            cerr << "cant find the dash in '" << word << "'"
                 << endl;
          }
          assert(posIndex != string::npos);
          string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);

          unsigned pos_id = vocab->GetOrAddEntry(pos, &vocab->posToInt,
                                                 &vocab->intToPos);
          unsigned num_words = vocab->CountWords(); // store for later check
          unsigned word_id = get_or_add_word(word);
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
          }

          current_sent.push_back(word_id);
          current_sent_pos.push_back(pos_id);
        } while (iss);
      }
      initial = false;
    } else if (count == 1) {
      int i = 0;
      bool found = false;
      for (auto a : vocab->actions) {
        if (a == lineS) {
          vector<unsigned> a = correct_act_sent[sentence];
          a.push_back(i);
          correct_act_sent[sentence] = a;
          found = true;
          break;
        }
        i++;
      }
      if (!found) {
        vocab->actions.push_back(lineS);
        vector<unsigned> a = correct_act_sent[sentence];
        a.push_back(vocab->actions.size() - 1);
        correct_act_sent[sentence] = a;
      }
      count = 0;
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentences[sentence] = current_sent;
    sentencesPos[sentence] = current_sent_pos;
    sentence++;
  }

  actionsFile.close();

  cerr << "done." << "\n";
  for (auto a : vocab->actions) {
    cerr << a << "\n";
  }
  cerr << "# of actions: " << vocab->CountActions() << "\n";
  cerr << "# of words: " << vocab->CountWords() << "\n";
  for (unsigned i = 0; i < vocab->intToPos.size(); i++) {
    cerr << i << ":" << vocab->intToPos[i] << "\n";
  }
}

void Corpus::load_correct_actionsDev(const string& file) {
  cerr << "Loading dev corpus from " << file << "...";
  ifstream actionsFile(file);
  string lineS;

  assert(vocab->posToInt.size() > 1);
  assert(vocab->wordsToInt.size() > 3);
  int count = -1;
  int sentence = -1;
  bool initial = false;
  bool first = true;
  vector<unsigned> current_sent;
  vector<unsigned> current_sent_pos;
  vector<string> current_sent_str;
  while (getline(actionsFile, lineS)) {
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      // an empty line marks the end of a sentence.
      count = 0;
      if (!first) {
        sentences[sentence] = current_sent;
        sentencesPos[sentence] = current_sent_pos;
        sentencesSurfaceForms[sentence] = current_sent_str;
      }

      sentence++;

      initial = true;
      current_sent.clear();
      current_sent_pos.clear();
      current_sent_str.clear();
    } else if (count == 0) {
      first = false;
      //stack and buffer, for now, leave it like this.
      count = 1;
      if (initial) {
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
          assert(posIndex != string::npos);
          string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);
          // new POS tag
          unsigned pos_id = vocab->GetOrAddEntry(pos, &vocab->posToInt,
                                                 &vocab->intToPos);
          // add an empty string for any token except OOVs (it is easy to
          // recover the surface form of non-OOV using intToWords(id)).
          current_sent_str.push_back("");
          unsigned word_id;
          // OOV word
          if (USE_SPELLING) {
            word_id = get_or_add_word(word);
          } else {
            auto word_iter = vocab->wordsToInt.find(word);
            if (word_iter == vocab->wordsToInt.end()) {
              // save the surface form of this OOV before overwriting it.
              current_sent_str[current_sent_str.size() - 1] = word;
              word_id = vocab->wordsToInt[vocab->UNK];
            } else {
              word_id = word_iter->second;
            }
          }
          current_sent.push_back(word_id);
          current_sent_pos.push_back(pos_id);
        } while (iss);
      }
      initial = false;
    } else if (count == 1) {
      auto action_iter = find(vocab->actions.begin(), vocab->actions.end(),
                              lineS);
      if (action_iter != vocab->actions.end()) {
        unsigned actionIndex = distance(vocab->actions.begin(), action_iter);
        correct_act_sent[sentence].push_back(actionIndex);
      } else {
        // TODO: right now, new actions which haven't been observed in
        // training are not added to correct_act_sentDev. This may be a
        // problem if the training data is little.
      }
      count = 0;
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentences[sentence] = current_sent;
    sentencesPos[sentence] = current_sent_pos;
    sentencesSurfaceForms[sentence] = current_sent_str;
    sentence++;
  }

  actionsFile.close();
}

} // namespace cpyp
