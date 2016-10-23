#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>

namespace cpyp {

class ParserVocabulary {
  friend class Corpus;
public:
  typedef std::map<std::string, unsigned> StrToIntMap;

  // String literals
  static constexpr const char* UNK = "<UNK>";
  static constexpr const char* BAD0 = "<BAD0>";

  StrToIntMap wordsToInt;
  std::vector<std::string> intToWords;

  StrToIntMap posToInt;
  std::vector<std::string> intToPos;

  StrToIntMap charsToInt;
  std::vector<std::string> intToChars;

  std::vector<std::string> actions;

  ParserVocabulary() {
    AddEntry(BAD0, &wordsToInt, &intToWords);
    AddEntry(UNK, &wordsToInt, &intToWords);
    // For some reason, original LSTM parser had char and POS lists starting at
    // index 1.
    AddEntry("", &posToInt, &intToPos);
    AddEntry("", &charsToInt, &intToChars);
    AddEntry(BAD0, &charsToInt, &intToChars);
  }

  inline unsigned CountPOS() { return posToInt.size(); }
  inline unsigned CountWords() { return wordsToInt.size(); }
  inline unsigned CountChars() { return charsToInt.size(); }
  inline unsigned CountActions() { return actions.size(); }

private:
  static inline int AddEntry(const std::string& str, StrToIntMap* map,
                             std::vector<std::string>* indexed_list) {
    int new_id = indexed_list->size();
    map->insert({str, new_id});
    indexed_list->push_back(str);
    return new_id;
  }

  static inline unsigned GetOrAddEntry(const std::string& str, StrToIntMap* map,
                                       std::vector<std::string>* indexed_list) {
    auto entry_iter = map->find(str);
    if (entry_iter == map->end()) {
      return AddEntry(str, map, indexed_list);
    } else {
      return entry_iter->second;
    }
  }
};

class Corpus {
  //typedef std::unordered_map<std::string, unsigned, std::hash<std::string> > Map;
  // typedef std::unordered_map<unsigned,std::string, std::hash<std::string> > ReverseMap;
public:
  bool USE_SPELLING = false;

  std::map<int, std::vector<unsigned>> correct_act_sent;
  std::map<int, std::vector<unsigned>> sentences;
  std::map<int, std::vector<unsigned>> sentencesPos;
  std::map<int, std::vector<std::string>> sentencesSurfaceForms;

  /*
  std::map<unsigned,unsigned>* headsTraining;
  std::map<unsigned,std::string>* labelsTraining;

  std::map<unsigned,unsigned>*  headsParsing;
  std::map<unsigned,std::string>* labelsParsing;
  //*/

  ParserVocabulary *vocab;

public:
  Corpus(const std::string& file, ParserVocabulary* vocab) : vocab(vocab) {
    load_correct_actions(file);
  }


  inline unsigned UTF8Len(unsigned char x) {
    if (x < 0x80) return 1;
    else if ((x >> 5) == 0x06) return 2;
    else if ((x >> 4) == 0x0e) return 3;
    else if ((x >> 3) == 0x1e) return 4;
    else if ((x >> 2) == 0x3e) return 5;
    else if ((x >> 1) == 0x7e) return 6;
    else return 0;
  }


  inline void load_correct_actions(const std::string& file) {
    std::cerr << "Loading corpus from " << file << "...";
    std::ifstream actionsFile(file);
    //correct_act_sent=new vector<vector<unsigned>>();
    std::string lineS;

    int count = -1;
    int sentence = -1;
    bool initial = false;
    bool first = true;

    std::vector<unsigned> current_sent;
    std::vector<unsigned> current_sent_pos;
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
          std::istringstream iss(lineS);
          do {
            std::string word;
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
            if (posIndex == std::string::npos) {
              std::cerr << "cant find the dash in '" << word << "'"
                        << std::endl;
            }
            assert(posIndex != std::string::npos);
            std::string pos = word.substr(posIndex + 1);
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
                std::string next_utf8_char = word.substr(j, char_utf8_len);
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
            std::vector<unsigned> a = correct_act_sent[sentence];
            a.push_back(i);
            correct_act_sent[sentence] = a;
            found = true;
            break;
          }
          i++;
        }
        if (!found) {
          vocab->actions.push_back(lineS);
          std::vector<unsigned> a = correct_act_sent[sentence];
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
    /*  std::string oov="oov";
     vocab->posToInt[oov]=maxPos;
     vocab->intToPos[maxPos]=oov;
     npos=maxPos;
     maxPos++;
     wordsToInt[oov]=max;
     intToWords[max]=oov;
     nwords=max;
     max++;*/

    std::cerr << "done." << "\n";
    for (auto a : vocab->actions) {
      std::cerr << a << "\n";
    }
    std::cerr << "# of actions: " << vocab->CountActions() << "\n";
    std::cerr << "# of words: " << vocab->CountWords() << "\n";
    for (unsigned i = 0; i < vocab->intToPos.size(); i++) {
      std::cerr << i << ":" << vocab->intToPos[i] << "\n";
    }
  }

  inline unsigned get_or_add_word(const std::string& word) {
    return vocab->GetOrAddEntry(word, &vocab->wordsToInt, &vocab->intToWords);
  }

  inline void load_correct_actionsDev(std::string file) {
    std::cerr << "Loading dev corpus from " << file << "...";
    std::ifstream actionsFile(file);
    std::string lineS;

    assert(vocab->posToInt.size() > 1);
    assert(vocab->wordsToInt.size() > 3);
    int count = -1;
    int sentence = -1;
    bool initial = false;
    bool first = true;
    std::vector<unsigned> current_sent;
    std::vector<unsigned> current_sent_pos;
    std::vector<std::string> current_sent_str;
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
          std::istringstream iss(lineS);
          do {
            std::string word;
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
            assert(posIndex != std::string::npos);
            std::string pos = word.substr(posIndex + 1);
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
        auto actionIter = std::find(vocab->actions.begin(),
                                    vocab->actions.end(), lineS);
        if (actionIter != vocab->actions.end()) {
          unsigned actionIndex = std::distance(vocab->actions.begin(),
                                               actionIter);
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

  void ReplaceStringInPlace(std::string& subject, const std::string& search,
                            const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
      subject.replace(pos, search.length(), replace);
      pos += replace.length();
    }
  }


/*  inline unsigned max() const { return words_.size(); }
  inline unsigned size() const { return words_.size(); }
  inline unsigned count(const std::string& word) const { return d_.count(word); }*/

/*  static bool is_ws(char x) {
    return (x == ' ' || x == '\t');
  }

  inline void ConvertWhitespaceDelimitedLine(const std::string& line, std::vector<unsigned>* out) {
    size_t cur = 0;
    size_t last = 0;
    int state = 0;
    out->clear();
    while(cur < line.size()) {
      if (is_ws(line[cur++])) {
        if (state == 0) continue;
        out->push_back(Convert(line.substr(last, cur - last - 1)));
        state = 0;
      } else {
        if (state == 1) continue;
        last = cur - 1;
        state = 1;
      }
    }
    if (state == 1)
      out->push_back(Convert(line.substr(last, cur - last)));
  }

  inline unsigned Convert(const std::string& word, bool frozen = false) {
    Map::iterator i = d_.find(word);
    if (i == d_.end()) {
      if (frozen)
        return 0;
      words_.push_back(word);
      d_[word] = words_.size();
      return words_.size();
    } else {
      return i->second;
    }
  }

  inline const std::string& Convert(const unsigned id) const {
    if (id == 0) return b0_;
    return words_[id-1];
  }
  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar & b0_;
    ar & words_;
    ar & d_;
  }
 private:
  std::string b0_;
  std::vector<std::string> words_;
  Map d_;*/
};

/*void ReadFromFile(const std::string& filename,
                  Corpus* d,
                  std::vector<std::vector<unsigned> >* src,
                  std::set<unsigned>* src_vocab) {
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int lc = 0;
  while(getline(in, line)) {
    ++lc;
    src->push_back(std::vector<unsigned>());
    d->ConvertWhitespaceDelimitedLine(line, &src->back());
    for (unsigned i = 0; i < src->back().size(); ++i) src_vocab->insert(src->back()[i]);
  }
}

void ReadParallelCorpusFromFile(const std::string& filename,
                                Corpus* d,
                                std::vector<std::vector<unsigned> >* src,
                                std::vector<std::vector<unsigned> >* trg,
                                std::set<unsigned>* src_vocab,
                                std::set<unsigned>* trg_vocab) {
  src->clear();
  trg->clear();
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int lc = 0;
  std::vector<unsigned> v;
  const unsigned kDELIM = d->Convert("|||");
  while(getline(in, line)) {
    ++lc;
    src->push_back(std::vector<unsigned>());
    trg->push_back(std::vector<unsigned>());
    d->ConvertWhitespaceDelimitedLine(line, &v);
    unsigned j = 0;
    while(j < v.size() && v[j] != kDELIM) {
      src->back().push_back(v[j]);
      src_vocab->insert(v[j]);
      ++j;
    }
    if (j >= v.size()) {
      std::cerr << "Malformed input in parallel corpus: " << filename << ":" << lc << std::endl;
      abort();
    }
    ++j;
    while(j < v.size()) {
      trg->back().push_back(v[j]);
      trg_vocab->insert(v[j]);
      ++j;
    }
  }
}*/

} // namespace

#endif
