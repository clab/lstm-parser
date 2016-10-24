#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <boost/serialization/split_member.hpp>
#include <stddef.h>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cpyp {

class ParserVocabulary {
  friend class Corpus;
public:
  typedef std::map<std::string, unsigned> StrToIntMap;

  // String literals
  static const std::string UNK;
  static const std::string BAD0;

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
    AddEntry(BAD0, &charsToInt, &intToChars);
  }

  inline unsigned CountPOS() { return posToInt.size(); }
  inline unsigned CountWords() { return wordsToInt.size(); }
  inline unsigned CountChars() { return charsToInt.size(); }
  inline unsigned CountActions() { return actions.size(); }

  inline unsigned GetOrAddWord(const std::string& word) {
    return GetOrAddEntry(word, &wordsToInt, &intToWords);
  }

private:
  friend class boost::serialization::access;

  template<class Archive, class ParserType>
  // Shared code: serialize the number-to-string mappings, from which the
  // reverse mappings can be reconstructed.
  static void serializeLists(Archive& ar, const unsigned int version,
                             ParserType* parser) {
    ar & parser->intToWords;
    ar & parser->intToPos;
    ar & parser->intToChars;
    ar & parser->actions;
  }

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    serializeLists(ar, version, this);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    unsigned num_existing_words = intToWords.size();

    wordsToInt.clear();
    intToWords.clear();
    posToInt.clear();
    intToPos.clear();
    charsToInt.clear();
    intToChars.clear();

    serializeLists(ar, version, this);
    if (intToWords.size() < num_existing_words) {
      std::cerr << "WARNING: lost " << num_existing_words - intToWords.size()
                << " words when loading model" << std::endl;
    }

    // Now reconstruct the reverse mappings.
    for (size_t i = 0; i < intToWords.size(); ++i)
      wordsToInt[intToWords[i]] = i;
    for (size_t i = 0; i < intToPos.size(); ++i)
      posToInt[intToPos[i]] = i;
    for (size_t i = 0; i < intToChars.size(); ++i)
      charsToInt[intToChars[i]] = i;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

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

  std::vector<std::vector<unsigned>> correct_act_sent;
  std::vector<std::vector<unsigned>> sentences;
  std::vector<std::vector<unsigned>> sentencesPos;
  std::vector<std::vector<std::string>> sentencesSurfaceForms;

  /*
  std::map<unsigned,unsigned>* headsTraining;
  std::map<unsigned,std::string>* labelsTraining;

  std::map<unsigned,unsigned>*  headsParsing;
  std::map<unsigned,std::string>* labelsParsing;
  //*/

  ParserVocabulary *vocab;

  Corpus(const std::string& file, ParserVocabulary* vocab, bool is_training) :
      vocab(vocab) {
    load_correct_actions(file, is_training);
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

private:
  void load_correct_actions(const std::string& file, bool is_training);

  static inline void ReplaceStringInPlace(std::string& subject,
                                          const std::string& search,
                                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
      subject.replace(pos, search.length(), replace);
      pos += replace.length();
    }
  }
};

} // namespace

#endif
