#ifndef CORPUS_H
#define CORPUS_H

#include <boost/serialization/split_member.hpp>
#include <stddef.h>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace lstm_parser {

class CorpusVocabulary {
  friend class Corpus;
public:
  typedef std::map<std::string, unsigned> StrToIntMap;

  // String literals
  static const std::string UNK;
  static const std::string BAD0;

  StrToIntMap words_to_int;
  std::vector<std::string> int_to_words;
  std::vector<bool> int_to_training_word; // Stores whether each word is OOV

  StrToIntMap pos_to_int;
  std::vector<std::string> int_to_pos;

  StrToIntMap chars_to_int;
  std::vector<std::string> int_to_chars;

  std::vector<std::string> actions;

  CorpusVocabulary() : int_to_training_word({true, true}) {
    AddEntry(BAD0, &words_to_int, &int_to_words);
    AddEntry(UNK, &words_to_int, &int_to_words);
    AddEntry(BAD0, &chars_to_int, &int_to_chars);
  }

  inline unsigned CountPOS() { return pos_to_int.size(); }
  inline unsigned CountWords() { return words_to_int.size(); }
  inline unsigned CountChars() { return chars_to_int.size(); }
  inline unsigned CountActions() { return actions.size(); }

  inline unsigned GetOrAddWord(const std::string& word,
                               bool record_as_training=false) {
    unsigned num_words = CountWords();
    unsigned word_id = GetOrAddEntry(word, &words_to_int, &int_to_words);
    if (CountWords() > num_words) { // a word was added
      int_to_training_word.push_back(record_as_training);
    } else {
      // Should get optimized out when record_as_training is literal false.
      int_to_training_word[word_id] = int_to_training_word[word_id]
          || record_as_training;
    }
    return word_id;
  }

private:
  friend class boost::serialization::access;

  template<class Archive, class VocabType>
  // Shared code: serialize the number-to-string mappings, from which the
  // reverse mappings can be reconstructed.
  static void serializeLists(Archive& ar, const unsigned int version,
                             VocabType* vocab) {
    ar & vocab->int_to_words;
    ar & vocab->int_to_pos;
    ar & vocab->int_to_chars;
    ar & vocab->int_to_training_word;
    ar & vocab->actions;
  }

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    serializeLists(ar, version, this);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    unsigned num_existing_words = int_to_words.size();

    words_to_int.clear();
    int_to_words.clear();
    pos_to_int.clear();
    int_to_pos.clear();
    chars_to_int.clear();
    int_to_chars.clear();

    serializeLists(ar, version, this);
    if (int_to_words.size() < num_existing_words) {
      std::cerr << "WARNING: lost " << num_existing_words - int_to_words.size()
                << " words when loading model" << std::endl;
    }

    // Now reconstruct the reverse mappings.
    for (size_t i = 0; i < int_to_words.size(); ++i)
      words_to_int[int_to_words[i]] = i;
    for (size_t i = 0; i < int_to_pos.size(); ++i)
      pos_to_int[int_to_pos[i]] = i;
    for (size_t i = 0; i < int_to_chars.size(); ++i)
      chars_to_int[int_to_chars[i]] = i;
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
    // assert(intToWords.size() == wordsToInt.size());
    auto entry_iter = map->find(str);
    if (entry_iter == map->end()) {
      return AddEntry(str, map, indexed_list);
    } else {
      return entry_iter->second;
    }
  }
};


class Corpus {
  // typedef std::unordered_map<std::string, unsigned,
  //                            std::hash<std::string> > Map;
  // typedef std::unordered_map<unsigned, std::string,
  //                            std::hash<std::string> > ReverseMap;
public:
  bool USE_SPELLING = false;

  std::vector<std::vector<unsigned>> correct_act_sent;
  std::vector<std::vector<unsigned>> sentences;
  std::vector<std::vector<unsigned>> sentences_pos;
  std::vector<std::vector<std::string>> sentences_surface_forms;
  std::set<unsigned> singletons;

  /*
  std::map<unsigned,unsigned>* headsTraining;
  std::map<unsigned,std::string>* labelsTraining;

  std::map<unsigned,unsigned>*  headsParsing;
  std::map<unsigned,std::string>* labelsParsing;
  //*/

  CorpusVocabulary *vocab;

  Corpus(const std::string& file, CorpusVocabulary* vocab, bool is_training) :
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

  void CountSingletons();
};

} // namespace lstm_parser

#endif
