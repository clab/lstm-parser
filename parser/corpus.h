#ifndef CORPUS_H
#define CORPUS_H

#include <boost/algorithm/string/predicate.hpp>
#include <boost/serialization/split_member.hpp>
#include <exception>
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
  static const std::string BAD0;
  static const std::string UNK;
  static const std::string ROOT;

  StrToIntMap words_to_int;
  std::vector<std::string> int_to_words;
  std::vector<bool> int_to_training_word; // Stores whether each word is OOV

  StrToIntMap pos_to_int;
  std::vector<std::string> int_to_pos;

  StrToIntMap chars_to_int;
  std::vector<std::string> int_to_chars;

  std::vector<std::string> actions;
  std::vector<std::string> actions_to_arc_labels;

  CorpusVocabulary() : int_to_training_word({true, true}) {
    AddEntry(BAD0, &words_to_int, &int_to_words);
    AddEntry(UNK, &words_to_int, &int_to_words);
    AddEntry(BAD0, &chars_to_int, &int_to_chars);
  }

  inline unsigned CountPOS() { return pos_to_int.size(); }
  inline unsigned CountWords() { return words_to_int.size(); }
  inline unsigned CountChars() { return chars_to_int.size(); }
  inline unsigned CountActions() { return actions.size(); }

  inline unsigned GetWord(const std::string& word) const {
    auto word_iter = words_to_int.find(word);
    if (word_iter == words_to_int.end()) {
      return words_to_int.find(CorpusVocabulary::UNK)->second;
    } else {
      return word_iter->second;
    }
  }

  inline unsigned GetPOS(const std::string& word) const {
    auto pos_iter = pos_to_int.find(word);
    if (pos_iter == pos_to_int.end()) {
      return -1;
    } else {
      return pos_iter->second;
    }
  }

  inline unsigned GetOrAddWord(const std::string& word,
                               bool record_as_training=false) {
    unsigned num_words = CountWords();
    unsigned word_id = GetOrAddEntry(word, &words_to_int, &int_to_words);
    if (CountWords() > num_words) { // a word was added
      int_to_training_word.push_back(record_as_training);
    } else {
      // Should get optimized out when record_as_training is literal false.
      int_to_training_word[word_id] =
          (int_to_training_word[word_id] || record_as_training);
    }
    return word_id;
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

  static inline std::string GetLabelForAction(const std::string& action) {
    if (boost::starts_with(action, "RIGHT-ARC") ||
        boost::starts_with(action, "LEFT-ARC")) {
      size_t first_char_in_rel = action.find('(') + 1;
      size_t last_char_in_rel = action.rfind(')') - 1;
      return action.substr(
          first_char_in_rel, last_char_in_rel - first_char_in_rel + 1);
    } else {
      return "NONE";
    }
  }

private:
  friend class boost::serialization::access;

  template<class Archive, class VocabType>
  // Shared code: serialize the number-to-string mappings, from which the
  // reverse mappings can be reconstructed.
  static void SerializeLists(Archive& ar, const unsigned int version,
                             VocabType* vocab) {
    ar & vocab->int_to_words;
    ar & vocab->int_to_pos;
    ar & vocab->int_to_chars;
    ar & vocab->int_to_training_word;
    ar & vocab->actions;
  }

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    SerializeLists(ar, version, this);
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

    SerializeLists(ar, version, this);
    if (int_to_words.size() < num_existing_words) {
      std::cerr << "WARNING: lost " << num_existing_words - int_to_words.size()
                << " words when loading model" << std::endl;
    }

    // Now reconstruct the reverse mappings...
    for (size_t i = 0; i < int_to_words.size(); ++i)
      words_to_int[int_to_words[i]] = i;
    for (size_t i = 0; i < int_to_pos.size(); ++i)
      pos_to_int[int_to_pos[i]] = i;
    for (size_t i = 0; i < int_to_chars.size(); ++i)
      chars_to_int[int_to_chars[i]] = i;

    // ...and the arc labels.
    for (const std::string& action : actions) {
      actions_to_arc_labels.push_back(GetLabelForAction(action));
    }
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

  static inline int AddEntry(const std::string& str, StrToIntMap* map,
                             std::vector<std::string>* indexed_list) {
    int new_id = indexed_list->size();
    map->insert({str, new_id});
    indexed_list->push_back(str);
    return new_id;
  }
};


class Corpus; // forward declaration

class CorpusReader {
public:
  virtual void ReadSentences(const std::string& file, Corpus* corpus) const = 0;
  virtual ~CorpusReader() {};
};


class ConllUCorpusReader : public CorpusReader {
public:
  class ConllFormatException : public std::logic_error {
  public:
    ConllFormatException(const std::string& what) : std::logic_error(what) {}
  };

  virtual void ReadSentences(const std::string& file, Corpus* corpus) const;
  virtual ~ConllUCorpusReader() {};
};


class Corpus {
public:
  // Store root tokens with unsigned ID -1 internally to make root come last
  // when iterating over a list of tokens in order of IDs.
  static constexpr unsigned ROOT_TOKEN_ID = -1;

  std::vector<std::map<unsigned, unsigned>> sentences;
  std::vector<std::map<unsigned, unsigned>> sentences_pos;
  std::vector<std::map<unsigned, std::string>> sentences_unk_surface_forms;
  CorpusVocabulary* vocab;

  Corpus(CorpusVocabulary* vocab, const CorpusReader& reader,
         const std::string& file) :
      vocab(vocab) {
    reader.ReadSentences(file, this);
  }

protected:
  // Corpus for subclasses to inherit and use. Subclasses are then responsible
  // for doing any corpus-reading or setup.
  Corpus(CorpusVocabulary* vocab) : vocab(vocab) {}

};


class TrainingCorpus : public Corpus {
public:
  friend class OracleTransitionsCorpusReader;

  bool USE_SPELLING = false;

  std::vector<std::vector<unsigned>> correct_act_sent;
  std::set<unsigned> singletons;

  TrainingCorpus(CorpusVocabulary* vocab, const std::string& file,
                 bool is_training) :
      Corpus(vocab) {
    OracleTransitionsCorpusReader reader(is_training);
    reader.ReadSentences(file, this);
  }

private:
  class OracleTransitionsCorpusReader : public CorpusReader {
  public:
    OracleTransitionsCorpusReader(bool is_training) :
        is_training(is_training) {}

    virtual void ReadSentences(const std::string& file, Corpus* corpus) const {
      TrainingCorpus* training_corpus = static_cast<TrainingCorpus *>(corpus);
      LoadCorrectActions(file, training_corpus);
    }

    virtual ~OracleTransitionsCorpusReader() {};

    static inline unsigned UTF8Len(unsigned char x) {
      if (x < 0x80) return 1;
      else if ((x >> 5) == 0x06) return 2;
      else if ((x >> 4) == 0x0e) return 3;
      else if ((x >> 3) == 0x1e) return 4;
      else if ((x >> 2) == 0x3e) return 5;
      else if ((x >> 1) == 0x7e) return 6;
      else return 0;
    }
  private:
    bool is_training; // can be dev rather than actual training
    void LoadCorrectActions(const std::string& file,
                            TrainingCorpus* corpus) const;
  };

  static inline void ReplaceStringInPlace(std::string* subject,
                                          const std::string& search,
                                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject->find(search, pos)) != std::string::npos) {
      subject->replace(pos, search.length(), replace);
      pos += replace.length();
    }
  }

  void CountSingletons();
};

} // namespace lstm_parser

#endif
