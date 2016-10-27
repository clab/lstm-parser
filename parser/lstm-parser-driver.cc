#include <boost/algorithm/string/predicate.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/program_options.hpp>
#include <signal.h>
#include <stddef.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <ratio>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../cnn/cnn/cnn.h"
#include "../cnn/cnn/init.h"
#include "../cnn/cnn/model.h"
#include "../cnn/cnn/tensor.h"
#include "../cnn/cnn/training.h"
#include "c2.h"
#include "lstm-parser.h"

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;
namespace io = boost::iostreams;

volatile bool requested_stop = false;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(),
         "List of transitions - training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1),
         "Unknown word strategy: 1 = singletons become UNK with probability"
         " unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2),
         "Probably with which to replace singletons with UNK in training data")
         ("model,m", po::value<string>(), "Load saved model from this file")
         ("compress,c", "Whether to compress the model when saving")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("layers,l", po::value<unsigned>()->default_value(2),
         "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16),
         "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32),
         "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64),
         "hidden dimension")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(10),
         "relation dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60),
         "LSTM input dimension")
        ("train,t", "Whether training should be run")
        ("test,e", "Whether the model should be tested on dev data")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --training_data (-T): this is required to determine"
            " the vocabulary mapping, even if the parser is used in prediction"
            " mode.\n";
    exit(1);
  }
}


void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}


unsigned compute_correct(const map<int, int>& ref, const map<int, int>& hyp,
                         unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second)
      ++res;
  }
  return res;
}


void output_conll(const vector<unsigned>& sentence, const vector<unsigned>& pos,
                  const vector<string>& sentenceUnkStrings,
                  const vector<string>& intToWords,
                  const vector<string>& intToPos,
                  const map<string, unsigned>& wordsToInt,
                  const map<int, int>& hyp, const map<int, string>& rel_hyp) {
  for (unsigned i = 0; i < (sentence.size() - 1); ++i) {
    auto index = i + 1;
    const unsigned int unk_word =
        wordsToInt.find(cpyp::ParserVocabulary::UNK)->second;
    assert(i < sentenceUnkStrings.size() &&
           ((sentence[i] == unk_word && sentenceUnkStrings[i].size() > 0) ||
            (sentence[i] != unk_word && sentenceUnkStrings[i].size() == 0 &&
             intToWords.size() > sentence[i])));
    string wit = (sentenceUnkStrings[i].size() > 0) ?
				  sentenceUnkStrings[i] : intToWords[sentence[i]];
    const string& pos_tag = intToPos[pos[i]];
    assert(hyp.find(i) != hyp.end());
    auto hyp_head = hyp.find(i)->second + 1;
    if (hyp_head == (int) sentence.size())
      hyp_head = 0;
    auto hyp_rel_it = rel_hyp.find(i);
    assert(hyp_rel_it != rel_hyp.end());
    auto hyp_rel = hyp_rel_it->second;
    size_t first_char_in_rel = hyp_rel.find('(') + 1;
    size_t last_char_in_rel = hyp_rel.rfind(')') - 1;
    hyp_rel = hyp_rel.substr(first_char_in_rel,
                             last_char_in_rel - first_char_in_rel + 1);
    cout << index << '\t'       // 1. ID
         << wit << '\t'         // 2. FORM
         << "_" << '\t'         // 3. LEMMA
         << "_" << '\t'         // 4. CPOSTAG
         << pos_tag << '\t'     // 5. POSTAG
         << "_" << '\t'         // 6. FEATS
         << hyp_head << '\t'    // 7. HEAD
         << hyp_rel << '\t'     // 8. DEPREL
         << "_" << '\t'         // 9. PHEAD
         << "_" << endl;        // 10. PDEPREL
  }
  cout << endl;
}


void do_train(ParserBuilder* parser, const cpyp::Corpus& corpus,
              const cpyp::Corpus& dev_corpus, const double unk_prob,
              const string& fname, bool compress) {
  bool softlinkCreated = false;
  int best_correct_heads = 0;
  unsigned status_every_i_iterations = 100;
  signal(SIGINT, signal_callback_handler);
  SimpleSGDTrainer sgd(&parser->model);
  //MomentumSGDTrainer sgd(model);
  sgd.eta_decay = 0.08;
  //sgd.eta_decay = 0.05;
  unsigned num_sentences = corpus.sentences.size();
  vector<unsigned> order(corpus.sentences.size());
  for (unsigned i = 0; i < corpus.sentences.size(); ++i)
    order[i] = i;
  double tot_seen = 0;
  status_every_i_iterations = min(status_every_i_iterations, num_sentences);
  cerr << "NUMBER OF TRAINING SENTENCES: " << num_sentences << endl;
  unsigned trs = 0;
  double right = 0;
  double llh = 0;
  bool first = true;
  int iter = -1;
  time_t time_start = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::now());
  cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z")
       << endl;

  unsigned si = num_sentences;
  while (!requested_stop) {
    ++iter;
    for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
      if (si == num_sentences) {
        si = 0;
        if (first) {
          first = false;
        } else {
          sgd.update_epoch();
        }
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }
      tot_seen += 1;
      const vector<unsigned>& sentence = corpus.sentences[order[si]];
      vector<unsigned> tsentence(sentence);
      if (parser->options.unk_strategy == 1) {
        for (auto& w : tsentence) {
          if (corpus.singletons.count(w) && cnn::rand01() < unk_prob) {
            w = parser->kUNK;
          }
        }
      }
      const vector<unsigned>& sentencePos = corpus.sentencesPos[order[si]];
      const vector<unsigned>& actions = corpus.correct_act_sent[order[si]];
      ComputationGraph hg;
      parser->LogProbParser(&hg, sentence, tsentence, sentencePos, actions,
                            corpus.vocab->actions, corpus.vocab->intToWords,
                            &right);
      double lp = as_scalar(hg.incremental_forward());
      if (lp < 0) {
        cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp
            << endl;
        assert(lp >= 0.0);
      }
      hg.backward();
      sgd.update(1.0);
      llh += lp;
      ++si;
      trs += actions.size();
    }
    sgd.status();
    time_t time_now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    cerr << "update #" << iter << " (epoch " << (tot_seen / num_sentences)
         << " |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "
         << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs
         << endl;
    llh = trs = right = 0;
    static int logc = 0;
    ++logc;
    if (logc % 25 == 1) {
      // report on dev set
      unsigned dev_size = dev_corpus.sentences.size();
      // dev_size = 100;
      double llh = 0;
      double trs = 0;
      double right = 0;
      double correct_heads = 0;
      double total_heads = 0;
      auto t_start = std::chrono::high_resolution_clock::now();
      for (unsigned sii = 0; sii < dev_size; ++sii) {
        const vector<unsigned>& sentence = dev_corpus.sentences[sii];
        const vector<unsigned>& sentencePos = dev_corpus.sentencesPos[sii];
        const vector<unsigned>& actions = dev_corpus.correct_act_sent[sii];
        vector<unsigned> tsentence(sentence); // sentence with OOVs replaced
        for (unsigned& word_id : tsentence) {
          if (!parser->vocab.intToTrainingWord[word_id]) {
            word_id = parser->kUNK;
          }
        }
        ComputationGraph hg;
        vector<unsigned> pred = parser->LogProbParser(
            &hg, sentence, tsentence, sentencePos, vector<unsigned>(),
            dev_corpus.vocab->actions, dev_corpus.vocab->intToWords, &right);

        double lp = 0;
        llh -= lp;
        trs += actions.size();
        map<int, int> ref = parser->ComputeHeads(sentence.size(), actions,
                                                 dev_corpus.vocab->actions);
        map<int, int> hyp = parser->ComputeHeads(sentence.size(), pred,
                                                 dev_corpus.vocab->actions);
        correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
        total_heads += sentence.size() - 1;
      }
      auto t_end = std::chrono::high_resolution_clock::now();
      cerr << "  **dev (iter=" << iter << " epoch="
           << (tot_seen / num_sentences) << ")\tllh=" << llh << " ppl: "
           << exp(llh / trs) << " err: " << (trs - right) / trs << " uas: "
           << (correct_heads / total_heads) << "\t[" << dev_size << " sents in "
           << std::chrono::duration<double, std::milli>(t_end - t_start).count()
           << " ms]" << endl;
      if (correct_heads > best_correct_heads) {
        best_correct_heads = correct_heads;
        ofstream out_file(fname);
        if (compress) {
          io::filtering_streambuf<io::output> filter;
          filter.push(io::gzip_compressor());
          filter.push(out_file);
          boost::archive::binary_oarchive oa(filter);
          oa << *parser;
        } else {
          boost::archive::text_oarchive oa(out_file);
          oa << *parser;
        }
        // Create a soft link to the most recent model in order to make it
        // easier to refer to it in a shell script.
        if (!softlinkCreated) {
          string softlink = "latest_model.params";
          if (compress)
            softlink += ".gz";
          if (system((string("rm -f ") + softlink).c_str()) == 0
              && system((string("ln -s ") + fname + " " + softlink).c_str())
                  == 0) {
            cerr << "Created " << softlink << " as a soft link to " << fname
                 << " for convenience." << endl;
          }
          softlinkCreated = true;
        }
      }
    }
  }
}


void do_test(ParserBuilder* parser, const cpyp::Corpus& corpus) {
  // do test evaluation
  double llh = 0;
  double trs = 0;
  double right = 0;
  double correct_heads = 0;
  double total_heads = 0;
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned corpus_size = corpus.sentences.size();
  for (unsigned sii = 0; sii < corpus_size; ++sii) {
    const vector<unsigned>& sentence = corpus.sentences[sii];
    const vector<unsigned>& sentencePos = corpus.sentencesPos[sii];
    const vector<string>& sentenceUnkStr = corpus.sentencesSurfaceForms[sii];
    const vector<unsigned>& actions = corpus.correct_act_sent[sii];
    vector<unsigned> tsentence(sentence); // sentence with OOVs replaced
    for (unsigned& word_id : tsentence) {
      if (!parser->vocab.intToTrainingWord[word_id]) {
        word_id = parser->kUNK;
      }
    }
    ComputationGraph cg;
    double lp = 0;
    vector<unsigned> pred;
    pred = parser->LogProbParser(&cg, sentence, tsentence, sentencePos,
                                 vector<unsigned>(), corpus.vocab->actions,
                                 corpus.vocab->intToWords, &right);
    llh -= lp;
    trs += actions.size();
    map<int, string> rel_ref;
    map<int, string> rel_hyp;
    map<int, int> ref = parser->ComputeHeads(sentence.size(), actions,
                                             corpus.vocab->actions, &rel_ref);
    map<int, int> hyp = parser->ComputeHeads(sentence.size(), pred,
                                             corpus.vocab->actions, &rel_hyp);
    output_conll(sentence, sentencePos, sentenceUnkStr,
                 corpus.vocab->intToWords, corpus.vocab->intToPos,
                 corpus.vocab->wordsToInt, hyp, rel_hyp);
    correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
    total_heads += sentence.size() - 1;
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs) << " err: "
       << (trs - right) / trs << " uas: " << (correct_heads / total_heads)
       << "\t[" << corpus_size << " sents in "
       << std::chrono::duration<double, std::milli>(t_end - t_start).count()
       << " ms]" << endl;
}


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
    cerr << ' ' << argv[i];
  cerr << endl;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  // Training options and operation options
  const double unk_prob = conf["unk_prob"].as<double>();
  const bool train = conf.count("train");
  const bool test = conf.count("test");
  const bool compress = conf.count("compress");

  ParserOptions cmd_options {
    conf.count("use_pos_tags"),
    conf["layers"].as<unsigned>(),
    conf["input_dim"].as<unsigned>(),
    conf["hidden_dim"].as<unsigned>(),
    conf["action_dim"].as<unsigned>(),
    conf["lstm_input_dim"].as<unsigned>(),
    conf["pos_dim"].as<unsigned>(),
    conf["rel_dim"].as<unsigned>(),
    conf["unk_strategy"].as<unsigned>()
  };

  cerr << "Unknown word strategy: ";
  if (cmd_options.unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    cerr << "INVALID SELECTION";
    abort();
  }
  assert(unk_prob >= 0. && unk_prob <= 1.);


  ParserBuilder parser(conf["words"].as<string>(), cmd_options, false);
  if (conf.count("model")) {
    const string& model_path = conf["model"].as<string>();
    cerr << "Loading model from " << model_path << endl;
    if (boost::algorithm::ends_with(model_path, ".gz")) {
      // It's a compressed stream.
      ifstream model_stream(model_path.c_str());
      io::filtering_streambuf<io::input> filter;
      filter.push(io::gzip_decompressor());
      filter.push(model_stream);
      boost::archive::binary_iarchive archive(filter);
      archive >> parser;
    } else {
      ifstream model_stream(model_path.c_str());
      boost::archive::text_iarchive archive(model_stream);
      archive >> parser;
    }

    if (parser.options != cmd_options) {
      // TODO: make this recognize the difference between a default option and
      // one that was actually specified on the command line, and only warn for
      // the latter.
      cerr << "WARNING: overriding command line network options with saved"
              " options" << endl;
    }
  }

  // TODO: make this conditional on whether we load training data (which now we
  // don't need to).
  cpyp::Corpus training_corpus(conf["training_data"].as<string>(),
                               &parser.vocab, true);
  parser.FinalizeVocab();

  cerr << "Total number of words: " << training_corpus.vocab->CountWords()
       << endl;

  // OOV words will be replaced by UNK tokens
  cpyp::Corpus dev_corpus(conf["dev_data"].as<string>(), &parser.vocab,
                          false);
  if (train) {
    ostringstream os;
    os << "parser_" << (parser.options.use_pos ? "pos" : "nopos")
       << '_' << parser.options.layers
       << '_' << parser.options.input_dim
       << '_' << parser.options.hidden_dim
       << '_' << parser.options.action_dim
       << '_' << parser.options.lstm_input_dim
       << '_' << parser.options.pos_dim
       << '_' << parser.options.rel_dim
       << "-pid" << getpid() << ".params";
    if (compress)
      os << ".gz";
    const string fname = os.str();
    cerr << "Writing parameters to file: " << fname << endl;
    do_train(&parser, training_corpus, dev_corpus, unk_prob, fname, compress);
  }
  if (test) { // do test evaluation
    do_test(&parser, dev_corpus);
  }

  /*
  for (unsigned i = 0; i < corpus.actions.size(); ++i) {
    cerr << corpus.actions[i] << '\t' << parser.p_r->values[i].transpose()
         << endl;
    cerr << corpus.actions[i] << '\t'
         << parser.p_p2a->values.col(i).transpose() << endl;
  }
  //*/
}
