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
#include "corpus.h"
#include "lstm-parser.h"

using namespace cnn::expr;
using namespace cnn;
using namespace std;
using namespace lstm_parser;
namespace po = boost::program_options;
namespace io = boost::iostreams;

volatile bool requested_stop = false;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,t", po::value<string>(),
         "List of transitions - training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus path")
        ("test_data,T", po::value<string>(), "Test corpus path")
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
        ("train,r", "Whether training should be run")
        ("test,e", "Whether the model should be tested."
                   " If train is true, this tests on dev data.")
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
    cerr << "Please specify --training_data (-t): this is required to determine"
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
  const bool load_model = conf.count("model");

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
    cerr << "INVALID SELECTION" << endl;
    abort();
  }
  if (unk_prob < 0. || unk_prob > 1.) {
    cerr << "Invalid unknown word substitution probability: " << unk_prob
         << endl;
    abort();
  }
  // If we're testing, we have to either be loading or training a model.
  if (test && !load_model && !train) {
    cerr << "No model specified for testing!" << endl;
    abort();
  }

  const string words = load_model ? "" : conf["words"].as<string>();
  LSTMParser parser(cmd_options, words, false);
  if (load_model) {
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

  if (train) {
    signal(SIGINT, signal_callback_handler);
    Corpus training_corpus(conf["training_data"].as<string>(),
                           &parser.vocab, true);
    parser.FinalizeVocab();
    cerr << "Total number of words: " << training_corpus.vocab->CountWords()
         << endl;
    // OOV words will be replaced by UNK tokens
    Corpus dev_corpus(conf["dev_data"].as<string>(), &parser.vocab, false);

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
    parser.Train(training_corpus, dev_corpus, unk_prob, fname, compress,
                 &requested_stop);
    if (test) { // do test evaluation
      parser.Test(dev_corpus);
    }
  }
  else if (test) { // actually run the parser
    // TODO: make this run parser on test data.
    parser.FinalizeVocab();
    Corpus dev_corpus(conf["dev_data"].as<string>(), &parser.vocab, false);
    parser.Test(dev_corpus);

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
