#include <boost/program_options.hpp>
#include <signal.h>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "corpus.h"
#include "lstm-parser.h"

using namespace std;
using namespace lstm_parser;
namespace po = boost::program_options;

volatile bool requested_stop = false;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,t", po::value<string>(),
         "List of transitions - training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus path")
        ("test_data,T", po::value<string>(), "Test corpus path")
        ("input_format,f", po::value<string>()->default_value("conll"),
         "Test corpus input format: currently supports only 'conll'")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1),
         "Unknown word strategy: 1 = singletons become UNK with probability"
         " unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2),
         "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
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
        ("test,s", "Whether the model should be tested on test_data.")
        ("evaluate,e", "Whether to evaluate the trained model on dev_data.")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(0);
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
  const bool train = conf.count("train");
  const bool test = conf.count("test");
  const bool evaluate = conf.count("evaluate");
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
    conf["unk_strategy"].as<unsigned>(),
    conf["unk_prob"].as<double>()
  };

  cerr << "Unknown word strategy: ";
  if (cmd_options.unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    cerr << "INVALID SELECTION" << endl;
    abort();
  }
  if (cmd_options.unk_prob < 0. || cmd_options.unk_prob > 1.) {
    cerr << "Invalid unknown word substitution probability: "
         << cmd_options.unk_prob << endl;
    abort();
  }
  // If we're testing/evaluating, we have to either be loading or training a
  // model.
  if ((test || evaluate) && !load_model && !train) {
    cerr << "No model specified for testing!" << endl;
    abort();
  }

  const string words = load_model ? "" : conf["words"].as<string>();
  unique_ptr<LSTMParser> parser;
  if (load_model) {
    parser.reset(new LSTMParser(conf["model"].as<string>()));
    if (parser->options != cmd_options) {
      // TODO: make this recognize the difference between a default option and
      // one that was actually specified on the command line, and only warn for
      // the latter.
      cerr << "WARNING: overriding command line neural network options with"
              " options saved in model" << endl;
    }
  } else { // generate a new model; don't load from file
    parser.reset(new LSTMParser(cmd_options, words, false));
  }

  unique_ptr<ParserTrainingCorpus> dev_corpus; // shared by train/evaluate

  if (train) {
    if (!conf.count("training_data") || !conf.count("dev_data")) {
      cerr << "Training requested, but training and dev data were not both"
              " specified!" << endl;
      abort();
    }

    signal(SIGINT, signal_callback_handler);
    ParserTrainingCorpus training_corpus(&parser->vocab,
                                         conf["training_data"].as<string>(),
                                         true);
    parser->FinalizeVocab();
    cerr << "Total number of words: " << training_corpus.vocab->CountWords()
         << endl;
    // OOV words will be replaced by UNK tokens
    dev_corpus.reset(
        new ParserTrainingCorpus(&parser->vocab, conf["dev_data"].as<string>(),
                                 false));

    ostringstream os;
    os << "parser_" << (parser->options.use_pos ? "pos" : "nopos")
       << '_' << parser->options.layers
       << '_' << parser->options.input_dim
       << '_' << parser->options.hidden_dim
       << '_' << parser->options.action_dim
       << '_' << parser->options.lstm_input_dim
       << '_' << parser->options.pos_dim
       << '_' << parser->options.rel_dim
       << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Writing parameters to file: " << fname << endl;
    parser->Train(training_corpus, *dev_corpus, parser->options.unk_prob, fname,
                  &requested_stop);
  }

  if (evaluate) {
    if (!conf.count("dev_data")) {
      cerr << "Evaluation requested, but no dev data was specified!" << endl;
      abort();
    }

    parser->FinalizeVocab();
    cerr << "Evaluating model on " << conf["dev_data"].as<string>() << endl;
    if (!train) { // Didn't already load dev corpus for training
      dev_corpus.reset(
          new ParserTrainingCorpus(&parser->vocab,
                                   conf["dev_data"].as<string>(), false));
    }
    parser->Evaluate(*dev_corpus);
  }

  if (test) { // actually run the parser on test data
    if (!conf.count("test_data")) {
      cerr << "Test requested, but no test data was specified!" << endl;
      abort();
    }

    parser->FinalizeVocab();
    // Set up reader as pointer to make it easier to add different reader types
    // later.
    unique_ptr<CorpusReader> reader;
    if (conf["input_format"].as<string>() == "conll") {
      reader.reset(new ConllUCorpusReader());
    } else {
      cerr << "Unrecognized input format: " << conf["input_format"].as<string>()
           << endl;
      abort();
    }
    Corpus test_corpus(&parser->vocab, *reader, conf["test_data"].as<string>());
    parser->Test(test_corpus);
  }
}
