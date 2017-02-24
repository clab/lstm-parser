#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <chrono>

#include <unordered_map>
#include <unordered_set>

#include <unistd.h>
#include <signal.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "c2.h"

cpyp::Corpus corpus;
volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned REL_DIM = 8;


unsigned LSTM_CHAR_OUTPUT_DIM = 100; //Miguel
bool USE_SPELLING = false;

bool USE_POS = false;

constexpr const char* ROOT_SYMBOL = "ROOT";
unsigned kROOT_SYMBOL = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned POS_SIZE = 0;

unsigned CHAR_SIZE = 255; //size of ascii chars... Miguel

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
        ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("rel_dim", po::value<unsigned>()->default_value(10), "relation dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("maxit,M", po::value<unsigned>()->default_value(8000), "Maximum number of training iterations")
        ("tolerance", po::value<double>()->default_value(-1.0), "Tolerance on dev uas for stopping training")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("use_spelling,S", "Use spelling model") //Miguel. Spelling model
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {

  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder buffer_lstm;
  LSTMBuilder action_lstm;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_r; // relation embeddings
  LookupParameters* p_p; // pos tag embeddings
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_H; // head matrix for composition function
  Parameters* p_D; // dependency matrix for composition function
  Parameters* p_R; // relation matrix for composition function
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_p2l; // POS to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack

  Parameters* p_start_of_word;//Miguel -->dummy <s> symbol
  Parameters* p_end_of_word; //Miguel --> dummy </s> symbol
  LookupParameters* char_emb; //Miguel-> mapping of characters to vectors 


  LSTMBuilder fw_char_lstm; // Miguel
  LSTMBuilder bw_char_lstm; //Miguel


  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      buffer_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM, 1})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM, 1})),
      p_r(model->add_lookup_parameters(ACTION_SIZE, {REL_DIM, 1})),
      p_pbias(model->add_parameters({HIDDEN_DIM, 1})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_H(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_D(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_R(model->add_parameters({LSTM_INPUT_DIM, REL_DIM})),
      p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model->add_parameters({LSTM_INPUT_DIM, 1})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM, 1})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM, 1})),
      p_abias(model->add_parameters({ACTION_SIZE, 1})),

      p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM, 1})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM, 1})),

      p_start_of_word(model->add_parameters({LSTM_INPUT_DIM, 1})), //Miguel
      p_end_of_word(model->add_parameters({LSTM_INPUT_DIM, 1})), //Miguel 

      char_emb(model->add_lookup_parameters(CHAR_SIZE, {INPUT_DIM, 1})),//Miguel

//      fw_char_lstm(LAYERS, LSTM_CHAR_OUTPUT_DIM, LSTM_INPUT_DIM, model), //Miguel
//      bw_char_lstm(LAYERS, LSTM_CHAR_OUTPUT_DIM, LSTM_INPUT_DIM,  model), //Miguel

      fw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM/2, model), //Miguel 
      bw_char_lstm(LAYERS, LSTM_INPUT_DIM, LSTM_CHAR_OUTPUT_DIM/2, model) /*Miguel*/ {
    if (USE_POS) {
      p_p = model->add_lookup_parameters(POS_SIZE, {POS_DIM, 1});
      p_p2l = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
    }
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM, 1});
      for (auto it : pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
    }
  }

static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize, vector<int> stacki) {
  if (a[1]=='W' && ssize<3) return true; //MIGUEL

  if (a[1]=='W') { //MIGUEL

        int top=stacki[stacki.size()-1];
        int sec=stacki[stacki.size()-2];

        if (sec>top) return true;
  }

  bool is_shift = (a[0] == 'S' && a[1]=='H');  //MIGUEL
  bool is_reduce = !is_shift;
  if (is_shift && bsize == 1) return true;
  if (is_reduce && ssize < 3) return true;
  if (bsize == 2 && // ROOT is the only thing remaining on buffer
      ssize > 2 && // there is more than a single element on the stack
      is_shift) return true;
  // only attach left to ROOT
  if (bsize == 1 && ssize == 3 && a[0] == 'R') return true;
  return false;
}

/*static bool IsActionForbidden(const string& a, unsigned bsize, unsigned ssize) {
  bool is_shift = (a[0] == 'S');
  bool is_reduce = !is_shift;
  if (is_shift && bsize == 1) return true;
  if (is_reduce && ssize < 3) return true;
  if (bsize == 2 && // ROOT is the only thing remaining on buffer
      ssize > 2 && // there is more than a single element on the stack
      is_shift) return true;
  // only attach left to ROOT
  if (bsize == 1 && ssize == 3 && a[0] == 'R') return true;
  return false;
}*/

static map<int,int> compute_heads(unsigned sent_len, const vector<unsigned>& actions, const vector<string>& setOfActions, map<int,string>* pr = nullptr) {
  map<int,int> heads;
  map<int,string> r;
  map<int,string>& rels = (pr ? *pr : r);
  for(unsigned i=0;i<sent_len;i++) { heads[i]=-1; rels[i]="ERROR"; }
  vector<int> bufferi(sent_len + 1, 0), stacki(1, -999);
  for (unsigned i = 0; i < sent_len; ++i)
    bufferi[sent_len - i] = i;
  bufferi[0] = -999;
  for (auto action: actions) { // loop over transitions for sentence
    const string& actionString=setOfActions[action];
    const char ac = actionString[0];
    const char ac2 = actionString[1];
    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
      stacki.push_back(bufferi.back());
      bufferi.pop_back();
    } 
   else if (ac=='S' && ac2=='W') {
        assert(stacki.size() > 2);

//	std::cout<<"SWAP"<<"\n";
        unsigned ii = 0, jj = 0;
        jj=stacki.back();
        stacki.pop_back();

        ii=stacki.back();
        stacki.pop_back();

        bufferi.push_back(ii);

        stacki.push_back(jj);
    }

    else { // LEFT or RIGHT
      assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
      assert(ac == 'L' || ac == 'R');
      unsigned depi = 0, headi = 0;
      (ac == 'R' ? depi : headi) = stacki.back();
      stacki.pop_back();
      (ac == 'R' ? headi : depi) = stacki.back();
      stacki.pop_back();
      stacki.push_back(headi);
      heads[depi] = headi;
      rels[depi] = actionString;
    }
  }
  assert(bufferi.size() == 1);
  //assert(stacki.size() == 2);
  return heads;
}


// given the first character of a UTF8 block, find out how wide it is
// see http://en.wikipedia.org/wiki/UTF-8 for more info
inline unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else return 0;
}


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// OOV handling: raw_sent will have the actual words
//               sent will have words replaced by appropriate UNK tokens
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const vector<unsigned>& raw_sent,  // raw sentence
                     const vector<unsigned>& sent,  // sent with oovs replaced
                     const vector<unsigned>& sentPos,
                     const vector<unsigned>& correct_actions,
                     const vector<string>& setOfActions,
                     const map<unsigned, std::string>& intToWords,
                     double *right) {
  //for (unsigned i = 0; i < sent.size(); ++i) cerr << ' ' << intToWords.find(sent[i])->second;
  //cerr << endl;
    vector<unsigned> results;
    const bool build_training_graph = correct_actions.size() > 0;

    stack_lstm.new_graph(*hg);
    buffer_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    stack_lstm.start_new_sequence();
    buffer_lstm.start_new_sequence();
    action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression H = parameter(*hg, p_H);
    Expression D = parameter(*hg, p_D);
    Expression R = parameter(*hg, p_R);
    Expression cbias = parameter(*hg, p_cbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression ib = parameter(*hg, p_ib);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (USE_POS)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right


    Expression word_end = parameter(*hg, p_end_of_word); //Miguel
    Expression word_start = parameter(*hg, p_start_of_word); //Miguel

    if (USE_SPELLING){
       fw_char_lstm.new_graph(*hg);
        //    fw_char_lstm.add_parameter_edges(hg);

       bw_char_lstm.new_graph(*hg);
       //    bw_char_lstm.add_parameter_edges(hg);
    }



    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < VOCAB_SIZE);
      //Expression w = lookup(*hg, p_w, sent[i]);

      unsigned wi=sent[i];
      std::string ww=intToWords.at(wi);
      Expression w;
      /**********SPELLING MODEL*****************/
      if (USE_SPELLING) {
        //std::cout<<"using spelling"<<"\n";
        if (ww.length()==4  && ww[0]=='R' && ww[1]=='O' && ww[2]=='O' && ww[3]=='T'){
          w=lookup(*hg, p_w, sent[i]); //we do not need a LSTM encoding for the root word, so we put it directly-.
        }
        else {

            fw_char_lstm.start_new_sequence();
            //cerr<<"start_new_sequence done"<<"\n";

            fw_char_lstm.add_input(word_start);
            //cerr<<"added start of word symbol"<<"\n";
            /*for (unsigned j=0;j<w.length();j++){

                //cerr<<j<<":"<<w[j]<<"\n"; 
                Expression cj=lookup(*hg, char_emb, w[j]);
                fw_char_lstm.add_input(cj, hg);
        
               //std::cout<<"Inputdim:"<<LSTM_INPUT_DIM<<"\n";  
               //hg->incremental_forward();

            }*/
	    std::vector<int> strevbuffer;
            for (unsigned j=0;j<ww.length();j+=UTF8Len(ww[j])){

                //cerr<<j<<":"<<w[j]<<"\n"; 
                std::string wj;
                for (unsigned h=j;h<j+UTF8Len(ww[j]);h++) wj+=ww[h];
                //std::cout<<"fw"<<wj<<"\n";
                int wjint=corpus.charsToInt[wj];
		//std::cout<<"fw:"<<wjint<<"\n";
		strevbuffer.push_back(wjint);
                Expression cj=lookup(*hg, char_emb, wjint);
                fw_char_lstm.add_input(cj);

               //std::cout<<"Inputdim:"<<LSTM_INPUT_DIM<<"\n";  
               //hg->incremental_forward();

            }
            fw_char_lstm.add_input(word_end);
            //cerr<<"added end of word symbol"<<"\n";



            Expression fw_i=fw_char_lstm.back();

            //cerr<<"fw_char_lstm.back() done"<<"\n";

            bw_char_lstm.start_new_sequence();
            //cerr<<"bw start new sequence done"<<"\n";

            bw_char_lstm.add_input(word_end);
	    //for (unsigned j=w.length()-1;j>=0;j--){
            /*for (unsigned j=w.length();j-->0;){
               //cerr<<j<<":"<<w[j]<<"\n";
               Expression cj=lookup(*hg, char_emb, w[j]);
               bw_char_lstm.add_input(cj); 
            }*/

	    while(!strevbuffer.empty()) {
		int wjint=strevbuffer.back();
		//std::cout<<"bw:"<<wjint<<"\n";
		Expression cj=lookup(*hg, char_emb, wjint);
                bw_char_lstm.add_input(cj);
		strevbuffer.pop_back();
	    }
	    
            /*for (unsigned j=w.length()-1;j>0;j=j-UTF8Len(w[j])) {

                //cerr<<j<<":"<<w[j]<<"\n"; 
                std::string wj;
                for (unsigned h=j;h<j+UTF8Len(w[j]);h++) wj+=w[h];
                std::cout<<"bw"<<wj<<"\n";
                int wjint=corpus.charsToInt[wj];
                Expression cj=lookup(*hg, char_emb, wjint);
                bw_char_lstm.add_input(cj);

               //std::cout<<"Inputdim:"<<LSTM_INPUT_DIM<<"\n";  
               //hg->incremental_forward();

            }*/
            bw_char_lstm.add_input(word_start);
            //cerr<<"start symbol in bw seq"<<"\n";     

            Expression bw_i=bw_char_lstm.back();

            vector<Expression> tt = {fw_i, bw_i};
            w=concatenate(tt); //and this goes into the buffer...
            //cerr<<"fw and bw done"<<"\n";
         }

	}
      /**************************************************/
      //cerr<<"concatenate?"<<"\n";

      /***************NO SPELLING*************************************/

      // Expression w = lookup(*hg, p_w, sent[i]);
      else { //NO SPELLING
          //Don't use SPELLING
          //std::cout<<"don't use spelling"<<"\n";
          w=lookup(*hg, p_w, sent[i]);
      }

      Expression i_i;
      if (USE_POS) {
        Expression p = lookup(*hg, p_p, sentPos[i]);
        i_i = affine_transform({ib, w2l, w, p2l, p});
      } else {
        i_i = affine_transform({ib, w2l, w});
      }
      if (p_t && pretrained.count(raw_sent[i])) {
        Expression t = const_lookup(*hg, p_t, raw_sent[i]);
        i_i = affine_transform({i_i, t2l, t});
      }
      buffer[sent.size() - i] = rectify(i_i);
      bufferi[sent.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto& b : buffer)
      buffer_lstm.add_input(b);

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction
    while(stack.size() > 2 || buffer.size() > 1) {

      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a: possible_actions) {
        if (IsActionForbidden(setOfActions[a], buffer.size(), stack.size(), stacki))
          continue;
        current_valid_actions.push_back(a);
      }

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression p_t = affine_transform({pbias, S, stack_lstm.back(), B, buffer_lstm.back(), A, action_lstm.back()});
      Expression nlp_t = rectify(p_t);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());
      double best_score = adist[current_valid_actions[0]];
      unsigned best_a = current_valid_actions[0];
      for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
        if (adist[current_valid_actions[i]] > best_score) {
          best_score = adist[current_valid_actions[i]];
          best_a = current_valid_actions[i];
        }
      }
      unsigned action = best_a;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        action = correct_actions[action_count];
        if (best_a == action) { (*right)++; }
      }
      ++action_count;
      // action_log_prob = pick(adist, action)
      log_probs.push_back(pick(adiste, action));
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

      // get relation embedding from action (TODO: convert to relation from action?)
      Expression relation = lookup(*hg, p_r, action);

      // do action
      const string& actionString=setOfActions[action];
      //cerr << "A=" << actionString << " Bsize=" << buffer.size() << " Ssize=" << stack.size() << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];


      if (ac =='S' && ac2=='H') {  // SHIFT
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
        stack.push_back(buffer.back());
        stack_lstm.add_input(buffer.back());
        buffer.pop_back();
        buffer_lstm.rewind_one_step();
        stacki.push_back(bufferi.back());
        bufferi.pop_back();
      } 
      else if (ac=='S' && ac2=='W'){ //SWAP --- Miguel
          assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)

          //std::cout<<"SWAP: "<<"stack.size:"<<stack.size()<<"\n";

          Expression toki, tokj;
          unsigned ii = 0, jj = 0;
          tokj=stack.back();
          jj=stacki.back();
          stack.pop_back();
          stacki.pop_back();

          toki=stack.back();
          ii=stacki.back();
          stack.pop_back();
          stacki.pop_back();

          buffer.push_back(toki);
          bufferi.push_back(ii);

          stack_lstm.rewind_one_step();
          stack_lstm.rewind_one_step();


          buffer_lstm.add_input(buffer.back());

          stack.push_back(tokj);
          stacki.push_back(jj);

          stack_lstm.add_input(stack.back());

          //stack_lstm.rewind_one_step();
          //buffer_lstm.rewind_one_step();
	}
      else { // LEFT or RIGHT
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(ac == 'L' || ac == 'R');
        Expression dep, head;
        unsigned depi = 0, headi = 0;
        (ac == 'R' ? dep : head) = stack.back();
        (ac == 'R' ? depi : headi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        (ac == 'R' ? head : dep) = stack.back();
        (ac == 'R' ? headi : depi) = stacki.back();
        stack.pop_back();
        stacki.pop_back();
        if (headi == sent.size() - 1) rootword = intToWords.find(sent[depi])->second;
        // composed = cbias + H * head + D * dep + R * relation
        Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
        Expression nlcomposed = tanh(composed);
        stack_lstm.rewind_one_step();
        stack_lstm.rewind_one_step();
        stack_lstm.add_input(nlcomposed);
        stack.push_back(nlcomposed);
        stacki.push_back(headi);
      }
    }
    assert(stack.size() == 2); // guard symbol, root
    assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return results;
  }

struct ParserState {
  LSTMBuilder stack_lstm;
  LSTMBuilder buffer_lstm;
  LSTMBuilder action_lstm;
  vector<Expression> buffer;
  vector<int> bufferi;
  vector<Expression> stack;
  vector<int> stacki;
  vector<unsigned> results;  // sequence of predicted actions
  bool complete;

  double score;
};

struct ParserStateCompare {
  bool operator()(const ParserState& a, const ParserState& b) const {
    return a.score > b.score;
  }
};

static void prune(vector<ParserState>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

static bool all_complete(const vector<ParserState>& pq) {
  for (auto& ps : pq) if (!ps.complete) return false;
  return true;
}

// run beam search
vector<unsigned> log_prob_parser_beam(ComputationGraph* hg,
                     const vector<unsigned>& raw_sent,  // raw sentence
                     const vector<unsigned>& sent,  // sent with OOVs replaced
                     const vector<unsigned>& sentPos,
                     const vector<string>& setOfActions,
                     unsigned beam_size, double* log_prob) {
    abort();
#if 0
    vector<unsigned> results;
    ParserState init;

    stack_lstm.new_graph(hg);
    buffer_lstm.new_graph(hg);
    action_lstm.new_graph(hg);
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression H = parameter(*hg, p_H);
    Expression D = parameter(*hg, p_D);
    Expression R = parameter(*hg, p_R);
    Expression cbias = parameter(*hg, p_cbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression ib = parameter(*hg, p_ib);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (USE_POS)
      i_p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (p_t2l)
      i_t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    action_lstm.add_input(i_action_start, hg);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings (possibly including POS info)
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence

    // precompute buffer representation from left to right
    for (unsigned i = 0; i < sent.size(); ++i) {
      assert(sent[i] < VOCAB_SIZE);
      Expression w = lookup(*hg, p_w, sent[i]);
      Expression i;
      if (USE_POS) {
        Expression p = lookup(*hg, p_p, sentPos[i]);
        i_i = hg->add_function<AffineTransform>({i_ib, i_w2l, i_w, i_p2l, i_p});
      } else {
        i_i = hg->add_function<AffineTransform>({i_ib, i_w2l, i_w});
      }
      if (p_t && pretrained.count(raw_sent[i])) {
        Expression t = hg->add_const_lookup(p_t, sent[i]);
        i_i = hg->add_function<AffineTransform>({i_i, i_t2l, i_t});
      }
      Expression inl = hg->add_function<Rectify>({i_i});
      buffer[sent.size() - i] = i_inl;
      bufferi[sent.size() - i] = i;
    }
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto& b : buffer)
      buffer_lstm.add_input(b, hg);

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back(), hg);

    init.stack_lstm = stack_lstm;
    init.buffer_lstm = buffer_lstm;
    init.action_lstm = action_lstm;
    init.buffer = buffer;
    init.bufferi = bufferi;
    init.stack = stack;
    init.stacki = stacki;
    init.results = results;
    init.score = 0;
    if (init.stacki.size() ==1 && init.bufferi.size() == 1) { assert(!"bad0"); }

    vector<ParserState> pq;
    pq.push_back(init);
    vector<ParserState> completed;
    while (pq.size() > 0) {
      const ParserState cur = pq.back();
      pq.pop_back();
      if (cur.stack.size() == 2 && cur.buffer.size() == 1) {
        completed.push_back(cur);
        if (completed.size() == beam_size) break;
        continue;
      }

      // get list of possible actions for the current parser state
      vector<unsigned> current_valid_actions;
      for (auto a: possible_actions) {
        if (IsActionForbidden(setOfActions[a], cur.buffer.size(), cur.stack.size(), stacki))
          continue;
        current_valid_actions.push_back(a);
      }

      // p_t = pbias + S * slstm + B * blstm + A * almst
      Expression p_t = hg->add_function<AffineTransform>({i_pbias, i_S, cur.stack_lstm.back(), i_B, cur.buffer_lstm.back(), i_A, cur.action_lstm.back()});

      // nlp_t = tanh(p_t)
      Expression nlp_t = hg->add_function<Rectify>({i_p_t});

      // r_t = abias + p2a * nlp
      Expression r_t = hg->add_function<AffineTransform>({i_abias, i_p2a, i_nlp_t});

      //cerr << "CVAs: " << current_valid_actions.size() << " (cur.buf=" << cur.bufferi.size() << " buf.sta=" << cur.stacki.size() << ")\n";
      // adist = log_softmax(r_t)
      hg->add_function<RestrictedLogSoftmax>({i_r_t}, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());

      for (auto action : current_valid_actions) {
        pq.resize(pq.size() + 1);
        ParserState& ns = pq.back();
        ns = cur;  // copy current state to new state
        ns.score += adist[action];
        ns.results.push_back(action);

        // add current action to action LSTM
        Expression action = lookup(*hg, p_a, action);
        ns.action_lstm.add_input(i_action, hg);

        // do action
        const string& actionString=setOfActions[action];
        //cerr << "A=" << actionString << " Bsize=" << buffer.size() << " Ssize=" << stack.size() << endl;
        const char ac = actionString[0];
        if (ac =='S') {  // SHIFT
          assert(ns.buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
          ns.stack.push_back(ns.buffer.back());
          ns.stack_lstm.add_input(ns.buffer.back(), hg);
          ns.buffer.pop_back();
          ns.buffer_lstm.rewind_one_step();
          ns.stacki.push_back(cur.bufferi.back());
          ns.bufferi.pop_back();
        } else { // LEFT or RIGHT
          assert(ns.stack.size() > 2); // dummy symbol means > 2 (not >= 2)
          assert(ac == 'L' || ac == 'R');
          Expression dep, head;
          unsigned depi = 0, headi = 0;
          (ac == 'R' ? dep : head) = ns.stack.back();
          (ac == 'R' ? depi : headi) = ns.stacki.back();
          ns.stack.pop_back();
          ns.stacki.pop_back();
          (ac == 'R' ? head : dep) = ns.stack.back();
          (ac == 'R' ? headi : depi) = ns.stacki.back();
          ns.stack.pop_back();
          ns.stacki.pop_back();
          // get relation embedding from action (TODO: convert to relation from action?)
          Expression relation = lookup(*hg, p_r, action);

          // composed = cbias + H * head + D * dep + R * relation
          Expression composed = affine_transform({cbias, H, head, D, dep, R, relation});
          // nlcomposed = tanh(composed)
          Expression nlcomposed = tanh(composed);
          ns.stack_lstm.rewind_one_step();
          ns.stack_lstm.rewind_one_step();
          ns.stack_lstm.add_input(i_nlcomposed, hg);
          ns.stack.push_back(i_nlcomposed);
          ns.stacki.push_back(headi);
        }
      } // all curent actions
      prune(pq, beam_size);
    } // beam search
    assert(completed.size() > 0);
    prune(completed, 1);
    results = completed.back().results;
    assert(completed.back().stack.size() == 2); // guard symbol, root
    assert(completed.back().stacki.size() == 2);
    assert(completed.back().buffer.size() == 1); // guard symbol
    assert(completed.back().bufferi.size() == 1);
    *log_prob = completed.back().score;
    return results;
#endif
  }
};

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

template<typename T>
unsigned compute_correct(const map<int,T>& ref, const map<int,T>& hyp, unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto ri = ref.find(i);
    auto hi = hyp.find(i);
    assert(ri != ref.end());
    assert(hi != hyp.end());
    if (ri->second == hi->second) ++res;
  }
  return res;
}

template<typename T1, typename T2>
unsigned compute_correct(const map<int,T1>& ref1, const map<int,T1>& hyp1,
                         const map<int,T2>& ref2, const map<int,T2>& hyp2, unsigned len) {
  unsigned res = 0;
  for (unsigned i = 0; i < len; ++i) {
    auto r1 = ref1.find(i);
    auto h1 = hyp1.find(i);
    auto r2 = ref2.find(i);
    auto h2 = hyp2.find(i);
    assert(r1 != ref1.end());
    assert(h1 != hyp1.end());
    assert(r2 != ref2.end());
    assert(h2 != hyp2.end());
    if (r1->second == h1->second && r2->second == h2->second) ++res;
  }
  return res;
}

void output_conll(const vector<unsigned>& sentence, const vector<unsigned>& pos,
                  const vector<string>& sentenceUnkStrings,
                  const map<unsigned, string>& intToWords,
                  const map<unsigned, string>& intToPos,
                  const map<int,int>& hyp, const map<int,string>& rel_hyp) {
  for (unsigned i = 0; i < (sentence.size()-1); ++i) {
    auto index = i + 1;
    assert(i < sentenceUnkStrings.size() &&
           ((sentence[i] == corpus.get_or_add_word(cpyp::Corpus::UNK) &&
             sentenceUnkStrings[i].size() > 0) ||
            (sentence[i] != corpus.get_or_add_word(cpyp::Corpus::UNK) &&
             sentenceUnkStrings[i].size() == 0 &&
             intToWords.find(sentence[i]) != intToWords.end())));
    string wit = (sentenceUnkStrings[i].size() > 0)?
      sentenceUnkStrings[i] : intToWords.find(sentence[i])->second;
    auto pit = intToPos.find(pos[i]);
    assert(hyp.find(i) != hyp.end());
    auto hyp_head = hyp.find(i)->second + 1;
    if (hyp_head == (int)sentence.size()) hyp_head = 0;
    auto hyp_rel_it = rel_hyp.find(i);
    assert(hyp_rel_it != rel_hyp.end());
    auto hyp_rel = hyp_rel_it->second;
    size_t first_char_in_rel = hyp_rel.find('(') + 1;
    size_t last_char_in_rel = hyp_rel.rfind(')') - 1;
    hyp_rel = hyp_rel.substr(first_char_in_rel, last_char_in_rel - first_char_in_rel + 1);
    cout << index << '\t'       // 1. ID
         << wit << '\t'         // 2. FORM
         << "_" << '\t'         // 3. LEMMA
         << "_" << '\t'         // 4. CPOSTAG
         << pit->second << '\t' // 5. POSTAG
         << "_" << '\t'         // 6. FEATS
         << hyp_head << '\t'    // 7. HEAD
         << hyp_rel << '\t'     // 8. DEPREL
         << "_" << '\t'         // 9. PHEAD
         << "_" << endl;        // 10. PDEPREL
  }
  cout << endl;
}

void init_pretrained(istream &in) {
  string line;
  vector<float> v(PRETRAINED_DIM, 0);
  string word;
  while (getline(in, line)) {
    if (word.empty() && line.find('.') == std::string::npos)
      continue; // first line contains vocabulary size and dimensions
    istringstream lin(line);
    lin >> word;
    for (unsigned i = 0; i < PRETRAINED_DIM; ++i) lin >> v[i];
    unsigned id = corpus.get_or_add_word(word);
    pretrained[id] = v;
  }
}


int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  USE_POS = conf.count("use_pos_tags");

  USE_SPELLING=conf.count("use_spelling"); //Miguel
  corpus.USE_SPELLING=USE_SPELLING;

  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  REL_DIM = conf["rel_dim"].as<unsigned>();
  const unsigned beam_size = conf["beam_size"].as<unsigned>();
  const unsigned unk_strategy = conf["unk_strategy"].as<unsigned>();
  cerr << "Unknown word strategy: ";
  if (unk_strategy == 1) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  const double unk_prob = conf["unk_prob"].as<double>();
  assert(unk_prob >= 0.); assert(unk_prob <= 1.);
  const unsigned maxit = conf["maxit"].as<unsigned>();
  cerr << "Maximum number of iterations: " << maxit << "\n";
  const double tolerance = conf["tolerance"].as<double>();
  if (tolerance > 0.0) {
    cerr << "Optimization tolerance: " << tolerance << "\n";
  }
  ostringstream os;
  os << "parser_" << (USE_POS ? "pos" : "nopos")
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << '_' << POS_DIM
     << '_' << REL_DIM
     << "-pid" << getpid() << ".params";
  int best_correct_heads = 0;
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;
  bool softlinkCreated = false;
  corpus.load_correct_actions(conf["training_data"].as<string>());
  const unsigned kUNK = corpus.get_or_add_word(cpyp::Corpus::UNK);
  kROOT_SYMBOL = corpus.get_or_add_word(ROOT_SYMBOL);

  if (conf.count("words")) {
    pretrained[kUNK] = vector<float>(PRETRAINED_DIM, 0);
    const string& words_fname = conf["words"].as<string>();
    cerr << "Loading from " << words_fname << " with " << PRETRAINED_DIM << " dimensions\n";
    if (boost::algorithm::ends_with(words_fname, ".gz")) {
      ifstream file(words_fname.c_str(), ios_base::in | ios_base::binary);
      boost::iostreams::filtering_streambuf<boost::iostreams::input> zip;
      zip.push(boost::iostreams::zlib_decompressor());
      zip.push(file);
      istream in(&zip);
      init_pretrained(in);
    } else {
      ifstream in(words_fname.c_str());
      init_pretrained(in); // read as normal text
    }
  }

  set<unsigned> training_vocab; // words available in the training corpus
  set<unsigned> singletons;
  {  // compute the singletons in the parser's training data
    map<unsigned, unsigned> counts;
    for (auto sent : corpus.sentences)
      for (auto word : sent.second) { training_vocab.insert(word); counts[word]++; }
    for (auto wc : counts)
      if (wc.second == 1) singletons.insert(wc.first);
  }

  cerr << "Number of words: " << corpus.nwords << endl;
  VOCAB_SIZE = corpus.nwords + 1;

  cerr << "Number of UTF8 chars: " << corpus.maxChars << endl;
  if (corpus.maxChars>255) CHAR_SIZE=corpus.maxChars;

  ACTION_SIZE = corpus.nactions + 1;
  //POS_SIZE = corpus.npos + 1;
  POS_SIZE = corpus.npos + 10;
  possible_actions.resize(corpus.nactions);
  for (unsigned i = 0; i < corpus.nactions; ++i)
    possible_actions[i] = i;

  Model model;
  ParserBuilder parser(&model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  // OOV words will be replaced by UNK tokens
  corpus.load_correct_actionsDev(conf["dev_data"].as<string>());
  if (USE_SPELLING) VOCAB_SIZE = corpus.nwords + 1;
  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;
    cerr << "Training started."<<"\n";
    vector<unsigned> order(corpus.nsentences);
    for (unsigned i = 0; i < corpus.nsentences; ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.nsentences);
    unsigned si = corpus.nsentences;
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.nsentences << endl;
    unsigned trs = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    unsigned iter = 0;
    double uas = -1;
    double prev_uas = -1;
    while(!requested_stop && iter < maxit &&
        (tolerance < 0 || uas < 0 || prev_uas < 0 || abs(prev_uas - uas) > tolerance)) {
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.nsentences) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           const vector<unsigned>& sentence=corpus.sentences[order[si]];
           vector<unsigned> tsentence=sentence;
           if (unk_strategy == 1 && !USE_SPELLING) {
             for (auto& w : tsentence)
               if (singletons.count(w) && cnn::rand01() < unk_prob) w = kUNK;
           }
           const vector<unsigned>& sentencePos=corpus.sentencesPos[order[si]];
           const vector<unsigned>& actions=corpus.correct_act_sent[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,actions,corpus.actions,corpus.intToWords,&right);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
      }
      sgd.status();
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.nsentences) << ")\tllh: "<< llh<<" ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << endl;
      llh = trs = right = 0;

      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = corpus.nsentencesDev;
        // dev_size = 100;
        double llh = 0;
        double trs = 0;
        double right = 0;
        double correct_heads = 0;
        double total_heads = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const vector<unsigned>& sentence=corpus.sentencesDev[sii];
           const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii];
           const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
           vector<unsigned> tsentence=sentence;
	   if (!USE_SPELLING) {
                for (auto& w : tsentence)
                    if (training_vocab.count(w) == 0) w = kUNK;
           }

           ComputationGraph hg;
           vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,tsentence,sentencePos,vector<unsigned>(),corpus.actions,corpus.intToWords,&right);
           double lp = 0;
           //vector<unsigned> pred = parser.log_prob_parser_beam(&hg,sentence,sentencePos,corpus.actions,beam_size,&lp);
           llh -= lp;
           trs += actions.size();
           map<int,int> ref = parser.compute_heads(sentence.size(), actions, corpus.actions);
           map<int,int> hyp = parser.compute_heads(sentence.size(), pred, corpus.actions);
           //output_conll(sentence, corpus.intToWords, ref, hyp);
           correct_heads += compute_correct(ref, hyp, sentence.size() - 1);
           total_heads += sentence.size() - 1;
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        prev_uas = uas;
        uas = correct_heads / total_heads;
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.nsentences) << ")\tllh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " uas: " << uas << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
        if (correct_heads > best_correct_heads) {
          best_correct_heads = correct_heads;
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 &&
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }
        }
      }
      ++iter;
    }
    if (iter >= maxit) {
      cerr << "\nMaximum number of iterations reached (" << iter << "), terminating optimization...\n";
    } else if (!requested_stop) {
      cerr << "\nScore tolerance reached (" << tolerance << "), terminating optimization...\n";
    }
  } // should do training?
  if (true) { // do test evaluation
    double llh = 0;
    double trs = 0;
    double right = 0;
    double correct_heads_unlabeled = 0;
    double correct_heads_labeled = 0;
    double total_heads = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned corpus_size = corpus.nsentencesDev;
    for (unsigned sii = 0; sii < corpus_size; ++sii) {
      const vector<unsigned>& sentence=corpus.sentencesDev[sii];
      const vector<unsigned>& sentencePos=corpus.sentencesPosDev[sii];
      const vector<string>& sentenceUnkStr=corpus.sentencesStrDev[sii];
      const vector<unsigned>& actions=corpus.correct_act_sentDev[sii];
      vector<unsigned> tsentence=sentence;
      if (!USE_SPELLING) {
        for (auto& w : tsentence)
	  if (training_vocab.count(w) == 0) w = kUNK;
      }
      ComputationGraph cg;
      double lp = 0;
      vector<unsigned> pred;
      if (beam_size == 1)
        pred = parser.log_prob_parser(&cg,sentence,tsentence,sentencePos,vector<unsigned>(),corpus.actions,corpus.intToWords,&right);
      else
        pred = parser.log_prob_parser_beam(&cg,sentence,tsentence,sentencePos,corpus.actions,beam_size,&lp);
      llh -= lp;
      trs += actions.size();
      map<int, string> rel_ref, rel_hyp;
      map<int,int> ref = parser.compute_heads(sentence.size(), actions, corpus.actions, &rel_ref);
      map<int,int> hyp = parser.compute_heads(sentence.size(), pred, corpus.actions, &rel_hyp);
      output_conll(sentence, sentencePos, sentenceUnkStr, corpus.intToWords, corpus.intToPos, hyp, rel_hyp);
      correct_heads_unlabeled += compute_correct(ref, hyp, sentence.size() - 1);
      correct_heads_labeled += compute_correct(ref, hyp, rel_ref, rel_hyp, sentence.size() - 1);
      total_heads += sentence.size() - 1;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    cerr << "TEST llh=" << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " uas: " << (correct_heads_unlabeled / total_heads) << " las: " << (correct_heads_labeled / total_heads) << "\t[" << corpus_size << " sents in " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << " ms]" << endl;
  }
  for (unsigned i = 0; i < corpus.actions.size(); ++i) {
    //cerr << corpus.actions[i] << '\t' << parser.p_r->values[i].transpose() << endl;
    //cerr << corpus.actions[i] << '\t' << parser.p_p2a->values.col(i).transpose() << endl;
  }
}
