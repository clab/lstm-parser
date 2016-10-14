# lstm-parser with character-based representations.
Transition based dependency parser with state embeddings computed by LSTM RNNs. The difference with the master branch is what is explained in this [paper](http://arxiv.org/pdf/1508.00657.pdf) which is activated with the -S option (see below).

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)
 * [gcc](https://gcc.gnu.org/gcc-5/) (only tested with gcc version 5.3.0, may be incompatible with earlier versions)

#### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

#### Train a parsing model

Having a training.conll file and a development.conll formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat), to train a parsing model with the LSTM parser type the following at the command line prompt:

    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c training.conll > trainingOracle.txt
    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c development.conll > devOracle.txt

    parser/lstm-parse -T trainingOracle.txt -d devOracle.txt --hidden_dim 100 --lstm_input_dim 100 --pretrained_dim 100 --rel_dim 20 --action_dim 20 --input_dim 100 -t -P -S

Note-1: these model can be either run with or without pretrained word embeddings, although all the experiments reported in the EMNLP paper are run without.

Note-2: the training process should be stopped when the development result does not substantially improve anymore. Normally, after 5500 iterations.

Note-3: the parser reports (after each iteration) results including punctuation symbols while in the ACL-15 paper we report results excluding them (as it is common practice in those data sets). You can find eval.pl script from the CoNLL-X Shared Task to get the correct numbers.

#### Parse data with your parsing model

Having a test.conll file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat)

    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c test.conll > testOracle.txt
    
    parser/lstm-parse -T trainingOracle.txt -d testOracle.txt --hidden_dim 100 --lstm_input_dim 100 --pretrained_dim 100 --rel_dim 20 --action_dim 20 --input_dim 100 -P -S -m parser_pos_2_100_100_20_100_12_20-pidXXXXX.params

The model name/id is stored where the parser has been trained.
The parser will output the conll file with the parsing result.

#### Pretrained models

TODO

#### Citation

If you make use of this software, please cite the following:

    @inproceedings{ballesteros:2015emnlp,
      author={Miguel Ballesteros and Chris Dyer and Noah A. Smith},
      title={Improved Transition-Based Parsing by Modeling Characters instead of Words with LSTMs}
      booktitle={Proc. EMNLP},
      year=2015,
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact miguel.ballesteros@upf.edu and cdyer@cs.cmu.edu 

