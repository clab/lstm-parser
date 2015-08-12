# lstm-parser
Transition based dependency parser with state embeddings computed by LSTM RNNs

For the [EMNLP character-based model](http://arxiv.org/pdf/1508.00657.pdf), please check out from the branch "char-based" and follow the instructions shown in the readme file of the new branch. Here, [the branch](https://github.com/clab/lstm-parser/tree/char-based).

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)

#### Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

#### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

#### Train a parsing model

Having a training.conll file and a development.conll formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat), to train a parsing model with the LSTM parser type the following at the command line prompt:

    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c training.conll > trainingOracle.txt
    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c development.conll > devOracle.txt

    parser/lstm-parse -T trainingOracle.txt -d devOracle.txt --hidden_dim 100 --lstm_input_dim 100 -w sskip.100.vectors --pretrained_dim 100 --rel_dim 20 --action_dim 20 -t -P
    
Link to the word vectors that we used in the ACL 2015 paper for English:  [sskip.100.vectors](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

Note-1: you can also run it without word embeddings by removing the -w option for both training and parsing.

Note-2: the training process should be stopped when the development result does not substantially improve anymore. Normally, after 5500 iterations.

Note-3: the parser reports (after each iteration) results including punctuation symbols while in the ACL-15 paper we report results excluding them (as it is common practice in those data sets). You can find eval.pl script from the CoNLL-X Shared Task to get the correct numbers.

#### Parse data with your parsing model

Having a test.conll file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat)

    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c test.conll > testOracle.txt

    parser/lstm-parse -T trainingOracle.txt -d testOracle.txt --hidden_dim 100 --lstm_input_dim 100 -w sskip.100.vectors --pretrained_dim 100 --rel_dim 20 --action_dim 20 -P -m parser_pos_2_32_100_20_100_12_20-pidXXXX.params

The model name/id is stored where the parser has been trained.
The parser will output the conll file with the parsing result.

#### Pretrained models

TODO

#### Citation

If you make use of this software, please cite the following:

    @inproceedings{dyer:2015acl,
      author={Chris Dyer and Miguel Ballesteros and Wang Ling and Austin Matthews and Noah A. Smith},
      title={Transition-based Dependeny Parsing with Stack Long Short-Term Memory}
      booktitle={Proc. ACL},
      year=2015,
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact cdyer@cs.cmu.edu and miguel.ballesteros@upf.edu

