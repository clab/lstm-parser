# lstm-parser
Transition based dependency parser with state embeddings computed by LSTM RNNs. This version is an easier-to-use refactoring of the original code from the ACL 2015 paper.

For the [EMNLP character-based model](http://arxiv.org/pdf/1508.00657.pdf), please check out from the branch [`char-based`](https://github.com/clab/lstm-parser/tree/char-based) and follow the instructions in the README file of that branch.

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

Given a `training.conll` file and a `development.conll` formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat), to train a parsing model with the LSTM parser type the following at the command line prompt:

    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c training.conll > trainingOracle.txt
    java -jar ParserOracleArcStdWithSwap.jar -t -1 -l 1 -c development.conll > devOracle.txt

    parser/lstm-parse --train -t trainingOracle.txt -d devOracle.txt --hidden_dim 100 --lstm_input_dim 100 --words sskip.100.vectors --rel_dim 20 --action_dim 20 --use_pos_tags

Link to the word vectors used in the ACL 2015 paper for English:  [sskip.100.vectors](https://drive.google.com/file/d/0B8nESzOdPhLsdWF2S1Ayb1RkTXc/view?usp=sharing).

Note-1: you can also run it without word embeddings by removing the `-w` option during training.

Note-2: the training process should be manually stopped when the development result does not substantially improve anymore. Normally, after 5500 iterations.

Note-3: the parser reports (after each iteration) results including punctuation symbols while in the ACL-15 paper we report results excluding them (as it is common practice in those data sets). You can find the `eval.pl` script from the CoNLL-X Shared Task to get the correct numbers.

#### Pretrained models

There is a pretrained model for English [here](http://www.cs.cmu.edu/~jdunietz/hosted/english_pos_2_32_100_20_100_12_20.params). It was trained as described above (though training was continued for about 6800 iterations) on the standard train/dev split from the Penn Treebank, converted to [Universal Dependencies](http://universaldependencies.org/) using the [Stanford dependency converter](http://nlp.stanford.edu/software/stanford-dependencies.shtml#Universal). When tested on Section 23 of the Penn Treebank, it achieves a LAS of 92.16% and a UAS of 93.55%.

#### Parse data with your parsing model

Given a `test.conll` file formatted according to the [CoNLL data format](http://ilk.uvt.nl/conll/#dataformat):

    parser/lstm-parse -m english_pos_2_32_100_20_100_12_20.params -t test.conll

If you are not using the pretrained model, you will need to replace the `.params` argument with the name of your own trained model file.

The parser will output the CoNLL file with the parsing result.

#### Citation

If you make use of this software, please cite the following:

    @inproceedings{dyer:2015acl,
      author={Chris Dyer and Miguel Ballesteros and Wang Ling and Austin Matthews and Noah A. Smith},
      title={Transition-based Dependeny Parsing with Stack Long Short-Term Memory},
      booktitle={Proc. ACL},
      year=2015,
    }

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

The [EOS binary archive library](https://epa.codeplex.com/) for Boost is included under the [MIT License](https://epa.codeplex.com/license).

#### Contact

For questions and usage issues, please contact cdyer@cs.cmu.edu and miguel.ballesteros@upf.edu.
For questions specifically about this easier-to-use version, please contact jdunietz@cs.cmu.edu.
