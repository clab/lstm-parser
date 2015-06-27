# lstm-parser
Transition based dependency parser with state embeddings computed by LSTM RNNs

### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)

### Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

### Pretrained models

TODO

### Citation

If you make use of this software, please cite the following:

    @inproceedings{dyer:2015acl,
      author={Chris Dyer and Miguel Ballesteros and Wang Ling and Austin Matthews and Noah A. Smith},
      title={Transition-based Dependeny Parsing with Stack Long Short-Term Memory}
      booktitle={Proc. ACL},
      year=2015,
    }

# License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

