# lstm-parser
Transition based dependency parser with state embeddings computed by LSTM RNNs

# Required software

 * A C++ compiler with supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)

# Checking out the project for the first time

The first time you clone the repository, you need to sync the `cnn/` submodule.

    git submodule init
    git submodule update

# Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

# Pretrained models

TODO

# Citation

TODO

# License

TODO

