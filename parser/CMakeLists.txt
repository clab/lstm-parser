PROJECT(lstm-parser:parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_LIBRARY(lstm-parser-core lstm-parser.cc corpus.cc neural-transition-tagger.cpp)
target_link_libraries(lstm-parser-core cnn ${Boost_LIBRARIES})

ADD_EXECUTABLE(lstm-parse lstm-parser-driver.cc)
target_link_libraries(lstm-parse lstm-parser-core ${Boost_LIBRARIES})
