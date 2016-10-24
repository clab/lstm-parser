PROJECT(lstm-parser:parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(lstm-parse lstm-parse.cc lstm-parse-driver.cc c2.cpp)
target_link_libraries(lstm-parse cnn ${Boost_LIBRARIES})

