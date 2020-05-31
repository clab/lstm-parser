PROJECT(lstm-parser:parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

add_library(lstm-parser-core STATIC lstm-parser.cc corpus.cc 
            neural-transition-tagger.cpp)
ADD_EXECUTABLE(lstm-parse lstm-parser-driver.cc)

if(WITH_CUDA_BACKEND)
  add_dependencies(lstm-parser-core cnncuda)
  target_link_libraries(lstm-parser-core cnncuda)
  CUDA_ADD_CUBLAS_TO_TARGET(lstm-parser-core)

  add_dependencies(lstm-parse cnncuda)
  target_link_libraries(lstm-parse cnncuda)
  CUDA_ADD_CUBLAS_TO_TARGET(lstm-parse)
endif(WITH_CUDA_BACKEND)

target_link_libraries(lstm-parser-core cnn ${Boost_LIBRARIES})
target_link_libraries(lstm-parse lstm-parser-core ${Boost_LIBRARIES})