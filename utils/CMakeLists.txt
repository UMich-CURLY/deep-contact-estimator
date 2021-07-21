cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(common_dir /usr/src/tensorrt/samples/common)
INCLUDE_DIRECTORIES(${common_dir})

SET(cuda_dir /usr/local/cuda-10.2/targets/x86_64-linux/include)
INCLUDE_DIRECTORIES(${cuda_dir})

SET(LOG_CPP /usr/src/tensorrt/samples/common/logger.cpp)

ADD_EXECUTABLE(serialize_engine serialize_engine.cpp ${LOG_CPP})

find_library(LIBNVINFER nvinfer /usr/lib/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(serialize_engine ${LIBNVINFER})

find_library(LIBNVONNXPARSER nvonnxparser /usr/include/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(serialize_engine ${LIBNVONNXPARSER})

find_library(LIBCUDART cudart /usr/local/cuda-10.2/lib64/)
TARGET_LINK_LIBRARIES(serialize_engine ${LIBCUDART})
