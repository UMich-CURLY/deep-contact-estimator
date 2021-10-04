cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(common_dir /usr/src/tensorrt/samples/common)
INCLUDE_DIRECTORIES(${common_dir})

# SET(cuda_dir /usr/local/cuda-10.2/targets/aarch64-linux/include)
SET(cuda_dir /usr/local/cuda-10.2/targets/x86_64-linux/include)
INCLUDE_DIRECTORIES(${cuda_dir})

SET(LOG_CPP /usr/src/tensorrt/samples/common/logger.cpp)

# SET(include_dir /media/jetson256g/code/LCM_CNN_INTERFACE/deep-contact-estimator/include)
SET(include_dir /home/tingjun/Desktop/mini_cheetah/deep-contact-estimator/include)
INCLUDE_DIRECTORIES(${include_dir})


ADD_EXECUTABLE(serialize_engine serialize_engine.cpp ${LOG_CPP})

# find_library(LIBNVINFER nvinfer /usr/lib/aarch64-linux-gnu/)
find_library(LIBNVINFER nvinfer /usr/lib/x86_64-linux-gnu/)
TARGET_LINK_LIBRARIES(serialize_engine ${LIBNVINFER})

# find_library(LIBNVONNXPARSER nvonnxparser /usr/include/aarch64-linux-gnu/)
find_library(LIBNVONNXPARSER nvonnxparser /usr/include/x86_64-linux-gnu/)
TARGET_LINK_LIBRARIES(serialize_engine ${LIBNVONNXPARSER})

find_library(LIBCUDART cudart /usr/local/cuda-10.2/lib64/)
TARGET_LINK_LIBRARIES(serialize_engine ${LIBCUDART})

TARGET_LINK_LIBRARIES(serialize_engine tensorrt_acc)
