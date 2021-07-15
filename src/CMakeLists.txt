cmake_minimum_required(VERSION 3.10)

SET(common_dir /usr/src/tensorrt/samples/common)
INCLUDE_DIRECTORIES(${common_dir})

SET(cuda_dir /usr/local/cuda-10.2/targets/aarch64-linux/include)
INCLUDE_DIRECTORIES(${cuda_dir})

SET(include_dir /home/tingjun/Desktop/Cheetah_Proj/deep-contact-estimator/src)
INCLUDE_DIRECTORIES(${include_dir})

SET(LOG_CPP /usr/src/tensorrt/samples/common/logger.cpp)

ADD_EXECUTABLE(lcm_cnn_interface lcm_cnn_interface.cpp ${LOG_CPP})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_library(LIBNVINFER nvinfer /usr/lib/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBNVINFER})

find_library(LIBNVONNXPARSER nvonnxparser /usr/include/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBNVONNXPARSER})

find_library(LIBCUDART cudart /usr/local/cuda-10.2/lib64/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBCUDART})

find_library(LIBLCM lcm /usr/local/include/lcm/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBLCM})


