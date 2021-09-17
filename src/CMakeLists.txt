cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(common_dir /usr/src/tensorrt/samples/common)
INCLUDE_DIRECTORIES(${common_dir})

SET(cuda_dir /usr/local/cuda-10.2/targets/aarch64-linux/include)
INCLUDE_DIRECTORIES(${cuda_dir})

SET(include_dir /media/jetson256g/code/LCM_CNN_INTERFACE/deep-contact-estimator/include)
INCLUDE_DIRECTORIES(${include_dir})

SET(PROGRMA_PATH /media/jetson256g/code/LCM_CNN_INTERFACE/deep-contact-estimator)
INCLUDE_DIRECTORIES(${PROGRMA_PATH})

SET(LOG_CPP /usr/src/tensorrt/samples/common/logger.cpp)

ADD_LIBRARY(contact_estimation contact_estimation.cpp)

ADD_LIBRARY(lcm_handler communication/lcm_handler.cpp)

ADD_LIBRARY(tensorrt_acc ../utils/tensorrt_acc.cpp)

ADD_LIBRARY(lcm_cnn_interface lcm_cnn_interface.cpp)

ADD_EXECUTABLE(run_contact_estimator run_contact_estimator.cpp ${LOG_CPP})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_library(LIBNVINFER nvinfer /usr/lib/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBNVINFER})

find_library(LIBNVONNXPARSER nvonnxparser /usr/include/aarch64-linux-gnu/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBNVONNXPARSER})

find_library(LIBCUDART cudart /usr/local/cuda-10.2/lib64/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBCUDART})

find_package(  yaml-cpp REQUIRED )

find_library(LIBLCM lcm /usr/local/include/lcm/)
TARGET_LINK_LIBRARIES(lcm_cnn_interface ${LIBLCM})

TARGET_LINK_LIBRARIES(lcm_handler ${LIBLCM}
					yaml-cpp)

TARGET_LINK_LIBRARIES(tensorrt_acc ${LIBNVINFER}
				   ${LIBNVONNXPARSER}
				   ${LIBCUDART}
				   yaml-cpp)

TARGET_LINK_LIBRARIES(lcm_cnn_interface contact_estimation
					lcm_handler
					tensorrt_acc
					yaml-cpp)

TARGET_LINK_LIBRARIES(run_contact_estimator contact_estimation
					lcm_cnn_interface
					lcm_handler
					tensorrt_acc
					yaml-cpp)
