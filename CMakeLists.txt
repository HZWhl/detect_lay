cmake_minimum_required(VERSION 3.1)
project(detect_lay)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

set(OpenCV_DIR D:/Programs/opencv/build)
set(TRT_DIR D:/software/TensorRT-8.6.1.6)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Dirent_INCLUDE_DIRS})

#change to your GPU own compute_XX
###########################################################################################
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)
###########################################################################################

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}/include)
include_directories(${TRT_DIR}/samples/common)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
        NO_MODULE
        NO_DEFAULT_PATH
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_PACKAGE_REGISTRY
        NO_CMAKE_BUILDS_PATH
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS} "D:/projects/labviewProjects/detect_lay/include")
link_directories(${TRT_DIR}/lib)

file(GLOB CPPS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
message(STATUS "output dir: \n${PROJECT_SOURCE_DIR}/bin\n")

#add_executable(${PROJECT_NAME} ${CPPS})
#add_executable(detect_lay yolov8_tensorrt.cpp)
#add_library(detect_lay SHARED ${CPPS})
add_library(detect_lay SHARED yolov8_trt_lv.cpp yolo.hpp yolo.cu infer.cu infer.hpp cpm.hpp)
#add_library(detect_lay SHARED lv2cv.cpp yolov5_lv.cpp)
#target_link_libraries(yolo ${CONAN_LIBS})

target_link_libraries(detect_lay "nvinfer" "nvinfer_plugin" "nvonnxparser" "nvparsers")
target_link_libraries(detect_lay ${OpenCV_LIBS})
target_link_libraries(detect_lay ${CUDA_LIBRARIES})
target_link_libraries(detect_lay Threads::Threads)

#添加1行代码
target_link_libraries(detect_lay
#        ${OpenCV_LIBS}
        "D:/projects/labviewProjects/detect_lay/lib/nivisext.lib"
#        D:/Programs/opencv/build/x64/vc15/lib/opencv_world460d.lib
#        D:/opencv/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64/runtime/lib/intel64/Debug/openvinod.lib
#        D:\software\w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64\runtime\lib\intel64\Debug
        )

