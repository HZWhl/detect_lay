cmake_minimum_required(VERSION 3.19)
project(detect_lay)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

#add_executable(detect_lay yolov8_openvino.cpp)
add_library(detect_lay SHARED yolov8_ov_lv.cpp)
#add_library(detect_lay SHARED lv2cv.cpp yolov5_lv.cpp)
#target_link_libraries(yolo ${CONAN_LIBS})


#设置可执行文件输出目录
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/debug)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/release)
endif()


#添加1行代码
target_link_libraries(detect_lay
#        ${OpenCV_LIBS}
        "C:/Program Files/National Instruments/NIVision OpenCV Utilities/nivisionextlib/x64/vc90/lib/nivisext.lib"
        D:/opencv/opencv_vc15/opencv/build/x64/vc15/lib/opencv_world460d.lib
        D:/software/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64/runtime/lib/intel64/Debug/openvinod.lib
#        D:\software\w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64\runtime\lib\intel64\Debug
        )
include_directories(
#        ${OpenCV_INCLUDE_DIRS}
        "C:/Program Files/National Instruments/NIVision OpenCV Utilities/nivisionextlib/include"
        D:/software/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64/runtime/include
        D:/software/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64/runtime/include/ie
        D:/software/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64/runtime/include/ngraph
        D:/opencv/opencv_vc15/opencv/build/include
        D:/opencv/opencv_vc15/opencv/build/include/opencv2)

#link_libraries(
#        D:/software/w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64/runtime/lib/intel64
#        "C:/Program Files/National Instruments/NIVision OpenCV Utilities/nivisionextlib/x64/vc90/lib/nivisext.lib"
#        D:/opencv/opencv_vc15/opencv/build/x64/vc15/lib
#        "D:/software/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/lib/x64"
#        "D:/software/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/atlmfc/lib/x64"
#        "D:/software/Microsoft Visual Studio/2019/Community/VC/Auxiliary/VS/lib/x64"
#)