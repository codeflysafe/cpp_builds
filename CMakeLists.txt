cmake_minimum_required(VERSION 3.19)
project(cpp_builds)

set(CMAKE_CXX_STANDARD 11)
include_directories(yolox)
set(OpenCV_DIR /usr/local/Cellar/opencv/4.5.5_3)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(cpp_builds main.cpp yolox/yolox.cpp)
target_link_libraries(cpp_builds ${OpenCV_LIBS})

