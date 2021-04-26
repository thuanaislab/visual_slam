cmake_minimum_required(VERSION 2.8)
project(calibrate)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories("/usr/include/eigen3")


find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


add_executable(calibarate calibarate.cpp)
target_link_libraries(calibarate ${Pangolin_LIBRARIES})
target_link_libraries(calibarate ${OpenCV_LIBS})

