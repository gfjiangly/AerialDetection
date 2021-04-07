cmake_minimum_required(VERSION 3.8)
project(pixel_weights)

set(CMAKE_CXX_STANDARD 11)

# You should change it to your build pytorch lib
set(ATEN_DIR "/opt/conda/lib/python3.6/site-packages/torch/lib")
include_directories(${ATEN_DIR}/include)

# set(SOURCE_FILES main.cpp pixel_weights_cpu.cpp temp.h)
set(SOURCE_FILES pixel_weights_cpu.cpp temp.h)

add_executable(pixel_weights ${SOURCE_FILES})

target_link_libraries(pixel_weights ${ATEN_DIR}/libATen.so)
