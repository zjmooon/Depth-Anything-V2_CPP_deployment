cmake_minimum_required(VERSION 3.16)

# CMake may not find CUDACXX, specify it explicitly
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(surface_detection LANGUAGES CXX CUDA)

# We required CUDA, OpenCV, TensorRT and PCL
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_library(NVINFER nvinfer)

message(STATUS "${CUDA_INCLUDE_DIRS}")
message(STATUS "${CUDA_LIBRARIES}")
message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${PCL_INCLUDE_DIRS}")
message(STATUS "${PCL_LIBRARIES}")
message(STATUS "${NVINFER}")

add_library(lanemode src/depth_anythingv2.cpp)

cuda_add_library(yuv2rgb src/yuv2rgb.cu src/yuv2rgb.cuh)

target_compile_features(lanemode PUBLIC cxx_std_17)

target_compile_options(lanemode PUBLIC -Ofast)

target_include_directories(lanemode PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} include src)

target_link_libraries(lanemode PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${NVINFER} yuv2rgb 
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/libcudasegmentation.so)

# Genrate executable 
add_executable(surface_detection surface_detection_api_sample.cpp)

target_link_libraries(surface_detection lanemode)

# GeForce 4080s's compute version is 8,9
set_target_properties(surface_detection lanemode yuv2rgb PROPERTIES CUDAARCH 89)
