cmake_minimum_required(VERSION 3.10)
project(trt_feature_extractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED) # 추가

# Include 디렉토리 설정
include_directories(
    ${PROJECT_SOURCE_DIR}/
    ${CUDA_INCLUDE_DIRS}
    /usr/local/include/opencv4
    /usr/local/tensorrt/include
    ${PROJECT_SOURCE_DIR}/third_party/cnpy
)

# 라이브러리 경로 설정
link_directories(
    /usr/local/lib
    /usr/local/cuda/lib64
    /usr/local/tensorrt/lib
)

# cnpy 라이브러리 추가
add_library(cnpy STATIC third_party/cnpy/cnpy.cpp)

# 실행 파일 설정
add_executable(trt_feature_extractor main.cpp)

# 링크할 라이브러리 설정
target_link_libraries(trt_feature_extractor
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
    nvinfer
    nvonnxparser
    cnpy
    ZLIB::ZLIB   # 🔥 libz 추가
)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fdiagnostics-color=always")
