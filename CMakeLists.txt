cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)

set(ONNXRUNTIME_DIR /root/onnxruntime/build/Linux/Release)
set(OpenCV_DIR /root/opencv-4.6.0/build/installed/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

include_directories("include/")

pybind11_add_module(yolo_ort MODULE src/main.cpp src/detector.cpp src/utils.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolo_ort PRIVATE "/root/onnxruntime/include/onnxruntime/core/session" ${OpenCV_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})
target_compile_features(yolo_ort PRIVATE cxx_std_14)
target_link_libraries(yolo_ort PUBLIC ${OpenCV_LIBS} ${pybind11_LIBRARIES} ${PYTHON_LIBRARIES} ${ONNXRUNTIME_DIR}/libonnxruntime.so)
