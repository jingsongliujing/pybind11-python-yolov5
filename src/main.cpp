#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cmdline.h"
#include "utils.h"
#include "detector.h"

namespace py = pybind11;

std::vector<Detection> infer(std::string modelPath, std::string imagePath, std::string classNamesPath){
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    bool isGPU = false;
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        throw std::runtime_error("Empty class names file.");
    }

    YOLODetector detector {nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    try
    {
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
        std::cout << "模型初始化完成" << std::endl;

        image = cv::imread(imagePath);
        auto start = std::chrono::high_resolution_clock::now();
       // 模型推理
        result = detector.detect(image, confThreshold, iouThreshold);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "推理耗时： " << elapsed.count() << " seconds." << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    // 调用画图函数

    // utils::visualizeDetection(image, result, classNames);

    // cv::imwrite("result.jpg", image);
    return result;
}

// 新增模型初始化
class YOLODetectorWrapper {
public:
    YOLODetectorWrapper(const std::string& modelPath, bool useGPU, const cv::Size& inputSize = cv::Size(640, 640))
        : detector_(modelPath, useGPU, inputSize) {}

    std::vector<Detection> infer(const std::string& imagePath, float confThreshold = 0.3f, float iouThreshold = 0.4f) {
        try {
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                throw std::runtime_error("读图失败： " + imagePath);
            }
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<Detection> result = detector_.detect(image, confThreshold, iouThreshold);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "推理耗时： " << elapsed.count() << " seconds." << std::endl;
            return result;
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            throw;
        }
    }

private:
    YOLODetector detector_;
};

PYBIND11_MODULE(yolo_ort, m){
    m.doc() = "pybind11 example yolov5 infer";
    m.def("infer", &infer, "example yolov5 infer", py::arg("modelPath"), py::arg("imagePath"), py::arg("classNamesPath"));

    py::class_<YOLODetectorWrapper>(m, "YOLODetector")
        .def(py::init<const std::string&, bool>(), py::arg("modelPath"), py::arg("useGPU") = false)
        .def("infer", &YOLODetectorWrapper::infer, "Perform inference on an image",
             py::arg("imagePath"), py::arg("confThreshold") = 0.3f, py::arg("iouThreshold") = 0.4f);

    // 注册识别类型
    py::class_<Detection>(m, "Detection")
        .def(py::init<>())
        .def_readwrite("classId", &Detection::classId)
        .def_readwrite("confidence", &Detection::conf) // 注意置信度的参数类型
        // 由于python不支持Rect类型的数据，所以要把box数据转换成元组
        .def_property("box", [](const Detection& detection) {
            return py::make_tuple(detection.box.x, detection.box.y, detection.box.width, detection.box.height);
        }, [](Detection& detection, const std::tuple<int, int, int, int>& box) {
            detection.box.x = std::get<0>(box);
            detection.box.y = std::get<1>(box);
            detection.box.width = std::get<2>(box);
            detection.box.height = std::get<3>(box);
        });
}
