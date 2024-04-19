# yolov5-onnxruntime-pybind11 推理

## 项目背景

#### 1.现状
* 以往的项目所使用的编程语言是Python作为开发语言，但是实际上Python的执行效率低和占用内存大，因此使用c++做模型推理似乎是一个好的选择。



* 这让我想到了使用Cython,Cython使用了用C编写的独立共享库，并将它们转换为Python模块。实际还是在Python上写库。
* PyBind11相反，它提供C++ API，写的是C++代码库，然后转成Python模块。
* Pybind11只支持C++，使用C++ 11特性。
* 此外，Pybind11的另一个优点是可以轻松处理来自Python的底层数据和类型(如Numpy array)。对于计算机视觉代码库的编写，无疑是一个好的选择。
* 本项目把yolov5核心模块，预处理，前处理，后处理用C++来写，使用pybind11把C++ 库/代码包装为可以在 Python 中使用的模块（本项目把类别，坐标，置信度封装成一个类与Python绑定），供Python调用，这样就可以大大提高代码的执行效率。

* 本项目基于：https://github.com/itsnine/yolov5-onnxruntime 来进行修改，
* 增加pybind11进行封装，封装yolov5核心流程，使用C++版本的onnxruntime,opencv,python仅需要import yolo_ort,通过配置模型，数据，类别来进行调用。

#### 第一次尝试用C++写视觉类工程项目，难免出现代码内存泄漏等风险，代码丑陋，勿喷！！！
```
PYBIND11_MODULE(yolo_ort, m){
    m.doc() = "pybind11 example yolov5 infer";
    m.def("infer", &infer, "example yolov5 infer", py::arg("modelPath"), py::arg("imagePath"), py::arg("classNamesPath"));

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
```




## pybind11 C++ YOLO v5 ONNX Runtime 目标检测推理

## 依赖:
- OpenCV 4.x
- ONNXRuntime 1.7+
- OS:  Windows 10 ,Ubuntu 20.04,centos7
- CUDA 11+ [CPU]
- pybind11

## opencv编译安装（编译时间较长，耐心等待）
```
git clone https://github.com/opencv/opencv.git

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=ON ../opencv

make -j7
```

## onnxruntime安装（编译时间较长，耐心等待）
```
git clone --single-branch --branch rel-1.7.0 https://github.com/microsoft/onnxruntime.git
./build.sh --skip_tests --config Release --build_shared_lib --parallel
```
## pybind11安装（编译时间较长，耐心等待）
```
https://blog.csdn.net/qq_27149279/article/details/121352696

export PYTHONPATH=/root/miniconda3/envs/resa/lib/python3.7/site-packages


g++ your_source_files -o your_executable -L/path/to/onnxruntime/lib -lonnxruntime

```

## 程序编译
#### 使用makefile编译
```
make start
python test_yolo_pybind11.py
```
#### 使用cmake跨平台编译
```
mkdir build
cd build
cmake ..
cmake --build .

python test_yolo_pybind11.py
```

## 



## 编译成yolo_ort.so，与test_yolo_pybind11.py在同一目录下，test_yolo_pybind11.py调用
```
import yolo_ort

# 创建YOLODetectorWrapper实例
detector = yolo_ort.YOLODetectorWrapper("models/yolov5s.onnx", True)

# 读取图像
image = "/root/liujingsong/yolov5_pybind11/images/bus.jpg"

# 进行推理
result = detector.infer(image, 0.5, 0.5)

# 处理结果
for detection in result:
    print(detection)

```

## 测试结果

##### 在同一设备，同一模型，同一张图片下，pybind11-yolov5与python-yolov5占用内存对比

```
pybind11-yolov5模型使用的CPU内存（MB）： 53.48828125

python-yolov5模型使用的CPU内存（MB）： 188.41796875
```

##### 150张图片批量测试时间差
###### 150张数据测试yolov5s模型：

```
c++:  38.006s   平均每张0.253秒
python: 47.459s  平均每张0.449秒
```

###### 150张数据测试yolov5m模型：
```
c++:  95.975s   平均每张0.63秒
python: 142.574s  平均每张0.95秒
```
