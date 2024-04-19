import sys
import time
import psutil

# 调用c++编译好的包
from yolo_ort import *

# modelPath, imagePath, classNamesPath = sys.argv[1], sys.argv[2], sys.argv[3]
modelPath, imagePath, classNamesPath = "models/yolov5s.onnx", "images/zidane.jpg", "models/classes.names"

start_time = time.time()

#推理得到cpp封装好的类别，坐标，置信度的值
pred = infer(modelPath, imagePath, classNamesPath)

result = ((detection.classId, detection.box, detection.confidence) for detection in pred)

end_time = time.time()

print(list(result))  # 将生成器转换为列表以打印结果
print("-----耗时----", end_time - start_time)

# 计算模型使用的CPU内存
process = psutil.Process()
memory_info = process.memory_info()
memory_in_mb = memory_info.rss / 1024 / 1024 # 将内存大小从字节转换为MB
print("模型使用的CPU内存（MB）：", memory_in_mb)
