import ctypes

modelPath, imagePath, classNamesPath = "models/yolov5s.onnx", "images/zidane.jpg", "models/coco.names"
# 假设你的.so文件名为example.so，并且里面有一个int add(int, int)函数
infer = ctypes.cdll.LoadLibrary('./yolo_ort.so')
pred = infer(modelPath, imagePath, classNamesPath)

result = ((detection.classId, detection.box, detection.confidence) for detection in pred)

print(list(result))  # 将生成器转换为列表以打印结果

# # 计算模型使用的CPU内存
# process = psutil.Process()
# memory_info = process.memory_info()
# memory_in_mb = memory_info.rss / 1024 / 1024 # 将内存大小从字节转换为MB
# print("模型使用的CPU内存（MB）：", memory_in_mb)
print(result)  # 输出8，假设add函数正常工作