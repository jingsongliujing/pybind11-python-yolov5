import yolo_ort
import time
import os
# 创建YOLODetectorWrapper实例
detector = yolo_ort.YOLODetector("models/yolov5m.onnx", False)

imageDir = "./images"
start_time = time.time()
for imageFile in os.listdir(imageDir):
    if imageFile.endswith(".jpg") or imageFile.endswith(".png"):
        imagePath = os.path.join(imageDir, imageFile)
        # 读取图像
        image_path = imagePath
        # 进行推理
        pred = detector.infer(image_path, 0.5, 0.5)

        # 处理结果
        result = ((detection.classId, detection.box, detection.confidence) for detection in pred)
        print(list(result))
end_time = time.time()
print("-----总耗时----", end_time - start_time)