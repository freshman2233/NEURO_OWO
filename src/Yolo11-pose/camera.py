import cv2
import torch
from ultralytics import YOLO

# 载入 YOLO11 预训练模型（可以换成你的自定义 .pt 文件）
model = YOLO("yolo11m-pose.pt")

# 打开摄像头（0 代表默认摄像头，可以改成其他索引）
cap = cv2.VideoCapture(0)

# 设置视频宽高（确保摄像头支持）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取帧率
if fps == 0:  # 防止部分摄像头无法获取帧率
    fps = 60

# 定义视频编码格式并创建 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 选择编码格式（MP4）
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 运行 YOLO 目标检测
    results = model(frame)

    # 在帧上绘制检测结果
    annotated_frame = results[0].plot()

    # 将处理后的视频帧写入文件
    out.write(annotated_frame)

    # 显示结果
    cv2.imshow("YOLO11 Camera Detection", annotated_frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()  # 释放视频写入对象
cv2.destroyAllWindows()
