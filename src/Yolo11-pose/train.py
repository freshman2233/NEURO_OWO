
from ultralytics import YOLO

if __name__ == '__main__':
    # 使用更大的模型处理时序特征
    model = YOLO("yolov8n-pose.yaml")  # 中等规模模型

    # 调整训练参数以适应连续动作
    model.train(
        data="D:/Document/GitHub/neuro-owo/dataset/data.yaml",
        epochs=150,  # 增加训练轮次
        imgsz=640,
        batch=1,
        device=0,
        optimizer="AdamW",  # 使用AdamW优化器
        lr0=1e-3,  # 更小的学习率
        cos_lr=True,  # 启用余弦退火
        flipud=0.3,  # 增加上下翻转增强
        mixup=0.2,  # 启用MixUp增强
        name="pose_action_v1"
    )

