import os
import csv
import cv2

# 配置路径
CSV_PATH = "HSiPu2/HSiPu2/fwcce/0.csv"      # 你的CSV文件
IMAGES_DIR = "HSiPu2/HSiPu2/all_pictures/fwcce/"
OUTPUT_DIR = "./output_vis"    
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 如果坐标在 [0,1] 范围表示已经归一化，需要用图像实际宽高乘回来
IS_COORD_NORMALIZED = True

# 关键点数量（COCO 人体姿态一般是 17 个）
NUM_KEYPOINTS = 17


def remove_black_border(image):
    """检测并返回去除黑边后的图片和有效高度"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 只检测非黑色区域

    # 找到轮廓（获取非黑色区域的外框）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image, image.shape[0]  # 没有黑边，返回原图

    # 计算非黑边区域的边界框
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_img = image[y:y+h, x:x+w]  # 裁剪掉黑边

    return cropped_img, h  # 返回去黑边的图片和新的高度


with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    # 如果CSV第一行是表头，可先 next(reader) 跳过
    next(reader)

    for row in reader:
        # row[0] 是图像ID，如 "ywqzz1-1961"
        image_id = row[0]
        image_path = os.path.join(IMAGES_DIR, image_id + ".jpg")  # 若是 .png 则改成 .png
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Failed to read image: {image_path}")
            continue

        # c_img,h = remove_black_border(img)
        h, w = img.shape[:2]  # 图像高度、宽度
        h = 146
        # 逐个关键点解析 (x, y, score)
        # body-0 占用 row[1], row[2], row[3]
        # body-1 占用 row[4], row[5], row[6]
        # ...
        # body-i 占用 row[1 + i*3], row[2 + i*3], row[3 + i*3]
        for i in range(NUM_KEYPOINTS):
            base_idx = 2 + i*4
            try:
                x_val = float(row[base_idx])
                y_val = float(row[base_idx + 1])
                score_val = float(row[base_idx + 2])
            except ValueError:
                # CSV中若有空或非数字数据，可做异常处理
                continue

            # 若关键点置信度太低，可选择跳过
            # 例如: if score_val < 0.2: continue

            # 如果 CSV 中的 x,y 已经归一化，需要乘回图像宽高
            if IS_COORD_NORMALIZED:
                px = int(x_val * w)
                py = int(y_val * h) + 55
            else:
                px = int(x_val)
                py = int(y_val)

            # 在图像上绘制关键点（绿色圆点）
            cv2.circle(img, (px, py), 4, (0, 255, 0), thickness=-1)

            # 如果想显示 score，可以加个文本：
            # cv2.putText(img, f"{score_val:.2f}", (px+3, py-3),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 将可视化结果保存到 OUTPUT_DIR
        out_path = os.path.join(OUTPUT_DIR, image_id + "_vis.jpg")
        cv2.imwrite(out_path, img)

        # 若想在窗口中预览，可取消注释以下两行
        # cv2.imshow("Keypoints", img)
        # cv2.waitKey(0)

# cv2.destroyAllWindows()
print("Done! Visualized keypoints have been saved to:", OUTPUT_DIR)
