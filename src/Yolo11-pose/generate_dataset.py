import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# 配置参数
RAW_DATA_DIR = "HSiPu2/HSiPu2"         # 原始数据根目录
IMG_PATH = "HSiPu2/HSiPu2/all_pictures"  # 图片存储路径
OUTPUT_DIR = "dataset"                  # 输出数据集目录
CLASSES = ["fwcce", "fwcce_n", "fwcz", "fwcz_n", "ytc", "ytc_n", "ytz", "ytz_n"]  # 完整类别列表
VISIBILITY_THRESH = 0.5                 # 关键点可见性阈值
TEST_SIZE = 0.2                         # 验证集比例
IMG_EXT = ".jpg"                        # 图片扩展名

# 创建输出目录
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)

# 初始化统计变量
stats = {
    'total_csv': 0,
    'total_samples': 0,
    'class_counts': {cls: 0 for cls in CLASSES},
    'train_samples': 0,
    'val_samples': 0,
    'train_class_counts': {cls: 0 for cls in CLASSES},
    'val_class_counts': {cls: 0 for cls in CLASSES}
}

def process_csv(csv_path, class_name):
    """处理单个CSV文件，返回该文件所有样本数据"""
    samples = []
    file_samples = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过标题行
        for row_idx, row in enumerate(reader, 1):
            # 生成图片文件名（使用CSV中的文件名）
            img_name = row[0].strip()
            
            # 解析关键点
            kps = []
            for i in range(17):
                nx = float(row[1 + i*4 + 1])
                ny = float(row[1 + i*4 + 2])
                px = int(nx * 256)
                py = int(ny * 146) + 55
                x = float(px/256)
                y = float(py/256)
                score = float(row[1 + i*4 + 3])
                visible = 1 if score > VISIBILITY_THRESH else 0
                kps.append([x, y, visible])
            
            # 计算边界框
            visible_kps = [kp for kp in kps if kp[2] == 1]
            if not visible_kps:
                continue  # 跳过无关键点的样本
            
            x_coords = [kp[0] for kp in visible_kps]
            y_coords = [kp[1] for kp in visible_kps]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 生成YOLO标签行
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            label = [str(CLASSES.index(class_name))]
            label += [f"{x_center:.6f}", f"{y_center:.6f}", f"{width:.6f}", f"{height:.6f}"]
            for x, y, visible in kps:
                label += [f"{x:.6f}", f"{y:.6f}", str(visible)]
                
            samples.append((img_name, " ".join(label)))
            
            # 更新统计
            stats['total_samples'] += 1
            stats['class_counts'][class_name] += 1
            file_samples += 1
            
    print(f"处理文件 {os.path.basename(csv_path)}: {file_samples} 个样本")
    return samples

# 收集所有CSV文件路径
csv_files = []
for class_name in CLASSES:
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"警告：缺少类别目录 {class_name}")
        continue
    
    class_csvs = [f for f in os.listdir(class_dir) if f.endswith(".csv")]
    stats['total_csv'] += len(class_csvs)
    
    for fname in class_csvs:
        csv_files.append((os.path.join(class_dir, fname), class_name))

# 打印原始数据统计
print("\n=== 原始数据统计 ===")
print(f"总CSV文件数：{stats['total_csv']}")
print(f"总样本数：{stats['total_samples']}")
print("各类别样本分布：")
for cls in CLASSES:
    count = stats['class_counts'][cls]
    print(f"  {cls.ljust(8)}: {count} ") # ({count/stats['total_samples']:.1%})

# 按CSV文件划分训练集/验证集（保持动作序列完整性）
train_files, val_files = train_test_split(
    csv_files, 
    test_size=TEST_SIZE,
    random_state=42,
    stratify=[c for _, c in csv_files]  # 保持类别分布
)

print(f"\n划分结果：")
print(f"训练CSV文件数：{len(train_files)}")
print(f"验证CSV文件数：{len(val_files)}")

# 处理所有CSV文件
def process_and_save(file_list, split):
    for csv_path, class_name in file_list:
        samples = process_csv(csv_path, class_name)
        for img_name, label in samples:
            # 构建图片路径
            src_img = os.path.join(IMG_PATH, f"{img_name}{IMG_EXT}")
            if class_name[-len("_n"):] == '_n':
                class_name = class_name[:-len("_n")]

            src_img = f"{IMG_PATH}/{class_name}/{img_name}{IMG_EXT}"
            dst_img = os.path.join(OUTPUT_DIR, "images", split, f"{img_name}{IMG_EXT}")
            
            # 复制图片
            if os.path.exists(src_img):
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                try:
                    shutil.copy(src_img, dst_img)
                except Exception as e:
                    print(f"复制失败 {src_img} -> {dst_img}: {str(e)}")
                    continue
            else:
                print(f"警告：图片不存在 {src_img}")
                continue
            
            # 保存标签
            label_path = os.path.join(OUTPUT_DIR, "labels", split, f"{img_name}.txt")
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(label + "\n")
            
            # 更新划分统计
            if split == "train":
                stats['train_samples'] += 1
                stats['train_class_counts'][class_name] += 1
            else:
                stats['val_samples'] += 1
                stats['val_class_counts'][class_name] += 1

process_and_save(train_files, "train")
process_and_save(val_files, "val")

# 打印最终统计
print("\n=== 数据集统计 ===")
print(f"训练集样本数：{stats['train_samples']} ({stats['train_samples']/(stats['train_samples']+stats['val_samples']):.1%})")
print(f"验证集样本数：{stats['val_samples']} ({stats['val_samples']/(stats['train_samples']+stats['val_samples']):.1%})")

print("\n训练集类别分布：")
for cls in CLASSES:
    count = stats['train_class_counts'][cls]
    print(f"  {cls.ljust(8)}: {count} ({count/stats['train_samples']:.1%})")

print("\n验证集类别分布：")
for cls in CLASSES:
    count = stats['val_class_counts'][cls]
    print(f"  {cls.ljust(8)}: {count} ({count/stats['val_samples']:.1%})")

# 生成data.yaml
yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
kpt_shape: [17, 3]  # 17个关键点，每个点含x,y,visibility
"""

with open(f"{OUTPUT_DIR}/data.yaml", 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print("\n=== 数据集构建完成 ===")
print(f"输出目录：{os.path.abspath(OUTPUT_DIR)}")