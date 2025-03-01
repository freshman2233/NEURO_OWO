import os
import glob
import pandas as pd

def txt_to_csv_in_folder(folder_path):
    # 在指定文件夹中查找所有 .txt 文件
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    all_data = []
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # 每行应有 1 + 4 + 17*3 = 56 个数据
                if len(parts) != 56:
                    print(f"Warning: {txt_file} 中某行数据数量不等于 56: {parts}")
                    continue
                # 第一项转换为 int，其余转换为 float
                row = [int(parts[0])] + [float(x) for x in parts[1:]]
                all_data.append(row)
    
    # 定义 CSV 的列名
    columns = ["class", "x", "y", "width", "height"]
    for i in range(1, 18):
        columns.extend([f"px{i}", f"py{i}", f"p{i}_visibility"])
    
    # 构造 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(all_data, columns=columns)
    csv_file = os.path.join( f"{folder_path}_pose.csv")
    df.to_csv(csv_file, index=False)
    print(f"成功将 {folder_path} 中的所有 txt 文件转换为 {csv_file}")

# 针对 train 和 val 文件夹执行转换
folders = ['dataset/labels/train', 'dataset/labels/val']
for folder in folders:
    if os.path.exists(folder):
        txt_to_csv_in_folder(folder)
    else:
        print(f"未找到文件夹: {folder}")


train_df = pd.read_csv("dataset/labels/train_pose.csv")
val_df = pd.read_csv("dataset/labels/val_pose.csv")

# 使用 pd.concat 按行合并两个 DataFrame
combined_df = pd.concat([train_df, val_df], ignore_index=True)

# 保存合并后的 DataFrame 到新的 CSV 文件中
combined_df.to_csv("pose.csv", index=False)

print("合并完成，结果已保存为 combined.csv")
