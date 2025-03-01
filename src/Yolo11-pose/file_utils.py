import os

def rename_subfolders(parent_folder, new_name):
    # 获取父文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    
    for i, subfolder in enumerate(subfolders):
        old_path = os.path.join(parent_folder, subfolder)
        new_subfolder_name = subfolder[:-len("_n")]
        new_path = os.path.join(parent_folder, new_subfolder_name)
        os.rename(old_path, new_path)
        print(f"已将子文件夹 '{subfolder}' 重命名为 '{new_subfolder_name}'")

# 示例用法
parent_folder = 'HSiPu2/HSiPu2/not_standard'  # 请替换为实际路径
new_name = '_n'  # 请替换为您希望的子文件夹新名称
rename_subfolders(parent_folder, new_name)
