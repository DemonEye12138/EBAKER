import pandas as pd
import os


def extract_class_folder(filename: str) -> str:
    """从文件名提取类目录名（最后一个下划线前的内容）"""
    # 移除可能的文件扩展名
    base_name = os.path.splitext(filename)[0]

    # 查找最后一个下划线
    last_underscore = base_name.rfind('_')

    if last_underscore == -1:
        raise ValueError(f"文件名缺少下划线: {filename}")

    return base_name[:last_underscore]


def convert_nwpu_path(old_path: str) -> str:
    """NWPU路径转换函数"""
    # 统一路径格式
    old_path = old_path.replace("\\", "/")

    # 基础路径配置
    old_prefix = "/home/mcx/RS/Datasets/NWPU-RESISC45/"
    new_prefix = "data/NWPU-RESISC45/"

    # 验证原始路径格式
    if not old_path.startswith(old_prefix):
        raise ValueError(f"非NWPU路径: {old_path}")

    # 提取文件名并处理
    filename = os.path.basename(old_path)
    class_folder = extract_class_folder(filename)

    # 构建新路径
    new_path = os.path.join(
        new_prefix,
        class_folder,
        filename
    ).replace("\\", "/")

    return new_path


# 输入输出配置
input_csv = "ebaker/nwpu_test0.csv"
output_csv = "ebaker/nwpu_test.csv"

# 处理CSV文件
df = pd.read_csv(input_csv)
df["filename"] = df["filename"].apply(convert_nwpu_path)
df.to_csv(output_csv, index=False)

print(f"转换完成！转换示例：\n{df.head(3).to_string(index=False)}")