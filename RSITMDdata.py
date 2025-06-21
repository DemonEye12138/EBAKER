import pandas as pd
import os

# 输入输出路径配置
input_csv = "/root/autodl-tmp/EBAKER/ebaker/rsitmd_test0.csv"
output_csv = "/root/autodl-tmp/EBAKER/ebaker/rsitmd_test.csv"

# 新旧路径前缀映射 (注意保留尾部斜杠)
old_base = "/home/mcx/RS/Datasets/RSITMD/images/"
new_base = "/root/autodl-tmp/data/RSITMD/images/"

# 读取CSV文件
df = pd.read_csv(input_csv)


# 路径转换函数
def convert_rsitmd_path(old_path):
    # 替换基础路径
    new_path = old_path.replace(old_base, new_base)

    # 统一使用正斜杠 (适配Windows)
    new_path = new_path.replace("\\", "/")

    # 验证文件名一致性
    assert os.path.basename(old_path) == os.path.basename(new_path), "文件名不匹配"

    return new_path


# 应用路径转换
df["filename"] = df["filename"].apply(convert_rsitmd_path)

# 保存新CSV (保持原始格式)
df.to_csv(output_csv, index=False, sep=",")

# 打印转换结果示例
print(f"转换完成！新文件已保存至：{output_csv}")
print("\n转换前后路径对比示例：")
print("[原始路径]", df.loc[0, "filename"].replace(new_base, old_base))
print("[新路径]  ", df.loc[0, "filename"])