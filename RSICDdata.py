import pandas as pd
import os

# 输入输出配置
input_csv = "/root/autodl-tmp/EBAKER/ebaker/rsicd_test0.csv"
output_csv = "/root/autodl-tmp/EBAKER/ebaker/rsicd_test.csv"

# 新旧路径前缀映射 (注意保留尾部斜杠)
old_base = "/home/mcx/RS/Datasets/RSICD/RSICD_image/"
new_base = "/root/autodl-tmp/data/RSICD/RSICD_images/"

# 读取CSV文件
df = pd.read_csv(input_csv)


# 路径转换函数
def convert_rsicd_path(old_path):
    # 直接替换路径前缀
    new_path = old_path.replace(old_base, new_base)

    # 统一使用正斜杠 (处理Windows路径问题)
    new_path = new_path.replace("\\", "/")

    # 验证文件名是否保留
    assert os.path.basename(old_path) == os.path.basename(new_path), "文件名不一致"

    return new_path


# 应用转换
df["filename"] = df["filename"].apply(convert_rsicd_path)

# 保存新文件 (保持CSV格式一致)
df.to_csv(output_csv, index=False, sep=",")

print(f"转换完成！新文件保存至: {output_csv}")
print(f"修改记录示例: \n{df.head(2).to_string(index=False)}")