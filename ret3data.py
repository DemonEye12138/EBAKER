import pandas as pd
import os

# 输入输出配置
input_csv = "/root/autodl-tmp/EBAKER/ebaker/ret30.csv"
output_csv = "/root/autodl-tmp/EBAKER/ebaker/ret3.csv"

# 转换规则配置（按优先级排序）
PATH_RULES = [
    {  # RSITMD规则
        "old_prefix": "/home/mcx/RS/Datasets/RSITMD/images/",
        "new_prefix": "/root/autodl-tmp/data/RSITMD/images/",
        "type": "direct"
    },
    {  # RSICD规则
        "old_prefix": "/home/mcx/RS/Datasets/RSICD/RSICD_image/",
        "new_prefix": "/root/autodl-tmp/data/RSICD/RSICD_images/",
        "type": "direct"
    },
    {  # NWPU-RESISC45规则（支持多下划线文件名）
        "old_prefix": "/home/mcx/RS/Datasets/NWPU-RESISC45/",
        "new_prefix": "/root/autodl-tmp/data/NWPU-RESISC45/",
        "type": "dynamic_class"
    }
]


def extract_class_name(filename: str) -> str:
    """提取最后一个下划线前的所有字符作为类名"""
    # 找到最后一个下划线的位置
    last_underscore = filename.rfind('_')
    if last_underscore == -1:
        raise ValueError(f"文件名格式错误，缺少下划线: {filename}")

    # 提取类名（如 golf_course_123.jpg → golf_course）
    class_name = filename[:last_underscore]

    # 如果文件名存在扩展名（如 .jpg），需要二次验证
    if '.' in class_name:
        raise ValueError(f"类名包含非法字符: {filename}")
    return class_name


def convert_path(old_path: str) -> str:
    """智能路径转换函数"""
    # 统一路径格式
    old_path = old_path.replace("\\", "/")

    # 遍历所有规则进行匹配
    for rule in PATH_RULES:
        if old_path.startswith(rule["old_prefix"]):
            # 直接替换前缀的情况
            if rule["type"] == "direct":
                return old_path.replace(rule["old_prefix"], rule["new_prefix"])

            # 动态生成类目录的情况
            elif rule["type"] == "dynamic_class":
                filename = os.path.basename(old_path)
                class_name = extract_class_name(filename)
                return f"{rule['new_prefix']}{class_name}/{filename}"

    # 未匹配任何规则时抛出错误
    raise ValueError(f"未知路径格式: {old_path}")


# 读取并处理数据
df = pd.read_csv(input_csv)
df["filename"] = df["filename"].apply(convert_path)

# 保存结果
df.to_csv(output_csv, index=False)

print(f"转换完成！转换记录示例：\n{df.head(3).to_string(index=False)}")