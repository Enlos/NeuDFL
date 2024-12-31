import re

# 定义一个函数来处理日志文件中的数据
def process_log_file(log_file):
    # 用于存储结果，每条记录存储为 {class_id: [(timestamp, avg_value)]}
    results = {}

    # 优化后的正则表达式
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - INFO - Class (\d+) accuracies: \[(.*?)\]")

    with open(log_file, 'r') as file:
        for line in file:
            # 匹配目标行
            match = pattern.search(line)
            if match:
                # 打印匹配的行（调试用）
                print("Matched line:", line.strip())

                # 提取时间戳、类编号和准确率列表
                timestamp = match.group(1)
                class_id = int(match.group(2))
                accuracies = list(map(float, match.group(3).split(', ')))

                # 计算倒数第二个到倒数第十一个数据的平均值
                if len(accuracies) >= 10:
                    avg_value = sum(accuracies[-11:-1]) / len(accuracies[-11:-1])
                    # 将结果添加到对应的 class_id 中，记录时间戳
                    if class_id not in results:
                        results[class_id] = []
                    results[class_id].append((timestamp, avg_value))

    return results

# 使用方法
log_file_path = 'logs/GTSRB_40ma_0To5.log'  # 替换为实际日志文件路径
averages = process_log_file(log_file_path)

# 打印结果
for class_id, records in averages.items():
    print(f"Class {class_id}:")
    for timestamp, avg in records:
        print(f"  At {timestamp}, Average of last 2nd to 11th values = {avg:.2f}")
