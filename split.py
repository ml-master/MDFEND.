import json
import random

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def main():
    # 文件路径
    file1 = 'processed_data1.json'
    file2 = 'processed_data2.json'
    train_file = 'train.json'
    test_file = 'test.json'
    train_ratio = 0.8  # 训练集比例

    # 加载数据
    data1 = load_json(file1)
    data2 = load_json(file2)

    # 整合数据
    combined_data = data1 + data2

    # 分割数据集
    train_data, test_data = split_data(combined_data, train_ratio)

    # 保存训练集和测试集
    save_json(train_data, train_file)
    save_json(test_data, test_file)

    print(f"Data has been split into {train_file} and {test_file} with train ratio of {train_ratio}")

if __name__ == "__main__":
    main()
