import json
import torch
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm  # 用于显示进度条


# 定义领域标签
labels = [
    "politics",
    "entertainment",
    "health",
    "sports",
    "business",
    "technology",
    "science",
    "education",
    "general",
    "other"
]

# 加载本地模型
model_path = "/home/szu/fujianye/FaKnow-master/bart-large-mnli/"
print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 检查CUDA是否可用并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 使用zero-shot分类器预测领域标签
def predict_labels(texts, labels):
    predictions = []
    for text in tqdm(texts, desc="Predicting labels"):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            best_label = labels[np.argmax(probabilities)]
            predictions.append(labels.index(best_label))  # 获取最高得分的标签
        except RuntimeError as e:
            print(f"Error processing text: {e}")
            predictions.append(None)  # 如果出错，记录None以保持索引一致
    return predictions

# 读取和处理JSON文件
def process_json(input_file, output_file):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_data = []
    origin_texts = []
    generated_texts = []

    print("Preparing texts...")
    for item in data.values():
        origin_texts.append(item["origin_text"])
        generated_texts.append(item["generated_text"])

    # 预测原始文本和生成文本的领域标签
    print("Predicting domains for origin texts...")
    origin_domains = predict_labels(origin_texts, labels)

    print("Predicting domains for generated texts...")
    generated_domains = predict_labels(generated_texts, labels)

    print("Processing and saving results...")
    for i, item in enumerate(data.values()):
        origin_label = 1 if item["origin_label"] == "fake" else 0
        generated_label = 1 if item["generated_label"] == "fake" else 0
        
        if origin_domains[i] is not None:  # 只处理没有出错的文本
            output_data.append({"text": origin_texts[i], "domain": int(origin_domains[i]), "label": origin_label})
        
        if generated_domains[i] is not None:  # 只处理没有出错的文本
            output_data.append({"text": generated_texts[i], "domain": int(generated_domains[i]), "label": generated_label})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

# 输入文件和输出文件路径
input_file = 'megafake-1_style_based_fake.json'
output_file = 'processed_data.json'

# 处理JSON文件
process_json(input_file, output_file)
