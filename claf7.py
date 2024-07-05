import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    for text in texts:
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

    print("Processing documents...")
    for key, item in tqdm(data.items(), desc="Processing documents"):
        doc_1_text = item.get("doc_1_text", None)
        doc_2_text = item.get("doc_2_text", None)
        generated_text_t01 = item.get("generated_text_t01", None)
        
        if doc_1_text:
            doc_1_domain = predict_labels([doc_1_text], labels)[0]
            if doc_1_domain is not None:
                doc_1_label = 1 if item["doc_1_label"] == "fake" else 0
                output_data.append({"text": doc_1_text, "domain": int(doc_1_domain), "label": doc_1_label})

        if doc_2_text:
            doc_2_domain = predict_labels([doc_2_text], labels)[0]
            if doc_2_domain is not None:
                doc_2_label = 1 if item["doc_2_label"] == "fake" else 0
                output_data.append({"text": doc_2_text, "domain": int(doc_2_domain), "label": doc_2_label})

        if generated_text_t01:
            generated_domain = predict_labels([generated_text_t01], labels)[0]
            if generated_domain is not None:
                generated_label = 1 if item["generated_label"] == "fake" else 0
                output_data.append({"text": generated_text_t01, "domain": int(generated_domain), "label": generated_label})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_file}")

# 输入文件和输出文件路径
input_file = 'megafake-7_integration_based_legitimate_tn300.json'
output_file = 'processed_data2.json'

# 处理JSON文件
process_json(input_file, output_file)
