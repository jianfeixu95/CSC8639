import base64
import json
import os
import time

import requests
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# OpenAI API Key
api_key = 'sk-None-ShiiU4k5S9L9PuSiADy3T3BlbkFJrBGBDvoNqKSKzhBX80pi'


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_chest_image_findings(image_path, save_path):
    image_name, _ = os.path.splitext(os.path.basename(image_path))
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a chest X-ray diagnostic report findings based on this image.(only show findings)"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    time.sleep(1)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response_json = response.json()
    findings_json = response_json['choices'][0]['message']['content'].replace('-', '').replace('\n', '')

    output_response_json_file_path = os.path.join(save_path + "/response", f'{image_name}.txt')
    output_findings_json_file_path = os.path.join(save_path + "/finding", f'{image_name}.txt')

    with open(output_response_json_file_path, 'w') as file:
        file.write(json.dumps(response_json, indent=4))

    with open(output_findings_json_file_path, 'w') as file:
        file.write(findings_json)

    print(f"Content successfully written to {image_name}")
    return findings_json


def get_model_metrics(generated_findings_list, ground_truth_list):
    assert len(generated_findings_list) == len(
        ground_truth_list), "The number of generated findings must match the number of ground truth examples."

    # 初始化指标累加器
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge_l_scores = []
    cider_scores = []

    # 初始化 CIDEr scorer
    cider_scorer = Cider()

    for generated_findings, ground_truth in zip(generated_findings_list, ground_truth_list):
        # 对文本进行分词
        ground_truth_tokens = word_tokenize(ground_truth.lower())
        generated_findings_tokens = word_tokenize(generated_findings.lower())

        # 计算 BLEU 分数
        smooth = SmoothingFunction().method1
        bleu1 = sentence_bleu([ground_truth_tokens], generated_findings_tokens, weights=(1, 0, 0, 0),
                              smoothing_function=smooth)
        bleu2 = sentence_bleu([ground_truth_tokens], generated_findings_tokens, weights=(0.5, 0.5, 0, 0),
                              smoothing_function=smooth)
        bleu3 = sentence_bleu([ground_truth_tokens], generated_findings_tokens, weights=(0.33, 0.33, 0.33, 0),
                              smoothing_function=smooth)
        bleu4 = sentence_bleu([ground_truth_tokens], generated_findings_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smooth)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)

        # 计算 METEOR 分数（需要原始字符串）
        meteor = meteor_score([ground_truth_tokens], generated_findings_tokens)
        meteor_scores.append(meteor)

        # 计算 ROUGE-L 分数
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_score = scorer.score(ground_truth, generated_findings)['rougeL'].fmeasure
        rouge_l_scores.append(rouge_score)

        # 计算 CIDEr 分数
        cider_score, _ = cider_scorer.compute_score({0: [ground_truth]}, {0: [generated_findings]})
        cider_scores.append(cider_score)

    # 计算各指标的平均值
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    avg_cider = sum(cider_scores) / len(cider_scores) if cider_scores else 0

    # 打印各指标的平均值
    print("Average BLEU-1:", avg_bleu1)
    print("Average BLEU-2:", avg_bleu2)
    print("Average BLEU-3:", avg_bleu3)
    print("Average BLEU-4:", avg_bleu4)
    print("Average METEOR:", avg_meteor)
    print("Average ROUGE-L:", avg_rouge_l)
    print("Average CIDEr:", avg_cider)


def check_file_exists(directory, target_filename):
    # 确保目标文件名以 .txt 结尾
    if not target_filename.endswith('.txt'):
        target_filename += '.txt'

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename == target_filename:
            return True

    return False


if __name__ == '__main__':
    csv_file_path = '/Volumes/G-RAID/个人库/代码/jupyter-notebook/workspace/CSC8639/mimic-cxr-jpg-v2.1.0/Model/gpt4-api-generation-model/data/test.csv'
    df = pd.read_csv(csv_file_path)
    save_path = f'/Volumes/G-RAID/个人库/代码/jupyter-notebook/workspace/CSC8639/mimic-cxr-jpg-v2.1.0/Model/gpt4-api-generation-model/data'
    generated_findings = []
    ground_truth = []

    for index, row in df.iterrows():
        image_path = row['mimic_image_file_path']
        filename, _ = os.path.splitext(os.path.basename(image_path))
        if not check_file_exists(save_path + "/finding", filename):
            generated_finding = get_chest_image_findings(image_path, save_path)

    for index, row in df.iterrows():
        image_path = row['mimic_image_file_path']
        filename, _ = os.path.splitext(os.path.basename(image_path))
        generated_finding_path = os.path.join(save_path + "/finding", filename + ".txt")
        ground_truth_findings = row['reference_report']
        if os.path.exists(generated_finding_path):
            with open(generated_finding_path, 'r', encoding='utf-8') as file:
                generated_finding = file.read()

            generated_findings.append(generated_finding)
            ground_truth.append(ground_truth_findings)

    get_model_metrics(generated_findings, ground_truth)
