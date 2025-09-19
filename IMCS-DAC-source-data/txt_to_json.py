#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/18 11:31
@source from: 
"""
import json

# 文件路径
json_file = "/Volumes/PSSD/NetThink/pythonProject/7-19-Project/imcs21-cblue-main/task4/IMCS-DAC-source-data/IMCS-DAC_test.json"
pred_file = "/Volumes/PSSD/predict_results.txt"
output_file = "IMCS-DAC_test_with_pred.json"

# id2label 映射表
id2label = {
    0: "Request-Etiology",
    1: "Request-Precautions",
    2: "Request-Medical_Advice",
    3: "Inform-Etiology",
    4: "Diagnose",
    5: "Request-Basic_Information",
    6: "Request-Drug_Recommendation",
    7: "Inform-Medical_Advice",
    8: "Request-Existing_Examination_and_Treatment",
    9: "Inform-Basic_Information",
    10: "Inform-Precautions",
    11: "Inform-Existing_Examination_and_Treatment",
    12: "Inform-Drug_Recommendation",
    13: "Request-Symptom",
    14: "Inform-Symptom",
    15: "Other"
}

# 1. 读取原始 JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 读取预测结果（逐行读取，去掉空行）
with open(pred_file, "r", encoding="utf-8") as f:
    preds = [line.strip() for line in f if line.strip()]

print(f"✅ 原始句子数: {sum(len(v) for v in data.values())}, 预测结果数: {len(preds)}")

# 3. 将预测结果逐一写回 dialogue_act
idx = 0
for conv_id, turns in data.items():
    for t in turns:
        if idx < len(preds):
            pred = preds[idx]
            pred = int(pred.split('\t')[1])
            # 如果预测是数字 ID → 转成 int，再映射到 label
            if isinstance(pred,int):
                label = id2label.get(int(pred), "Other")
            else:
                # 如果已经是标签字符串，直接用
                label = pred
            t["dialogue_act"] = label
            idx += 1

# 4. 保存新 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存结果到 {output_file}")