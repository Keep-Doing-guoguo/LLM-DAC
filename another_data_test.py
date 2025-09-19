#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/12 17:17
@source from: 
"""
import json

result_path = "/Volumes/PSSD/NetThink/pythonProject/7-19-Project/imcs21-cblue-main/task4/LLM-DAC/success1.json"
source_path = "/Volumes/PSSD/NetThink/pythonProject/7-19-Project/imcs21-cblue-main/task4/IMCS-DAC-source-data/IMCS-DAC_test.json"
output_path = "unmatched_test1.json"

# 1. 读取 success1.json
with open(result_path, "r", encoding="utf-8") as f:
    raw_pred = json.load(f)

# success 里有哪些对话 ID
predicted_ids = set()
for item in raw_pred:
    for k in item.keys():
        predicted_ids.add(str(k))

# 2. 读取 test 数据
with open(source_path, "r", encoding="utf-8") as f:
    dialogues = json.load(f)
print(f"总计{len(dialogues)}s数据")
print(f"成功运行了{len(raw_pred)}数据")
# 3. 找到没匹配到的
unmatched = {}
for dialog_id, utterances in dialogues.items():
    if str(dialog_id) not in predicted_ids:
        unmatched[dialog_id] = utterances

# 4. 保存
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(unmatched, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存未匹配的对话，共 {len(unmatched)} 条 → {output_path}")


