#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/15 11:34
@source from: 
"""
import json

# 假设 data 就是你贴的那个大字典
with open("/Volumes/PSSD/NetThink/pythonProject/7-19-Project/imcs21-cblue-main/task4/LLM-DAC/success_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# with open("/Volumes/PSSD/NetThink/pythonProject/7-19-Project/imcs21-cblue-main/task4/LLM-DAC/result/success1.json", "r", encoding="utf-8") as f:
#     data1 = json.load(f)

total_dialogs = len(data)
dialog_with_value = 0
dialog_without_value = 0

for item in data:   # data 是 list
    (dialog_id, utterances), = item.items()  # 解包成 dialog_id, utterances
    has_value = any(str(x).strip() != "" for x in utterances)
    if has_value:
        dialog_with_value += 1
    else:
        dialog_without_value += 1

print(f"总对话数: {total_dialogs}")
print(f"至少有一个 dialogue_act 有值: {dialog_with_value}")
print(f"全部 dialogue_act 为空: {dialog_without_value}")
