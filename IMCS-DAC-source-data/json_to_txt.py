#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/18 11:19
@source from: 
"""
import json

# 输入输出路径
input_path = "IMCS-DAC_test.json"
output_path = "IMCS-DAC_test_formatted.txt"

# 读取 JSON 数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_path, "w", encoding="utf-8") as f_out:
    for dialog_id, utterances in data.items():
        for utt in utterances:
            role = utt.get("speaker", "未知")
            text = utt.get("sentence", "").strip()
            # 如果内容为空，替换为（空）
            if text == "":
                text = "（空）"
            # 这里随便加一个占位数字（比如 -1），你后面可以换成预测结果
            f_out.write(f"{role}：{text}\t-1\n")

print(f"✅ 已保存到 {output_path}")