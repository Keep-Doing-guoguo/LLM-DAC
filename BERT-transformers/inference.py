#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/15 17:04
@source from: 
"""
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ckpt_dir = "/home/model/public/real_zhangguowen/other_code/imcs21-cblue-main/BERT-transformers/checkpoints/checkpoint-27603"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
model.to(device)
model.eval()

# 读取测试文件（每行一条）
test_file = "/home/model/public/real_zhangguowen/other_code/imcs21-cblue-main/IMCS-DAC_test_formatted.txt"
with open(test_file, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

# ===== 小批量推理 =====
BATCH_SIZE = 32          # 视显存酌情改小如 16/8
MAX_LEN    = 128

pred_ids_all = []

with torch.inference_mode():  # 比 no_grad 还能再省一点
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}

        # FP16 自动混合精度（如果是 A100/V100/T4 等都 OK）
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(**enc).logits   # [B, num_labels]

        pred_ids = logits.argmax(dim=-1).tolist()
        pred_ids_all.extend(pred_ids)

        # 及时清理，避免碎片化
        del enc, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 保存结果（按 “原文本<TAB>预测ID”）
out_path = "predict_results.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for t, pid in zip(texts, pred_ids_all):
        f.write(f"{t}\t{pid}\n")