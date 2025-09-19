#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/15 17:22
@source from: 
"""
from collections import Counter
import os

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# ===== 路径（按需修改） =====
train_path = '/7-19-Project/imcs21-cblue-main/task4/LLM-DAC/BERT-DAC-BYMY/data/train.txt'
test_path  = '/7-19-Project/imcs21-cblue-main/task4/LLM-DAC/BERT-DAC-BYMY/data/test.txt'
val_path   = '/7-19-Project/imcs21-cblue-main/task4/LLM-DAC/BERT-DAC-BYMY/data/test.txt'
model_path = '/Volumes/mac_win/models/tiansz/bert-base-chinese'   # 也可 'bert-base-chinese'

NUM_LABELS = 16  # <<< 你的类别数

# ===== Dataset =====
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        # 多分类：label 必须是 0..NUM_LABELS-1 的整数
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
# 如果你的标签是字符串，先做全局映射；若本来就是整数，下面会自动跳过
LABEL2ID, ID2LABEL = None, None

def build_label_mapping(paths):
    """扫描所有文件，若发现非整型标签，构建 str -> id 的映射"""
    global LABEL2ID, ID2LABEL
    all_labels = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                lin = line.rstrip("\n")
                if not lin:
                    continue
                # 从右侧切一次，避免文本中间的 \t 干扰
                parts = lin.rsplit("\t", 1)
                if len(parts) != 2:
                    continue  # 或者 raise
                _, lab = parts
                lab = lab.strip()
                # 判断是不是整数
                try:
                    int(lab)
                except ValueError:
                    all_labels.append(lab)
    if all_labels:
        classes = sorted(set(all_labels))
        LABEL2ID = {c: i for i, c in enumerate(classes)}
        ID2LABEL = {i: c for c, i in LABEL2ID.items()}
        print(f"[LabelMap] detected string labels -> {len(classes)} classes")
    else:
        LABEL2ID = None
        ID2LABEL = None
        print("[LabelMap] integer labels detected; no mapping needed.")

def load_text_classification_data(path):
    """
    行格式: text \\t label
    - 支持 text 中包含制表符（用 rsplit('\t', 1) 只切最后一个）
    - label 若是字符串 -> 用 LABEL2ID 映射
    - label 若是整数 -> 直接转 int
    """
    texts, labels = [], []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lin = line.rstrip("\n")
            if not lin:
                continue
            parts = lin.rsplit("\t", 1)
            if len(parts) != 2:
                bad += 1
                continue
            content, lab = parts[0], parts[1].strip()
            if LABEL2ID is None:
                # 期望是整数
                try:
                    lab_id = int(lab)
                except ValueError:
                    bad += 1
                    continue
            else:
                # 字符串标签 -> 映射
                if lab not in LABEL2ID:
                    bad += 1
                    continue
                lab_id = LABEL2ID[lab]
            texts.append(content)
            labels.append(lab_id)

    # 简单分布检查
    cnt = Counter(labels)
    print(f"[Load] {path} -> samples={len(labels)}  bad_lines={bad}  classes={len(cnt)}")
    print(f"[Dist] top counts: {cnt.most_common(5)}")
    return texts, labels

def get_datasets(tokenizer):
    # 先根据三个文件建立一次映射（如果需要）
    build_label_mapping([train_path, val_path, test_path])

    train_texts, train_labels = load_text_classification_data(train_path)
    val_texts,   val_labels   = load_text_classification_data(val_path)
    test_texts,  test_labels  = load_text_classification_data(test_path)

    # 断言范围（如果你坚持用固定 NUM_LABELS）
    if 'NUM_LABELS' in globals() and NUM_LABELS is not None:
        max_id = max(train_labels + val_labels + test_labels) if (train_labels or val_labels or test_labels) else -1
        assert max_id < NUM_LABELS, f"标签 id={max_id} 超过 NUM_LABELS={NUM_LABELS}，请调整或自动推断。"
        num_labels = NUM_LABELS
    else:
        # 自动推断类别数
        num_labels = len(set(train_labels) | set(val_labels) | set(test_labels))

    # 如果你想让模型的 id2label/label2id 更直观（字符串标签）
    if ID2LABEL is None:
        id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        label2id = {v: k for k, v in id2label.items()}
    else:
        id2label = {i: ID2LABEL[i] for i in range(num_labels)}
        label2id = {v: k for k, v in id2label.items()}

    # 构造 Dataset
    train_ds = MyDataset(train_texts, train_labels, tokenizer)
    val_ds   = MyDataset(val_texts,   val_labels,   tokenizer)
    test_ds  = MyDataset(test_texts,  test_labels,  tokenizer)

    # 为后续创建模型时使用
    return train_ds, val_ds, test_ds, test_texts, test_labels, num_labels, id2label, label2id


tokenizer = BertTokenizer.from_pretrained(model_path)
train_dataset, val_dataset, test_dataset, test_texts, test_labels, NUM_LABELS_AUTO, ID2LABEL_MAP, LABEL2ID_MAP = get_datasets(tokenizer)

# 用“数据里真正的类别数和映射”来初始化模型（不要手写 NUM_LABELS=16，避免不一致）
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=NUM_LABELS_AUTO,
    id2label=ID2LABEL_MAP,
    label2id=LABEL2ID_MAP,
)