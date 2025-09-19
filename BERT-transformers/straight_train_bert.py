#!/usr/bin/env python
# coding=utf-8
"""
@author: zgw
@date: 2025/6/25 16:28
@desc: BERT 多分类（15 类）
"""
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

def load_text_classification_data(path):
    """
    假设：第1列=文本，第2列=整数标签(0..N-1)。
    如果你是字符串标签，先做映射再返回整数。
    """
    texts = []
    labels = []
    texts1 = []
    labels1 = []
    with open(path,'r') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            conten1t, label1 = lin.split('\t')
            content, label = line.rstrip("\n").rsplit("\t", 1)
            texts.append(content)
            labels.append(int(label))
            texts1.append(conten1t)
            labels1.append(int(label1))
    # df = pd.read_excel(path)
    # texts = df.iloc[:, 0].astype(str).tolist()
    # labels = df.iloc[:, 1].astype(int).tolist()
    print('debug')
    return texts1, labels1#texts是list类型，其中放的是str。labels是list类型，其中放的是int。

def get_datasets(tokenizer):
    train_texts, train_labels = load_text_classification_data(train_path)
    val_texts,   val_labels   = load_text_classification_data(val_path)
    test_texts,  test_labels  = load_text_classification_data(test_path)
    return (
        MyDataset(train_texts, train_labels, tokenizer),
        MyDataset(val_texts,   val_labels,   tokenizer),
        MyDataset(test_texts,  test_labels,  tokenizer),
        test_texts, test_labels
    )

# ===== 多分类指标 =====
def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(logits, tuple):  # 兼容某些返回 (logits,)
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        # 需要的话可加 micro/weighted：
        # "micro_f1": f1_score(labels, preds, average="micro"),
        # "weighted_f1": f1_score(labels, preds, average="weighted"),
    }

# ===== Tokenizer & 数据 =====
tokenizer = BertTokenizer.from_pretrained(model_path)
train_dataset, val_dataset, test_dataset, test_texts, test_labels = get_datasets(tokenizer)

# ===== 模型（多分类头，CE损失自动处理） =====
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=NUM_LABELS,
    # 可选：设置 id2label/label2id（便于导出/推理更清晰）
    id2label={i: f"LABEL_{i}" for i in range(NUM_LABELS)},
    label2id={f"LABEL_{i}": i for i in range(NUM_LABELS)},
)

# ===== 训练参数 =====
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available(),  # 有GPU则开混合精度
    report_to="none",                # 不上报到 wandb 等
    # no_cuda=True,  # 如需强制CPU，打开此行
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 训练
trainer.train()

# （可选）测试集预测并保存
# pred_output = trainer.predict(test_dataset)
# logits = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions
# preds  = np.argmax(logits, axis=-1)
#
# df_out = pd.DataFrame({
#     "text": test_texts,
#     "true_label": test_labels,
#     "pred_label": preds
# })
# df_out.to_csv("bert_multiclass_predictions.csv", index=False, encoding="utf-8-sig")
# print("✅ 保存预测到 bert_multiclass_predictions.csv")