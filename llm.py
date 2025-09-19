#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/9/10 17:52
@source from: 
"""
import time
import re
import dashscope
import requests
from http import HTTPStatus
import json
from prompt import prompt
#data_path = "/Volumes/PSSD/NetThink/pythonProject/7-19-Project/imcs21-cblue-main/task4/IMCS-DAC-source-data/IMCS-DAC_train.json"
test_path = "/Volumes/PSSD/NetThink/pythonProject/7-19-Project/imcs21-cblue-main/task4/IMCS-DAC-source-data/IMCS-DAC_test.json"

API_URL = ""
API_KEY = ""   # 替换成你的实际 key

# 全局变量：记录调用次数与起始时间
_last_reset_time = time.time()
_request_count = 0
MAX_CALLS_PER_MIN = 5  # 每分钟最多 5 次

def _check_rate_limit():
    global _last_reset_time, _request_count
    now = time.time()
    # 如果过了一分钟，重置计数
    if now - _last_reset_time >= 60:
        _last_reset_time = now
        _request_count = 0

    if _request_count >= MAX_CALLS_PER_MIN:
        # 本分钟次数已满，需要等待
        sleep_time = 60 - (now - _last_reset_time)
        if sleep_time > 0:
            print(f"⚠️ 已达到限制，等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
        # 重置计数器
        _last_reset_time = time.time()
        _request_count = 0

    _request_count += 1




def call_qwen_local(message):
    """
    调用本地 Qwen 模型接口
    :param message: [{'role': 'user', 'content': 'xxx'}]
    :return: 模型回复文本
    """
    API_KEY = "sk-test"  # 这里随便写就行，server默认不验证

    API_URL = ""

    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + message

    payload = {
        "model": "qwen2.5-7b-instruct",
        "messages": messages,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == HTTPStatus.OK:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        return f"API 调用失败: {response.status_code}, {response.text}"


def call_qwen_siliconflow(message):
    """
    调用 SiliconFlow Qwen 模型接口
    :param message: [{'role': 'user', 'content': 'xxx'}] 这样的消息数组
    :return: 模型回复文本
    """
    _check_rate_limit()
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + message

    payload = {
        "model": "Qwen/QwQ-32B",
        "messages": messages
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == HTTPStatus.OK:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        return f"API 调用失败: {response.status_code}, {response.text}"

def call_qwen_dashscope(message):
    _check_rate_limit()
    your_api_key = ''  # 您的通义千问大模型API_key
    dashscope.api_key = your_api_key


    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                message[0]]
    response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_plus,
            dashscope.Generation.Models.qwen_plus,
            api_key=your_api_key,
            messages=messages,
            result_format='message',  # 将返回结果格式设置为 message

        )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0]['message']['content']
    else:

        return '当前调用API失败'

def after_process(dialog_id,text):
    # 方法 1：用正则取出 “数字.类别”
    try:
        result = re.findall(r'\d+\.(.+)', text)
        #print(result)
        return result
    except:
        print('模型结果，后处理错误，记录当前ID。')
        return dialog_id

with open(test_path, "r", encoding="utf-8") as f:
    data = json.load(f)


error = []
success = []
i = 1
save_interval = 10   # 每多少条保存一次
pred_map = {}
# 遍历所有对话
for dialog_id, sentences in data.items():
    dialog = []
    print(f"对话ID: {dialog_id}, 共 {len(sentences)} 句,{type(dialog_id)},正在处理第{i}条数据！")
    i = i+1
    # 遍历该对话里的每句话
    for index,utt in enumerate(sentences):
        sid = utt["sentence_id"]
        role = utt["speaker"]
        text = utt["sentence"]
        intent = utt["dialogue_act"]
        #print(f"  {sid} | {role}: {text}  -->  {intent}")
        dialog.append(f"{index+1}.{role}: {text}")
        #print(f"{index+1}.{role}: {text}")
    output = "\n".join(f"{line}" for i, line in enumerate(dialog))

    message = [{'role': 'user', 'content': f'{prompt.format(input=output)}'}]
    result = call_qwen_dashscope(message)
    #print(result)
    labels = after_process(dialog_id, result)#这个结果是一个长度和sentences相同的，并且需要将其保存到原始文件中。和原始数据进行拼接到一起。
    print(len(labels) ,len(sentences))
    # 严格校验长度：不一致就跳过或补齐
    n = len(sentences)
    if not isinstance(labels, list):
        # 解析失败，给个占位（比如 "Other"），避免打乱后续对齐
        labels = ["Other"] * n
    elif len(labels) != n:
        # 长度不一致：截断或补齐到 n
        if len(labels) > n:
            labels = labels[:n]
        else:
            labels = labels + ["Other"] * (n - len(labels))

    pred_map[str(dialog_id)] = labels


# —— 回填到原始 data 里，并保存 merged.json ——
merged = {}
for dialog_id, sentences in data.items():
    labels = pred_map.get(str(dialog_id), None)
    if labels is None:
        # 该对话没预测结果时，保持原样或统一填默认
        # 这里保留原样（不改 dialogue_act）
        merged[dialog_id] = sentences
        continue

    # 把每句的 dialogue_act 写回去
    new_sentences = []
    for i, utt in enumerate(sentences):
        utt_new = dict(utt)  # 拷贝避免原地修改（可选）
        utt_new["dialogue_act"] = labels[i]
        new_sentences.append(utt_new)

    merged[dialog_id] = new_sentences

# —— 最终写盘 ——
with open("merged.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print("✅ 已保存回填后的结果到 merged.json")

'''

llm load:
modelscope download --model Qwen/Qwen2.5-14B-Instruct-GGUF qwen2.5-14b-instruct-fp16.gguf --local_dir /home/model/public/real_zhangguowen/models


nohup python run_qwen.py > run_qwen.log 2>&1 &
'''




