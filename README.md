
1. 项目介绍

本项目面向 医疗对话场景的意图识别任务。

传统方法

常见的做法包括：

	•	特征工程 + 传统机器学习（如 TF-IDF + SVM、CRF）；
	•	深度学习序列模型（如 BiLSTM + Attention、TextCNN）；
	•	预训练模型微调（如 BERT、RoBERTa）。

本实验方法

本实验基于 大语言模型（LLM），通过 提示词设计（Prompt Tuning），结合整段对话的上下文，生成每句话对应的意图标签。这样能更好地利用对话语境信息，相比传统单句分类方法有明显优势。



2. 环境准备

确保已安装：

	•	Python ≥ 3.8

	•	modelscope

	•	Docker（需支持 GPU，安装 NVIDIA Container Toolkit）



3. 模型下载

使用 modelscope 下载 Qwen2.5-7B-Instruct-GGUF 模型（分片格式）：

modelscope download --model Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-fp16-00001-of-00004.gguf --local_dir /home/model/public/real_zhangguowen/models/qwen2.5-7b-instruct-gguf
modelscope download --model Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-fp16-00002-of-00004.gguf --local_dir /home/model/public/real_zhangguowen/models/qwen2.5-7b-instruct-gguf
modelscope download --model Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-fp16-00003-of-00004.gguf --local_dir /home/model/public/real_zhangguowen/models/qwen2.5-7b-instruct-gguf
modelscope download --model Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-fp16-00004-of-00004.gguf --local_dir /home/model/public/real_zhangguowen/models/qwen2.5-7b-instruct-gguf




4. 模型启动

使用 llama.cpp server 启动模型（指定 GPU 设备）：

docker run --privileged --name llama9001 --gpus device=2 \
  -v /home/model/public/real_zhangguowen/models/qwen2.5-7b-instruct-gguf/:/models \
  -p 9997:8000 ghcr.nju.edu.cn/ggml-org/llama.cpp:server-cuda-b5726 \
  -m /models/qwen2.5-7b-instruct-fp16-00001-of-00004.gguf \
  --port 8000 --host 0.0.0.0 -c 16384 --n-gpu-layers 100 -n 4096 -np 8

此时模型服务会监听在 http://<IP>:9997/v1/。



5. 模型测试

使用 OpenAI SDK 进行接口调用：

from openai import OpenAI

client = OpenAI(
    api_key="sk-test",   # 任意值即可，llama.cpp 默认不校验
    base_url="http://<IP>:9997/v1/"
)

resp = client.chat.completions.create(
    model="qwen2.5-7b-instruct",
    messages=[{"role": "user", "content": "介绍一下朱元璋"}],
    temperature=0.7,
)

print(resp.choices[0].message.content)




6. Baseline

6.1. 大模型Prompt方案：

实验运行流程

	1.	在 llm.py 中配置模型服务的 IP 和 API_KEY。

	2.	执行 llm.py，会调用模型完成对话意图识别，并将结果保存到 success.json。直接提交该文件即可达到70%的准确率。

优化方向：

    1.可以继续调优提示词。
    2.可以尝试更大模型，如 Qwen2.5-14B。这里的模型是自己进行部署的。使用的llama.cpp。


6.2. Bert微调方案：

实验运行流程

    0.  使用json_to_txt.py将text.json转换为txt文件，后续拿着该txt文件去训练，配置到straight_train_bert。
    1.	在 straight_train_bert 中配置数据集路径和模型路径。即可开始执行训练。
    2.	在inference中将微调好的模型配置好路径，（微调后的模型可以达到结果为85.）
    3.  使用json_to_txt.py将IMCS-DAC_test.json转换为txt文件，拿着该txt文件去inference预测，得到结果txt文件。然后使用txt_to_json.py将结果txt文件转换为json格式，得到文件IMCS-DAC_test_with_pred.json作为最终提交结果。
优化方向：

    1.可以更换ernie。
    2.也可以进行大模型微调方案。

7. 提交结果
