# self-instruct
本项目实现了一个循环工作流，核心功能包括：
加载带标签的敏感句和任务池数据
自动选择敏感句并匹配相关任务
基于任务生成新指令和对应文本
通过 ROUGE 相似度过滤重复文本
循环执行直至达到目标文本数量（10000 条）
技术栈：LangGraph 工作流编排、Qwen2.5 预训练模型、ROUGE 文本相似度计算、vllm 高效推理
安装指南
环境要求
Python 3.8+
CUDA 11.7+（推荐 GPU 显存≥16GB）
依赖包：见requirements.txt
安装步骤
克隆仓库
bash
运行
git clone <仓库地址>
cd <仓库目录>
安装依赖
bash
运行
pip install -r requirements.txt
# 其中关键依赖包括：
# transformers vllm langgraph rouge-chinese jieba torch
准备模型
下载 Qwen2.5-7B-Instruct-Uncensored 模型（或其他兼容模型）
修改代码中model_path为本地模型路径：
python
运行
# 在main_workflow.py和self_instruction.py中
model_path = "/path/to/your/model"
使用说明
数据准备
需准备以下输入文件（JSON Lines 格式）：
敏感句文件（sensitive_words.jsonl）：
jsonl
{"text": "敏感内容1", "label": "标签A"}
{"text": "敏感内容2", "label": "标签B"}
任务池文件（task_pool.jsonl）：
jsonl
{"instruction": "任务指令1", "label": "标签A"}
{"instruction": "任务指令2", "label": "标签B"}
运行流程
bash
运行
python main_workflow.py \
  --sensitive_sentences_path /path/to/sensitive_words.jsonl \
  --tasks_path /path/to/task_pool.jsonl \
  --output_dir /path/to/output
命令行参数说明
参数名	默认值	说明
--sensitive_sentences_path	data/input/sensitive_words.jsonl	敏感句输入文件路径
--tasks_path	data/input/task_pool.jsonl	任务池输入文件路径
--output_dir	data/output	生成文本的输出目录
工作流程详解
加载数据（load_original_data）读取敏感句和任务池文件，验证数据格式并存储到状态中。
选择敏感句（select_sensitive_sentence）
首轮按顺序遍历敏感句（基于round_idx）
遍历完成后随机选取敏感句
筛选相关任务（filter_relevant_tasks）从任务池中筛选与当前敏感句同标签的任务：
随机选择 1 条作为必选任务
按 ROUGE 相似度从剩余任务中选取 Top2
生成新指令（generate_new_instructions）基于筛选的任务，调用模型生成 3 条新指令（保持标签一致）。
生成文本（generate_new_texts）根据新指令生成对应文本内容。
过滤重复文本（filter_by_rouge）
计算新生成文本与历史文本的 ROUGE-L 相似度
过滤相似度≥0.75 的文本，保留通过的文本
循环判断（check_completion）若累计通过文本数≥10000 则终止，否则返回步骤 2 继续循环。
输出文件
输出目录下会生成：
filtered_new_texts_1125.jsonl：通过过滤的文本（含标签、指令等元数据）
rejected_texts.jsonl：被过滤的重复文本（含 ROUGE 分数）
注意事项
模型加载需要较大显存，建议使用≥24GB 显存的 GPU
敏感句和任务池的标签需保持一致，否则会导致筛选失败
可通过调整SamplingParams（温度、top_p 等）控制生成文本的多样性
ROUGE 过滤阈值（0.75）可在filter_by_rouge函数中修改
