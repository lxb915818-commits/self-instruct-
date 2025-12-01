本仓库实现了一个基于 本地大模型（vLLM / Transformers） 与 LangGraph 工作流 的自监督文本生成流水线，功能包括：

根据带标签的敏感句与任务池自动生成新的指令（Self-Instruct）。

使用生成的指令进一步生成文本数据。

使用 ROUGE-L 相似度策略去重、过滤并保存合格文本。

自动迭代，持续生成直到达到预期数量（默认 10,000 条）。

核心代码包括：

self_instruction.py：文本生成、相似度过滤、状态管理等主要逻辑

main_workflow.py：基于 LangGraph 的主流程编排

📁 目录结构示例
.
├── README.md
├── self_instruction.py
├── main_workflow.py
├── data/
│   ├── input/
│   │   ├── sensitive_words.jsonl
│   │   └── task_pool.jsonl
│   └── output/
│       ├── filtered_new_texts_1125.jsonl
│       └── rejected_texts.jsonl
└── requirements.txt

🚀 快速开始
1. 环境要求

Python 3.8+

GPU（建议）+ CUDA

依赖包：

transformers
vllm
langgraph
rouge
jieba
torch
tqdm

2. 安装依赖
python -m venv venv
source venv/bin/activate

pip install transformers vllm langgraph rouge jieba torch tqdm

3. 模型准备

代码初始化模型时需要指定本地路径，例如：

model_path = "/path/to/your/Qwen2.5-7B-Instruct"


请根据自己的环境修改 self_instruction.py 或 main_workflow.py 中的路径。

📦 输入数据格式
1. sensitive_words.jsonl

每行包含一个敏感句及其标签：

{"text": "示例敏感句 A", "label": "类别1"}

2. task_pool.jsonl

任务池需包含任务指令与对应标签：

{"instruction": "写一句关于X的描述。", "label": "类别1"}


程序会根据“标签”匹配任务与敏感句。

🔧 Command-line 参数（来自 main_workflow.py）
--sensitive_sentences_path   输入敏感句路径
--tasks_path                 输入任务池路径
--output_dir                 输出目录


示例：

python main_workflow.py \
  --sensitive_sentences_path ./data/input/sensitive_words.jsonl \
  --tasks_path ./data/input/task_pool.jsonl \
  --output_dir ./data/output

🔄 整体工作流程说明

整体流程由 LangGraph 构建：

load_original_data
加载敏感句和任务池（自动格式校验）

select_sensitive_sentence
从敏感句集合中轮询/随机选择一条

filter_relevant_tasks
从任务池中筛选 3 条同标签任务：

1 条随机必选

2 条 ROUGE-L Top2

generate_new_instructions
使用大模型基于任务 + 敏感句生成新指令（Self-Instruct）

generate_new_texts
使用生成的指令生成文本

filter_by_rouge
使用 ROUGE-L（阈值 0.75）过滤重复文本并写入文件

check_completion
判断是否达到目标数量（默认 10,000）

未完成则继续循环，直到满足生成目标。

🧠 核心模块摘要（self_instruction.py）
1. init_model(model_path)

加载 vLLM 模型和 tokenizer，设置生成参数（temperature、top_p、top_k 等）。

2. load_original_data

加载 jsonl，检查字段格式是否正确，解析为内部结构。

3. filter_relevant_tasks

按标签筛选任务 + 计算 ROUGE-L 排序选 Top2。

4. generate_new_instructions

将敏感句与任务模板输入大模型，生成指令。

5. generate_new_texts

根据指令生成输出文本。

6. filter_by_rouge

过滤与历史文本高度相似的文本（默认阈值 0.75）。

📤 输出文件说明
1. filtered_new_texts_1125.jsonl

通过过滤的文本，将用于训练或下游任务。

包括字段：

text

instruction

label

round_idx

rouge_score

timestamp

2. rejected_texts.jsonl

被过滤掉的文本，包括相似度等信息。
