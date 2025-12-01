# Self-Instruct Text Generation & Filtering Pipeline

本仓库实现了一个基于 **本地大模型（vLLM / Transformers）** 与 **LangGraph 工作流** 的自监督文本生成流水线，支持：

- 基于带标签的敏感句自动生成新指令（Self-Instruct）
- 根据生成的指令生成文本
- 使用 ROUGE-L 相似度进行去重过滤
- 持续迭代，直到达到设定文本数量（默认 10,000）

核心文件包括：

- `self_instruction.py`：主要逻辑（生成、过滤、状态管理）
- `main_workflow.py`：基于 LangGraph 的主流程控制
