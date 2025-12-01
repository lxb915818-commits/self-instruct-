import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置使用的 GPU 设备
import json
import argparse
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any, Optional
import self_instruction as si
from self_instruction import State


# --------------------------
# 构建流程
# --------------------------
def build_workflow() -> StateGraph:
    workflow = StateGraph(State)
    

    workflow.add_node("load_original_data", si.load_original_data)
    workflow.add_node("select_sensitive_sentence", si.select_sensitive_sentence)
    workflow.add_node("filter_relevant_tasks", si.filter_relevant_tasks)
    workflow.add_node("generate_new_instructions", si.generate_new_instructions)
    workflow.add_node("generate_new_texts", si.generate_new_texts)
    workflow.add_node("filter_by_rouge", si.filter_by_rouge_node) 
    
    # 定义流程路径
    workflow.add_edge(START, "load_original_data")
    workflow.add_edge("load_original_data", "select_sensitive_sentence")
    workflow.add_edge("select_sensitive_sentence", "filter_relevant_tasks")
    workflow.add_edge("filter_relevant_tasks", "generate_new_instructions")
    workflow.add_edge("generate_new_instructions", "generate_new_texts")
    workflow.add_edge("generate_new_texts", "filter_by_rouge")  # 生成后进入过滤节点做后续处理
    
    # 条件分支
    workflow.add_conditional_edges(
        "filter_by_rouge",
        si.check_completion,
        {
            "continue": "select_sensitive_sentence",
            "end": END
        }
    )
    
    return workflow.compile()


model_path = "/work/jawy/models/Qwen2.5-7B-Instruct-Uncensored"  # 替换为你的本地模型路径
# --------------------------
# 主函数
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument("--sensitive_sentences_path", default="/work/jawy/11-06-self-instruct/data/input/sensitive_words.jsonl", type=str, help="原始敏感句文件（每行一条）")
    parser.add_argument("--tasks_path", type=str, default="/work/jawy/11-06-self-instruct/data/input/task_pool.jsonl", help="带标签的task文件（jsonl）")
    parser.add_argument("--output_dir", type=str, default="/work/jawy/11-06-self-instruct/data/output", help="输出目录")

    args = parser.parse_args()

    
    # 加载模型
    # print(f"加载模型: {args.model_path}")
    # model, tokenizer, device = load_local_model(args.model_path, args.device)
    # state.model = model
    # state.tokenizer = tokenizer
    # state.device = device
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行流程（目标10000条文本）
    print("启动流程，目标生成10000条过滤后的文本...")
    app = build_workflow()

    initial_state = State(
        original_sensitive_sentences = [],
        all_tasks = [],
        selected_sensitive_sentence = "",
        selected_sensitive_label = "",
        relevant_tasks = [],
        new_instructions = [],
        all_generated_instructions = [],
        generated_texts = [],
        filtered_texts = [],
        all_accepted_texts = [],
        total_texts_count = 0,
        config = vars(args),
        round_idx = 1497 ,
    )
        
    final_state = app.invoke(initial_state,
    config={"recursion_limit": 50000})
    
    # 输出最终结果
    print("\n流程结束：")
    print(f"总轮次：{final_state['round_idx']}") 
    print(f"最终累计文本数：{final_state['total_texts_count']}")
    print(f"生成新指令总数：{len(final_state['all_generated_instructions'])}")
    print(f"结果保存至：{final_state['config']['output_dir']}")  

if __name__ == "__main__":
    si.init_model(model_path)
    main()