import json
from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from datasets import Dataset
import pandas as pd
import numpy as np

from study_buddy.utils.llm import llm 

EVALUATION_RESULTS_PATH = Path("evaluation_results.json")
ANALYSIS_OUTPUT_PATH = Path("analysis_report.txt")

def calculate_retrieval_metrics(results, output_file):
    report_header = "\n--- Running Classic Retrieval Evaluation ---\n"
    print(report_header)
    output_file.write(report_header)

    rag_items = [item for item in results if item["type"] == "RAG"]
    k = 4
    precisions = []
    reciprocal_ranks = []

    for item in rag_items:
        gt_chunks = set(item["ground_truth"]["chunks"])
        predicted_chunks = item["predicted_output"].get("predicted_chunks", [])[:k]
        
        correct_predictions = [p for p in predicted_chunks if p in gt_chunks]
        precision_at_k = len(correct_predictions) / k if k > 0 else 0
        precisions.append(precision_at_k)

        rr = 0.0
        for i, p_chunk in enumerate(predicted_chunks):
            if p_chunk in gt_chunks:
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)

    mean_precision_at_k = np.mean(precisions) if precisions else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    results_str = (
        f"Mean Precision@{k}: {mean_precision_at_k:.4f}\n"
        f"Mean Reciprocal Rank (MRR): {mrr:.4f}\n"
    )
    print(results_str)
    output_file.write(results_str)


def analyze_rag_tasks(results, output_file):
    report_header = "\n--- Running RAGAS Semantic Evaluation ---\n"
    print(report_header)
    output_file.write(report_header)
    
    rag_items = [item for item in results if item["type"] == "RAG"]
    if not rag_items:
        no_tasks_msg = "No RAG tasks found to evaluate.\n"
        print(no_tasks_msg)
        output_file.write(no_tasks_msg)
        return

    rag_data = {
        "question": [item["task"] for item in rag_items],
        "answer": [item["predicted_output"].get("answer", "") for item in rag_items],
        "contexts": [item["predicted_output"].get("retrieved_contexts", []) for item in rag_items],
        "ground_truth": [item["ground_truth"]["reference_answer"] for item in rag_items]
    }
    rag_dataset = Dataset.from_dict(rag_data)

    sequential_run_config = RunConfig(max_workers=1)

    try:
        result = evaluate(
            rag_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            run_config=sequential_run_config
        )
        results_df = result.to_pandas()

        results_header = "\n--- RAGAS Semantic Evaluation Results ---\n"
        results_str = results_df[['question', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].to_string()
        print(results_header)
        print(results_str)
        output_file.write(results_header + results_str + "\n")

        avg_header = "\n--- Average RAGAS Scores ---\n"
        avg_str = str(results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())
        print(avg_header)
        print(avg_str)
        output_file.write(avg_header + avg_str + "\n")

    except Exception as e:
        error_msg = f"An error occurred during RAGAS evaluation: {e}\n"
        print(error_msg)
        output_file.write(error_msg)


def analyze_tool_tasks(results, output_file):
    report_header = "\n--- Running Tool Use Evaluation ---\n"
    print(report_header)
    output_file.write(report_header)
    
    tool_tasks = [item for item in results if item["type"] == "TOOL"]
    if not tool_tasks:
        no_tasks_msg = "No TOOL tasks found to evaluate.\n"
        print(no_tasks_msg)
        output_file.write(no_tasks_msg)
        return
        
    correct_calls = 0
    for item in tool_tasks:
        ground_truth = item["ground_truth"]
        predicted = item["predicted_output"].get("tool_call")
        
        args_match = False
        if predicted and predicted.get("arguments") is not None and ground_truth.get("arguments") is not None:
            predicted_args_str = json.dumps(predicted["arguments"], sort_keys=True)
            gt_args_str = json.dumps(ground_truth["arguments"], sort_keys=True)
            if predicted_args_str == gt_args_str:
                args_match = True

        if predicted and predicted.get("tool_name") == ground_truth.get("tool_name") and args_match:
            correct_calls += 1
            
    accuracy = (correct_calls / len(tool_tasks)) * 100 if tool_tasks else 0
    
    accuracy_str = f"Tool Use Accuracy: {accuracy:.2f}% ({correct_calls}/{len(tool_tasks)} correct calls)\n"
    print(accuracy_str)
    output_file.write(accuracy_str)

def main():
    with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    
    with open(ANALYSIS_OUTPUT_PATH, "w", encoding="utf-8") as report_file:
        calculate_retrieval_metrics(results_data, report_file)
        analyze_rag_tasks(results_data, report_file)
        analyze_tool_tasks(results_data, report_file)
        
    print(f"\nAnalysis complete. Full report saved to: {ANALYSIS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()