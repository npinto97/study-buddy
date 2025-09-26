import json
from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from datasets import Dataset
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

EVALUATION_RESULTS_PATH = Path("evaluation_results.json")
ANALYSIS_OUTPUT_PATH = Path("analysis_report.txt")
RAGAS_CACHE_PATH = Path("ragas_scores.csv")


def calculate_retrieval_metrics(results, output_file):

    report_header = "\n--- Running Classic Retrieval Evaluation ---\n"
    print(report_header)
    output_file.write(report_header)

    rag_items = [item for item in results if item.get("type") == "RAG"]
    k = 3
    precisions = []
    reciprocal_ranks = []

    for item in rag_items:
        gt_chunks = set(item.get("ground_truth", {}).get("chunks", []))
        predicted_chunks = item.get("predicted_output", {}).get("predicted_chunks", [])[:k]
        
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

    if RAGAS_CACHE_PATH.exists():
        print(f"Found cached RAGAS results. Loading from {RAGAS_CACHE_PATH}...")
        results_df = pd.read_csv(RAGAS_CACHE_PATH)
    else:
        print("No cached results found. Running RAGAS evaluation (this will take a while)...")
        rag_items = [item for item in results if item.get("type") == "RAG"]
        if not rag_items:
            output_file.write("No RAG tasks found to evaluate.\n")
            print("No RAG tasks found to evaluate.")
            return

        rag_data_list = []
        for item in rag_items:
            if "task" in item and "predicted_output" in item and "ground_truth" in item:
                rag_data_list.append({
                    "question": item["task"],
                    "answer": item["predicted_output"].get("answer", ""),
                    "contexts": item["predicted_output"].get("retrieved_contexts", []),
                    "ground_truth": item["ground_truth"].get("reference_answer", "")
                })

        if not rag_data_list:
            print("No valid RAG data to process after filtering.")
            return

        rag_dataset = Dataset.from_list(rag_data_list)
        sequential_run_config = RunConfig(max_workers=1)

        try:
            result = evaluate(
                rag_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                run_config=sequential_run_config
            )
            results_df = result.to_pandas()
            results_df.to_csv(RAGAS_CACHE_PATH, index=False)
            print(f"RAGAS results saved to cache file: {RAGAS_CACHE_PATH}")
        except Exception as e:
            error_msg = f"An error occurred during RAGAS evaluation: {e}\n"
            print(error_msg)
            output_file.write(error_msg)
            return
    
    display_cols = ['user_input', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    known_metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    
    results_header = "\n--- RAGAS Semantic Evaluation Results ---\n"

    if all(col in results_df.columns for col in display_cols):
        results_str = results_df[display_cols].to_string()
    else:
        results_str = f"Error: One or more expected columns not in the dataframe. Available columns: {list(results_df.columns)}"

    print(results_header)
    print(results_str)
    output_file.write(results_header + results_str + "\n")

    avg_header = "\n--- Average RAGAS Scores ---\n"
    avg_str = str(results_df[known_metric_cols].mean())
    print(avg_header)
    print(avg_str)
    output_file.write(avg_header + avg_str + "\n")


def analyze_tool_tasks(results, output_file):
    """Valuta la Tool Use Accuracy."""
    report_header = "\n--- Running Tool Use Evaluation ---\n"
    print(report_header)
    output_file.write(report_header)
    
    tool_tasks = [item for item in results if item.get("type") == "TOOL"]
    if not tool_tasks:
        no_tasks_msg = "No TOOL tasks found to evaluate.\n"
        print(no_tasks_msg)
        output_file.write(no_tasks_msg)
        return
        
    correct_calls = 0
    for item in tool_tasks:
        ground_truth = item.get("ground_truth", {})
        predicted = item.get("predicted_output", {}).get("tool_call")
        
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
    """Carica i risultati, avvia le analisi e salva il report."""
    try:
        with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {EVALUATION_RESULTS_PATH} was not found. Please run 'run_evaluation.py' first.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {EVALUATION_RESULTS_PATH} is not a valid JSON.")
        return

    with open(ANALYSIS_OUTPUT_PATH, "w", encoding="utf-8") as report_file:
        calculate_retrieval_metrics(results_data, report_file)
        analyze_rag_tasks(results_data, report_file)
        analyze_tool_tasks(results_data, report_file)
        
    print(f"\nAnalysis complete. Full report saved to: {ANALYSIS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()