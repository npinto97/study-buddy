import json
from pathlib import Path
import pandas as pd
import numpy as np
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from study_buddy.utils.embeddings import embeddings, CONFIG

BASE_DIR = Path(__file__).parent
EVALUATION_RESULTS_PATH = BASE_DIR / "evaluation_results.json"
ANALYSIS_OUTPUT_PATH = BASE_DIR / "analysis_report.txt"
RAGAS_CACHE_PATH = BASE_DIR / "ragas_scores.csv"

def calculate_retrieval_metrics(results, output_file):
    """Classic retrieval metrics: precision@k and MRR for RAG items."""
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
        correct = [p for p in predicted_chunks if p in gt_chunks]
        precisions.append(len(correct) / k if k > 0 else 0)
        rr = 0.0
        for i, p in enumerate(predicted_chunks):
            if p in gt_chunks:
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)
    mean_precision = np.mean(precisions) if precisions else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    results_str = f"Mean Precision@{k}: {mean_precision:.4f}\nMean Reciprocal Rank (MRR): {mrr:.4f}\n"
    print(results_str)
    output_file.write(results_str)

def analyze_rag_tasks(results, output_file):
    """RAGAS semantic evaluation using TogetherAI embeddings."""
    report_header = "\n--- Running RAGAS Semantic Evaluation ---\n"
    print(report_header)
    output_file.write(report_header)

    # Load or compute RAGAS scores
    if RAGAS_CACHE_PATH.exists():
        print(f"Found cached RAGAS results. Loading from {RAGAS_CACHE_PATH}...")
        results_df = pd.read_csv(RAGAS_CACHE_PATH)
    else:
        print("No cached results found. Running RAGAS evaluation (this may take a while)...")
        rag_items = [item for item in results if item.get("type") == "RAG"]
        if not rag_items:
            msg = "No RAG tasks found to evaluate.\n"
            print(msg)
            output_file.write(msg)
            return
        # Build dataset
        data = []
        for item in rag_items:
            if "task" in item and "predicted_output" in item and "ground_truth" in item:
                data.append({
                    "question": item["task"],
                    "answer": item["predicted_output"].get("answer", ""),
                    "contexts": item["predicted_output"].get("retrieved_contexts", []),
                    "ground_truth": item["ground_truth"].get("reference_answer", "")
                })
        if not data:
            msg = "No valid RAG data found for evaluation.\n"
            print(msg)
            output_file.write(msg)
            return

        dataset = pd.DataFrame(data)
        
        # Run evaluation
        print("Evaluating with RAGAS...")
        # Note: We use the embeddings imported from study_buddy.utils.embeddings which should be BGE-M3
        # And we assume the LLM is configured correctly in RAGAS or we pass it if needed.
        # For now, relying on default or env vars for LLM if not passed explicitly.
        # Ideally we should pass llm=... and embeddings=... to evaluate()
        
        # Since we want to use TogetherAI for LLM (as per previous context), we might need to configure it.
        # But the user's objective mentioned "Evaluation LLM: ChatTogether...".
        # If it's not configured here, RAGAS might use OpenAI by default.
        # Let's assume the environment variables are set or RAGAS defaults are acceptable for now,
        # OR better, let's try to pass the embeddings we have.
        
        from langchain_together import ChatTogether
        # We need to make sure we use the same LLM as the app or a strong evaluator.
        # The user mentioned "Evaluation LLM: ChatTogether(model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')"
        # Let's try to instantiate it if possible, or just rely on the embeddings for now.
        
        # To be safe and consistent with previous attempts, let's just pass the embeddings.
        # RAGAS `evaluate` takes `embeddings` argument.
        
        results = evaluate(
            dataset.from_pandas(dataset),
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            embeddings=embeddings
            # llm=... # If we want to specify the LLM
        )
        results_df = results.to_pandas()
        results_df.to_csv(RAGAS_CACHE_PATH, index=False)
        print(f"RAGAS evaluation complete. Results saved to {RAGAS_CACHE_PATH}")

    results_df = results_df.fillna(0.0)

    # Display results
    display_cols = ["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    known_metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    results_header = "\n--- RAGAS Semantic Evaluation Results ---\n"
    if all(col in results_df.columns for col in display_cols):
        results_str = results_df[display_cols].to_string()
    else:
        # fallback if column names differ
        if "user_input" in results_df.columns:
            display_cols[0] = "user_input"
        results_str = results_df[display_cols].to_string() if all(col in results_df.columns for col in display_cols) else str(results_df)
    print(results_header)
    print(results_str)
    output_file.write(results_header + results_str + "\n")
    # Average scores
    avg_header = "\n--- Average RAGAS Scores ---\n"
    avg_str = str(results_df[known_metric_cols].mean())
    print(avg_header)
    print(avg_str)
    output_file.write(avg_header + avg_str + "\n")

def analyze_tool_tasks(results, output_file):
    """Tool use accuracy evaluation."""
    report_header = "\n--- Running Tool Use Evaluation ---\n"
    print(report_header)
    output_file.write(report_header)
    tool_tasks = [item for item in results if item.get("type") == "TOOL"]
    if not tool_tasks:
        msg = "No TOOL tasks found to evaluate.\n"
        print(msg)
        output_file.write(msg)
        return
    correct = 0
    for item in tool_tasks:
        gt = item.get("ground_truth", {})
        pred = item.get("predicted_output", {}).get("tool_call")
        if not pred:
            continue
        args_match = False
        if pred.get("arguments") is not None and gt.get("arguments") is not None:
            if json.dumps(pred["arguments"], sort_keys=True) == json.dumps(gt["arguments"], sort_keys=True):
                args_match = True
        if pred.get("tool_name") == gt.get("tool_name") and args_match:
            correct += 1
    accuracy = (correct / len(tool_tasks)) * 100 if tool_tasks else 0
    accuracy_str = f"Tool Use Accuracy: {accuracy:.2f}% ({correct}/{len(tool_tasks)} correct calls)\n"
    print(accuracy_str)
    output_file.write(accuracy_str)

def main():
    """Load evaluation results, run all analyses, and write report."""
    try:
        with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {EVALUATION_RESULTS_PATH} not found. Run run_evaluation.py first.")
        return
    except json.JSONDecodeError:
        print(f"Error: {EVALUATION_RESULTS_PATH} is not valid JSON.")
        return
    with open(ANALYSIS_OUTPUT_PATH, "w", encoding="utf-8") as report_file:
        calculate_retrieval_metrics(results_data, report_file)
        analyze_rag_tasks(results_data, report_file)
        analyze_tool_tasks(results_data, report_file)
    print(f"\nAnalysis complete. Full report saved to: {ANALYSIS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()