import json
import os
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from study_buddy.utils.retriever import HybridRetriever
from study_buddy.utils.llm import llm
from study_buddy.config import logger

def evaluate_ragas():
    print("Loading test set...")
    try:
        with open("test_set.json", "r", encoding="utf-8") as f:
            test_set = json.load(f)
    except FileNotFoundError:
        print("test_set.json not found!")
        return

    print("Initializing HybridRetriever...")
    retriever = HybridRetriever(k=5, fetch_k=20)

    # Define a simple RAG prompt for generation
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"Running evaluation on {len(test_set)} questions...")
    
    for item in test_set:
        question = item["question"]
        ground_truth = item["expected_answer"]
        
        print(f"Processing: {question[:50]}...")
        
        # 1. Retrieve
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        
        # 2. Generate
        context_text = "\n\n".join(contexts)
        answer = chain.invoke({"context": context_text, "question": question})
        
        # 3. Collect data
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(ground_truth)

    # Convert to Dataset
    dataset = Dataset.from_dict(data)

    print("Running RAGAS evaluation...")
    
    # Initialize embeddings
    from study_buddy.utils.embeddings import initialize_gpu_embeddings
    embeddings = initialize_gpu_embeddings()
    
    # Run evaluation with custom LLM and embeddings
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(results)
    
    # Try to print scores directly if possible
    try:
        print("Scores:", results)
        
        # Save detailed scores to CSV
        output_df = results.to_pandas()
        os.makedirs("study_buddy/evaluation", exist_ok=True)
        output_df.to_csv("study_buddy/evaluation/ragas_scores.csv", index=False)
        print("Saved detailed scores to study_buddy/evaluation/ragas_scores.csv")

        # Save summary to JSON
        # Calculate averages from the dataframe for the summary
        scores = output_df.mean(numeric_only=True).to_dict()
        with open("evaluation_results_final.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        print("Saved summary to evaluation_results_final.json")

        # Generate and save Markdown Report
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        report = f"""# RAGAS Evaluation Report (Final)

**Date:** {date_str}
**Status:** Success (Re-run)

## Metrics
| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Faithfulness** | **{scores.get('faithfulness', 0):.2f}** | Measures how well the generated answer is grounded in the retrieved context. |
| **Answer Relevancy** | **{scores.get('answer_relevancy', 0):.2f}** | Measures how relevant the answer is to the question. |
| **Context Recall** | **{scores.get('context_recall', 0):.2f}** | Measures if the retrieved context contains the answer. |
| **Context Precision** | **{scores.get('context_precision', 0):.2f}** | Measures if relevant documents are ranked highly. |

## Conclusion
Evaluation run completed successfully.
"""
        with open("ragas_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("Saved report to ragas_report.md")

    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    evaluate_ragas()
