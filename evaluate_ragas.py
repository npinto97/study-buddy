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
    except:
        pass

if __name__ == "__main__":
    evaluate_ragas()
