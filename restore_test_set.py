import json
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "study_buddy", "evaluation", "evaluation_dataset.json")
TARGET_FILE = os.path.join(BASE_DIR, "test_set.json")

def restore_test_set():
    print(f"Reading from: {SOURCE_FILE}")
    
    try:
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at {SOURCE_FILE}")
        return

    # Filter for RAG tasks
    rag_questions = []
    for item in data:
        if item.get("type") == "RAG":
            rag_questions.append({
                "question": item["task"],
                "expected_answer": item["ground_truth"]["reference_answer"]
            })

    print(f"Found {len(rag_questions)} RAG questions.")

    # Write to target file
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        json.dump(rag_questions, f, indent=4)
    
    print(f"Successfully wrote {len(rag_questions)} questions to {TARGET_FILE}")

if __name__ == "__main__":
    restore_test_set()
