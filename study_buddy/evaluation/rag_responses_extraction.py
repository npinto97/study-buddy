import os
import json
from loguru import logger
from study_buddy.agent import build_compiled_graph
from study_buddy.config import EVAL_DATA_DIR

TEST_SET_PATH = EVAL_DATA_DIR / "test_set.json"
OUTPUT_PATH = EVAL_DATA_DIR / "rag_responses.json"


with open(TEST_SET_PATH, 'r', encoding="utf-8") as f:
    test_set = json.load(f)

rag_responses = []

for item in test_set:
    question = item["question"]
    logger.info(f"Processing quesiton: {question}")

    compiled_graph = build_compiled_graph()

    config = {"configurable": {"thread_id": "test_session"}}

    events = compiled_graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="values",
    )

    response_text = None
    retrieved_docs = []

    for event in events:
        messages = event.get("messages", [])
    
        for msg in reversed(messages):
            if type(msg).__name__ == "AIMessage":
                response_text = msg.content
                break

        # extracts file paths from event log
        for msg in messages:
            if type(msg).__name__ == "ToolMessage" and msg.name == "retrieve_tool":
                print(msg) # stampa di debug
                if hasattr(msg, "artifact") and isinstance(msg.artifact, list):
                    for doc in msg.artifact:
                        if isinstance(doc, dict) and "metadata" in doc:
                            retrieved_docs.append(doc["metadata"].get("file_path", "Unknown"))
                        elif hasattr(doc, "metadata"):
                            retrieved_docs.append(doc.metadata.get("file_path", "Unknown"))

    # clean retrieved_docs, extracting file_names and removing duplicates
    unique_files = []
    for path in retrieved_docs:
        file_name = os.path.basename(path)
        if file_name not in unique_files:
            unique_files.append(file_name)
    retrieved_docs = unique_files

    rag_responses.append({
        "question": question,
        "expected_answer": item["expected_answer"],
        "generated_answer": response_text,
        "target_doc": item["filename"],
        "retrieved_docs": retrieved_docs
    })

with open(OUTPUT_PATH, 'w', encoding="utf-8") as f:
    json.dump(rag_responses, f, indent=4, ensure_ascii=False)

logger.info(f"Response saved to {OUTPUT_PATH}")