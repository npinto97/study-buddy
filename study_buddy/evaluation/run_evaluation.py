import json
import asyncio
from pathlib import Path
from tqdm import tqdm

from study_buddy.agent import compiled_graph
from study_buddy.vectorstore_pipeline.vector_store_builder import get_vector_store
from study_buddy.config import FAISS_INDEX_DIR

EVALUATION_DATASET_PATH = Path("evaluation_dataset.json")
EVALUATION_RESULTS_PATH = Path("evaluation_results.json")

async def run_agent(task_input, config):
    # config = {"configurable": {"thread_id": "evaluation-thread"}}
    final_answer = ""
    tool_calls = []

    async for event in compiled_graph.astream({"messages": [{"role": "user", "content": task_input}]}, config=config):
        for key, value in event.items():
            if key == "agent" and value.get("messages"):
                message = value["messages"][-1]
                if message.content:
                    final_answer += message.content
                if message.tool_calls:
                    tool_calls.extend(message.tool_calls)
    
    return final_answer, tool_calls

def get_retrieved_chunks(query: str, k: int = 4):
    
    vector_store = get_vector_store(FAISS_INDEX_DIR)
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    # Estraiamo gli ID dei chunk dai metadati o dai nomi dei file
    retrieved_ids = []
    for doc in retrieved_docs:
        source_file = Path(doc.metadata.get("file_path", "")).stem
        # Troviamo il numero del chunk dal contenuto, se il nostro script lo ha salvato
        # In alternativa, usiamo una logica per derivarlo se necessario
        # Per ora, usiamo il nome del file come proxy
        retrieved_ids.append(source_file) # Semplificazione, da affinare se serve più precisione
        
    return retrieved_docs

async def main():

    with open(EVALUATION_DATASET_PATH, "r", encoding="utf-8") as f:
        evaluation_dataset = json.load(f)

    evaluation_results = []

    for task_item in tqdm(evaluation_dataset, desc="Running Evaluation"):
        task_id = task_item["id"]
        task_type = task_item["type"]
        task_prompt = task_item["task"]
        
        config = {"configurable": {"thread_id": task_id}}

        final_answer, tool_calls = await run_agent(task_prompt, config)

        result_entry = {
            "id": task_id,
            "type": task_type,
            "task": task_prompt,
            "ground_truth": task_item["ground_truth"],
            "predicted_output": {}
        }

        if task_type == "RAG":
            retrieved_docs = get_retrieved_chunks(task_prompt)
            retrieved_contexts = [doc.page_content for doc in retrieved_docs]
            
            # Per il confronto, usiamo i nomi dei file sorgente come ID
            base_filename = Path(retrieved_docs[0].metadata.get("file_path")).stem if retrieved_docs else ""
            predicted_chunks_ids = [f"{base_filename}_chunk_{i+1}" for i in range(len(retrieved_docs))]


            result_entry["predicted_output"] = {
                "answer": final_answer,
                "retrieved_contexts": retrieved_contexts,
                "predicted_chunks": predicted_chunks_ids
            }
        
        elif task_type == "TOOL":
            if tool_calls:
                # Salviamo solo la prima chiamata al tool per semplicità
                predicted_tool_call = {
                    "tool_name": tool_calls[0].get("name"),
                    "arguments": tool_calls[0].get("args")
                }
                result_entry["predicted_output"] = {"tool_call": predicted_tool_call}
            else:
                result_entry["predicted_output"] = {"tool_call": None}

        evaluation_results.append(result_entry)

    with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation run complete. Results saved to {EVALUATION_RESULTS_PATH}")

if __name__ == "__main__":
    asyncio.run(main())