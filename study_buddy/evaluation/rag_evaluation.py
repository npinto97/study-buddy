import os
import json
from sentence_transformers import SentenceTransformer, util

INPUT_JSON = "rag_responses.json"
OUTPUT_EVAL = "evaluation_results.json"

with open(INPUT_JSON, 'r', encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return cosine_sim.item()

def evaluate_recommendation(target_doc, retrieved_docs):

    target_file = os.path.splitext(os.path.basename(target_doc))[0].lower()
    retrieved_files = [os.path.splitext(os.path.basename(doc))[0].lower() for doc in retrieved_docs]

    if target_file in retrieved_files:
        rank = retrieved_files.index(target_file) + 1  
        hit = 1
        mrr = 1.0 / rank
    else:
        rank = None
        hit = 0
        mrr = 0.0

    return hit, rank, mrr

results = []
similarity_scores = []
hits = []
mrrs = []

for item in data:
    sim_score = semantic_similarity(item['expected_answer'], item['generated_answer'])
    hit, rank, mrr = evaluate_recommendation(item['target_doc'], item['retrieved_docs'])

    similarity_scores.append(sim_score)
    hits.append(hit)
    mrrs.append(mrr)

    results.append({
        "question": item["question"],
        "semantic_similarity": sim_score,
        "target_doc": item["target_doc"],
        "retrieved_docs": item["retrieved_docs"],
        "recommendation_hit": hit,
        "recommendation_rank": rank,
        "recommendation_mrr": mrr
    })

avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
hit_rate = sum(hits) / len(hits) if hits else 0.0
avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

print(f"Average semantic similarity: {avg_similarity}")
print(f"Hit rate: {hit_rate}")
print(f"Average MRR: {avg_mrr}")

with open(OUTPUT_EVAL, 'w', encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
