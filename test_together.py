import os
from dotenv import load_dotenv
from langchain_together import TogetherEmbeddings

load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    print("Error: TOGETHER_API_KEY not found in environment.")
else:
    print(f"TOGETHER_API_KEY found: {api_key[:4]}...")

try:
    embeddings = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5",
        api_key=api_key
    )
    vector = embeddings.embed_query("Hello world")
    print(f"Embedding successful. Vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
except Exception as e:
    print(f"Error generating embedding: {e}")
