from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

try:
    print("Testing gemini-flash-latest...")
    response = llm.invoke("Hello, how are you?")
    print("Response:", response.content)
    
    print("Testing embeddings...")
    emb = embeddings.embed_query("Hello world")
    print("Embedding length:", len(emb))
    print("Embeddings working!")
except Exception as e:
    print("Error:", e)
