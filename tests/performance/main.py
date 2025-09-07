from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import asyncio
import time
import os
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

from study_buddy.utils.tools import (
    VectorStoreRetriever,
    DocumentSummarizer,
    DataVisualizer,
    TavilySearch,
    AudioProcessor
)

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 4

class VisualizationRequest(BaseModel):
    csv_path: str
    query: str

class FilePathRequest(BaseModel):
    file_path: str

class TextToSpeechRequest(BaseModel):
    text: str

tools = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tools on startup, cleanup on shutdown"""
    global tools
    print("Initializing tools...")
    
    try:
        tools['retriever'] = VectorStoreRetriever()
        tools['web_search'] = TavilySearch(max_results=3)
        tools['summarizer'] = DocumentSummarizer()
        tools['text_to_speech'] = AudioProcessor()
        tools['visualizer'] = DataVisualizer()
        print("All tools initialized")
    except Exception as e:
        print(f"Error initializing tools: {e}")
    
    yield
    
    print("Cleanup completed")

app = FastAPI(
    title="StudyBuddy Load Test API",
    description="API endpoints for load testing StudyBuddy core tools",
    version="1.0.0",
    lifespan=lifespan
)

request_metrics = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'avg_response_time': 0,
    'tool_usage': {}
}

def track_request(tool_name: str, success: bool, duration: float):
    """Track request metrics"""
    request_metrics['total_requests'] += 1
    if success:
        request_metrics['successful_requests'] += 1
    else:
        request_metrics['failed_requests'] += 1
    
    # Update average response time
    current_avg = request_metrics['avg_response_time']
    total = request_metrics['total_requests']
    if total > 0:
        request_metrics['avg_response_time'] = ((current_avg * (total - 1)) + duration) / total
    
    # Track tool usage
    if tool_name not in request_metrics['tool_usage']:
        request_metrics['tool_usage'][tool_name] = {'calls': 0, 'avg_time': 0, 'errors': 0}
    
    tool_stats = request_metrics['tool_usage'][tool_name]
    tool_stats['calls'] += 1
    if not success:
        tool_stats['errors'] += 1
    
    # Update tool average time
    current_tool_avg = tool_stats['avg_time']
    tool_calls = tool_stats['calls']
    if tool_calls > 0:
        tool_stats['avg_time'] = ((current_tool_avg * (tool_calls - 1)) + duration) / tool_calls

@app.get("/")
async def root():
    return {"message": "StudyBuddy Load Test API is running", "metrics": request_metrics}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "tools_loaded": list(tools.keys()),
        "timestamp": time.time()
    }

@app.get("/metrics")
async def get_metrics():
    """Get current performance metrics"""
    return request_metrics

@app.post("/test/vector-search")
async def test_vector_search(request: QueryRequest):
    """Test vector store retrieval"""
    start_time = time.time()
    success = False
    
    try:
        if 'retriever' not in tools:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        result, docs, file_paths = tools['retriever'].retrieve(request.query, request.k)
        success = True
        
        return {
            "query": request.query,
            "result_length": len(result),
            "docs_found": len(docs),
            "file_paths": file_paths[:3],
            "execution_time": time.time() - start_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        track_request("vector_search", success, time.time() - start_time)

@app.post("/test/web-search")
async def test_web_search(request: QueryRequest):
    """Test web search functionality"""
    start_time = time.time()
    success = False
    
    try:
        if 'web_search' not in tools:
            raise HTTPException(status_code=503, detail="Web search not available")
        
        result = tools['web_search'].run(request.query)
        success = True
        
        return {
            "query": request.query,
            "result_length": len(result),
            "execution_time": time.time() - start_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        track_request("web_search", success, time.time() - start_time)

@app.post("/test/summarize")
async def test_summarize(request: FilePathRequest):
    """Test document summarization"""
    start_time = time.time()
    success = False
    
    try:
        if 'summarizer' not in tools:
            raise HTTPException(status_code=503, detail="Summarizer not available")
        
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        result = tools['summarizer'].summarize(request.file_path)
        success = not result.startswith("Error")
        
        if not success:
            raise HTTPException(status_code=500, detail=result)
        
        return {
            "file_path": request.file_path,
            "summary_length": len(result),
            "execution_time": time.time() - start_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        track_request("summarize", success, time.time() - start_time)

@app.post("/test/text-to-speech")
async def test_text_to_speech(request: TextToSpeechRequest):
    """Test text-to-speech conversion"""
    start_time = time.time()
    success = False
    
    try:
        if 'text_to_speech' not in tools:
            raise HTTPException(status_code=503, detail="Text-to-speech not available")
        
        # Validate text length (ElevenLabs has limits)
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = tools['text_to_speech'].text_to_speech(request.text)
        success = not result.startswith("Error")
        
        if not success:
            raise HTTPException(status_code=500, detail=result)
        
        # Check if file was created successfully
        if not os.path.exists(result):
            raise HTTPException(status_code=500, detail="Audio file was not created")
        
        file_size = os.path.getsize(result)
        
        return {
            "text_length": len(request.text),
            "audio_file_path": result,
            "audio_file_size": file_size,
            "execution_time": time.time() - start_time,
            "success": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        track_request("text_to_speech", success, time.time() - start_time)

@app.post("/test/visualization")
async def test_visualization(request: VisualizationRequest):
    """Test data visualization generation"""
    start_time = time.time()
    success = False
    
    try:
        if 'visualizer' not in tools:
            raise HTTPException(status_code=503, detail="Visualizer not available")
        
        if not os.path.exists(request.csv_path):
            raise HTTPException(status_code=404, detail="CSV file not found")
        
        result = tools['visualizer'].create_visualization(request.csv_path, request.query)
        success = result.get('success', False)
        
        if not success:
            raise HTTPException(status_code=500, detail=result.get('error', 'Visualization failed'))
        
        return {
            "csv_path": request.csv_path,
            "query": request.query,
            "images_generated": len(result.get('image_paths', [])),
            "execution_time": time.time() - start_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        track_request("visualization", success, time.time() - start_time)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )