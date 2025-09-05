from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import asyncio
import time
import os
from typing import Optional, List
import uvicorn
from contextlib import asynccontextmanager

from study_buddy.utils.tools import (
    VectorStoreRetriever,
    DocumentProcessor, 
    DocumentSummarizer,
    CodeInterpreter,
    DataVisualizer,
    TavilySearch
)

# Pydantic models for requests
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 4

class CodeRequest(BaseModel):
    code: str

class VisualizationRequest(BaseModel):
    csv_path: str
    query: str

class FilePathRequest(BaseModel):
    file_path: str

tools = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tools on startup, cleanup on shutdown"""
    global tools
    print("Initializing tools...")
    
    try:
        tools['retriever'] = VectorStoreRetriever()
        tools['web_search'] = TavilySearch(max_results=3)
        tools['doc_processor'] = DocumentProcessor()
        tools['summarizer'] = DocumentSummarizer()
        tools['code_interpreter'] = CodeInterpreter()
        tools['visualizer'] = DataVisualizer()
        print("All tools initialized")
    except Exception as e:
        print(f"Error initializing tools: {e}")
        # Continue anyway for partial testing
    
    yield
    
    if 'code_interpreter' in tools:
        tools['code_interpreter'].close()
    print("Cleanup completed")

app = FastAPI(
    title="StudyBuddy Load Test API",
    description="API endpoints for load testing critical StudyBuddy tools",
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
            "file_paths": file_paths[:3],  # Limit output for performance
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

@app.post("/test/extract-text")
async def test_extract_text(request: FilePathRequest):
    """Test document text extraction"""
    start_time = time.time()
    success = False
    
    try:
        if 'doc_processor' not in tools:
            raise HTTPException(status_code=503, detail="Document processor not available")
        
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        result = tools['doc_processor'].extract_text(request.file_path)
        success = not result.startswith("Error")
        
        if not success:
            raise HTTPException(status_code=500, detail=result)
        
        return {
            "file_path": request.file_path,
            "text_length": len(result),
            "execution_time": time.time() - start_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        track_request("extract_text", success, time.time() - start_time)

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

@app.post("/test/execute-code")
async def test_execute_code(request: CodeRequest):
    """Test code execution in sandbox"""
    start_time = time.time()
    success = False
    
    try:
        if 'code_interpreter' not in tools:
            raise HTTPException(status_code=503, detail="Code interpreter not available")
        
        result = tools['code_interpreter'].run_code(request.code)
        success = not result.get('error')
        
        return {
            "code_length": len(request.code),
            "has_results": len(result.get('results', [])) > 0,
            "has_stdout": bool(result.get('stdout')),
            "has_error": bool(result.get('error')),
            "execution_time": time.time() - start_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        track_request("execute_code", success, time.time() - start_time)

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

@app.post("/test/batch")
async def test_batch_operations(background_tasks: BackgroundTasks):
    """Execute multiple operations in parallel to test concurrent load"""
    start_time = time.time()
    
    async def run_concurrent_tests():
        # Create test tasks
        tasks = []
        
        # Vector search tasks
        for i in range(3):
            tasks.append(asyncio.create_task(
                test_vector_search(QueryRequest(query=f"machine learning test {i}"))
            ))
        
        # Web search tasks  
        for i in range(2):
            tasks.append(asyncio.create_task(
                test_web_search(QueryRequest(query=f"AI research {i}"))
            ))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        return {
            "total_operations": len(results),
            "successful": successful,
            "failed": failed,
            "execution_time": time.time() - start_time
        }
    
    # Run in background
    background_tasks.add_task(run_concurrent_tests)
    
    return {"message": "Batch test started", "timestamp": time.time()}

# Endpoint to upload test files
@app.post("/upload-test-file")
async def upload_test_file(file: UploadFile = File(...)):
    """Upload a file for testing document processing"""
    try:
        upload_dir = "./test_files"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "filename": file.filename,
            "file_path": file_path,
            "size": len(content)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable for load testing
        workers=1      # Single worker for consistent metrics
    )