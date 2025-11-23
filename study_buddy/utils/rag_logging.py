"""
RAG-specific logging utilities to track retrieval and generation process.
"""
from typing import List, Dict, Any, Optional
from loguru import logger

class RAGLogger:
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.logger = logger.bind(context=f"rag_{context_id}")
        
    def log_query(self, query: str):
        """Log the initial query to the RAG system."""
        self.logger.info(f"""
🔍 Processing RAG Query:
   Query: {query}
""")
        
    def log_retrieval_results(self, results: List[Dict[str, Any]], scores: List[float]):
        """Log detailed retrieval results with similarity scores."""
        self.logger.info(f"""
📚 Retrieved {len(results)} documents:""")
        
        for i, (doc, score) in enumerate(zip(results, scores)):
            self.logger.info(f"""
   Document {i+1}:
   - Score: {score:.4f}
   - Source: {doc.get('metadata', {}).get('source', 'Unknown')}
   - Content Preview: {doc.get('content', '')[:200]}...
""")
            
    def log_tool_selection(self, selected_tool: str, reason: str):
        """Log which tool was selected and why."""
        self.logger.info(f"""
🔧 Tool Selection:
   Tool: {selected_tool}
   Reason: {reason}
""")
        
    def log_no_local_results(self, query: str):
        """Log when no results found in local knowledge base."""
        self.logger.warning(f"""
⚠️ No Results in Local Knowledge Base:
   Query: {query}
   Proceeding with external tools...
""")
        
    def log_generation_step(self, 
                          prompt_tokens: int,
                          completion_tokens: int,
                          sources_used: List[str]):
        """Log details about the generation step."""
        self.logger.info(f"""
🤖 Generation Step:
   Prompt Tokens: {prompt_tokens}
   Completion Tokens: {completion_tokens}
   Sources Used: {len(sources_used)}
""")
        for src in sources_used:
            self.logger.debug(f"   - {src}")
            
    def log_response_validation(self, 
                              has_citations: bool,
                              used_tools: List[str],
                              response_length: int):
        """Validate and log response quality metrics."""
        self.logger.info(f"""
✅ Response Validation:
   Citations Included: {has_citations}
   Tools Used: {', '.join(used_tools)}
   Response Length: {response_length} chars
""")
        
    def log_error(self, error: Exception, step: str):
        """Log errors in the RAG process."""
        self.logger.error(f"""
❌ Error in RAG Process:
   Step: {step}
   Error: {str(error)}
""")

# Create singleton instance for global access
rag_logger = RAGLogger("main")