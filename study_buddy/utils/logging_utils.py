from loguru import logger
import time
import inspect
from functools import wraps
from typing import Any, Callable

def log_tool_call(func: Callable) -> Callable:
    """
    Decorator to log tool execution with detailed information.
    Shows start, completion, arguments, timing and any API calls.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        # Get caller context
        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name if caller_frame else "unknown"
        
        # Log start of tool execution
        logger.info(f"🔧 Tool '{tool_name}' called from {caller_name}")
        logger.debug(f"  Arguments: args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful execution
            logger.success(f"✅ Tool '{tool_name}' completed in {execution_time:.2f}s")
            logger.debug(f"  Result type: {type(result)}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            # Log failure with detailed error
            logger.error(f"❌ Tool '{tool_name}' failed after {execution_time:.2f}s")
            logger.error(f"  Error: {str(e)}")
            raise

    return wrapper

def log_api_call(func: Callable) -> Callable:
    """
    Decorator to log external API calls.
    Shows request details, timing and response status.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        service_name = func.__name__
        
        # Log API request
        logger.info(f"🌐 Calling external API: {service_name}")
        logger.debug(f"  Request arguments: args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            response = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful API call
            logger.success(f"✅ API call to {service_name} succeeded in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            # Log API failure
            logger.error(f"❌ API call to {service_name} failed after {execution_time:.2f}s")
            logger.error(f"  Error: {str(e)}")
            raise
            
    return wrapper

def log_state_change(func: Callable) -> Callable:
    """
    Decorator to log internal state changes.
    Useful for tracking the reasoning process.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        operation = func.__name__
        
        logger.info(f"🔄 State change: {operation}")
        logger.debug(f"  Current arguments: args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            
            if isinstance(result, dict):
                logger.debug(f"  New state keys: {list(result.keys())}")
            else:
                logger.debug(f"  State change result type: {type(result)}")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ State change {operation} failed")
            logger.error(f"  Error: {str(e)}")
            raise
            
    return wrapper

def log_llm_interaction(func: Callable) -> Callable:
    """
    Decorator to log LLM interactions.
    Shows prompts, completions and tokens used.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        operation = func.__name__
        
        logger.info(f"🤖 LLM interaction: {operation}")
        
        # Log prompt if present in kwargs
        if 'prompt' in kwargs:
            logger.debug(f"  Prompt: {kwargs['prompt'][:200]}...")
        elif len(args) > 0 and isinstance(args[0], str):
            logger.debug(f"  Prompt: {args[0][:200]}...")
            
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log completion
            logger.success(f"✅ LLM {operation} completed in {execution_time:.2f}s")
            if isinstance(result, str):
                logger.debug(f"  Completion: {result[:200]}...")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ LLM {operation} failed after {execution_time:.2f}s")
            logger.error(f"  Error: {str(e)}")
            raise
            
    return wrapper