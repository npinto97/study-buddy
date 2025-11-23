"""
Enhanced logging configuration for the Univox system.
Provides detailed logging with timing, process info, and tool usage tracking.
"""

import sys
from typing import Optional
from pathlib import Path
from loguru import logger
import time

# Remove default logger
logger.remove()

# Rich console logging with emojis and colors
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {extra[context]} | <level>{message}</level>",
    filter=lambda record: record.levelno >= 20  # INFO and above
)

# Detailed file logging with process info and call stack
logs_dir = Path(__file__).parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

logger.add(
    str(logs_dir / "univox_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {process} | {thread} | {name}:{function}:{line} | {extra[context]}: {message}",
    rotation="00:00",  # New file at midnight
    retention="30 days",
    compression="zip",
    enqueue=True,  # Thread-safe logging
    backtrace=True,  # Include full stack trace
    diagnose=True    # Include variable values in tracebacks
)

class LogContext:
    """Context manager for tracking operation timing and logging."""
    def __init__(self, context_name: str, logger_instance: Optional[logger] = None):
        self.context_name = context_name
        self.logger = logger_instance or logger.bind(context=context_name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"🔄 Starting {self.context_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.success(f"✅ {self.context_name} completed in {duration:.2f}s")
        else:
            self.logger.error(f"❌ {self.context_name} failed after {duration:.2f}s")
            self.logger.error(f"Error: {exc_val}")
        return False  # Don't suppress exceptions

# Performance metrics tracking
class PerformanceMetrics:
    """Tracks performance metrics for the system."""
    def __init__(self):
        self.start_time = None
        self.metrics = {
            "tool_calls": 0,
            "api_calls": 0,
            "llm_tokens": 0,
            "reasoning_steps": 0
        }
    
    def start_tracking(self):
        """Start tracking a new interaction."""
        self.start_time = time.time()
        self.metrics = {k: 0 for k in self.metrics}
    
    def increment(self, metric: str, amount: int = 1):
        """Increment a metric counter."""
        if metric in self.metrics:
            self.metrics[metric] += amount
    
    def get_summary(self) -> str:
        """Get a formatted summary of metrics."""
        if not self.start_time:
            return "No tracking session active"
        
        duration = time.time() - self.start_time
        return f"""📊 Performance Summary:
Duration: {duration:.2f}s
Tool Calls: {self.metrics['tool_calls']}
API Calls: {self.metrics['api_calls']}
LLM Tokens: {self.metrics['llm_tokens']}
Reasoning Steps: {self.metrics['reasoning_steps']}"""

# Create global metrics tracker
metrics = PerformanceMetrics()

# Initialize default logger with main context
logger = logger.bind(context="main")

def get_logger(context: str = "main") -> logger:
    """Get a logger instance for a specific context."""
    return logger.bind(context=context)