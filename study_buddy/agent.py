from loguru import logger
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from study_buddy.utils.memory import MemorySaver
from study_buddy.config import IMAGES_DIR
from study_buddy.utils.state import AgentState
from study_buddy.utils.nodes import call_model, tool_node, grade_documents, transform_query
from langchain_core.messages import ToolMessage
import os

print("[DEBUG] agent.py imports finished")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the graph
print("Defining workflow graph...")
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition
)

def should_grade_or_continue(state):
    """
    Check if we should grade documents or continue to agent.
    Only grade if the last tool call was 'retrieve_knowledge'.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, ToolMessage):
        # We need to check which tool generated this message.
        # ToolMessage has 'name' attribute which corresponds to tool name.
        if last_message.name == "retrieve_knowledge":
            return "grade_documents"
            
    return "agent"

def check_relevance(state):
    """
    Check the outcome of grading.
    If relevant -> agent
    If irrelevant -> transform_query
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If grade_documents returned a SystemMessage saying "NOT relevant", we go to transform
    # But grade_documents returns a dict update.
    # We can check the content of the last message added.
    
    if isinstance(last_message, ToolMessage):
        # This means grade_documents returned empty list (relevant)
        return "agent"
        
    if "NOT relevant" in last_message.content:
        return "transform_query"
        
    return "agent"

workflow.add_conditional_edges(
    "tools",
    should_grade_or_continue,
    {
        "grade_documents": "grade_documents",
        "agent": "agent"
    }
)

workflow.add_conditional_edges(
    "grade_documents",
    check_relevance,
    {
        "transform_query": "transform_query",
        "agent": "agent"
    }
)

workflow.add_edge("transform_query", "agent")

# Initialize memory
print("Initializing memory...")
memory = MemorySaver()

# Compile the graph
print("Compiling graph...")
compiled_graph = workflow.compile(checkpointer=memory)
print("Graph compiled successfully.")
