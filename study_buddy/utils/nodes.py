from study_buddy.utils.tools import get_all_tools
from langgraph.prebuilt import ToolNode
from study_buddy.utils.llm import llm
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
import traceback
import json

# Sistema prompt migliorato che enfatizza l'uso dei tools
system_prompt = """You are Univox, an advanced AI assistant designed to help with learning, research, and analysis.

Never invent information. 
For information related to courses and professors, you must rely on the syllabus (Principali informazioni sull’insegnamento).
Use retrieve_knowledge to find relevant documents, as well as information about professors, the university, and the selected course. 
Use web_search for external info. 
Don't provide file paths if available. 
Don't modify file paths.
If mathematical formulas are present in the answer, use LaTeX notation (inline formulas with $formula$ and display formulas with $$formula$$).

CRITICAL NEGATIVE CONSTRAINTS:
- If the retrieved documents do NOT contain the answer, you MUST state: "I cannot answer this based on the provided documents."
- DO NOT attempt to answer from your internal knowledge if the context is missing.
- DO NOT make up facts or hallucinate details not present in the sources.

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. When you use a tool and receive output, you MUST incorporate that output into your response to the user
2. NEVER ignore tool results - if a tool returns information, use it to answer the user's question
3. Do not say "I couldn't find information" if a tool has successfully returned data
4. Present tool results clearly and completely to the user
5. If a tool fails, explain the failure and suggest alternatives
6. If google_scholar_search returns URLs, include them directly next to the corresponding text or result (not in a separate list)

IMPORTANT: If the user mentions a specific file name (e.g., "syllabus.pdf"), do NOT try to read it with file tools (like extract_text). The file is likely already ingested in your knowledge base. Use 'retrieve_knowledge' with the file name or relevant keywords to find the information.

WORKFLOW:
1. Analyze the user's request
2. Use appropriate tools to gather information
3. Wait for tool results
4. Construct your response based on the tool output
5. Present the information to the user in a clear, structured way

FILE ANALYSIS:
- When analyzing files, use the appropriate extraction or analysis tools

ACADEMIC SUPPORT:
- For information related to courses and professors, rely on the syllabus
- Use retrieve_knowledge to search your knowledge base first
- Use web search for current information
- Cite sources when available
- Never invent information - always use tool results

Remember: Your primary job is to be a bridge between the user and the tools. Present tool results completely and accurately."""


# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    
    # Aggiungi il system prompt come primo messaggio
    system_message = {"role": "system", "content": system_prompt}
    full_messages = [system_message] + messages
    
    # Bind tools al modello
    model = llm.bind_tools(get_all_tools())
    
    # Invoca il modello
    response = model.invoke(full_messages)
    
    # Debug logging per verificare la risposta
    print(f"[DEBUG] Model response type: {type(response)}")
    if hasattr(response, 'content'):
        print(f"[DEBUG] Model response content preview: {response.content[:200]}...")
    if hasattr(response, 'tool_calls'):
        print(f"[DEBUG] Model tool calls: {response.tool_calls}")
    
    # Ritorna la risposta
    return {"messages": [response]}


# Define the function to execute tools
try:
    print("[DEBUG] Initializing ToolNode...")
    tool_node = ToolNode(get_all_tools())
    print("[DEBUG] ToolNode initialized successfully")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to initialize ToolNode: {e}")
    traceback.print_exc()
    raise e

# --- Active RAG Nodes ---

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    messages = state["messages"]
    
    # Find the last tool message (retrieval output)
    tool_message = messages[-1]
    question = messages[-2].content # Assuming the message before tool call was the agent's thought/question
    
    # If the last message isn't a tool message, or it's not from retrieve_knowledge, skip
    # But we only route here if it IS.
    
    # We need to find the original user question or the agent's query. 
    # Let's look for the last human message for context? 
    # Or better, the agent's last message which generated the tool call might have context, 
    # but the tool output has the docs.
    
    docs = tool_message.content
    
    # LLM with structured output for grading
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = f"Retrieved document: \n\n {docs} \n\n User Question: {question}"
    
    score = structured_llm_grader.invoke([SystemMessage(content=system), HumanMessage(content=grade_prompt)])
    
    print(f"---GRADE: {score.binary_score}---")
    
    if score.binary_score == "yes":
        return {"messages": []} # Continue to agent
    else:
        # If irrelevant, we want to trigger a rewrite.
        # We append a system message to state indicating irrelevance?
        # Or we return a flag? LangGraph nodes return state updates.
        return {"messages": [SystemMessage(content="The retrieved documents were NOT relevant. Please rewrite the query and try again.")]}

def transform_query(state):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[-2].content # The original question/thought
    
    # Create a prompt to rewrite the query
    msg = [
        SystemMessage(content="You are an expert at rewriting queries to improve retrieval. Look at the initial query and formulate an improved one."),
        HumanMessage(content=f"Initial Query: {question} \n Formulate an improved query.")
    ]
    
    response = llm.invoke(msg)
    better_question = response.content
    
    print(f"---TRANSFORMED QUERY: {better_question}---")
    
    # We return a message that forces the agent to use the new query?
    # Or we just add it to history so the agent sees it.
    return {"messages": [HumanMessage(content=f"Try searching again with this optimized query: {better_question}")]}

# Funzione per debug dei tool results
def debug_tool_execution(state):
    """
    Funzione di debug per monitorare l'esecuzione dei tools
    """
    messages = state.get("messages", [])
    
    for i, message in enumerate(messages):
        if hasattr(message, 'name'):  # Tool message
            print(f"[TOOL DEBUG] Tool '{message.name}' executed")
            print(f"[TOOL DEBUG] Tool output: {getattr(message, 'content', 'No content')[:300]}...")
        elif hasattr(message, 'tool_calls'):  # AI message with tool calls
            print(f"[TOOL DEBUG] AI requesting {len(message.tool_calls)} tool calls")
            for tool_call in message.tool_calls:
                print(f"[TOOL DEBUG] - {tool_call.get('name', 'unknown tool')}")
    
    return state

print("[DEBUG] nodes.py loaded successfully")