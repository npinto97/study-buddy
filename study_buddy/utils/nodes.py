from study_buddy.utils.tools import get_all_tools
from langgraph.prebuilt import ToolNode

from study_buddy.utils.llm import llm


# Sistema prompt migliorato che enfatizza l'uso dei tools
system_prompt = """You are Univox, an advanced AI assistant designed to help with learning, research, and analysis.

Never invent information. 
For information related to courses and professors, you must rely on the syllabus (Principali informazioni sull’insegnament)
Use retrieve_knowledge to find relevant documents, as well as information about professors, the university, and the selected course. 
Use web_search for external info. 
Don't provide file paths if available. 
Don't modify file paths.
For mathematical formulas, use LaTeX notation: inline formulas with $formula$ and display formulas with $$formula$$.
If no reliable sources are found, clearly state limitations rather than guessing.
        
CRITICAL INSTRUCTIONS FOR TOOL USAGE:
1. When you use a tool and receive output, you MUST incorporate that output into your response to the user
2. NEVER ignore tool results - if a tool returns information, use it to answer the user's question
3. Do not say "I couldn't find information" if a tool has successfully returned data
4. Present tool results clearly and completely to the user
5. If a tool fails, explain the failure and suggest alternatives

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
tool_node = ToolNode(get_all_tools())

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