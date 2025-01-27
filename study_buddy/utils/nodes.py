from study_buddy.utils.tools import tools
from langgraph.prebuilt import ToolNode

from study_buddy.utils.llm import llm


# SOSTITUIBILE CON TOOL_CONDITION
# # Define the function that determines whether to continue or not
# def should_continue(state):
#     messages = state["messages"]
#     last_message = messages[-1]
#     # If there are no tool calls, then we finish
#     if not last_message.tool_calls:
#         return "end"
#     # Otherwise if there is, we continue
#     else:
#         return "continue"


system_prompt = """Be a helpful assistant"""


# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode(tools)
