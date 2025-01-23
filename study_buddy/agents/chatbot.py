from study_buddy.modules.graph_builder import State
from study_buddy.modules.llm import llm
from study_buddy.modules.tools import retrieve_tool, web_search_tool

chatbot_tools = [retrieve_tool, web_search_tool]
llm_with_tools = llm.bind_tools(chatbot_tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
