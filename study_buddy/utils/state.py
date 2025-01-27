from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage

from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
