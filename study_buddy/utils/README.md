# `utils/`

This directory contains the core utility scripts that power the agent. These components provide essential services such as managing the LLM, handling embeddings, defining the agent's state, and integrating external tools.

## Components

* **`llm.py`**
    This script is responsible for initializing and configuring the **Large Language Model (LLM)** used by the agent. It sets up the model, including parameters like temperature and streaming, ensuring the agent has a properly configured language model to generate responses.

* **`embeddings.py`**
    This component handles the setup of the **embedding model**. It is crucial for the RAG system, as it's used to convert text into numerical vectors. The script is optimized to leverage GPU acceleration for faster and more efficient embedding generation.

* **`state.py`**
    Defines the **`AgentState`**, a `TypedDict` that represents the state of the LangGraph agent at any given time. This state typically includes the history of messages and other relevant information needed to maintain context throughout the conversation.

* **`tools.py`**
    This file defines all the **tools** the agent can use to perform specific actions, such as searching the web, retrieving information, or interpreting code. It serves as a central registry for all available functionalities, allowing the agent to dynamically decide which tool to call based on the user's query.

* **`nodes.py`**
    This script defines the **nodes** of the LangGraph graph. It contains the logic for the agent's main functions, such as `call_model` to invoke the LLM and the `tool_node` which executes the tools defined in `tools.py`. This is where the core logic of the agent's workflow resides.

* **`memory.py`**
    Manages the **memory and checkpointing** for the LangGraph agent. It ensures that the state of the conversation is saved and can be retrieved, allowing for continuity across interactions and turns.