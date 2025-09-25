# `study_buddy/`

The `study_buddy/` package contains the **core logic** of Univox.
It defines the **multi-agent system**, the **configuration layer**, and the **pipelines** for managing documents, building vectorstores, and enabling retrieval-augmented generation (RAG).

---

## 📂 Structure

```
study_buddy/
│
├── utils/                  # Utility functions (with its own README)
├── vectorstore_pipeline/   # RAG pipeline: parsing, indexing, audio handling (with its own README)
│
├── agent.py                # LangGraph-based multi-agent orchestration
├── config.py               # Global configuration, file loaders, logging, environment
└── README.md               # This file
```

---

## 🧩 Components

### 1. `agent.py`

Defines the **LangGraph agent workflow**:

* Builds a **state graph** (`StateGraph(AgentState)`), representing the agent’s reasoning steps.
* Defines two main nodes:

  * **`agent`** → Handles LLM reasoning.
  * **`tools`** → Executes external tools (e.g., document retrieval, search, RAG).
* Configures **conditional edges** to decide when to call tools based on the query.
* Uses a **memory saver checkpoint** to maintain conversational state.

This enables a **loop between reasoning and tool usage**, powering Univox’s multi-agent architecture.

---

### 2. `config.py`

Central configuration file, handling:

* **File loaders** for multiple formats:
  `.pdf`, `.docx`, `.pptx`, `.csv`, `.epub`, `.html`, `.xlsx`, `.txt`, `.md`, plus audio/video formats.
* **Image OCR** (`pytesseract`) for extracting text from images.
* **Environment setup**: loads `.env`, configures **logging** with Loguru, and manages **API keys**.
* **Paths**: defines all main directories (raw data, metadata, processed data, FAISS index, etc.).
* **AppConfig models**: strongly typed configs (`LLMConfig`, `EmbeddingsConfig`, `VectorStoreConfig`).
* **YAML config loader**: loads `config.yaml` into a unified `CONFIG` object.
* **LangSmith tracing** support (optional).

In short: `config.py` is the **control center** for data paths, file processing, and model configuration.

---

## 🔗 Related Modules

* [`utils/`](../utils/README.md): Shared helper functions.
* [`vectorstore_pipeline/`](../vectorstore_pipeline/README.md): The ingestion and FAISS indexing pipeline for the RAG system.

---

## 🚀 Role in Univox

Together, `agent.py` and `config.py` enable:

1. **Dynamic reasoning** via LangGraph.
2. **Configurable document ingestion and retrieval**.
3. **RAG-powered answers** with references and downloadable study material.

---

Vuoi che lo arricchisca anche con un **diagrammino ASCII** del flusso agent → tools → FAISS retrieval → enriched answer, per rendere più visivo come funziona la parte `agent.py`?
