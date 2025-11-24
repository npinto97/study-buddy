# **Univox**

**Univox** is an intelligent AI-powered study assistant.
It leverages **advanced natural language processing**, **computer vision**, and **graph-based reasoning** to help students learn more effectively through **document analysis**, **content extraction**, and **personalized support**.

The system is built on top of **LangGraph**, enabling a **modular, agent-driven architecture** where different AI components collaborate seamlessly.
With Univox, students can interact naturally — through **text** or **voice** — and access a wide range of academic resources in an intuitive way.

## Retrieval-Augmented Generation
A core feature of Univox is its **RAG pipeline**, which enhances responses with relevant documents retrieved from the student’s course materials:
- **Document retrieval**: Every query is matched against a FAISS-based vectorstore built from parsed syllabi, books, slides, notes, and multimedia.
- **Context enrichment**: Retrieved documents are injected into the LLM prompt, ensuring answers are grounded in the actual study material.
- **Downloadable resources**: In addition to enriched answers, students are given direct links to download useful files (e.g., lecture slides, exam papers, reference books).

This means Univox is not just a conversational agent — it acts as a personal knowledge navigator, combining semantic search with LLM reasoning to maximize learning.

## Project Structure
```bash
study-buddy/
│
├── data/                   # Raw (only locally), metadata, and processed datasets
│   └── README.md
│
├── faiss_index/            # Vectorstore index
│
├── images/
│
├── study_buddy/            # Core source code
│   └── README.md
│
├── tests/
│   ├── performance/
│   │    └── README.md
│   └── tool_tests/         # Unit and load tests
│
├── streamlit_frontend.py   # Streamlit web app entrypoint
├── config.yaml             # Configurable models, embeddings, vectorstore
├── langgraph.json
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt
├── setup.cfg
├── .gitignore
├── LICENSE
└── README.md
```

## **What You Can Do with Univox**

Univox acts as your **personal academic companion**, combining **course-specific knowledge** with **powerful AI tools** to assist you throughout your learning journey.

### **Course-Specific Assistance**

Need quick access to important information? Univox can:

* Provide **contact details** for professors and teaching assistants
* Show **exam dates, office hours**, and course schedules
* Extract information directly from syllabi, announcements, and handouts

For example:

> “When are the midterm exams for this course?”

> “How can I contact Professor Lops?”


### **Exam Preparation and Practice**

Univox helps you **study smarter**, not harder:

* Retrieve **past exam questions** and exercises from your uploaded materials
* Generate **custom practice problems** tailored to your curriculum
* Suggest relevant topics and resources for your upcoming tests

For example:

> “Give me practice problems for cosine similarity”
> “Show me past exams of MRI course”

### **Topic Clarification and Learning Support**

Stuck on a concept? Univox explains it clearly and links you back to the **exact source**:

* Summarizes **difficult topics** from lecture notes
* Highlights **key formulas and definitions**
* Points you to the relevant **pages and documents**

For example:

> “Explain mean reciprocal rank in simple terms”

> “What’s the difference between item based and user based recommender systems?”

### **Advanced Research and Analysis**

Univox isn’t just a Q\&A bot — it’s also a **research companion**:

**Document Processing**

* Extracts text from scanned PDFs and images using **Tesseract OCR**
* Summarizes long research papers and textbooks
* Handles multiple document formats effortlessly

**Data Analysis & Visualization**

* Analyzes datasets and generates insights
* Creates visualizations from CSV files using natural language
* Runs custom **Python code** for statistical tasks

**Research Enhancement**

* Searches academic sources like **ArXiv**, **Google Scholar**, and **Wikidata**
* Finds books, papers, and supplementary materials
* Accesses **up-to-date information** via web search

### **Voice Interaction and Accessibility**

Univox also offers **hands-free interaction**:

* Ask questions using **voice commands**
* Receive **audio-based responses**
* Transcribe recorded lectures into **searchable text** (supports Italian)
* Convert summaries and explanations into **natural-sounding audio**

This makes Univox highly useful for students with **visual impairments**, **reading difficulties**, or those who prefer **multimodal learning**.

### **Interactive Learning Experience**

* Ask natural language or voice questions about your uploaded materials
* Get **direct citations** with every answer
* Manage and browse your content via an **intuitive Streamlit-based web app**
* Integrate multimedia: process text, images, audio, and datasets seamlessly

### **Transparency and Source Attribution**

Univox is designed to be **trustworthy**:

* Every answer includes **clear references** to the source documents
* Displays **page numbers** and relevant sections where possible
* Ensures traceability so you can always **verify information**

---

## **Technologies Used**

* **Python 3.12**
* **LangGraph** for orchestrating multi-agent workflows
* **Tesseract OCR** for document text extraction
* **Configurable Embedding Models** — default: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
* **Together.ai** for LLM integration
* **FAISS** for fast vector indexing
* **Transformers** for model inference
* **TensorFlow / PyTorch** with CUDA acceleration
* **Streamlit** for the interactive web interface

---

## **Prerequisites**

* **Windows 11 64-bit** (tested environment)
* **Python 3.12**
* **NVIDIA GPU** with CUDA support (RTX 5080 recommended)
* **8GB+ RAM**
* **\~5GB disk space** for models and dependencies
* A **Together.ai API key**

---

## **Installation**

### Quick Start (Recommended)

For the easiest installation experience, use the automated setup script:

```powershell
.\setup.ps1
```

This script will:
- ✓ Detect your GPU and install the appropriate PyTorch version
- ✓ Create and configure the virtual environment
- ✓ Install all required dependencies automatically
- ✓ Verify the installation

### Detailed Installation Guide

For manual installation, troubleshooting, or more details, see **[INSTALLATION.md](INSTALLATION.md)**.

The guide covers:
- Prerequisites and system requirements
- Manual installation steps
- GPU configuration
- Common issues and solutions
- API key configuration

### Quick Manual Setup

If you prefer manual installation:

```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install PyTorch (GPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install additional packages
pip install ffmpeg-python youtube-transcript-api wikipedia google-search-results
```

---

### GPU Configuration (Optional but Recommended)

Check CUDA availability:

```bash
nvidia-smi
```

The setup script automatically detects your GPU and installs the correct PyTorch version.
For manual installation with GPU support, use CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

### 4. Initialize the System

1. Parse your course metadata:

   ```bash
   python parse_course_metadata.py
   ```
2. Build the FAISS index:

   ```bash
   python update_faiss_index.py
   ```

---

## **Configuration**

The system uses `config.yaml` to customize **LLM**, **embeddings**, and **vector stores**:

```yaml
llm:
  model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"

embeddings:
  model: "BAAI/bge-m3"

vector_store:
  type: "faiss"
```

You can replace the model names with any alternative available on [Together.ai](https://together.ai) or [Hugging Face](https://huggingface.co).

---

## **Usage**

Start the Streamlit app:

```bash
streamlit run streamlit_frontend.py
```

By default, it launches at:
[http://localhost:8501](http://localhost:8501)

**Workflow**:

1. Upload your study materials
2. Ask questions via **chat** or **voice**
3. View responses with **source citations**

---

## **Future Enhancements**

* More lightweight embedding models for CPU-only setups
* Enhanced semantic search and document summarization
* Richer multimodal learning experience
