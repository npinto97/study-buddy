# **Univox**

**Univox** is an intelligent AI-powered study assistant developed as part of the **SIIA-RS** course project.
It leverages **advanced natural language processing**, **computer vision**, and **graph-based reasoning** to help students learn more effectively through **document analysis**, **content extraction**, and **personalized support**.

The system is built on top of **LangGraph**, enabling a **modular, agent-driven architecture** where different AI components collaborate seamlessly.
With Univox, students can interact naturally — through **text** or **voice** — and access a wide range of academic resources in an intuitive way.


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

### 1. Install Tesseract OCR

1. Download the installer:
   [Tesseract OCR – UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install it in the default directory.
3. Add these paths to **System Environment Variables**:

   ```
   C:\Program Files\Tesseract-OCR
   C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
4. Verify installation:

   ```bash
   tesseract --version
   ```

---

### 2. Set Up the Python Environment

```bash
git clone https://github.com/npinto97/univox.git
cd univox

python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac

pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. GPU Configuration (Optional but Recommended)

Check CUDA availability:

```bash
nvidia-smi
```

Then install the correct **CUDA** and **cuDNN** versions for your GPU.
For RTX 5080, **CUDA 12.x** is recommended.
See [NVIDIA CUDA downloads](https://developer.nvidia.com/cuda-downloads) for more details.

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

* Deeper integration with **LangGraph** for multi-agent workflows
* More lightweight embedding models for CPU-only setups
* Enhanced semantic search and document summarization
* Richer multimedia learning experience
