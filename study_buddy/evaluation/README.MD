# `evaluation/`

This directory contains the complete pipeline for the quantitative evaluation of the **Univox** system. The framework is designed to be a multi-faceted and rigorous assessment of the agent's core capabilities, including its Retrieval-Augmented Generation (RAG) pipeline and its proficiency in tool utilization.

The evaluation process is divided into a series of sequential scripts that automate testing and generate a comprehensive performance report.

## Directory Structure

```
evaluation/
│
├── chunked_documents/          # (Generated) Human-readable text chunks for annotation
│   ├── document1_chunk_1.txt
│   └── ...
│
├── README.md                   # This file
├── evaluation_dataset.json     # The ground truth dataset (manual annotation)
├── evaluation_results.json     # (Generated) The agent's predictions
├── analysis_report.txt         # (Generated) The final report with calculated metrics
│
├── generate_evaluation_chunks.py # Script to create identifiable chunks from source docs
├── run_evaluation.py             # Script to run the agent against the dataset
└── analyze_results.py            # Script to calculate metrics and generate the report
```

## The Evaluation Pipeline

The evaluation is performed in a four-step process, combining a one-time manual annotation with automated script execution.

### 1\. Generate Chunks for Annotation

The first step is to create a human-readable version of the document chunks that the RAG system uses. The `generate_evaluation_chunks.py` script processes all source materials, splits them according to the configured chunking strategy, and saves each chunk as a uniquely named `.txt` file.

**To run:**

```bash
python -m study_buddy.evaluation.generate_evaluation_chunks
```

This populates the `chunked_documents/` folder, enabling the manual creation of the ground truth dataset.

### 2\. Create the Ground Truth Dataset (Manual Step)

Using the output from the previous step, a "gold standard" dataset named `evaluation_dataset.json` is created manually. This file contains a list of tasks and the ideal outcomes:

  * **For `RAG` tasks:** It specifies the `task` (question), the exact `chunks` that contain the correct answer, and a `reference_answer` (the ideal, synthesized response).
  * **For `TOOL` tasks:** It specifies the `task` (instruction) and the `ground_truth` tool call, including the correct `tool_name` and `arguments`.

### 3\. Run the Agent Evaluation

The `run_evaluation.py` script automates the core testing process. It iterates through every task in `evaluation_dataset.json`, runs the Univox agent on each one, and logs the agent's actual output. The results, including the retrieved contexts, the generated answer, and the tool calls made by the agent, are saved to `evaluation_results.json`.

**To run:**

```bash
python -m study_buddy.evaluation.run_evaluation
```

### 4\. Analyze Results and Generate Report

The final step is to calculate the performance metrics. The `analyze_results.py` script compares the agent's predictions in `evaluation_results.json` with the ground truth from `evaluation_dataset.json`.

It computes a comprehensive set of metrics and saves them in a clean, readable format to `analysis_report.txt`.

**To run:**

```bash
python -m study_buddy.evaluation.analyze_results
```

## Metrics Calculated

The analysis script evaluates three key aspects of the system's performance:

  * **Classic Retrieval Metrics:**
      * `Mean Precision@K`: Measures the proportion of relevant chunks among the top K retrieved.
      * `Mean Reciprocal Rank (MRR)`: Measures how highly the first correct chunk is ranked.
  * **RAGAS Semantic Metrics:**
      * `faithfulness`: Measures how well the generated answer is grounded in the retrieved context, penalizing hallucinations.
      * `answer_relevancy`: Measures how relevant the generated answer is to the original question.
      * `context_precision` and `context_recall`: Semantic equivalents of the classic retrieval metrics.
  * **Agent Performance Metrics:**
      * `Tool Use Accuracy`: Measures the percentage of times the agent correctly selects and invokes the right tool with the correct arguments.