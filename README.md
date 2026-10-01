# Historical News Question Answering

***Project Status:** Completed (academic project 2025)*

## Project Overview
The goal of this project is to develop a specialized search and Question Answering (QA) system capable of retrieving, ranking, and synthesizing information from historical newspaper passages in the *Chronicling America* digitized archive (1800–1920). The system answers factoid natural language questions while navigating the unique challenges of historical data, including archaic language, temporal ambiguities, and noisy text.

A primary focus was evaluating the **"OCR gap"** (the degradation caused by Optical Character Recognition errors) and implementing a multi-stage pipeline:
1. **Lexical Retrieval (Phase I):** BM25 vs. TF-IDF baselines on clean vs. raw OCR text.
2. **Advanced Re-ranking (Phase II):** Neural cross-encoders (semantic similarity) and custom domain heuristics (publication year proximity + Named Entity Recognition).
3. **Generative RAG & Sandbox (Phase III):** Grounded question answering using an instruction-tuned local Large Language Model (LLM) and an interactive in-notebook testing sandbox.

---

## Technologies and Tools
* **Environment:** Python (Google Colab / Jupyter)
* **Information Retrieval:** [PyTerrier](https://github.com/terrier-team/pyterrier), BM25, TF-IDF
* **NLP & Re-ranking:**
  * Neural: [sentence-transformers](https://www.sbert.net/) (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  * Heuristic: [spaCy](https://spacy.io/) (`en_core_web_sm`), Regular Expressions (temporal matching)
* **Generative RAG & Local Inference:**
  * [Hugging Face Transformers](https://huggingface.co/docs/transformers), [Accelerate](https://github.com/huggingface/accelerate), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (4-bit quantization)
  * Default Models: `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`
* **Interactive Sandbox:** `ipywidgets`

---

## Methodology

### 1. Indexing and Baseline (Phase I)
We built a custom index to ensure reproducibility and control, evaluating two primary lexical retrievers:
* **BM25 vs. TF-IDF:** Evaluated as strong baselines for initial candidate retrieval.
* **Data Quality Impact:** A comparative analysis was conducted between raw OCR text and corrected text to quantify the impact of transcription noise on retrieval effectiveness.

### 2. Advanced Re-ranking Systems (Phase II)
To move beyond simple keyword matching, we implemented two re-ranking strategies:
* **Neural Re-ranking (E1):** Utilizes a cross-encoder architecture (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score semantic similarity between the query and the top 10 documents retrieved by BM25.
* **Temporal and Entity-based Re-ranking (E2):** A custom heuristic function that boosts document scores based on the proximity of their `publication_date` to the query's temporal focus and verifies the presence of query Named Entities (using spaCy).

### 3. Generative RAG & Interactive Sandbox (Phase III)
In the final phase, the system extends retrieval into an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline:
* **Context Assembly:** The top re-ranked historical passages are combined with their publication date and document ID into an evidence-grounded prompt.
* **Factual Synthesis:** A causal instruction-tuned LLM reads the context and generates a concise, factual answer strictly constrained by the 19th-century source material.
* **Interactive Sandbox:** An embedded `ipywidgets` interface allows users to submit natural language historical questions and view both the synthesized answer and the underlying source passages directly in the notebook.

### 4. Evaluation Setup
The retrieval and re-ranking systems were evaluated on the test collection across 8 metrics, including **P@1, P@5, P@10, R@5, R@10, nDCG@5, nDCG@10, and MAP**.

---

## Results
The evaluation demonstrated that while lexical models provide a robust baseline, neural re-ranking offers a substantial edge in semantic understanding.

*Results E1 (Neural Cross-Encoder):*
<img width="956" height="210" alt="image" src="https://github.com/user-attachments/assets/2db08c18-d021-4ec8-92c7-6b4cadc0026f" />

*Results E2 (Custom Heuristic Re-ranker):*
<img width="967" height="217" alt="image" src="https://github.com/user-attachments/assets/771e84c0-ed9a-4e54-b8af-f34bd838beeb" />

**Key Insights:**
* Transitioning from raw OCR to clean text improved retrieval performance by ~32%.
* The neural cross-encoder achieved the highest precision (**P@1: 0.678**, **MAP: 0.728**), successfully identifying the relevant passage even with minimal keyword overlap.
* Temporal and entity heuristics provided significant disambiguation improvements over baseline BM25.

---

## Quickstart Instructions

The complete pipeline is implemented in [`Historical_News_Question_Answering.ipynb`](Historical_News_Question_Answering.ipynb).

### Running in Google Colab (Recommended)
1. **Open the Notebook:** Upload or open [`Historical_News_Question_Answering.ipynb`](Historical_News_Question_Answering.ipynb) in [Google Colab](https://colab.research.google.com/).
2. **Enable GPU Acceleration:**
   * Go to `Runtime` > `Change runtime type`.
   * Select **T4 GPU** (required for smooth neural cross-encoder and LLM inference).
3. **Execute the Cells:**
   * Run cells sequentially from the beginning (dataset download, indexing, and baseline execution).
   * In **Phase III**, the notebook loads the generative model and launches the interactive QA sandbox.
4. **Interact with the Sandbox:**
   * Call `ask_historical_question("Your historical question here?")` in a code cell, or use the interactive input box and click **Ask Historical QA** to see the generated answer and retrieved source passages.

### Running Locally
1. **Prerequisites:**
   * Python 3.10 – 3.12
   * Java 11+ (required by PyTerrier / Terrier backend)
2. **Launch Jupyter:**
   ```bash
   jupyter notebook Historical_News_Question_Answering.ipynb
   ```
3. **Run All Cells:** Ensure CUDA is available if running the LLM with 4-bit quantization, or allow the automatic CPU fallback.

---

## Issues Encountered & Solutions
* **Hardware & Memory Constraints:** Standard Google Colab environments have strict RAM limits. To avoid out-of-memory errors during cross-encoder inference, neural re-ranking was restricted to a shallow pool (top 10 BM25 documents).
* **Library Integration:** Interfacing PyTerrier with Hugging Face transformers required custom transformation wrappers to manage memory usage cleanly.
* **LLM Quantization:** For Phase III, `bitsandbytes` 4-bit quantization (`BitsAndBytesConfig`) was integrated to enable efficient local LLM inference without requiring high-end GPU hardware.

---

## Project Recap & Conclusion
In summary, this project indexed over 130,000 historical records, evaluated the impact of OCR degradation, and built a full multi-stage pipeline uniting classical information retrieval (BM25) with neural semantic re-ranking and generative RAG. The system successfully demonstrates how modern NLP techniques can bridge the vocabulary gap and temporal complexities of 19th-century English.

---

## Credits
* Developed by **Ecaterina Tsuhuy**, **Giacomo Colombo**, and **Lorenzo Goatelli** for the *"Information Retrieval and Recommender Systems"* course.
* Dataset provided by the **Library of Congress** (*Chronicling America* collection).
