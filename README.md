# Drug-RAG: Reducing Hallucination in Drug Question Answering using RAG

## Overview

Drug-RAG is a Retrieval-Augmented Generation (RAG) based question answering system designed to reduce hallucination in Large Language Models (LLMs) for drug-related queries.

The project compares:

* Baseline LLM (Llama 3.2)
* Llama 3.2 + Retrieval-Augmented Generation

The goal is to improve factual reliability by grounding AI responses with retrieved drug information.

---

## Problem

Large Language Models can generate incorrect but convincing answers (hallucinations), especially in sensitive domains like healthcare.

This project investigates whether Retrieval-Augmented Generation can reduce unsupported responses.

---

## Architecture

```
User Question
      |
      v
Sentence Transformer Embeddings
      |
      v
ChromaDB Vector Database
      |
      v
Relevant Drug Context Retrieval
      |
      v
Llama 3.2 Generation
      |
      v
Final Answer
```

---

## Tech Stack

* Python
* LangChain
* Ollama
* Llama 3.2
* Sentence Transformers
* ChromaDB
* Pandas
* NumPy

---

## Features

✅ Drug Question Answering
✅ Vector-based retrieval
✅ Local LLM inference
✅ Hallucination comparison
✅ Baseline vs RAG evaluation

---

## Project Results

The system evaluates:

* Answer correctness
* Retrieval grounding
* Hallucination reduction

Screenshots:

(Add your screenshots here)

---

## Installation

Clone repository:

```bash
git clone YOUR_REPO_LINK
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python experiment.py
```

---

## Research Paper

The complete research paper is available in:

```
paper/Drug_RAG_Research_Paper.pdf
```

---

## Future Improvements

* Larger medical datasets
* Expert evaluation
* Better medical LLMs
* Hybrid RAG + fine tuning approaches

---

## Author

Pratik Kumar
<img width="791" height="630" alt="Screenshot 2026-06-28 121728" src="https://github.com/user-attachments/assets/ee9a1305-721e-47ec-934f-22da261dd595" />
<img width="1383" height="927" alt="Screenshot 2026-06-28 120705" src="https://github.com/user-attachments/assets/6ad8579c-6b9b-4dcc-ac21-ef9e1620797e" />

