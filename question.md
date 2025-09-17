Q: 1
LoRA/QLoRA Fine-tuning Assignment: RAG vs Fine-tuning
Overview
Two-day assignment to implement LoRA/QLoRA fine-tuning and compare it with RAG (Retrieval-Augmented Generation) using small models and datasets.

Day 1: RAG + Data Prep
Part 1: RAG Setup
Dataset: Choose 2–3 documents (research papers, Hugging Face datasets, or custom docs).
Deliverables: Standardized documents (1000–3000 words each).
Pipeline: Chunking → Embeddings → Retrieval → Generation.
Output: Working RAG system with vector DB + query interface.
Part 2: Data Preprocessing
Tasks: Clean/normalize text, remove noise, standardize format, fix encoding.
Metrics: Document length distribution, vocab coverage, topic diversity, quality issues.
Part 3: QA Dataset Creation
QA Generation: Use LLM to produce 50–100 QAs per doc (factual, inferential, analytical).
Requirements: Extractive + abstractive pairs, validate quality.
Curation: Manual review, filter low-quality, balance types, split (70/15/15).
Output: 150–300 curated QAs in JSON/CSV + quality report.
Submission
Functional RAG pipeline
Cleaned document collection
QA dataset (150–300 pairs)
Data quality assessment report