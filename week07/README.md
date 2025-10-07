# Week 7: Retrieval-Augmented Generation (RAG)

## Overview
- **Topic:** Retrieval-Augmented Generation (RAG) and document-based QA with reranking and PDF ingestion
- **Purpose:** Demonstrate how to combine retrieval (dense or sparse) with generative models to answer questions grounded in external documents. Show practical pipelines including simple RAG and a more advanced RAG with reranking and PDF support.

## RAG - Retrieval Augmented Generation 
- Help mitigate but not eliminate hallucinations and bias
- Allows LLM to make use of user designated external data sources as context to respond to user queries 
- Combines "retriever" of info with "generator" of output
- But could fail due to: 
  - Ineffective context chunking 
    - Context not in the chunk 
  - Noisy retrieval 
    - Extra chunks are irrelevant and add noise
  - Missing retrieval
    - Target chunk is outside of the top N chunks retrieved 

## Notebooks in this folder
- `simple_rag.ipynb` - A minimal RAG pipeline that shows:
  - Creating embeddings for a small document collection
  - Indexing embeddings in a vector store (FAISS or in-memory)
  - Retrieving top-k documents for a query
  - Passing retrieved contexts to a generative model to produce grounded answers
  - Cross encoder reranker
  - Cohere

- `Topic7_RAG_with_Reranking_and_PDFs.ipynb` - An advanced RAG pipeline that includes:
  - PDF ingestion and text extraction
  - Document chunking, cleaning and metadata handling
  - Creating dense embeddings for chunks
  - Hybrid retrieval + reranking (BM25/TF-IDF + dense reranker or cross-encoder)
  - Passage-level reranking to improve precision of retrieved evidence
  - Final answer generation using retrieved & re-ranked passages

## Key Steps (typical flow)
1. Ingest documents (plain text, PDFs) and extract raw text
2. Chunk documents into passages with overlap (chunking is needed for PDF)
3. Clean and add metadata (source, page numbers)
4. Create embeddings for passages (e.g., OpenAI, SentenceTransformers)
5. Build a vector index (FAISS, Milvus, or in-memory) and store metadata
6. At query time: retrieve top-N candidates using vector similarity
7. Optionally rerank retrieved passages with a cross-encoder or BM25
8. Concatenate top passages into a context prompt and generate an answer with an LLM
9. Return answer with citations/links to source passages

## Practical Tips
- Use overlap when chunking (e.g., 100–200 tokens) so that passages preserve context
- Store useful metadata (source filename, page number, chunk index) to enable citations
- Rerank with a cross-encoder when you need high precision (but it's more expensive)
- Use FAISS for small/medium datasets; switch to Milvus or an external vector DB for larger corpora
- Cache embeddings to avoid repeated computation during development

## Example Code Snippets
- Chunking (pseudo-python):

```python
# ...existing code...
chunks = []
for doc in docs:
    tokens = tokenizer(doc['text'])
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+chunk_size]
        chunks.append({'text': chunk_text, 'meta': doc_meta})
```

- Retrieval + generation (pseudo-python):

```python
# ...existing code...
query_emb = embedder.encode(query)
ids, scores = index.search(query_emb, top_k)
passages = [index.get_metadata(i) for i in ids]
context = "\n\n".join([p['text'] for p in passages])
answer = llm.generate(f"Context: {context}\n\nQuestion: {query}")
```