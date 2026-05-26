# RAG on DuckDB

A **Retrieval-Augmented Generation (RAG)** pipeline built with [DuckDB](https://duckdb.org/) as the vector store and [OpenAI](https://platform.openai.com/) for embeddings and answer generation.

The pipeline ingests raw text (Helen Keller's autobiography), chunks it, generates embeddings, and answers natural-language questions grounded in the source material.

---

## Architecture

```
Raw Text (.txt)
    │
    ▼
┌─────────────────────────┐
│  Step 1: Setup & Parse  │  ──►  data/sample_data.parquet
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Step 2: Load Parquet   │  ──►  DuckDB 'documents' table
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Step 3: Text Chunking  │  ──►  DuckDB 'chunks' table
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Step 4: Embeddings     │  ──►  DuckDB 'embeddings' table
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Step 5: Retrieval      │  ──►  Top-K similar chunks
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Step 6: Generation     │  ──►  LLM-generated answer
└─────────────────────────┘
```

---

## Prerequisites

- **Python 3.9+**
- **OpenAI API key** — set as an environment variable:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
- **Python packages**:
  ```bash
  pip install duckdb openai numpy pandas tqdm pyarrow
  ```
- **Input data**: `data_helen_Keller.txt` in the project root (Helen Keller's autobiography from Project Gutenberg).

---

## Pipeline Scripts — Execution Sequence

Run the scripts **in order** (Steps 1 → 4) to build the index, then use Steps 5–8 for querying and validation.

### Step 1: `scr_setup_and_create_data.py`

> **Setup & Create Sample Data**

- Creates the `data/` and `db/` directories.
- Reads the raw text file (`data_helen_Keller.txt`).
- Parses it into structured rows (id, section, subsection, text) by splitting on chapter headings.
- Saves the result as `data/sample_data.parquet`.

```bash
python scr_setup_and_create_data.py
```

### Step 2: `scr_load_parquet_to_duckdb.py`

> **Load Parquet into DuckDB**

- Opens (or creates) the persistent DuckDB database at `db/rag.duckdb`.
- Reads the Parquet file from Step 1 using DuckDB's native `read_parquet()`.
- Creates a `documents` table with all rows.

```bash
python scr_load_parquet_to_duckdb.py
```

### Step 3: `scr_create_text_chunks.py`

> **Text Chunking**

- Reads every document from the `documents` table.
- Splits each document's text into **overlapping chunks** (512 chars, 64 char overlap) using a sliding window.
- Stores all chunks in a `chunks` table with metadata linking back to the source document.

```bash
python scr_create_text_chunks.py
```

### Step 4: `scr_generate_embeddings.py`

> **Generate & Store Embeddings**

- Reads all text chunks from the `chunks` table.
- Sends chunk texts to the **OpenAI Embeddings API** in batches (`text-embedding-3-large`, 3072 dims).
- Stores the resulting embedding vectors in an `embeddings` table.

```bash
python scr_generate_embeddings.py
```

### Step 5: `scr_retrieve_similar_chunks.py`

> **Similarity Search / Retrieval**

- Embeds a user query using the same OpenAI model.
- Computes **cosine similarity** between the query vector and all stored chunk embeddings.
- Returns the **top-K most relevant chunks** ranked by similarity.
- Includes demo queries for testing.

```bash
python scr_retrieve_similar_chunks.py
```

### Step 6: `scr_generate_answer.py`

> **Answer Generation (RAG)**

- Combines retrieval (Step 5) and generation into a single flow.
- Retrieves top-K relevant chunks, formats them as context, and sends a prompt to **GPT-4o-mini**.
- The LLM generates an answer grounded in the retrieved context.
- Includes multiple demo queries.

```bash
python scr_generate_answer.py
```

> **Note:** `scr_generate_answer.orig.py` is the original version using `text-embedding-3-small` (1536 dims). The current `scr_generate_answer.py` uses `text-embedding-3-large` (3072 dims).

### Step 7: `scr_rag_query_pipeline.py`

> **Complete End-to-End Pipeline**

- Wraps Steps 2–6 into a single `RAGPipeline` class with a CLI interface.
- Supports four modes:

| Command | Description |
|---------|-------------|
| `python scr_rag_query_pipeline.py index` | Run the full indexing pipeline (load → chunk → embed) |
| `python scr_rag_query_pipeline.py query "your question"` | Answer a single question |
| `python scr_rag_query_pipeline.py interactive` | Interactive Q&A loop (REPL) |
| `python scr_rag_query_pipeline.py demo` | Run pre-defined demo questions |

### Step 8: `scr_run_tests.py`

> **Testing & Validation**

- Comprehensive test suite with four categories:

| Category | What it tests | Requirements |
|----------|---------------|--------------|
| **Unit Tests** | `chunk_text()` logic, cosine similarity math | None |
| **DuckDB Tests** | Parquet round-trips, `FLOAT[]` storage, table schemas | None (in-memory) |
| **Integration Tests** | `documents`, `chunks`, `embeddings` tables exist with correct data | Steps 1–4 completed |
| **API Tests** | Embedding dimensions, semantic similarity ordering | `OPENAI_API_KEY` set |

```bash
python scr_run_tests.py
```

---

## Quick Start

```bash
# 1. Set your API key
export OPENAI_API_KEY="sk-..."

# 2. Install dependencies
pip install duckdb openai numpy pandas tqdm pyarrow

# 3. Run the full pipeline (Steps 1-4)
python scr_setup_and_create_data.py
python scr_load_parquet_to_duckdb.py
python scr_create_text_chunks.py
python scr_generate_embeddings.py

# 4. Ask a question
python scr_rag_query_pipeline.py query "How did Helen Keller learn to communicate?"

# 5. Or use interactive mode
python scr_rag_query_pipeline.py interactive

# 6. Run tests
python scr_run_tests.py
```

---

## Project Structure

```
DuckDBonRAG/
├── data_helen_Keller.txt           # Raw input text (Project Gutenberg)
├── scr_setup_and_create_data.py    # Step 1: Parse text → Parquet
├── scr_load_parquet_to_duckdb.py   # Step 2: Parquet → DuckDB
├── scr_create_text_chunks.py       # Step 3: Documents → Chunks
├── scr_generate_embeddings.py      # Step 4: Chunks → Embeddings
├── scr_retrieve_similar_chunks.py  # Step 5: Query → Similar chunks
├── scr_generate_answer.orig.py     # Step 6: RAG answer (original, small model)
├── scr_generate_answer.py          # Step 6: RAG answer (updated, large model)
├── scr_rag_query_pipeline.py       # Step 7: Unified pipeline + CLI
├── scr_run_tests.py                # Step 8: Test suite
├── data/                           # Generated: Parquet files
│   └── sample_data.parquet
└── db/                             # Generated: DuckDB database
    └── rag.duckdb
```

---

## Key Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `EMBEDDING_DIM` | `3072` | Vector dimensionality |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
