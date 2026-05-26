"""
Step 7: Complete RAG Query Pipeline
=====================================
End-to-end RAG pipeline that combines all previous steps into a single
orchestrator class. Supports indexing (load → chunk → embed) and querying.

Run: python scr_rag_query_pipeline.py
Requires:
  - OPENAI_API_KEY environment variable to be set.
  - For query-only mode: previous scripts (1-4) to have been run.
  - For index mode: data/sample_data.parquet to exist.

Usage:
  python scr_rag_query_pipeline.py index       # Full indexing pipeline
  python scr_rag_query_pipeline.py query "..."  # Ask a question
  python scr_rag_query_pipeline.py interactive  # Interactive Q&A loop

Overview:
    This is the SEVENTH script and serves as the unified orchestrator for
    the entire RAG pipeline. It wraps Steps 2–6 into a single RAGPipeline
    class with two main capabilities:

    1. **Indexing** (run once): Loads Parquet → creates chunks → generates
       embeddings. This replaces running Steps 2, 3, and 4 individually.
    2. **Querying** (run many times): Embeds a question → retrieves relevant
       chunks via cosine similarity → generates an answer via OpenAI LLM.

    The class provides a clean API:
      - RAGPipeline.index()  — full indexing pipeline
      - RAGPipeline.query()  — answer a single question

    CLI modes: index, query, interactive (REPL), and demo.

Dependencies:
    - os: For environment variable access and directory creation.
    - sys: For CLI argument parsing and exit codes.
    - json: For structured output formatting.
    - duckdb: Embedded OLAP database engine.
    - numpy: Cosine similarity computation.
    - openai: OpenAI API client for embeddings and chat completions.
    - tqdm: Progress bars for batch embedding.
"""
import os           # Standard library: environment variables, directory ops
import sys          # Standard library: CLI arguments and exit codes
import json         # Standard library: JSON output formatting
import duckdb       # Third-party: embedded OLAP database engine
import numpy as np  # Third-party: numerical computing for vector operations
import openai       # Third-party: OpenAI API client for embeddings and chat
from tqdm import tqdm  # Third-party: progress bar for batch processing

# ── Configuration ──────────────────────────────────────────────────────────
# OPENAI_API_KEY: Required for embeddings and chat completion API calls.
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")

# EMBEDDING_MODEL: OpenAI model for generating text embeddings.
EMBEDDING_MODEL = "text-embedding-3-large"

# EMBEDDING_DIM: Output dimensionality of the chosen embedding model.
EMBEDDING_DIM   = 3072

# CHAT_MODEL: OpenAI LLM model for generating answers from context.
CHAT_MODEL      = "gpt-4o-mini"

# DB_PATH: Persistent DuckDB database storing documents, chunks, and embeddings.
DB_PATH         = "db/rag.duckdb"

# PARQUET_PATH: Default path to the Parquet file for indexing.
PARQUET_PATH    = "data/sample_data.parquet"

# CHUNK_SIZE / CHUNK_OVERLAP: Parameters for the sliding-window text chunker.
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64

# TOP_K: Number of most-similar chunks to retrieve per query.
TOP_K           = 5

# SYSTEM_PROMPT: Instruction sent to the LLM as the system message.
# Constrains answers to the provided context only, preventing hallucination.
SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's question
using ONLY the provided context. If the context does not contain enough
information to answer, say "I don't have enough information to answer that."

Rules:
- Be concise and factual.
- Cite the section names when relevant.
- Do not make up facts beyond what is in the context."""


class RAGPipeline:
    """End-to-end RAG pipeline backed by DuckDB and OpenAI.

    This class encapsulates the entire RAG workflow: indexing documents
    (loading, chunking, embedding) and querying (retrieval + generation).
    It maintains a persistent DuckDB connection and an OpenAI client
    throughout its lifecycle.
    """

    def __init__(self, db_path=DB_PATH):
        # Ensure the database directory exists before connecting
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Open a persistent DuckDB connection (creates the file if needed)
        self.con = duckdb.connect(db_path)
        # Initialize the OpenAI client for embeddings and chat completions
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def close(self):
        """Close the DuckDB connection."""
        self.con.close()

    # ── INDEXING (run once) ────────────────────────────────────────────────

    def index(self, parquet_path=PARQUET_PATH):
        """Full indexing pipeline: load → chunk → embed.

        Orchestrates all three indexing steps in sequence:
          1. Load Parquet data into a 'documents' table.
          2. Split each document's text into overlapping chunks.
          3. Generate OpenAI embeddings for all chunks.

        Args:
            parquet_path (str): Path to the Parquet file to index.
        """
        print("=" * 50)
        # Step 1: Load the Parquet file into the 'documents' table
        print("STEP 1/3: Loading Parquet into DuckDB...")
        self._load_parquet(parquet_path)

        # Step 2: Split document text into overlapping chunks
        print("\nSTEP 2/3: Creating text chunks...")
        self._create_chunks()

        # Step 3: Generate embedding vectors for all chunks via OpenAI API
        print("\nSTEP 3/3: Generating & storing embeddings...")
        self._store_embeddings()

        print("=" * 50)
        print("Indexing complete!")

    def _load_parquet(self, parquet_path):
        """Load Parquet data into the 'documents' table (drop-and-replace)."""
        self.con.execute("DROP TABLE IF EXISTS documents")
        self.con.execute(f"""
            CREATE TABLE documents AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        count = self.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        print(f"  Loaded {count} rows into 'documents' table.")

    def _chunk_text(self, text):
        """Split text into overlapping chunks using a sliding window."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _create_chunks(self):
        """Read all documents, split into chunks, and store in 'chunks' table."""
        # Fetch all document rows from the database
        rows = self.con.execute(
            "SELECT id, section, subsection, text FROM documents"
        ).fetchall()

        # Build list of chunk tuples to bulk-insert
        all_chunks = []
        chunk_id = 0
        for doc_id, section, subsection, text in rows:
            for i, chunk in enumerate(self._chunk_text(text)):
                chunk_id += 1
                all_chunks.append((chunk_id, doc_id, section, subsection, i, chunk))

        # Create the chunks table (drop first for idempotent re-runs)
        self.con.execute("DROP TABLE IF EXISTS chunks")
        self.con.execute("""
            CREATE TABLE chunks (
                chunk_id     INTEGER PRIMARY KEY,
                doc_id       INTEGER,
                section      VARCHAR,
                subsection   VARCHAR,
                chunk_index  INTEGER,
                chunk_text   VARCHAR
            )
        """)
        # Bulk-insert all chunks in a single transaction for speed
        self.con.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", all_chunks
        )
        print(f"  Created {len(all_chunks)} chunks from {len(rows)} documents.")

    def _get_embeddings_batch(self, texts, batch_size=100):
        """Generate embeddings for a list of texts in batches via OpenAI API."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
            batch = [t.replace("\n", " ").strip() for t in texts[i:i + batch_size]]
            batch = [t if t else " " for t in batch]
            response = self.client.embeddings.create(
                input=batch, model=EMBEDDING_MODEL
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def _store_embeddings(self):
        """Generate and store embeddings for all chunks in the 'embeddings' table."""
        # Fetch all chunk IDs and texts
        rows = self.con.execute(
            "SELECT chunk_id, chunk_text FROM chunks ORDER BY chunk_id"
        ).fetchall()

        # Separate into parallel lists for batch embedding
        chunk_ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        print(f"  Generating embeddings for {len(texts)} chunks...")
        embeddings = self._get_embeddings_batch(texts)

        # Create the embeddings table (drop first for idempotency)
        self.con.execute("DROP TABLE IF EXISTS embeddings")
        self.con.execute("""
            CREATE TABLE embeddings (
                chunk_id   INTEGER PRIMARY KEY,
                embedding  FLOAT[]
            )
        """)

        # Insert each embedding individually (FLOAT[] binding requires this)
        for cid, emb in zip(chunk_ids, embeddings):
            self.con.execute(
                "INSERT INTO embeddings VALUES (?, ?)", [cid, emb]
            )
        print(f"  Stored {len(embeddings)} embeddings in DuckDB.")

    # ── QUERYING ───────────────────────────────────────────────────────────

    def query(self, question, top_k=TOP_K):
        """Answer a question using Retrieval-Augmented Generation.

        This method orchestrates the query pipeline:
          1. Embeds the question into a vector.
          2. Retrieves the top-k most similar chunks from DuckDB.
          3. Sends the chunks as context to the LLM to generate an answer.

        Args:
            question (str): The user's natural language question.
            top_k (int): Number of chunks to retrieve as context.

        Returns:
            dict: A dictionary with keys:
                - 'question' (str): The original question.
                - 'answer' (str): The LLM-generated answer.
                - 'sources' (list[dict]): Top-k chunks with similarity scores.
        """
        # Retrieve the most relevant chunks for this question
        chunks = self._retrieve(question, top_k=top_k)

        # Generate an answer using the retrieved context
        answer = self._generate(question, chunks)

        return {
            "question": question,
            "answer":   answer,
            "sources":  [
                {
                    "chunk_id":   c["chunk_id"],
                    "section":    c["section"],
                    "subsection": c["subsection"],
                    "chunk_text": c["chunk_text"][:200] + "...",
                    "similarity": round(c["similarity"], 4),
                }
                for c in chunks
            ],
        }

    def _get_embedding(self, text):
        """Get the embedding vector for a single text string."""
        # Clean text: replace newlines and strip whitespace
        text = text.replace("\n", " ").strip()
        # Return zero vector for empty input
        if not text:
            return [0.0] * EMBEDDING_DIM
        # Call the OpenAI Embeddings API
        response = self.client.embeddings.create(
            input=text, model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    @staticmethod
    def _cosine_similarity(a, b):
        """Compute cosine similarity between two vectors (with epsilon for stability)."""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _retrieve(self, query_text, top_k=TOP_K):
        """Embed the query and retrieve the top-k most similar chunks."""
        # Embed the query into the same vector space as the stored chunks
        query_embedding = self._get_embedding(query_text)

        # Fetch all embeddings joined with chunk metadata
        rows = self.con.execute("""
            SELECT e.chunk_id, c.doc_id, c.section, c.subsection,
                   c.chunk_text, e.embedding
            FROM embeddings e
            JOIN chunks c ON e.chunk_id = c.chunk_id
        """).fetchall()

        # Compute cosine similarity for each chunk
        results = []
        for chunk_id, doc_id, section, subsection, chunk_text, emb in rows:
            sim = self._cosine_similarity(query_embedding, emb)
            results.append({
                "chunk_id": chunk_id, "doc_id": doc_id,
                "section": section, "subsection": subsection,
                "chunk_text": chunk_text, "similarity": sim,
            })

        # Sort by descending similarity and return top-k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _build_context(self, retrieved_chunks):
        """Format retrieved chunks into a numbered context string for the LLM."""
        parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            section = chunk.get("section", "")
            subsection = chunk.get("subsection", "")
            label = section
            if subsection:
                label += f" > {subsection}"
            parts.append(f"[{i}] ({label})\n{chunk['chunk_text']}")
        return "\n\n---\n\n".join(parts)

    def _generate(self, query_text, retrieved_chunks):
        """Send context + question to the LLM and return the generated answer."""
        # Build the context string from retrieved chunks
        context = self._build_context(retrieved_chunks)
        # Construct the chat messages: system prompt + user message
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {query_text}\n\nAnswer:"}
        ]
        # Call the Chat Completions API with low temperature for factual answers
        response = self.client.chat.completions.create(
            model=CHAT_MODEL, messages=messages,
            temperature=0.2, max_tokens=1024,
        )
        return response.choices[0].message.content


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for the RAG pipeline with multiple modes."""
    # Check for the required API key before proceeding
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Show usage if no command is provided
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scr_rag_query_pipeline.py index              # Index data")
        print("  python scr_rag_query_pipeline.py query \"question\"    # Ask a question")
        print("  python scr_rag_query_pipeline.py interactive        # Interactive Q&A")
        print("  python scr_rag_query_pipeline.py demo               # Run demo queries")
        sys.exit(1)

    # Parse the CLI command and instantiate the pipeline
    cmd = sys.argv[1]
    rag = RAGPipeline()

    try:
        if cmd == "index":
            # Index mode: run the full indexing pipeline (load → chunk → embed)
            path = sys.argv[2] if len(sys.argv) > 2 else PARQUET_PATH
            rag.index(path)

        elif cmd == "query":
            # Query mode: answer a single question from CLI arguments
            q = " ".join(sys.argv[2:])
            if not q:
                print("Please provide a question after 'query'.")
                sys.exit(1)
            result = rag.query(q)
            print(f"\nQ: {result['question']}")
            print(f"\nA: {result['answer']}")
            print(f"\nSources ({len(result['sources'])}):")
            for s in result["sources"]:
                print(f"  [{s['similarity']}] {s['section']} > {s['subsection']}")

        elif cmd == "interactive":
            # Interactive mode: REPL loop for continuous Q&A
            print("RAG Interactive Mode (type 'quit' to exit)\n")
            while True:
                q = input("You: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                result = rag.query(q)
                print(f"\nAssistant: {result['answer']}\n")

        elif cmd == "demo":
            # Demo mode: run a set of pre-defined questions
            print("=" * 60)
            print("RAG Pipeline Demo")
            print("=" * 60)

            demo_questions = [
                "How did Helen Keller learn to communicate?",
                "What role did Anne Sullivan play in Helen's life?",
                "What were Helen Keller's major achievements?",
            ]

            for q in demo_questions:
                print(f"\n{'═' * 60}")
                print(f"Q: {q}")
                result = rag.query(q)
                print(f"\nA: {result['answer']}")
                print(f"\nSources ({len(result['sources'])}):")
                for s in result["sources"]:
                    print(f"  [{s['similarity']}] {s['section']} > {s['subsection']}")
                print(f"{'═' * 60}")

        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)

    finally:
        rag.close()


if __name__ == "__main__":
    main()
