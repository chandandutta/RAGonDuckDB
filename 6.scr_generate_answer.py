"""
Step 6: Generate Answer (LLM Integration) — Updated Version
=============================================================
Takes a user query, retrieves relevant chunks, builds a RAG prompt, and
sends it to OpenAI chat completion to generate an answer.

Run: python scr_generate_answer.py
Requires:
  - scr_generate_embeddings.py to have been run first.
  - OPENAI_API_KEY environment variable to be set.

Overview:
    This is the UPDATED version of Step 6 in the RAG pipeline. It uses
    'text-embedding-3-large' (3072 dims) instead of the original's
    'text-embedding-3-small' (1536 dims), and includes additional test queries.

    The pipeline:
      1. Embeds each user query via the OpenAI Embeddings API.
      2. Computes cosine similarity against all stored chunk embeddings.
      3. Selects the top-K most relevant chunks as context.
      4. Constructs a prompt with context + question and sends it to GPT-4o-mini.
      5. Prints the LLM-generated answer along with source citations.

Dependencies:
    - os: For reading the OPENAI_API_KEY environment variable.
    - duckdb: For fetching stored embeddings and chunk metadata.
    - numpy: For cosine similarity computation.
    - openai: For embedding queries and generating chat completions.
"""
import os           # Standard library: environment variable access
import duckdb       # Third-party: embedded OLAP database engine
import numpy as np  # Third-party: numerical computing for vector operations
import openai       # Third-party: OpenAI API client for embeddings and chat

# ── Configuration ──────────────────────────────────────────────────────────
# DB_PATH: Path to the persistent DuckDB database with chunks and embeddings.
DB_PATH         = "db/rag.duckdb"

# OPENAI_API_KEY: Required for both embedding queries and generating answers.
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")

# EMBEDDING_MODEL: The embedding model for query vectors.
#                  Uses 'text-embedding-3-large' (3072 dims) for higher quality.
EMBEDDING_MODEL = "text-embedding-3-large"

# EMBEDDING_DIM: Vector dimensionality — must match the embedding model output.
EMBEDDING_DIM   = 3072

# CHAT_MODEL: The LLM used for generating answers from retrieved context.
CHAT_MODEL      = "gpt-4o-mini"

# TOP_K: Number of most-similar chunks to include as context for the LLM.
TOP_K           = 5

# SYSTEM_PROMPT: Instructions sent to the LLM as the system message.
# This prompt constrains the model to answer ONLY from the provided context,
# preventing hallucination and encouraging factual, citation-backed responses.
SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's question
using ONLY the provided context. If the context does not contain enough
information to answer, say "I don't have enough information to answer that."

Rules:
- Be concise and factual.
- Cite the section names when relevant.
- Do not make up facts beyond what is in the context."""


def get_connection(db_path=DB_PATH):
    """Return a persistent DuckDB connection.

    Args:
        db_path (str): File path to the DuckDB database.

    Returns:
        duckdb.DuckDBPyConnection: An open connection to the database.
    """
    return duckdb.connect(db_path)


def get_embedding(client, text):
    """Get the embedding vector for a single text string.

    Args:
        client (openai.OpenAI): Initialized OpenAI API client.
        text (str): The text to embed.

    Returns:
        list[float]: Embedding vector of length EMBEDDING_DIM.
    """
    # Clean text: replace newlines with spaces and strip whitespace
    text = text.replace("\n", " ").strip()
    # Return zero vector for empty input
    if not text:
        return [0.0] * EMBEDDING_DIM
    # Call the OpenAI Embeddings API
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite.
    A small epsilon (1e-10) prevents division by zero.

    Args:
        a (list[float]): First vector.
        b (list[float]): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve(con, client, query, top_k=TOP_K):
    """Retrieve the top-k most relevant chunks for a given query.

    Embeds the query, computes cosine similarity against all stored
    chunk embeddings, and returns the top-k most similar chunks.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        client (openai.OpenAI): OpenAI API client.
        query (str): The user's natural language question.
        top_k (int): Number of top results to return.

    Returns:
        list[dict]: Top-k chunks with similarity scores.
    """
    # Embed the query into the same vector space as the stored chunks
    query_embedding = get_embedding(client, query)

    # Fetch all embeddings joined with chunk metadata
    rows = con.execute("""
        SELECT
            e.chunk_id, c.doc_id, c.section, c.subsection,
            c.chunk_text, e.embedding
        FROM embeddings e
        JOIN chunks c ON e.chunk_id = c.chunk_id
    """).fetchall()

    # Compute cosine similarity between the query and each stored chunk
    results = []
    for chunk_id, doc_id, section, subsection, chunk_text, emb in rows:
        sim = cosine_similarity(query_embedding, emb)
        results.append({
            "chunk_id":   chunk_id,
            "doc_id":     doc_id,
            "section":    section,
            "subsection": subsection,
            "chunk_text": chunk_text,
            "similarity": sim,
        })

    # Sort by descending similarity and return the top-k results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def build_context(retrieved_chunks):
    """Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is numbered and labeled with its section/subsection.
    Chunks are separated by '---' dividers.

    Args:
        retrieved_chunks (list[dict]): Retrieved chunks from retrieve().

    Returns:
        str: Formatted context string.
    """
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        section = chunk.get("section", "")
        subsection = chunk.get("subsection", "")
        label = f"{section}"
        if subsection:
            label += f" > {subsection}"
        parts.append(f"[{i}] ({label})\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(client, query, retrieved_chunks):
    """Send the query + context to OpenAI and return the LLM-generated answer.

    Args:
        client (openai.OpenAI): OpenAI API client.
        query (str): The user's question.
        retrieved_chunks (list[dict]): Top-k relevant chunks for context.

    Returns:
        str: The LLM-generated answer text.
    """
    # Build a formatted context string from the retrieved chunks
    context = build_context(retrieved_chunks)

    # Construct the chat messages: system prompt + user message with context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""Context:
{context}

---

Question: {query}

Answer:"""}
    ]

    # Call the Chat Completions API with low temperature for factual responses
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    # Extract and return the assistant's response text
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("STEP 6: Generate Answer (LLM Integration)")
    print("=" * 60)

    # 1. Verify the OpenAI API key is available
    if not OPENAI_API_KEY:
        print("  ERROR: OPENAI_API_KEY environment variable is not set.")
        return
    print(f"  API key found (ends with ...{OPENAI_API_KEY[-4:]})")

    # 2. Initialize the OpenAI client and DuckDB connection
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    con = get_connection()

    # Verify that both required tables (embeddings, chunks) exist
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "embeddings" not in tables or "chunks" not in tables:
        print("  ERROR: Required tables not found.")
        print("  Run scr_generate_embeddings.py first.")
        con.close()
        return

    # 3. Run end-to-end RAG on multiple test queries
    test_queries = [
        "How did Helen Keller learn to communicate?",
        "What was Helen Keller's relationship with Anne Sullivan?",
"Why was Helen Keller’s journey to Boston different from her earlier trip to Baltimore?",
"Why did Helen call Boston “the City of Kind Hearts”?",
"what does she remembers of her visit to Dr. Holmes?"
    ]

    for query in test_queries:
        print(f"\n{'═' * 60}")
        print(f"  Question: {query}")
        print(f"{'═' * 60}")

        # Retrieve: embed the query and find the most similar chunks
        print("\n  Retrieving relevant chunks...")
        chunks = retrieve(con, client, query, top_k=TOP_K)

        print(f"  Found {len(chunks)} chunks (top similarity: {chunks[0]['similarity']:.4f})")

        # Generate: send context + question to the LLM for an answer
        print("  Generating answer...")
        answer = generate_answer(client, query, chunks)

        print(f"\n  Answer:\n  {answer}")

        print(f"\n  Sources:")
        for i, c in enumerate(chunks, 1):
            print(f"    [{i}] ({c['similarity']:.4f}) {c['section']} > {c['subsection']}")

    con.close()

    print("\n" + "=" * 60)
    print("Generation works! Next: python scr_rag_query_pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
