"""
Step 5: Retrieve Similar Chunks (Similarity Search)
=====================================================
Embeds a user query via OpenAI, computes cosine similarity against all
stored chunk embeddings, and returns the top-K most relevant chunks.

Run: python scr_retrieve_similar_chunks.py
Requires:
  - scr_generate_embeddings.py to have been run first.
  - OPENAI_API_KEY environment variable to be set.

Overview:
    This is the FIFTH script in the RAG pipeline. It demonstrates the
    retrieval phase of Retrieval-Augmented Generation:
      1. Embeds a user's natural-language query into a vector using the
         same OpenAI embedding model used for indexing.
      2. Fetches all stored chunk embeddings from DuckDB.
      3. Computes cosine similarity between the query vector and every
         chunk vector (brute-force nearest neighbor search).
      4. Returns the top-K most similar chunks, ranked by similarity.

    This brute-force approach works well for small-to-medium datasets.
    For larger corpora, approximate nearest neighbor (ANN) indices would
    be more efficient.

Dependencies:
    - os: For reading the OPENAI_API_KEY environment variable.
    - duckdb: For fetching stored embeddings and chunk metadata.
    - numpy: For vector math (dot product, norms) in cosine similarity.
    - openai: For embedding the user query via the OpenAI API.
"""
import os           # Standard library: environment variable access
import duckdb       # Third-party: embedded OLAP database engine
import numpy as np  # Third-party: numerical computing for vector operations
import openai       # Third-party: OpenAI API client for query embedding

# ── Configuration ──────────────────────────────────────────────────────────
# DB_PATH: Path to the persistent DuckDB database with chunks and embeddings.
DB_PATH         = "db/rag.duckdb"

# OPENAI_API_KEY: Required for embedding the user query via the OpenAI API.
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")

# EMBEDDING_MODEL: Must match the model used in scr_generate_embeddings.py
#                  to ensure query and chunk embeddings are in the same vector space.
EMBEDDING_MODEL = "text-embedding-3-large"

# EMBEDDING_DIM: Dimensionality of the vectors (must match the model output).
EMBEDDING_DIM   = 3072

# TOP_K: Number of most-similar chunks to retrieve per query.
TOP_K           = 5


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

    Preprocesses the text by replacing newlines with spaces and stripping
    whitespace. Returns a zero vector for empty strings to avoid API errors.

    Args:
        client (openai.OpenAI): Initialized OpenAI API client.
        text (str): The text to embed.

    Returns:
        list[float]: Embedding vector of length EMBEDDING_DIM.
    """
    # Clean the text: replace newlines and strip whitespace
    text = text.replace("\n", " ").strip()
    # Return zero vector for empty input to avoid API errors
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

    Cosine similarity measures the angle between two vectors:
      - 1.0 means identical direction (most similar)
      - 0.0 means orthogonal (unrelated)
      - -1.0 means opposite direction

    A small epsilon (1e-10) is added to the denominator to prevent
    division-by-zero errors when a vector has zero magnitude.

    Args:
        a (list[float]): First vector.
        b (list[float]): Second vector.

    Returns:
        float: Cosine similarity score in the range [-1, 1].
    """
    # Convert lists to numpy arrays for vectorized math
    a, b = np.array(a), np.array(b)
    # dot(a,b) / (||a|| * ||b||), with epsilon to avoid division by zero
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve(con, client, query, top_k=TOP_K):
    """
    Retrieve the top-k most relevant chunks for a given query.

    Returns a list of dicts with keys:
        chunk_id, doc_id, section, subsection, chunk_text, similarity
    """
    # 1. Embed the query text into a vector using the same model
    #    that was used to embed the stored chunks. This ensures both
    #    query and chunk vectors live in the same embedding space.
    query_embedding = get_embedding(client, query)

    # 2. Fetch all stored embeddings along with their chunk metadata.
    #    We JOIN embeddings with chunks to get the text and section info
    #    alongside each embedding vector.
    rows = con.execute("""
        SELECT
            e.chunk_id,
            c.doc_id,
            c.section,
            c.subsection,
            c.chunk_text,
            e.embedding
        FROM embeddings e
        JOIN chunks c ON e.chunk_id = c.chunk_id
    """).fetchall()

    # 3. Compute cosine similarity between the query embedding and every
    #    stored chunk embedding (brute-force nearest neighbor search).
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

    # 4. Sort results by similarity in descending order and return only top_k.
    #    The highest-similarity chunks are the most semantically relevant.
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def main():
    print("=" * 60)
    print("STEP 5: Retrieve Similar Chunks")
    print("=" * 60)

    # 1. Verify the OpenAI API key is set in the environment
    if not OPENAI_API_KEY:
        print("  ERROR: OPENAI_API_KEY environment variable is not set.")
        return
    print(f"  API key found (ends with ...{OPENAI_API_KEY[-4:]})")

    # 2. Initialize the OpenAI client and DuckDB connection
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    con = get_connection()

    # Verify that both required tables exist in the database
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "embeddings" not in tables or "chunks" not in tables:
        print("  ERROR: 'embeddings' or 'chunks' table not found.")
        print("  Run scr_generate_embeddings.py first.")
        con.close()
        return

    # 3. Run test queries to demonstrate the retrieval functionality.
    #    Each query is embedded, compared against all chunks, and the
    #    top-K most relevant chunks are displayed.
    test_queries = [
        "How did Helen Keller learn to communicate?",
        "What role did Anne Sullivan play?",
        "What was Helen Keller's childhood like?",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"  Query: \"{query}\"")
        print(f"{'─' * 60}")

        results = retrieve(con, client, query, top_k=TOP_K)

        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] Similarity: {r['similarity']:.4f}")
            print(f"      Section: {r['section']} > {r['subsection']}")
            print(f"      Chunk (first 150 chars): {r['chunk_text'][:150]}...")

    con.close()

    print("\n" + "=" * 60)
    print("Retrieval works! Next: python 6.scr_generate_answer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
