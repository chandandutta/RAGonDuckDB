"""
Step 4: Generate & Store Embeddings
=====================================
Reads all chunks from DuckDB, generates embeddings via OpenAI API, and
stores the vectors back in an 'embeddings' table.

Run: python scr_generate_embeddings.py
Requires:
  - scr_create_text_chunks.py to have been run first.
  - OPENAI_API_KEY environment variable to be set.

Overview:
    This is the FOURTH script in the RAG pipeline. It performs:
      1. Reads all text chunks from the 'chunks' table in DuckDB.
      2. Sends the chunk texts to OpenAI's Embeddings API in batches
         to generate dense vector representations (embeddings).
      3. Stores the resulting embedding vectors in an 'embeddings' table
         in DuckDB, keyed by chunk_id.

    Embeddings are numerical vector representations of text that capture
    semantic meaning. Similar texts produce vectors that are close together
    in the embedding space, enabling similarity-based retrieval.

    The embedding model used is 'text-embedding-3-large' which produces
    3072-dimensional vectors. Batching is used to minimize API calls.

Dependencies:
    - os: For reading the OPENAI_API_KEY environment variable.
    - duckdb: For reading chunks and storing embeddings.
    - openai: For calling the OpenAI Embeddings API.
    - tqdm: For displaying progress bars during batch processing.
"""
import os           # Standard library: environment variable access
import duckdb       # Third-party: embedded OLAP database engine
import openai       # Third-party: OpenAI API client for embeddings
from tqdm import tqdm  # Third-party: progress bar for batch processing loops

# ── Configuration ──────────────────────────────────────────────────────────
# DB_PATH: Path to the persistent DuckDB database containing the 'chunks' table.
DB_PATH         = "db/rag.duckdb"

# OPENAI_API_KEY: API key for authenticating with OpenAI's Embeddings API.
#                 Must be set as an environment variable before running this script.
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")

# EMBEDDING_MODEL: The OpenAI embedding model to use.
#                  'text-embedding-3-large' produces high-quality 3072-dim vectors.
EMBEDDING_MODEL = "text-embedding-3-large"

# EMBEDDING_DIM: Dimensionality of the embedding vectors produced by the model.
#                Must match the chosen EMBEDDING_MODEL's output size.
EMBEDDING_DIM   = 3072

# BATCH_SIZE: Number of text chunks to send per API call.
#             Larger batches are more efficient but may hit API rate/size limits.
BATCH_SIZE      = 100


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
        list[float]: A list of floats representing the embedding vector
                     (length = EMBEDDING_DIM).
    """
    # Clean up the text: replace newlines with spaces and strip leading/trailing whitespace
    text = text.replace("\n", " ").strip()
    # Return a zero vector for empty strings (avoids sending empty input to the API)
    if not text:
        return [0.0] * EMBEDDING_DIM
    # Call the OpenAI Embeddings API to generate the vector
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    # Extract and return the embedding vector from the API response
    return response.data[0].embedding


def get_embeddings_batch(client, texts, batch_size=BATCH_SIZE):
    """Get embeddings for a list of texts in batches.

    Processes texts in batches to minimize API calls. Each batch is sent
    as a single request to OpenAI's Embeddings API. A progress bar (tqdm)
    tracks the batch processing.

    Args:
        client (openai.OpenAI): Initialized OpenAI API client.
        texts (list[str]): List of text strings to embed.
        batch_size (int): Number of texts per API call.

    Returns:
        list[list[float]]: A list of embedding vectors, one per input text.
    """
    all_embeddings = []  # Accumulator for all embedding vectors
    # Process texts in batches of 'batch_size' with a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding batches"):
        # Preprocess each text in the batch: replace newlines and strip whitespace
        batch = [t.replace("\n", " ").strip() for t in texts[i:i + batch_size]]
        # Replace empty strings with a single space to avoid API errors
        # (OpenAI's API rejects empty string inputs)
        batch = [t if t else " " for t in batch]
        # Send the entire batch in one API call for efficiency
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL
        )
        # Extract embedding vectors from the response and add to accumulator
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def store_embeddings(con, client):
    """Generate and store embeddings for all chunks.

    Reads all chunk texts from the 'chunks' table, generates embedding
    vectors via the OpenAI API, and stores them in a new 'embeddings' table.
    The embeddings table is keyed by chunk_id for easy joining with the
    chunks table during retrieval.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        client (openai.OpenAI): Initialized OpenAI API client.

    Returns:
        int: Number of embeddings generated and stored.

    Raises:
        RuntimeError: If the 'chunks' table does not exist.
    """
    # Verify the 'chunks' table exists before attempting to read
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "chunks" not in tables:
        raise RuntimeError(
            "'chunks' table not found. Run scr_create_text_chunks.py first."
        )

    # Fetch all chunks ordered by chunk_id to maintain consistent ordering
    rows = con.execute(
        "SELECT chunk_id, chunk_text FROM chunks ORDER BY chunk_id"
    ).fetchall()

    # Separate chunk IDs and texts into parallel lists for batch processing
    chunk_ids = [r[0] for r in rows]  # List of integer chunk identifiers
    texts     = [r[1] for r in rows]  # List of text strings to embed

    print(f"  Generating embeddings for {len(texts)} chunks...")
    embeddings = get_embeddings_batch(client, texts)

    # Create the embeddings table, dropping any existing one for idempotency.
    # The table stores each embedding as a FLOAT[] (variable-length float array).
    con.execute("DROP TABLE IF EXISTS embeddings")
    con.execute("""
        CREATE TABLE embeddings (
            chunk_id   INTEGER PRIMARY KEY,
            embedding  FLOAT[]
        )
    """)

    # Insert each embedding row individually.
    # Note: executemany could be used here, but FLOAT[] parameter binding
    # with lists requires individual inserts in some DuckDB versions.
    for cid, emb in zip(chunk_ids, embeddings):
        con.execute(
            "INSERT INTO embeddings VALUES (?, ?)",
            [cid, emb]
        )

    print(f"  Stored {len(embeddings)} embeddings in DuckDB.")
    return len(embeddings)


def main():
    print("=" * 60)
    print("STEP 4: Generate & Store Embeddings")
    print("=" * 60)

    # 1. Verify that the OpenAI API key is available in the environment.
    #    Without it, we cannot call the Embeddings API.
    print("\n[1/4] Checking OpenAI API key...")
    if not OPENAI_API_KEY:
        print("  ERROR: OPENAI_API_KEY environment variable is not set.")
        print("  Run: export OPENAI_API_KEY='sk-...'")
        return
    print(f"  API key found (ends with ...{OPENAI_API_KEY[-4:]})")

    # 2. Create an OpenAI client instance using the API key.
    #    This client will be used for all embedding API calls.
    print("\n[2/4] Initializing OpenAI client...")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # 3. Connect to the DuckDB database that contains the 'chunks' table
    print(f"\n[3/4] Connecting to DuckDB at '{DB_PATH}'...")
    con = get_connection()

    # 4. Generate embeddings for all chunks and store them in DuckDB.
    #    This is the most time-consuming step as it involves API calls.
    print(f"\n[4/4] Generating and storing embeddings...")
    num_embeddings = store_embeddings(con, client)

    # Verify the stored embeddings by checking count and sample dimensions
    print(f"\n  Verification:")
    count = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"  Total embeddings stored: {count}")

    sample = con.execute("""
        SELECT chunk_id, LENGTH(embedding) AS vector_dim
        FROM embeddings LIMIT 3
    """).fetchdf()
    print(f"  Sample embedding dimensions:")
    print(sample.to_string(index=False))

    con.close()

    print("\n" + "=" * 60)
    print("Embeddings stored! Next: python 5.scr_retrieve_similar_chunks.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
