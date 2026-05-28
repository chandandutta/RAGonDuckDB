"""
Step 3: Text Chunking
======================
Reads the 'documents' table from DuckDB, splits each document's text into
overlapping chunks, and stores them in a 'chunks' table.

Run: python scr_create_text_chunks.py
Requires: scr_load_parquet_to_duckdb.py to have been run first.

Overview:
    This is the THIRD script in the RAG pipeline. It performs the following:
      1. Connects to the DuckDB database containing the 'documents' table.
      2. Reads every document row and splits its text into fixed-size,
         overlapping character chunks (sliding window approach).
      3. Stores all chunks in a new 'chunks' table with metadata linking
         each chunk back to its source document.

    Overlapping chunks ensure that information spanning chunk boundaries
    is not lost. For example, with chunk_size=512 and overlap=64, each
    successive chunk starts 448 characters after the previous one, sharing
    64 characters of context.

    The 'chunks' table is used by the next step (scr_generate_embeddings.py)
    to generate vector embeddings for each chunk.

Dependencies:
    - duckdb: For reading documents and storing chunks in the database.
"""
import duckdb  # Third-party: embedded OLAP database engine for SQL queries

# ── Configuration ──────────────────────────────────────────────────────────
# DB_PATH: Path to the persistent DuckDB database file created in Step 2.
DB_PATH       = "db/rag.duckdb"

# CHUNK_SIZE: Maximum number of characters per chunk.
#             Larger chunks retain more context but produce fewer, coarser chunks.
#             512 characters is a common choice that balances context and granularity.
CHUNK_SIZE    = 512     # characters per chunk

# CHUNK_OVERLAP: Number of characters shared between consecutive chunks.
#                This sliding-window overlap ensures that sentences or ideas
#                straddling a chunk boundary appear in at least one complete chunk.
CHUNK_OVERLAP = 64      # overlap between consecutive chunks


def get_connection(db_path=DB_PATH):
    """Return a persistent DuckDB connection.

    Args:
        db_path (str): File path to the DuckDB database.

    Returns:
        duckdb.DuckDBPyConnection: An open connection to the database.
    """
    return duckdb.connect(db_path)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of fixed character length.

    Uses a sliding window approach: each chunk is 'chunk_size' characters long,
    and the window advances by (chunk_size - overlap) characters. The last chunk
    may be shorter than chunk_size if the remaining text is insufficient.

    Args:
        text (str): The input text to split.
        chunk_size (int): Maximum number of characters per chunk.
        overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
        list[str]: A list of text chunks. Returns an empty list for empty input.
    """
    chunks = []   # Accumulator for the resulting text chunks
    start = 0     # Current starting character index for the sliding window
    while start < len(text):
        # Define the end boundary of the current chunk
        end = start + chunk_size
        # Slice the text from start to end (Python handles out-of-bounds gracefully)
        chunks.append(text[start:end])
        # Advance the window by (chunk_size - overlap) characters
        # This ensures 'overlap' characters are shared between consecutive chunks
        start += chunk_size - overlap
    return chunks


def create_chunks_table(con):
    """Read documents, chunk each row, and store in a 'chunks' table.

    This function:
      1. Verifies that the 'documents' table exists in the database.
      2. Reads all document rows (id, section, subsection, text).
      3. Applies chunk_text() to each document's text to produce chunks.
      4. Creates a new 'chunks' table and bulk-inserts all chunk rows.

    Each chunk is assigned a unique chunk_id (auto-incremented) and retains
    a reference to its parent document via doc_id. The chunk_index field
    indicates the chunk's position within its parent document.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.

    Returns:
        int: Total number of chunks created.

    Raises:
        RuntimeError: If the 'documents' table does not exist.
    """
    # Verify that the 'documents' table exists before attempting to read from it
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "documents" not in tables:
        raise RuntimeError(
            "'documents' table not found. Run scr_load_parquet_to_duckdb.py first."
        )

    # Check if the 'id' column exists in the documents table.
    # Some Parquet files may not have an 'id' column, so we fall back to
    # using ROW_NUMBER() as a window function to generate sequential IDs.
    cols = [r[0] for r in con.execute("DESCRIBE documents").fetchall()]
    if "id" in cols:
        id_expr = "id"
    else:
        id_expr = "ROW_NUMBER() OVER () AS id"

    rows = con.execute(f"""
        SELECT {id_expr}, section, subsection, text FROM documents
    """).fetchall()

    # Iterate over every document row and split its text into chunks.
    # Each chunk gets a globally unique chunk_id and inherits its parent's metadata.
    all_chunks = []   # Accumulator: list of tuples (chunk_id, doc_id, section, subsection, chunk_index, chunk_text)
    chunk_id = 0      # Global auto-incrementing chunk identifier
    for doc_id, section, subsection, text in rows:
        # chunk_text() returns a list of overlapping text fragments
        for i, chunk in enumerate(chunk_text(text)):
            chunk_id += 1
            # Append a tuple matching the chunks table schema
            all_chunks.append((chunk_id, doc_id, section, subsection, i, chunk))

    # Drop any pre-existing chunks table for idempotent re-runs
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute("""
        CREATE TABLE chunks (
            chunk_id     INTEGER PRIMARY KEY,
            doc_id       INTEGER,
            section      VARCHAR,
            subsection   VARCHAR,
            chunk_index  INTEGER,
            chunk_text   VARCHAR
        )
    """)

    # Bulk-insert all chunks using executemany for efficiency.
    # executemany sends all rows in a single transaction, which is much faster
    # than individual INSERT statements.
    con.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        all_chunks
    )

    print(f"  Created {len(all_chunks)} chunks from {len(rows)} documents.")
    return len(all_chunks)


def main():
    print("=" * 60)
    print("STEP 3: Text Chunking")
    print("=" * 60)

    # 1. Establish connection to the persistent DuckDB database
    print(f"\n[1/3] Connecting to DuckDB at '{DB_PATH}'...")
    con = get_connection()

    # 2. Read all documents, split each into overlapping chunks, and store them.
    #    The chunk_size and overlap parameters control the granularity of retrieval.
    print(f"\n[2/3] Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    num_chunks = create_chunks_table(con)

    # 3. Verify the chunks by printing a sample and aggregate statistics.
    #    This helps confirm that chunking worked correctly.
    print(f"\n[3/3] Verifying chunks...")
    # Fetch the first 5 chunks as a pandas DataFrame for display
    sample = con.execute("SELECT * FROM chunks LIMIT 5").fetchdf()
    print(f"\n  First 5 chunks:")
    print(sample.to_string(max_colwidth=80))

    # Compute and display aggregate statistics about the chunks:
    # total count, number of distinct source documents, and chunk length stats.
    stats = con.execute("""
        SELECT
            COUNT(*) AS total_chunks,
            COUNT(DISTINCT doc_id) AS total_docs,
            AVG(LENGTH(chunk_text)) AS avg_chunk_len,
            MIN(LENGTH(chunk_text)) AS min_chunk_len,
            MAX(LENGTH(chunk_text)) AS max_chunk_len
        FROM chunks
    """).fetchdf()
    print(f"\n  Chunk statistics:")
    print(stats.to_string(index=False))

    con.close()

    print("\n" + "=" * 60)
    print("Chunking done! Next: python 4.scr_generate_embeddings.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
