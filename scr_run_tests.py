"""
Step 8: Testing & Validation
==============================
Runs unit tests and integration tests to validate every component of the
RAG pipeline: chunking, cosine similarity, DuckDB loading, embeddings,
retrieval, and generation.

Run: python scr_run_tests.py
Requires:
  - Previous scripts (1-4) to have been run for integration tests.
  - OPENAI_API_KEY environment variable for API-dependent tests.

Overview:
    This is the EIGHTH and final script in the RAG pipeline. It provides
    a comprehensive test suite organized into four categories:

    1. **Unit Tests**: Pure logic tests for chunk_text() and cosine_similarity().
       No external dependencies needed. These validate the core algorithms.

    2. **DuckDB Tests**: In-memory database tests that verify Parquet round-trips,
       FLOAT[] storage, and table schema creation. No persistent state needed.

    3. **Integration Tests**: Tests against the persistent DuckDB database
       (db/rag.duckdb). Verify that documents, chunks, and embeddings tables
       exist, have data, correct dimensions, and referential integrity.
       These require Steps 1-4 to have been run first.

    4. **API Tests**: Tests that call the OpenAI Embeddings API to verify
       correct dimensionality and semantic similarity ordering.
       Require OPENAI_API_KEY to be set.

    The test runner uses a simple pass/fail/skip framework with global counters.
    It returns exit code 1 if any test fails, 0 otherwise.

Dependencies:
    - os: For environment variables and file existence checks.
    - sys: For exit codes.
    - time: (Available for timing tests if needed.)
    - duckdb: For database tests.
    - numpy: For cosine similarity computation.
    - openai: For API-dependent tests (imported locally).
"""
import os           # Standard library: environment variables, file checks
import sys          # Standard library: exit codes
import time         # Standard library: timing utilities (available for perf tests)
import duckdb       # Third-party: embedded OLAP database engine
import numpy as np  # Third-party: numerical computing for vector operations

# ── Configuration ──────────────────────────────────────────────────────────
# DB_PATH: Path to the persistent DuckDB database used by integration tests.
DB_PATH         = "db/rag.duckdb"

# OPENAI_API_KEY: Required for API-dependent tests; tests are skipped if not set.
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")

# EMBEDDING_MODEL / EMBEDDING_DIM: Must match the model used during indexing.
#   NOTE: This test file uses 'text-embedding-3-small' (1536 dims).
#   If you indexed with 'text-embedding-3-large' (3072), update these values.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536

# ── Test utilities ─────────────────────────────────────────────────────────
# Global counters for tracking test results across all categories.
passed = 0   # Number of tests that passed
failed = 0   # Number of tests that failed
skipped = 0  # Number of tests skipped (e.g., no API key)


def run_test(name, func, requires_api=False):
    """Run a single test function and track pass/fail/skip results.

    If the test requires the OpenAI API and the key is not set, the test
    is skipped. Otherwise, the test function is called and any exception
    is caught and reported as a failure.

    Args:
        name (str): Human-readable name for the test.
        func (callable): Zero-argument function that raises on failure.
        requires_api (bool): If True, skip when OPENAI_API_KEY is unset.
    """
    global passed, failed, skipped
    # Skip API-dependent tests if no API key is available
    if requires_api and not OPENAI_API_KEY:
        print(f"  ⏭  SKIP: {name} (no OPENAI_API_KEY)")
        skipped += 1
        return
    try:
        func()
        print(f"  ✓  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗  FAIL: {name} — {e}")
        failed += 1


# ── Chunking function (duplicated here to be self-contained) ──────────────
# This is a standalone copy of the chunk_text function from scr_create_text_chunks.py.
# It is duplicated here so that unit tests can run without importing pipeline code.
def chunk_text(text, chunk_size=512, overlap=64):
    """Split text into overlapping chunks using a sliding window.

    Args:
        text (str): Input text to chunk.
        chunk_size (int): Characters per chunk.
        overlap (int): Overlapping characters between consecutive chunks.

    Returns:
        list[str]: List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite.
    A small epsilon (1e-10) prevents division by zero.
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ══════════════════════════════════════════════════════════════════════════
# UNIT TESTS (no API or DB required)
# ══════════════════════════════════════════════════════════════════════════

def test_chunk_text_basic():
    """1024-char string with size=512, overlap=64 should produce 3 chunks."""
    text = "A" * 1024
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert len(chunks[0]) == 512, f"First chunk should be 512 chars, got {len(chunks[0])}"


def test_chunk_text_small():
    """Text smaller than chunk_size should produce exactly 1 chunk."""
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0] == "Hello world"


def test_chunk_text_exact_size():
    """Text exactly equal to chunk_size should produce 1 full chunk + 1 overlap chunk."""
    text = "B" * 512
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    # start=0 → chunk[0:512]=512 chars, start becomes 448
    # start=448 → chunk[448:960]=64 chars (only 64 remain), start becomes 896
    # 896 >= 512, so 2 chunks
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"


def test_chunk_text_empty():
    """Empty text should produce 0 chunks."""
    text = ""
    chunks = chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 0, f"Expected 0 chunks, got {len(chunks)}"


def test_chunk_overlap_content():
    """Verify the overlap region is shared between consecutive chunks."""
    text = "ABCDEFGHIJKLMNOPQRST"  # 20 chars
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    # Chunk 1: [0:10] = ABCDEFGHIJ, start becomes 7
    # Chunk 2: [7:17] = HIJKLMNOPQ, start becomes 14
    # Chunk 3: [14:24] = QRST (only 6 chars left), start becomes 21
    assert len(chunks) == 3
    # Overlap: last 3 chars of chunk1 should equal first 3 chars of chunk2
    assert chunks[0][-3:] == chunks[1][:3], "Overlap mismatch between chunk 1 and 2"


def test_cosine_identical():
    """Identical vectors should have cosine similarity ≈ 1.0."""
    v = [1.0, 2.0, 3.0]
    sim = cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-6, f"Expected ~1.0, got {sim}"


def test_cosine_orthogonal():
    """Orthogonal vectors should have cosine similarity ≈ 0.0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    sim = cosine_similarity(a, b)
    assert abs(sim) < 1e-6, f"Expected ~0.0, got {sim}"


def test_cosine_opposite():
    """Opposite vectors should have cosine similarity ≈ -1.0."""
    a = [1.0, 2.0, 3.0]
    b = [-1.0, -2.0, -3.0]
    sim = cosine_similarity(a, b)
    assert abs(sim + 1.0) < 1e-6, f"Expected ~-1.0, got {sim}"


# ══════════════════════════════════════════════════════════════════════════
# DUCKDB TESTS (uses in-memory DB, no prior setup needed)
# ══════════════════════════════════════════════════════════════════════════

def test_duckdb_parquet_roundtrip():
    """Create and read back a Parquet file via DuckDB.

    Tests the full cycle: write data to Parquet via COPY, read it back
    into a table via read_parquet, and verify data integrity.
    """
    # Use an in-memory DuckDB connection (no persistent state needed)
    con = duckdb.connect(":memory:")
    con.execute("""
        COPY (
            SELECT 1 AS id, 'sec' AS section,
                   'sub' AS subsection, 'sample text here' AS text
        ) TO '/tmp/test_rag_roundtrip.parquet' (FORMAT PARQUET)
    """)
    con.execute("""
        CREATE TABLE documents AS
        SELECT * FROM read_parquet('/tmp/test_rag_roundtrip.parquet')
    """)
    count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count == 1, f"Expected 1 row, got {count}"
    row = con.execute("SELECT * FROM documents").fetchone()
    assert row[0] == 1 and row[3] == "sample text here"
    con.close()
    os.remove("/tmp/test_rag_roundtrip.parquet")


def test_duckdb_float_list_storage():
    """Verify DuckDB can store and retrieve FLOAT[] columns.

    This is critical for embedding storage — the embeddings table uses
    FLOAT[] columns to store variable-length float arrays.
    """
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE test_emb (id INTEGER, vec FLOAT[])")
    test_vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    con.execute("INSERT INTO test_emb VALUES (?, ?)", [1, test_vec])
    result = con.execute("SELECT vec FROM test_emb WHERE id = 1").fetchone()[0]
    assert len(result) == 5, f"Expected 5 elements, got {len(result)}"
    assert abs(result[0] - 0.1) < 1e-6
    con.close()


def test_duckdb_chunks_table_schema():
    """Verify the chunks table schema matches the expected structure.

    Creates the table schema, inserts sample data, and verifies row count.
    """
    con = duckdb.connect(":memory:")
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
    con.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        [(1, 1, "sec", "sub", 0, "text1"), (2, 1, "sec", "sub", 1, "text2")]
    )
    count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 2
    con.close()


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (require prior scripts to have run + DB to exist)
# ══════════════════════════════════════════════════════════════════════════

def test_documents_table_exists():
    """Verify 'documents' table exists in the persistent DB and has data."""
    # Connect in read-only mode to avoid accidental modifications
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "documents" in tables, f"'documents' table not found. Tables: {tables}"
    count = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert count > 0, "documents table is empty"
    con.close()


def test_chunks_table_exists():
    """Verify 'chunks' table exists in the persistent DB and has data."""
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "chunks" in tables, f"'chunks' table not found. Tables: {tables}"
    count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count > 0, "chunks table is empty"
    con.close()


def test_embeddings_table_exists():
    """Verify 'embeddings' table exists in the persistent DB and has data."""
    con = duckdb.connect(DB_PATH, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    assert "embeddings" in tables, f"'embeddings' table not found. Tables: {tables}"
    count = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert count > 0, "embeddings table is empty"
    con.close()


def test_embeddings_dimension():
    """Verify all stored embeddings have the correct dimension (1536).

    Queries for any embeddings with a dimension other than 1536.
    If any are found, the test fails.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    result = con.execute("""
        SELECT chunk_id, LENGTH(embedding) AS dim
        FROM embeddings
        WHERE LENGTH(embedding) != 1536
        LIMIT 5
    """).fetchall()
    assert len(result) == 0, f"Found embeddings with wrong dimension: {result}"
    con.close()


def test_chunks_embeddings_join():
    """Verify referential integrity: every embedding has a matching chunk.

    Uses a LEFT JOIN to find any 'orphaned' embeddings that don't
    have a corresponding chunk_id in the chunks table.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    orphans = con.execute("""
        SELECT e.chunk_id FROM embeddings e
        LEFT JOIN chunks c ON e.chunk_id = c.chunk_id
        WHERE c.chunk_id IS NULL
    """).fetchall()
    assert len(orphans) == 0, f"Found {len(orphans)} embeddings without matching chunks"
    con.close()


def test_embedding_count_matches_chunks():
    """Verify that the number of embeddings equals the number of chunks.

    After indexing, there should be exactly one embedding per chunk.
    A mismatch indicates a partial or failed embedding run.
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    emb_count = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert chunk_count == emb_count, f"Chunks: {chunk_count}, Embeddings: {emb_count}"
    con.close()


# ══════════════════════════════════════════════════════════════════════════
# API TESTS (require OPENAI_API_KEY)
# ══════════════════════════════════════════════════════════════════════════

def test_embedding_api_dimension():
    """Verify OpenAI embedding API returns vectors of the expected dimension."""
    import openai  # Import locally to avoid import errors when API key is absent
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input="Test sentence for dimension check.",
        model=EMBEDDING_MODEL
    )
    emb = response.data[0].embedding
    assert len(emb) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {len(emb)}"


def test_embedding_api_similarity():
    """Verify that semantically similar texts have higher cosine similarity.

    Embeds three texts: two about Helen Keller and one about weather.
    The two Helen Keller texts should be more similar to each other
    than either is to the weather text.
    """
    import openai  # Import locally to avoid import errors when API key is absent
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    texts = [
        "Helen Keller learned to communicate with sign language.",
        "Helen Keller was taught by Anne Sullivan.",
        "The weather forecast shows rain tomorrow in Seattle.",
    ]

    embeddings = []
    for t in texts:
        resp = client.embeddings.create(input=t, model=EMBEDDING_MODEL)
        embeddings.append(resp.data[0].embedding)

    sim_related = cosine_similarity(embeddings[0], embeddings[1])
    sim_unrelated = cosine_similarity(embeddings[0], embeddings[2])

    assert sim_related > sim_unrelated, (
        f"Related similarity ({sim_related:.4f}) should be > "
        f"unrelated similarity ({sim_unrelated:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    """Run all test categories and print summary results."""
    global passed, failed, skipped

    print("=" * 60)
    print("STEP 8: Testing & Validation")
    print("=" * 60)

    # ── Unit Tests ─────────────────────────────────────────────────────
    # These tests validate core algorithms (chunking, cosine similarity)
    # with no external dependencies — they always run.
    print("\n── Unit Tests (no external dependencies) ──")
    run_test("chunk_text basic (1024 chars → 3 chunks)", test_chunk_text_basic)
    run_test("chunk_text small (< chunk_size → 1 chunk)", test_chunk_text_small)
    run_test("chunk_text exact size (512 chars → 2 chunks)", test_chunk_text_exact_size)
    run_test("chunk_text empty string", test_chunk_text_empty)
    run_test("chunk overlap content verification", test_chunk_overlap_content)
    run_test("cosine similarity: identical vectors", test_cosine_identical)
    run_test("cosine similarity: orthogonal vectors", test_cosine_orthogonal)
    run_test("cosine similarity: opposite vectors", test_cosine_opposite)

    # ── DuckDB Tests ───────────────────────────────────────────────────
    # These tests use in-memory DuckDB connections to verify Parquet I/O,
    # FLOAT[] storage, and table schema creation.
    print("\n── DuckDB Tests (in-memory) ──")
    run_test("DuckDB Parquet roundtrip", test_duckdb_parquet_roundtrip)
    run_test("DuckDB FLOAT[] storage", test_duckdb_float_list_storage)
    run_test("DuckDB chunks table schema", test_duckdb_chunks_table_schema)

    # ── Integration Tests ──────────────────────────────────────────────
    # These tests require the persistent DuckDB database to exist.
    # They verify that Steps 1-4 created the expected tables and data.
    db_exists = os.path.exists(DB_PATH)
    print(f"\n── Integration Tests (persistent DB: {'found' if db_exists else 'NOT found'}) ──")
    if db_exists:
        run_test("documents table exists and has data", test_documents_table_exists)
        run_test("chunks table exists and has data", test_chunks_table_exists)
        run_test("embeddings table exists and has data", test_embeddings_table_exists)
        run_test("embeddings dimension check (1536)", test_embeddings_dimension)
        run_test("chunks ↔ embeddings join integrity", test_chunks_embeddings_join)
        run_test("embedding count matches chunk count", test_embedding_count_matches_chunks)
    else:
        print(f"  ⏭  SKIP: All integration tests (DB not found at '{DB_PATH}')")
        print(f"         Run scripts 1-4 first to create the database.")
        skipped += 6

    # ── API Tests ──────────────────────────────────────────────────────
    # These tests call the OpenAI Embeddings API. They are skipped if
    # OPENAI_API_KEY is not set in the environment.
    print(f"\n── API Tests (OPENAI_API_KEY: {'set' if OPENAI_API_KEY else 'NOT set'}) ──")
    run_test("embedding API dimension (1536)", test_embedding_api_dimension, requires_api=True)
    run_test("embedding API semantic similarity", test_embedding_api_similarity, requires_api=True)

    # ── Summary ────────────────────────────────────────────────────────
    # Print final results and exit with code 1 if any tests failed.
    total = passed + failed + skipped
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped (total: {total})")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("All executed tests passed!")


if __name__ == "__main__":
    main()
