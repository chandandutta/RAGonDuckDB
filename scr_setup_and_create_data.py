"""
Step 1: Setup & Create Sample Data
====================================
Creates the project directory structure and generates a sample Parquet file
from the Helen Keller text data.

Run: python scr_setup_and_create_data.py

Overview:
    This is the FIRST script in the RAG (Retrieval-Augmented Generation) pipeline.
    It performs the following tasks:
      1. Creates the necessary project directory structure (data/ and db/ folders).
      2. Reads a raw text file containing Helen Keller's autobiography.
      3. Parses the raw text into structured rows (id, section, subsection, text)
         by splitting on chapter headings and paragraphs.
      4. Saves the structured data as a Parquet file for efficient downstream
         consumption by DuckDB.

    The output Parquet file (data/sample_data.parquet) serves as the input for
    the next step in the pipeline: scr_load_parquet_to_duckdb.py.

Dependencies:
    - os: For filesystem operations (directory creation, file existence checks).
    - re: For regular expression-based text parsing (chapter heading detection).
    - pandas: For creating DataFrames and writing Parquet files.
"""
import os       # Standard library: file system operations (makedirs, path checks)
import re       # Standard library: regular expressions for parsing chapter headings
import pandas as pd  # Third-party: DataFrame manipulation and Parquet I/O

# ── Configuration ──────────────────────────────────────────────────────────
# INPUT_FILE: Path to the raw Helen Keller text file that will be parsed.
#             This file is expected to be in the project root directory.
INPUT_FILE   = "data_helen_Keller.txt"

# OUTPUT_FILE: Destination path for the generated Parquet file.
#              This Parquet file will be consumed by scr_load_parquet_to_duckdb.py.
OUTPUT_FILE  = "data/sample_data.parquet"

# DB_DIR: Directory where the DuckDB database file will be stored (created here
#         so downstream scripts have the directory ready).
DB_DIR       = "db"

# DATA_DIR: Directory where the Parquet output file will be saved.
DATA_DIR     = "data"

# ── Create directories ─────────────────────────────────────────────────────
def setup_directories():
    """Create the required project directories if they don't exist.

    Ensures that both the 'data/' and 'db/' directories exist.
    Uses os.makedirs with exist_ok=True so it won't raise an error
    if the directories already exist.
    """
    # Iterate over both required directories and create them
    for d in [DATA_DIR, DB_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"  Directory ready: {d}/")

# ── Parse text file into structured rows ───────────────────────────────────
def parse_text_file(filepath):
    """Parse the Helen Keller text file into rows with id, section, subsection, text.

    This function reads the raw text file, strips Project Gutenberg header/footer
    boilerplate, then splits the remaining content by chapter headings (e.g.,
    "CHAPTER I", "CHAPTER II", etc.). Each paragraph within a chapter becomes
    a separate row in the output list.

    Args:
        filepath (str): Path to the raw text file to parse.

    Returns:
        list[dict]: A list of dictionaries, each with keys:
            - 'id' (int): A unique sequential row identifier.
            - 'section' (str): The high-level section name (always "The Story of My Life").
            - 'subsection' (str): The chapter identifier (e.g., "Chapter I") or "Introduction".
            - 'text' (str): The paragraph text, whitespace-normalized.
    """
    # Read the entire file content into memory as a single string
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip the Project Gutenberg header by locating the start of the actual book text.
    # The marker "I. THE STORY OF MY LIFE" signals where the real content begins.
    # If the marker is not found, we fall back to using the entire content from the start.
    start_marker = "I. THE STORY OF MY LIFE"
    start_idx = content.find(start_marker)
    if start_idx == -1:
        start_idx = 0
    content = content[start_idx:]

    # Remove the Project Gutenberg footer/license text if present.
    # Everything after this marker is boilerplate and not part of the book.
    end_marker = "*** END OF THE PROJECT GUTENBERG"
    end_idx = content.find(end_marker)
    if end_idx != -1:
        content = content[:end_idx]

    # Split the content into chapters using a regex that matches chapter headings.
    # The pattern captures headings like "CHAPTER I", "CHAPTER XIV", etc.
    # re.split with a capturing group returns a list alternating between:
    #   [text_before_ch1, "CHAPTER I", text_of_ch1, "CHAPTER II", text_of_ch2, ...]
    chapter_splits = re.split(r'\n+(CHAPTER\s+[IVXLC]+)\n+', content)

    rows = []                        # Accumulator for all parsed document rows
    row_id = 0                       # Auto-incrementing unique row identifier
    current_section = "Introduction"  # Default subsection for pre-chapter text

    # The first element in chapter_splits is any text that appears before the
    # first "CHAPTER" heading. This is the preamble/introduction.
    preamble = chapter_splits[0].strip()
    if preamble:
        # Split the preamble into paragraphs by double-newlines
        paragraphs = [p.strip() for p in preamble.split("\n\n") if p.strip()]
        for para in paragraphs:
            # Normalize all whitespace (newlines, tabs, multiple spaces) to single spaces
            para = re.sub(r'\s+', ' ', para).strip()
            # Only include paragraphs with meaningful content (>30 chars filters
            # out short headings, blank lines, and formatting artifacts)
            if len(para) > 30:
                row_id += 1
                rows.append({
                    "id": row_id,
                    "section": "The Story of My Life",
                    "subsection": current_section,
                    "text": para,
                })

    # Process each chapter pair in the split results.
    # The list alternates: [preamble, heading1, body1, heading2, body2, ...]
    # So we step by 2 starting at index 1 to get each (heading, body) pair.
    for i in range(1, len(chapter_splits), 2):
        # Extract the chapter heading string (e.g., "CHAPTER VII")
        chapter_heading = chapter_splits[i].strip()
        # Safely extract the chapter body text; guard against index out of range
        chapter_body = chapter_splits[i + 1].strip() if i + 1 < len(chapter_splits) else ""

        # Extract the Roman numeral part (e.g., "VII") from the heading
        chapter_num = chapter_heading.replace("CHAPTER ", "")
        section = "The Story of My Life"
        subsection = f"Chapter {chapter_num}"  # e.g., "Chapter VII"

        # Split the chapter body into individual paragraphs
        paragraphs = [p.strip() for p in chapter_body.split("\n\n") if p.strip()]
        for para in paragraphs:
            # Collapse whitespace into single spaces for clean storage
            para = re.sub(r'\s+', ' ', para).strip()
            # Filter out short/insignificant text fragments
            if len(para) > 30:
                row_id += 1
                rows.append({
                    "id": row_id,
                    "section": section,
                    "subsection": subsection,
                    "text": para,
                })

    # Return the complete list of structured document rows
    return rows


def main():
    print("=" * 60)
    print("STEP 1: Setup & Create Sample Data")
    print("=" * 60)

    # 1. Create the data/ and db/ directories if they don't already exist.
    #    This ensures downstream scripts have the correct folder structure.
    print("\n[1/3] Creating project directories...")
    setup_directories()

    # 2. Verify that the raw input text file exists in the current directory.
    #    Without this file, we cannot generate the Parquet dataset.
    print(f"\n[2/3] Checking input file: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"  ERROR: '{INPUT_FILE}' not found in current directory.")
        print(f"  Please ensure the file exists before running this script.")
        return

    # 3. Parse the raw text file into structured rows and save as Parquet.
    #    The Parquet format is chosen for its columnar storage efficiency,
    #    compression, and native compatibility with DuckDB.
    print(f"\n[3/3] Parsing text and creating Parquet file...")
    rows = parse_text_file(INPUT_FILE)  # Parse text into list of dicts
    df = pd.DataFrame(rows)             # Convert to a pandas DataFrame

    # Print summary statistics about the parsed data for verification
    print(f"\n  Parsed {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  Sample rows:")
    print(df.head(3).to_string(max_colwidth=80))
    # Show a breakdown of how many rows exist per section/subsection
    print(f"\n  Sections / Subsections:")
    print(df.groupby(["section", "subsection"]).size().to_string())

    # Write the DataFrame to Parquet format using the PyArrow engine.
    # index=False prevents writing the DataFrame index as a column.
    df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    print(f"\n  Parquet file written to '{OUTPUT_FILE}'")

    # 4. Read the Parquet file back to verify it was written correctly.
    #    This round-trip check ensures data integrity before moving to the next step.
    df_check = pd.read_parquet(OUTPUT_FILE)
    print(f"  Verification: read back {len(df_check)} rows from Parquet ✓")

    print("\n" + "=" * 60)
    print("Setup complete! Next: python scr_load_parquet_to_duckdb.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
