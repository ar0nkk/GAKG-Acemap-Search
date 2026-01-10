import pandas as pd
import numpy as np
import os
import glob
from config import CLEAN_MIN_DEGREE, CLEAN_MAX_TOKEN_LEN, DATA_DIR

def clean_gakg(gakg_df, min_degree: int = CLEAN_MIN_DEGREE, max_token_len: int = CLEAN_MAX_TOKEN_LEN):
    """Clean GAKG edges to reduce noise.

    Steps:
    1) Normalize to lowercase/stripped strings.
    2) Drop empty tokens and self-loops.
    3) Remove obviously noisy tokens by length (too short/long).
    4) Deduplicate edges.
    5) Prune low-degree nodes (keep edges where both endpoints have degree >= min_degree).
    """

    if gakg_df is None or len(gakg_df) == 0:
        return gakg_df

    print("Cleaning start...")
    df = gakg_df.copy()
    
    # 1. Normalize
    print("- Normalizing strings...")
    df['subject'] = df['subject'].astype(str).str.strip().str.lower()
    df['object'] = df['object'].astype(str).str.strip().str.lower()

    before_edges = len(df)

    # 2. Drop empty and self-loops
    print("- Removing self-loops...")
    df = df[(df['subject'] != '') & (df['object'] != '')]
    df = df[df['subject'] != df['object']]

    # 3. Length-based noise filter
    print("- Filtering by length...")
    df = df[(df['subject'].str.len() >= 2) & (df['object'].str.len() >= 2)]
    df = df[(df['subject'].str.len() <= max_token_len) & (df['object'].str.len() <= max_token_len)]

    # 4. Advanced Noise Filtering
    print("- Advanced regex filtering (removing numbers/symbols)...")
    # Reject entities starting with a number
    df = df[~df['subject'].str.match(r'^\d')]
    df = df[~df['object'].str.match(r'^\d')]

    # Reject entities containing non-standard concept characters
    valid_pattern = r'^[\w\s\.\-\']+$'
    df = df[df['subject'].str.match(valid_pattern)]
    df = df[df['object'].str.match(valid_pattern)]

    # Reject pure numbers
    df = df[~df['subject'].str.isnumeric()]
    df = df[~df['object'].str.isnumeric()]

    # Stopword Filtering (New)
    print("- Removing generic stopwords (area, region, system, etc.)...")
    STOP_CONCEPTS = {
        "area", "region", "system", "model", "use", "using", "used", "based", 
        "study", "analysis", "method", "result", "data", "time", "year", "new", 
        "paper", "research", "example", "case", "approach", "effect", "process", 
        "information", "development", "different", "high", "low", "type", "level", 
        "value", "number", "test", "first", "two", "well", "also", "one", "may", 
        "usa", "china", "world", "application", "problem", "solution", "types", 
        "part", "work", "significant", "found", "showed", "observed", "results",
        "methods", "studies", "models", "systems", "areas", "regions", 
        "nfg", "eia", "1 mgal accuracy" # Specific noise seen by user
    }
    df = df[~df['subject'].isin(STOP_CONCEPTS)]
    df = df[~df['object'].isin(STOP_CONCEPTS)]

    # 5. Deduplicate edges
    print("- Deduplicating...")
    df = df.drop_duplicates(subset=['subject', 'relation', 'object'])

    # 6. Degree-based pruning
    print(f"- Pruning nodes with degree < {min_degree}...")
    degree = df['subject'].value_counts()
    degree = degree.add(df['object'].value_counts(), fill_value=0)
    keep_mask = (df['subject'].map(degree) >= min_degree) & (df['object'].map(degree) >= min_degree)
    df = df[keep_mask].reset_index(drop=True)

    after_edges = len(df)
    print(f"Cleaned GAKG: edges {before_edges} -> {after_edges}")

    return df

def process_data():
    print(f"Looking for raw GAKG file in {DATA_DIR}...")
    # Find any parquet file that is NOT the output cleaned file
    raw_files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.parquet")) 
                 if "cleaned" not in f]
    
    if not raw_files:
        print("No raw parquet file found.")
        return

    # Use the first found file
    raw_file = raw_files[0]
    print(f"Found raw file: {os.path.basename(raw_file)}. Loading...")
    
    try:
        combined_df = pd.read_parquet(raw_file, columns=['subject', 'relation', 'object'])
    except Exception as e:
        print(f"Error loading {raw_file}: {e}")
        return

    cleaned_df = clean_gakg(combined_df)
    
    output_path = os.path.join(DATA_DIR, "gakg_cleaned.parquet")
    print(f"Saving cleaned dataset to {output_path}...")
    cleaned_df.to_parquet(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    process_data()
