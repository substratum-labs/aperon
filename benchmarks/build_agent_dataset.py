#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
import numpy as np
from datetime import datetime

BRAIN_DIR = Path("/Users/yong/.gemini/antigravity-cli/brain")
DATA_DIR = Path("/Users/yong/projects/aperon/benchmarks/data/agent_memory")
MAX_RECORDS = 20000  # Cap the dataset size for benchmarking speed
DIM = 128

def parse_iso_timestamp(ts_str):
    try:
        # e.g., "2026-06-12T06:58:52Z"
        dt = datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        return int(dt.timestamp())
    except Exception:
        return 1781254732  # Default mockup epoch timestamp if parse fails

def clean_text(text):
    if not text:
        return ""
    # Remove markdown formatting and long code blocks/logs to keep text concise
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)  # Remove XML/HTML-like tags
    return text.strip()

def extract_symbols(step):
    symbols = []
    # Extract file extensions or paths from tool calls if present
    if "tool_calls" in step and step["tool_calls"]:
        for tc in step["tool_calls"]:
            args = tc.get("args", {})
            for key, val in args.items():
                if isinstance(val, str) and ("/" in val or "." in val):
                    # Extract file name or suffix as a symbol
                    basename = os.path.basename(val.strip('"\''))
                    if basename:
                        symbols.append(f"file_{basename.lower().replace('.', '_')}")
                        if "." in basename:
                            ext = basename.split(".")[-1]
                            symbols.append(f"ext_{ext.lower()}")
    
    # Add step type and source as symbols/categories
    if "type" in step:
        symbols.append(f"type_{step['type'].lower()}")
    if "source" in step:
        symbols.append(f"src_{step['source'].lower()}")
        
    return list(set(symbols))

def build_dataset():
    print("=== STARTING AGENT MEMORY DATASET BUILDER ===")
    
    # 1. Scan and collect transcript lines
    raw_records = []
    transcript_files = list(BRAIN_DIR.glob("**/transcript.jsonl"))
    print(f"Found {len(transcript_files)} conversation folders containing transcripts.")

    for tf in transcript_files:
        if len(raw_records) >= MAX_RECORDS:
            break
        try:
            with open(tf, "r", encoding="utf-8") as f:
                for line in f:
                    if len(raw_records) >= MAX_RECORDS:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        step = json.loads(line)
                        content = clean_text(step.get("content", ""))
                        if len(content) < 15: # Skip very short snippets/commands
                            continue
                        
                        ts = parse_iso_timestamp(step.get("created_at", ""))
                        symbols = extract_symbols(step)
                        
                        raw_records.append({
                            "text": content,
                            "timestamp": ts,
                            "symbols": symbols,
                            "source_id": 1 if step.get("source") == "USER_EXPLICIT" else 2,
                            "confidence": 0.95 if step.get("status") == "DONE" else 0.5
                        })
                    except Exception:
                        continue
        except Exception as e:
            print(f"Warning: failed to read {tf}: {e}")

    total_records = len(raw_records)
    print(f"Successfully loaded {total_records} raw transcript records.")
    
    if total_records == 0:
        print("Error: No records found. Cannot generate dataset.")
        return

    # 2. TF-IDF Vocabulary Building
    print("Tokenizing and building TF-IDF index...")
    stopwords = {"the", "a", "of", "and", "to", "in", "is", "that", "it", "on", "for", "with", "as", "was", "at", "by", "an"}
    
    word_to_idx = {}
    vocab_size = 0
    
    tokenized_docs = []
    for r in raw_records:
        tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,15}\b', r["text"])]
        tokens = [t for t in tokens if t not in stopwords]
        tokenized_docs.append(tokens)
        for t in tokens:
            if t not in word_to_idx:
                word_to_idx[t] = vocab_size
                vocab_size += 1

    print(f"Vocabulary Size: {vocab_size}")

    # Compute TF-IDF
    doc_freq = np.zeros(vocab_size)
    for doc in tokenized_docs:
        unique_tokens = set(doc)
        for ut in unique_tokens:
            doc_freq[word_to_idx[ut]] += 1
            
    idf = np.log((total_records + 1) / (doc_freq + 1)) + 1.0

    # Build sparse TF-IDF vectors
    print("Vectorizing documents...")
    tfidf_matrix = []
    for idx, doc in enumerate(tokenized_docs):
        vec = np.zeros(vocab_size)
        if len(doc) > 0:
            for t in doc:
                vec[word_to_idx[t]] += 1.0
            vec = vec / len(doc)  # L1 normalize term frequencies
            vec = vec * idf
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        tfidf_matrix.append(vec)
    
    tfidf_matrix = np.array(tfidf_matrix)

    # 3. Random Projection to 128-dim
    print(f"Projecting vectors from dimension {vocab_size} to {DIM}...")
    np.random.seed(42)  # Deterministic projection
    projection_matrix = np.random.normal(0.0, 1.0 / np.sqrt(DIM), (vocab_size, DIM))
    
    dense_embeddings = np.dot(tfidf_matrix, projection_matrix)
    
    # L2 normalize projected embeddings
    for i in range(len(dense_embeddings)):
        norm = np.linalg.norm(dense_embeddings[i])
        if norm > 0:
            dense_embeddings[i] = dense_embeddings[i] / norm
        else:
            dense_embeddings[i] = np.zeros(DIM)

    # 4. Save to JSON
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "agent_memory_dataset.json"
    
    final_dataset = []
    for i, r in enumerate(raw_records):
        final_dataset.append({
            "record_id": i,
            "scope_id": 1,
            "timestamp": r["timestamp"],
            "source_id": r["source_id"],
            "confidence": r["confidence"],
            "text": r["text"][:200],  # Truncate text snippet for storage size
            "embedding": dense_embeddings[i].tolist(),
            "symbols": r["symbols"]
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2)

    print(f"Saved {len(final_dataset)} memory records to {out_file}")
    print("=== DATASET GENERATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    build_dataset()
