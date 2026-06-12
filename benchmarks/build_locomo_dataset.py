#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
import numpy as np
from datetime import datetime

LOCOMO_JSON = Path("/Users/yong/projects/aperon/benchmarks/data/locomo_repo/data/locomo10.json")
OUT_DIR = Path("/Users/yong/projects/aperon/benchmarks/data/locomo")
DIM = 128

def parse_locomo_date(date_str):
    # e.g., "1:56 pm on 8 May, 2023"
    match = re.search(r'on\s+(.*)', date_str)
    if match:
        d_str = match.group(1).strip()
        # strip any extra whitespaces
        d_str = re.sub(r'\s+', ' ', d_str)
        for fmt in ("%d %B, %Y", "%d %b, %Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                dt = datetime.strptime(d_str, fmt)
                return int(dt.timestamp())
            except Exception:
                continue
    return 1683500000  # Fallback to May 2023

def clean_text(text):
    if not text:
        return ""
    return text.strip()

def build_dataset():
    print("=== STARTING LOCOMO DATASET BUILDER ===")
    if not LOCOMO_JSON.exists():
        print(f"Error: LoCoMo json file not found at {LOCOMO_JSON}")
        return

    with open(LOCOMO_JSON, "r", encoding="utf-8") as f:
        samples = json.load(f)

    raw_records = []
    raw_queries = []
    
    # Mapping to locate record_id by (scope_id, dia_id)
    dia_to_record_idx = {}
    
    record_counter = 0
    query_counter = 0

    for sample_idx, sample in enumerate(samples):
        scope_id = sample_idx + 1  # Map each of the 10 samples to a distinct scope (tenant)
        conv = sample.get("conversation", {})
        
        # 1. Parse all sessions
        sessions = sorted([k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")])
        
        for sess in sessions:
            date_time_key = f"{sess}_date_time"
            date_time_str = conv.get(date_time_key, "")
            ts = parse_locomo_date(date_time_str)
            
            utterances = conv[sess]
            for u in utterances:
                speaker = u.get("speaker", "")
                dia_id = u.get("dia_id", "")
                text = clean_text(u.get("text", ""))
                
                if not text:
                    continue
                
                rec = {
                    "record_id": record_counter,
                    "scope_id": scope_id,
                    "timestamp": ts,
                    "symbols": [f"speaker_{speaker.lower()}", f"sess_{sess}", f"dia_{dia_id.lower().replace(':', '_')}"],
                    "source_id": 1 if speaker.lower() == "caroline" else 2,
                    "confidence": 0.95,
                    "text": text,
                    "dia_id": dia_id
                }
                raw_records.append(rec)
                dia_to_record_idx[(scope_id, dia_id)] = record_counter
                record_counter += 1

        # 2. Parse all QA pairs (queries)
        qas = sample.get("qa", [])
        for qa in qas:
            question = clean_text(qa.get("question", ""))
            evidence = qa.get("evidence", [])
            category = qa.get("category", 1)
            
            if not question:
                continue
            
            # Map evidence dia_ids to actual record_ids
            expected_ids = []
            for ev_id in evidence:
                key = (scope_id, ev_id)
                if key in dia_to_record_idx:
                    expected_ids.append(dia_to_record_idx[key])
            
            if not expected_ids:
                # Evidence not found (could be in a pruned session), skip query for benchmarking accuracy
                continue
                
            raw_queries.append({
                "query_id": query_counter,
                "scope_id": scope_id,
                "text": question,
                "expected_record_ids": expected_ids,
                "category": category
            })
            query_counter += 1

    print(f"Parsed {len(raw_records)} records and {len(raw_queries)} queries.")

    # 3. TF-IDF and Random Projection
    print("Tokenizing and vectorizing text...")
    all_texts = [r["text"] for r in raw_records] + [q["text"] for q in raw_queries]
    
    stopwords = {"the", "a", "of", "and", "to", "in", "is", "that", "it", "on", "for", "with", "as", "was", "at", "by", "an", "when", "did", "what", "where", "how"}
    word_to_idx = {}
    vocab_size = 0
    tokenized_docs = []
    
    for text in all_texts:
        tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,15}\b', text)]
        tokens = [t for t in tokens if t not in stopwords]
        tokenized_docs.append(tokens)
        for t in tokens:
            if t not in word_to_idx:
                word_to_idx[t] = vocab_size
                vocab_size += 1

    print(f"Vocabulary Size: {vocab_size}")

    # Compute TF-IDF
    total_docs = len(all_texts)
    doc_freq = np.zeros(vocab_size)
    for doc in tokenized_docs:
        unique_tokens = set(doc)
        for ut in unique_tokens:
            doc_freq[word_to_idx[ut]] += 1
            
    idf = np.log((total_docs + 1) / (doc_freq + 1)) + 1.0

    # Build sparse matrix
    tfidf_matrix = []
    for doc in tokenized_docs:
        vec = np.zeros(vocab_size)
        if len(doc) > 0:
            for t in doc:
                vec[word_to_idx[t]] += 1.0
            vec = vec / len(doc)
            vec = vec * idf
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        tfidf_matrix.append(vec)
    
    tfidf_matrix = np.array(tfidf_matrix)

    # Project to 128-dim
    print(f"Projecting vectors to {DIM} dimensions...")
    np.random.seed(1234)  # Deterministic projection
    projection_matrix = np.random.normal(0.0, 1.0 / np.sqrt(DIM), (vocab_size, DIM))
    
    dense_embeddings = np.dot(tfidf_matrix, projection_matrix)
    
    embeddings_list = []
    for i in range(len(dense_embeddings)):
        norm = np.linalg.norm(dense_embeddings[i])
        if norm > 0:
            embeddings_list.append((dense_embeddings[i] / norm).tolist())
        else:
            embeddings_list.append(np.zeros(DIM).tolist())

    # Split back to records and queries
    records_embeddings = embeddings_list[:len(raw_records)]
    queries_embeddings = embeddings_list[len(raw_records):]

    # Assemble final output
    final_records = []
    for idx, r in enumerate(raw_records):
        final_records.append({
            "record_id": r["record_id"],
            "scope_id": r["scope_id"],
            "timestamp": r["timestamp"],
            "source_id": r["source_id"],
            "confidence": r["confidence"],
            "text": r["text"],
            "embedding": records_embeddings[idx],
            "symbols": r["symbols"]
        })

    final_queries = []
    for idx, q in enumerate(raw_queries):
        final_queries.append({
            "query_id": q["query_id"],
            "scope_id": q["scope_id"],
            "text": q["text"],
            "expected_record_ids": q["expected_record_ids"],
            "embedding": queries_embeddings[idx],
            "category": q["category"]
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUT_DIR / "locomo_records.json", "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2)
    with open(OUT_DIR / "locomo_queries.json", "w", encoding="utf-8") as f:
        json.dump(final_queries, f, indent=2)

    print(f"Saved {len(final_records)} records to locomo_records.json")
    print(f"Saved {len(final_queries)} queries to locomo_queries.json")
    print("=== LOCOMO DATASET BUILDER COMPLETED ===")

if __name__ == "__main__":
    build_dataset()
