use aperon_core::{
    stable_memory_branch_id, MemoryManifestFile, MemoryManifestSegment, MemoryRecordInput,
    MemorySegment, MemorySpace, RecallQuery,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const MAIN_BRANCH_NAME: &str = "main";

#[derive(Debug, Deserialize)]
struct JsonMemoryRecord {
    segment_id: u64,
    record_id: u64,
    scope_id: u32,
    timestamp: i64,
    source_id: u16,
    confidence: f32,
    text: String,
    embedding: Vec<f32>,
    symbols: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct JsonRecallQuery {
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    symbols: Vec<String>,
    scope_id: Option<u32>,
    time_start: Option<i64>,
    time_end: Option<i64>,
    min_confidence: Option<f32>,
    #[serde(default = "default_limit")]
    limit: usize,
    candidate_budget: Option<usize>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    let command = args.get(1).map(String::as_str).unwrap_or("");
    match command {
        "build" => {
            let input = required_path(&args, "--input")?;
            let out = required_path(&args, "--out")?;
            build(&input, &out)
        }
        "recall" => {
            let manifest = required_path(&args, "--manifest")?;
            let query = required_path(&args, "--query")?;
            recall(&manifest, &query)
        }
        "fork" => {
            let manifest = required_path(&args, "--manifest")?;
            let branch = required_value(&args, "--branch")?;
            let out = required_path(&args, "--out")?;
            fork(&manifest, &branch, &out)
        }
        _ => {
            print_usage(&args[0]);
            Err("missing or unsupported command".to_string())
        }
    }
}

fn build(input: &Path, out: &Path) -> Result<(), String> {
    let records = read_jsonl(input)?;
    let mut by_segment = BTreeMap::<u64, Vec<MemoryRecordInput>>::new();
    let mut dim = None;

    for record in records {
        match dim {
            Some(expected) if expected != record.embedding.len() => {
                return Err(format!(
                    "record {} embedding dimension mismatch: expected {}, got {}",
                    record.record_id,
                    expected,
                    record.embedding.len()
                ));
            }
            None => dim = Some(record.embedding.len()),
            _ => {}
        }

        by_segment
            .entry(record.segment_id)
            .or_default()
            .push(MemoryRecordInput {
                record_id: record.record_id,
                scope_id: record.scope_id,
                timestamp: record.timestamp,
                source_id: record.source_id,
                confidence: record.confidence,
                text: record.text,
                embedding: record.embedding,
                symbols: record.symbols,
            });
    }

    let dim = dim.ok_or_else(|| "input JSONL contained no records".to_string())?;
    fs::create_dir_all(out).map_err(|err| format!("create {}: {err}", out.display()))?;

    let mut manifest_segments = Vec::new();
    for (segment_id, records) in by_segment {
        let segment = MemorySegment::build(segment_id, dim, records)?;
        let file_name = format!("segment-{segment_id}.apms");
        let segment_path = out.join(&file_name);
        segment
            .write(&segment_path)
            .map_err(|err| format!("write {}: {err}", segment_path.display()))?;
        manifest_segments.push(MemoryManifestSegment {
            segment_id,
            path: PathBuf::from(file_name),
        });
        println!(
            "wrote segment id={} records={} path={}",
            segment_id,
            segment.len(),
            segment_path.display()
        );
    }

    let manifest = MemoryManifestFile::new(
        None,
        stable_memory_branch_id(MAIN_BRANCH_NAME),
        manifest_segments,
    );
    let manifest_path = out.join("main.apmf");
    manifest
        .write(&manifest_path)
        .map_err(|err| format!("write {}: {err}", manifest_path.display()))?;
    println!(
        "wrote manifest id={} branch={} path={}",
        manifest.manifest_id,
        manifest.branch_id,
        manifest_path.display()
    );
    Ok(())
}

fn recall(manifest: &Path, query: &Path) -> Result<(), String> {
    let space = MemorySpace::open(manifest).map_err(|err| format!("open manifest: {err}"))?;
    let query = read_query(query)?;
    let result = space.recall(&query)?;

    println!("Memory SSTable demo recall");
    println!(
        "manifest_id={} branch_id={} considered={} scanned={} pruned={} semantic_evals={} returned={}",
        result.trace.manifest_id,
        result.trace.branch_id,
        result.trace.segments_considered,
        result.trace.segments_scanned,
        result.trace.segments_pruned,
        result.trace.semantic_evals,
        result.trace.returned
    );
    for segment in &result.trace.segment_traces {
        if segment.pruned {
            println!(
                "trace segment_id={} pruned=true reason={}",
                segment.segment_id,
                segment.prune_reason.unwrap_or("unknown")
            );
        } else if let Some(trace) = &segment.trace {
            println!(
                "trace segment_id={} access_paths={:?} records_total={} column_filters={} symbol_postings={} vector_generator={} vector_candidates={} semantic_rerank={} returned={}",
                trace.segment_id,
                trace.access_paths,
                trace.records_total,
                trace.candidates_after_filters,
                trace.candidates_after_symbols,
                trace.vector_generator,
                trace.vector_candidates,
                trace.semantic_evals,
                trace.returned
            );
        }
    }
    for hit in result.hits {
        println!(
            "hit record_id={} score={:.3} semantic_distance={:?} symbols={} confidence={:.2} text={}",
            hit.record_id,
            hit.score,
            hit.semantic_distance,
            hit.symbol_matches,
            hit.confidence,
            hit.text
        );
    }
    Ok(())
}

fn fork(manifest: &Path, branch: &str, out: &Path) -> Result<(), String> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("create {}: {err}", parent.display()))?;
    }
    let space = MemorySpace::open(manifest).map_err(|err| format!("open manifest: {err}"))?;
    space
        .fork(branch, out)
        .map_err(|err| format!("write fork manifest: {err}"))?;
    let child =
        MemoryManifestFile::read(out).map_err(|err| format!("read fork manifest: {err}"))?;
    println!(
        "forked manifest parent_id={} child_id={} branch={} branch_id={} out={}",
        space.manifest.manifest_id,
        child.manifest_id,
        branch,
        child.branch_id,
        out.display()
    );
    Ok(())
}

fn read_jsonl(path: &Path) -> Result<Vec<JsonMemoryRecord>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut records = Vec::new();
    for (line_id, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record = serde_json::from_str(line)
            .map_err(|err| format!("parse {} line {}: {err}", path.display(), line_id + 1))?;
        records.push(record);
    }
    Ok(records)
}

fn read_query(path: &Path) -> Result<RecallQuery, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let query: JsonRecallQuery =
        serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))?;
    Ok(RecallQuery {
        embedding: query.embedding,
        symbols: query.symbols,
        scope_id: query.scope_id,
        time_start: query.time_start,
        time_end: query.time_end,
        min_confidence: query.min_confidence,
        limit: query.limit,
        candidate_budget: query.candidate_budget,
    })
}

fn required_path(args: &[String], name: &str) -> Result<PathBuf, String> {
    required_value(args, name).map(PathBuf::from)
}

fn required_value(args: &[String], name: &str) -> Result<String, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
        .ok_or_else(|| format!("missing required argument {name}"))
}

fn default_limit() -> usize {
    10
}

fn print_usage(bin: &str) {
    eprintln!("usage:");
    eprintln!("  {bin} build --input examples/aperon_memory.jsonl --out target/memory-demo");
    eprintln!(
        "  {bin} recall --manifest target/memory-demo/main.apmf --query examples/query_prefix8.json"
    );
    eprintln!(
        "  {bin} fork --manifest target/memory-demo/main.apmf --branch prefix12-exp --out target/memory-demo/prefix12.apmf"
    );
}
