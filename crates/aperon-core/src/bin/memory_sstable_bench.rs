use aperon_core::{
    distance::l2_squared_unchecked, stable_memory_branch_id, MemoryManifestFile,
    MemoryManifestSegment, MemoryRecordInput, MemorySegment, MemorySpace, RecallQuery,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant};

const MAIN_BRANCH_NAME: &str = "main";

#[derive(Clone, Debug)]
struct BenchRecord {
    segment_id: u64,
    record: MemoryRecordInput,
}

#[derive(Clone, Debug)]
struct BenchQuery {
    query: RecallQuery,
    expected_record_id: u64,
}

#[derive(Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug)]
struct Args {
    records: usize,
    segments: usize,
    queries: usize,
    out: PathBuf,
    medium: bool,
}

#[derive(Clone, Debug)]
struct Scenario {
    name: String,
    records: Vec<BenchRecord>,
    queries: Vec<BenchQuery>,
}

#[derive(Clone, Debug)]
struct PathMetrics {
    path: &'static str,
    access_path: &'static str,
    build_time: Option<Duration>,
    bytes: Option<u64>,
    fork_time: Option<Duration>,
    fork_bytes: Option<u64>,
    total_latency: Duration,
    semantic_evals: usize,
    filter_candidates: Option<usize>,
    symbol_candidates: Option<usize>,
    correct: usize,
    queries: usize,
}

#[derive(Clone, Debug)]
struct FlatHit {
    record_id: u64,
    score: f32,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let tiny = tiny_scenario()?;
    run_scenario(&tiny, &args.out.join("tiny"))?;

    let small = synthetic_scenario(
        "synthetic-small",
        1_000,
        10,
        args.queries.min(20).max(10),
        16,
    );
    run_scenario(&small, &args.out.join("synthetic-small"))?;

    if args.medium || args.records != 100_000 || args.segments != 100 || args.queries != 100 {
        let synthetic = synthetic_scenario(
            "synthetic-custom",
            args.records,
            args.segments,
            args.queries,
            16,
        );
        run_scenario(&synthetic, &args.out.join("synthetic-custom"))?;
    } else {
        let medium = synthetic_scenario(
            "synthetic-medium",
            args.records,
            args.segments,
            args.queries,
            16,
        );
        run_scenario(&medium, &args.out.join("synthetic-medium"))?;
    }

    Ok(())
}

fn run_scenario(scenario: &Scenario, out: &Path) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|err| format!("create {}: {err}", out.display()))?;
    let jsonl_path = out.join("records.jsonl");
    write_jsonl(&scenario.records, &jsonl_path)?;

    let (manifest_path, mvp_build_time, mvp_bytes) = build_mvp(&scenario.records, out)?;
    let space = MemorySpace::open(&manifest_path).map_err(|err| format!("open manifest: {err}"))?;

    let fork_path = out.join("child.apmf");
    let fork_start = Instant::now();
    space
        .fork("bench-child", &fork_path)
        .map_err(|err| format!("fork manifest: {err}"))?;
    let fork_time = fork_start.elapsed();
    let fork_bytes = file_len(&fork_path)?;

    let records = read_jsonl(&jsonl_path)?;
    let jsonl_bytes = file_len(&jsonl_path)?;

    let mut rows = Vec::new();
    rows.push(run_mvp(
        &space,
        &scenario.queries,
        mvp_build_time,
        mvp_bytes,
        fork_time,
        fork_bytes,
    )?);
    rows.push(run_naive_jsonl(
        &jsonl_path,
        &scenario.queries,
        Some(jsonl_bytes),
    )?);
    rows.push(run_in_memory_flat(&records, &scenario.queries));
    rows.push(run_vector_only(&records, &scenario.queries));

    println!();
    println!(
        "scenario={} records={} queries={} out={}",
        scenario.name,
        scenario.records.len(),
        scenario.queries.len(),
        out.display()
    );
    println!(
        "{:<22} {:<48} {:>11} {:>11} {:>11} {:>11} {:>12} {:>12} {:>10} {:>11} {:>10}",
        "path",
        "access_path",
        "build_ms",
        "bytes",
        "lat_us/q",
        "sem/q",
        "filter/q",
        "symbol/q",
        "correct",
        "fork_ms",
        "fork_b"
    );
    for row in rows {
        println!(
            "{:<22} {:<48} {:>11} {:>11} {:>11} {:>11} {:>12} {:>12} {:>10} {:>11} {:>10}",
            row.path,
            row.access_path,
            fmt_duration_ms(row.build_time),
            fmt_bytes(row.bytes),
            duration_us_per_query(row.total_latency, row.queries),
            row.semantic_evals / row.queries.max(1),
            fmt_avg(row.filter_candidates, row.queries),
            fmt_avg(row.symbol_candidates, row.queries),
            format!("{}/{}", row.correct, row.queries),
            fmt_duration_ms(row.fork_time),
            fmt_bytes(row.fork_bytes)
        );
    }

    Ok(())
}

fn run_mvp(
    space: &MemorySpace,
    queries: &[BenchQuery],
    build_time: Duration,
    bytes: u64,
    fork_time: Duration,
    fork_bytes: u64,
) -> Result<PathMetrics, String> {
    let mut total_latency = Duration::ZERO;
    let mut semantic_evals = 0;
    let mut filter_candidates = 0;
    let mut symbol_candidates = 0;
    let mut correct = 0;

    for bench_query in queries {
        let start = Instant::now();
        let result = space.recall(&bench_query.query)?;
        total_latency += start.elapsed();
        semantic_evals += result.trace.semantic_evals;
        for segment in result.trace.segment_traces {
            if let Some(trace) = segment.trace {
                filter_candidates += trace.candidates_after_filters;
                symbol_candidates += trace.candidates_after_symbols;
            }
        }
        if result
            .hits
            .iter()
            .any(|hit| hit.record_id == bench_query.expected_record_id)
        {
            correct += 1;
        }
    }

    Ok(PathMetrics {
        path: "memory-sstable",
        access_path: "metadata filters + symbol postings + semantic rerank",
        build_time: Some(build_time),
        bytes: Some(bytes),
        fork_time: Some(fork_time),
        fork_bytes: Some(fork_bytes),
        total_latency,
        semantic_evals,
        filter_candidates: Some(filter_candidates),
        symbol_candidates: Some(symbol_candidates),
        correct,
        queries: queries.len(),
    })
}

fn run_naive_jsonl(
    jsonl_path: &Path,
    queries: &[BenchQuery],
    bytes: Option<u64>,
) -> Result<PathMetrics, String> {
    let mut total_latency = Duration::ZERO;
    let mut semantic_evals = 0;
    let mut filter_candidates = 0;
    let mut symbol_candidates = 0;
    let mut correct = 0;

    for bench_query in queries {
        let start = Instant::now();
        let records = read_jsonl(jsonl_path)?;
        let (hits, stats) = scan_records(&records, &bench_query.query, false);
        total_latency += start.elapsed();
        semantic_evals += stats.semantic_evals;
        filter_candidates += stats.filter_candidates;
        symbol_candidates += stats.symbol_candidates;
        if hits
            .iter()
            .any(|hit| hit.record_id == bench_query.expected_record_id)
        {
            correct += 1;
        }
    }

    Ok(PathMetrics {
        path: "naive-jsonl-scan",
        access_path: "full JSONL parse per query + metadata/symbol scan",
        build_time: None,
        bytes,
        fork_time: None,
        fork_bytes: None,
        total_latency,
        semantic_evals,
        filter_candidates: Some(filter_candidates),
        symbol_candidates: Some(symbol_candidates),
        correct,
        queries: queries.len(),
    })
}

fn run_in_memory_flat(records: &[BenchRecord], queries: &[BenchQuery]) -> PathMetrics {
    let build_start = Instant::now();
    let flat = records.to_vec();
    let build_time = build_start.elapsed();
    let mut total_latency = Duration::ZERO;
    let mut semantic_evals = 0;
    let mut filter_candidates = 0;
    let mut symbol_candidates = 0;
    let mut correct = 0;

    for bench_query in queries {
        let start = Instant::now();
        let (hits, stats) = scan_records(&flat, &bench_query.query, false);
        total_latency += start.elapsed();
        semantic_evals += stats.semantic_evals;
        filter_candidates += stats.filter_candidates;
        symbol_candidates += stats.symbol_candidates;
        if hits
            .iter()
            .any(|hit| hit.record_id == bench_query.expected_record_id)
        {
            correct += 1;
        }
    }

    PathMetrics {
        path: "in-memory-flat",
        access_path: "one-time Vec load + metadata/symbol scan",
        build_time: Some(build_time),
        bytes: None,
        fork_time: None,
        fork_bytes: None,
        total_latency,
        semantic_evals,
        filter_candidates: Some(filter_candidates),
        symbol_candidates: Some(symbol_candidates),
        correct,
        queries: queries.len(),
    }
}

fn run_vector_only(records: &[BenchRecord], queries: &[BenchQuery]) -> PathMetrics {
    let build_start = Instant::now();
    let flat = records.to_vec();
    let build_time = build_start.elapsed();
    let mut total_latency = Duration::ZERO;
    let mut semantic_evals = 0;
    let mut correct = 0;

    for bench_query in queries {
        let start = Instant::now();
        let (hits, stats) = scan_records(&flat, &bench_query.query, true);
        total_latency += start.elapsed();
        semantic_evals += stats.semantic_evals;
        if hits
            .iter()
            .any(|hit| hit.record_id == bench_query.expected_record_id)
        {
            correct += 1;
        }
    }

    PathMetrics {
        path: "vector-only-flat",
        access_path: "flat embedding distance; ignores memory metadata/symbols",
        build_time: Some(build_time),
        bytes: None,
        fork_time: None,
        fork_bytes: None,
        total_latency,
        semantic_evals,
        filter_candidates: None,
        symbol_candidates: None,
        correct,
        queries: queries.len(),
    }
}

#[derive(Clone, Debug, Default)]
struct ScanStats {
    filter_candidates: usize,
    symbol_candidates: usize,
    semantic_evals: usize,
}

fn scan_records(
    records: &[BenchRecord],
    query: &RecallQuery,
    vector_only: bool,
) -> (Vec<FlatHit>, ScanStats) {
    let limit = query.limit.max(1);
    let mut stats = ScanStats::default();
    let mut scored = Vec::new();

    for record in records {
        if !vector_only && !passes_filters(&record.record, query) {
            continue;
        }
        stats.filter_candidates += 1;

        let symbol_matches = if vector_only {
            0
        } else {
            symbol_match_count(&record.record.symbols, &query.symbols)
        };
        if !vector_only && symbol_matches != query.symbols.len() {
            continue;
        }
        stats.symbol_candidates += 1;

        let semantic_distance = query
            .embedding
            .as_deref()
            .map(|embedding| l2_squared_unchecked(embedding, &record.record.embedding).sqrt());
        stats.semantic_evals += usize::from(query.embedding.is_some());
        let score = if vector_only {
            semantic_distance.map_or(0.0, |dist| -dist)
        } else {
            score_record(&record.record, semantic_distance, symbol_matches)
        };
        scored.push(FlatHit {
            record_id: record.record.record_id,
            score,
        });
    }

    scored.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.record_id.cmp(&b.record_id))
    });
    scored.truncate(limit);
    (scored, stats)
}

fn build_mvp(records: &[BenchRecord], out: &Path) -> Result<(PathBuf, Duration, u64), String> {
    let build_dir = out.join("mvp");
    fs::create_dir_all(&build_dir)
        .map_err(|err| format!("create {}: {err}", build_dir.display()))?;

    let start = Instant::now();
    let mut by_segment = BTreeMap::<u64, Vec<MemoryRecordInput>>::new();
    let mut dim = None;
    for record in records {
        match dim {
            Some(expected) if expected != record.record.embedding.len() => {
                return Err("synthetic record dimension mismatch".to_string());
            }
            None => dim = Some(record.record.embedding.len()),
            _ => {}
        }
        by_segment
            .entry(record.segment_id)
            .or_default()
            .push(record.record.clone());
    }
    let dim = dim.ok_or_else(|| "scenario has no records".to_string())?;

    let mut manifest_segments = Vec::new();
    for (segment_id, segment_records) in by_segment {
        let segment = MemorySegment::build(segment_id, dim, segment_records)?;
        let file_name = format!("segment-{segment_id}.apms");
        segment
            .write(build_dir.join(&file_name))
            .map_err(|err| format!("write segment {segment_id}: {err}"))?;
        manifest_segments.push(MemoryManifestSegment {
            segment_id,
            path: PathBuf::from(file_name),
        });
    }
    let manifest = MemoryManifestFile::new(
        None,
        stable_memory_branch_id(MAIN_BRANCH_NAME),
        manifest_segments,
    );
    let manifest_path = build_dir.join("main.apmf");
    manifest
        .write(&manifest_path)
        .map_err(|err| format!("write manifest: {err}"))?;
    let build_time = start.elapsed();
    let bytes = dir_bytes(&build_dir)?;
    Ok((manifest_path, build_time, bytes))
}

fn tiny_scenario() -> Result<Scenario, String> {
    let records = read_jsonl(Path::new("examples/aperon_memory.jsonl"))?;
    let query = read_query(Path::new("examples/query_prefix8.json"))?;
    Ok(Scenario {
        name: "tiny-prefix8".to_string(),
        records,
        queries: vec![BenchQuery {
            query,
            expected_record_id: 173001,
        }],
    })
}

fn synthetic_scenario(
    name: &str,
    record_count: usize,
    segment_count: usize,
    query_count: usize,
    dim: usize,
) -> Scenario {
    let mut records = Vec::with_capacity(record_count);
    for id in 0..record_count {
        let segment_id = (id % segment_count.max(1)) as u64;
        let scope_id = (id % 32) as u32;
        let topic_id = (id / 17) % 64;
        let mut embedding = Vec::with_capacity(dim);
        for d in 0..dim {
            embedding.push(deterministic_value(id as u64, d as u64));
        }
        records.push(BenchRecord {
            segment_id,
            record: MemoryRecordInput {
                record_id: 1_000_000 + id as u64,
                scope_id,
                timestamp: 1_700_000_000 + id as i64,
                source_id: (id % 7) as u16,
                confidence: 0.70 + ((id % 30) as f32 * 0.01),
                text: format!("synthetic memory record {id} in scope {scope_id} topic {topic_id}"),
                embedding,
                symbols: vec![format!("scope-{scope_id}"), format!("topic-{topic_id}")],
            },
        });
    }

    let mut queries = Vec::with_capacity(query_count);
    for qid in 0..query_count {
        let target = (qid * 9_973 + 17) % record_count.max(1);
        let record = &records[target].record;
        queries.push(BenchQuery {
            query: RecallQuery {
                embedding: Some(record.embedding.clone()),
                symbols: record.symbols.clone(),
                scope_id: Some(record.scope_id),
                time_start: Some(record.timestamp - 32),
                time_end: Some(record.timestamp + 32),
                min_confidence: Some((record.confidence - 0.001).max(0.0)),
                limit: 10,
                candidate_budget: None,
            },
            expected_record_id: record.record_id,
        });
    }

    Scenario {
        name: name.to_string(),
        records,
        queries,
    }
}

fn deterministic_value(id: u64, dim: u64) -> f32 {
    let mut x = id
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(dim.wrapping_mul(0xbf58476d1ce4e5b9));
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    ((x % 10_000) as f32 / 5_000.0) - 1.0
}

fn passes_filters(record: &MemoryRecordInput, query: &RecallQuery) -> bool {
    if query
        .scope_id
        .is_some_and(|scope_id| record.scope_id != scope_id)
    {
        return false;
    }
    if query
        .time_start
        .is_some_and(|start| record.timestamp < start)
    {
        return false;
    }
    if query.time_end.is_some_and(|end| record.timestamp > end) {
        return false;
    }
    if query
        .min_confidence
        .is_some_and(|min| record.confidence < min)
    {
        return false;
    }
    true
}

fn symbol_match_count(record_symbols: &[String], query_symbols: &[String]) -> usize {
    query_symbols
        .iter()
        .filter(|query_symbol| {
            record_symbols
                .iter()
                .any(|symbol| symbol.eq_ignore_ascii_case(query_symbol))
        })
        .count()
}

fn score_record(
    record: &MemoryRecordInput,
    semantic_distance: Option<f32>,
    symbol_matches: usize,
) -> f32 {
    let semantic = semantic_distance.map_or(0.0, |dist| -dist);
    let symbol = symbol_matches as f32 * 2.0 + record.symbols.len() as f32 * 0.01;
    semantic + symbol + record.confidence
}

fn write_jsonl(records: &[BenchRecord], path: &Path) -> Result<(), String> {
    let mut text = String::new();
    for record in records {
        let json = JsonMemoryRecord {
            segment_id: record.segment_id,
            record_id: record.record.record_id,
            scope_id: record.record.scope_id,
            timestamp: record.record.timestamp,
            source_id: record.record.source_id,
            confidence: record.record.confidence,
            text: record.record.text.clone(),
            embedding: record.record.embedding.clone(),
            symbols: record.record.symbols.clone(),
        };
        let line = serde_json::to_string(&json).map_err(|err| format!("serialize jsonl: {err}"))?;
        text.push_str(&line);
        text.push('\n');
    }
    fs::write(path, text).map_err(|err| format!("write {}: {err}", path.display()))
}

fn read_jsonl(path: &Path) -> Result<Vec<BenchRecord>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut records = Vec::new();
    for (line_id, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: JsonMemoryRecord = serde_json::from_str(line)
            .map_err(|err| format!("parse {} line {}: {err}", path.display(), line_id + 1))?;
        records.push(BenchRecord {
            segment_id: record.segment_id,
            record: MemoryRecordInput {
                record_id: record.record_id,
                scope_id: record.scope_id,
                timestamp: record.timestamp,
                source_id: record.source_id,
                confidence: record.confidence,
                text: record.text,
                embedding: record.embedding,
                symbols: record.symbols,
            },
        });
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

fn dir_bytes(path: &Path) -> Result<u64, String> {
    let mut total = 0;
    for entry in fs::read_dir(path).map_err(|err| format!("read_dir {}: {err}", path.display()))? {
        let entry = entry.map_err(|err| format!("read_dir entry {}: {err}", path.display()))?;
        let metadata = entry
            .metadata()
            .map_err(|err| format!("metadata {}: {err}", entry.path().display()))?;
        if metadata.is_file() {
            total += metadata.len();
        }
    }
    Ok(total)
}

fn file_len(path: &Path) -> Result<u64, String> {
    Ok(fs::metadata(path)
        .map_err(|err| format!("metadata {}: {err}", path.display()))?
        .len())
}

fn parse_args() -> Result<Args, String> {
    let args = env::args().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage(&args[0]);
        process::exit(0);
    }
    Ok(Args {
        records: parse_usize(&args, "--records", 100_000)?,
        segments: parse_usize(&args, "--segments", 100)?,
        queries: parse_usize(&args, "--queries", 100)?,
        out: parse_path(&args, "--out", "target/memory-sstable-bench"),
        medium: args.iter().any(|arg| arg == "--medium"),
    })
}

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize, String> {
    match args.windows(2).find(|pair| pair[0] == name) {
        Some(pair) => pair[1]
            .parse::<usize>()
            .map_err(|err| format!("invalid {name}: {err}")),
        None => Ok(default),
    }
}

fn parse_path(args: &[String], name: &str, default: &str) -> PathBuf {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| PathBuf::from(&pair[1]))
        .unwrap_or_else(|| PathBuf::from(default))
}

fn duration_us_per_query(duration: Duration, queries: usize) -> String {
    let per_query = duration.as_secs_f64() * 1_000_000.0 / queries.max(1) as f64;
    format!("{per_query:.1}")
}

fn fmt_duration_ms(duration: Option<Duration>) -> String {
    duration
        .map(|duration| format!("{:.2}", duration.as_secs_f64() * 1_000.0))
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_bytes(bytes: Option<u64>) -> String {
    bytes
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn fmt_avg(value: Option<usize>, queries: usize) -> String {
    value
        .map(|value| (value / queries.max(1)).to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn default_limit() -> usize {
    10
}

fn print_usage(bin: &str) {
    eprintln!("usage:");
    eprintln!("  {bin} [--records 100000] [--segments 100] [--queries 100] [--out target/memory-sstable-bench] [--medium]");
    eprintln!("examples:");
    eprintln!("  cargo run -p aperon-core --bin memory_sstable_bench -- --records 100000 --segments 100 --queries 100");
}
