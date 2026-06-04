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
    category: &'static str,
    description: &'static str,
    required: bool,
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
    vector_candidates: Option<usize>,
    correct: usize,
    queries: usize,
}

#[derive(Debug, Serialize)]
struct BenchSummary<'a> {
    schema_version: u32,
    benchmark: &'static str,
    scenario: BenchScenarioInfo<'a>,
    artifacts: BenchArtifacts<'a>,
    rows: Vec<BenchMetricRow<'a>>,
}

#[derive(Debug, Serialize)]
struct BenchScenarioInfo<'a> {
    name: &'a str,
    category: &'a str,
    required: bool,
    description: &'a str,
    records: usize,
    queries: usize,
    segments: usize,
    embedding_dim: usize,
}

#[derive(Debug, Serialize)]
struct BenchArtifacts<'a> {
    out_dir: &'a str,
    records_jsonl: &'a str,
    manifest: &'a str,
    child_manifest: &'a str,
}

#[derive(Debug, Serialize)]
struct BenchMetricRow<'a> {
    schema_version: u32,
    benchmark: &'static str,
    scenario: &'a str,
    scenario_category: &'a str,
    required_scenario: bool,
    path: &'static str,
    access_path: &'static str,
    records: usize,
    queries: usize,
    build_ms: Option<f64>,
    bytes: Option<u64>,
    latency_us_per_query: f64,
    semantic_evals_per_query: f64,
    filter_candidates_per_query: Option<f64>,
    symbol_candidates_per_query: Option<f64>,
    vector_candidates_per_query: Option<f64>,
    correct: usize,
    fork_ms: Option<f64>,
    fork_bytes: Option<u64>,
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
    for scenario in required_scenarios()? {
        run_scenario(&scenario, &args.out.join(&scenario.name))?;
    }

    let small = synthetic_scenario(
        "synthetic-small",
        "scale-smoke",
        "small deterministic synthetic comparison for fast local smoke runs",
        1_000,
        10,
        args.queries.min(20).max(10),
        16,
    );
    run_scenario(&small, &args.out.join("synthetic-small"))?;

    if args.medium || args.records != 100_000 || args.segments != 100 || args.queries != 100 {
        let synthetic = synthetic_scenario(
            "synthetic-custom",
            "scale-custom",
            "caller-sized deterministic synthetic comparison",
            args.records,
            args.segments,
            args.queries,
            16,
        );
        run_scenario(&synthetic, &args.out.join("synthetic-custom"))?;
    } else {
        let medium = synthetic_scenario(
            "synthetic-medium",
            "scale-medium",
            "default deterministic synthetic comparison",
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

    write_machine_readable_summary(
        scenario,
        out,
        &jsonl_path,
        &manifest_path,
        &fork_path,
        &rows,
    )?;

    println!();
    println!(
        "scenario={} records={} queries={} out={}",
        scenario.name,
        scenario.records.len(),
        scenario.queries.len(),
        out.display()
    );
    println!(
        "{:<22} {:<48} {:>11} {:>11} {:>11} {:>11} {:>12} {:>12} {:>12} {:>10} {:>11} {:>10}",
        "path",
        "access_path",
        "build_ms",
        "bytes",
        "lat_us/q",
        "sem/q",
        "filter/q",
        "symbol/q",
        "vector/q",
        "correct",
        "fork_ms",
        "fork_b"
    );
    for row in rows {
        println!(
            "{:<22} {:<48} {:>11} {:>11} {:>11} {:>11} {:>12} {:>12} {:>12} {:>10} {:>11} {:>10}",
            row.path,
            row.access_path,
            fmt_duration_ms(row.build_time),
            fmt_bytes(row.bytes),
            duration_us_per_query(row.total_latency, row.queries),
            row.semantic_evals / row.queries.max(1),
            fmt_avg(row.filter_candidates, row.queries),
            fmt_avg(row.symbol_candidates, row.queries),
            fmt_avg(row.vector_candidates, row.queries),
            format!("{}/{}", row.correct, row.queries),
            fmt_duration_ms(row.fork_time),
            fmt_bytes(row.fork_bytes)
        );
    }

    Ok(())
}

fn write_machine_readable_summary(
    scenario: &Scenario,
    out: &Path,
    records_jsonl: &Path,
    manifest: &Path,
    child_manifest: &Path,
    rows: &[PathMetrics],
) -> Result<(), String> {
    let out_dir = out.display().to_string();
    let records_jsonl = records_jsonl.display().to_string();
    let manifest = manifest.display().to_string();
    let child_manifest = child_manifest.display().to_string();
    let metric_rows = rows
        .iter()
        .map(|row| metric_row(scenario, row))
        .collect::<Vec<_>>();
    let summary = BenchSummary {
        schema_version: 1,
        benchmark: "memory-sstable-five-layer-prep",
        scenario: BenchScenarioInfo {
            name: &scenario.name,
            category: scenario.category,
            required: scenario.required,
            description: scenario.description,
            records: scenario.records.len(),
            queries: scenario.queries.len(),
            segments: scenario_segment_count(scenario),
            embedding_dim: scenario_embedding_dim(scenario),
        },
        artifacts: BenchArtifacts {
            out_dir: &out_dir,
            records_jsonl: &records_jsonl,
            manifest: &manifest,
            child_manifest: &child_manifest,
        },
        rows: metric_rows,
    };

    let summary_text = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("serialize summary json: {err}"))?;
    fs::write(out.join("summary.json"), format!("{summary_text}\n"))
        .map_err(|err| format!("write {}: {err}", out.join("summary.json").display()))?;

    let mut jsonl = String::new();
    for row in &summary.rows {
        let line =
            serde_json::to_string(row).map_err(|err| format!("serialize metrics jsonl: {err}"))?;
        jsonl.push_str(&line);
        jsonl.push('\n');
    }
    fs::write(out.join("metrics.jsonl"), jsonl)
        .map_err(|err| format!("write {}: {err}", out.join("metrics.jsonl").display()))
}

fn metric_row<'a>(scenario: &'a Scenario, row: &'a PathMetrics) -> BenchMetricRow<'a> {
    BenchMetricRow {
        schema_version: 1,
        benchmark: "memory-sstable-five-layer-prep",
        scenario: &scenario.name,
        scenario_category: scenario.category,
        required_scenario: scenario.required,
        path: row.path,
        access_path: row.access_path,
        records: scenario.records.len(),
        queries: row.queries,
        build_ms: row.build_time.map(duration_ms),
        bytes: row.bytes,
        latency_us_per_query: duration_us_per_query_value(row.total_latency, row.queries),
        semantic_evals_per_query: avg_usize(row.semantic_evals, row.queries),
        filter_candidates_per_query: row
            .filter_candidates
            .map(|value| avg_usize(value, row.queries)),
        symbol_candidates_per_query: row
            .symbol_candidates
            .map(|value| avg_usize(value, row.queries)),
        vector_candidates_per_query: row
            .vector_candidates
            .map(|value| avg_usize(value, row.queries)),
        correct: row.correct,
        fork_ms: row.fork_time.map(duration_ms),
        fork_bytes: row.fork_bytes,
    }
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
    let mut vector_candidates = 0;
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
                vector_candidates += trace.vector_candidates;
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
        access_path:
            "metadata filters + symbol postings + flat vector candidates + semantic rerank",
        build_time: Some(build_time),
        bytes: Some(bytes),
        fork_time: Some(fork_time),
        fork_bytes: Some(fork_bytes),
        total_latency,
        semantic_evals,
        filter_candidates: Some(filter_candidates),
        symbol_candidates: Some(symbol_candidates),
        vector_candidates: Some(vector_candidates),
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
        vector_candidates: None,
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
        vector_candidates: None,
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
        vector_candidates: None,
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
        category: "required/tiny-prefix8",
        description: "tiny example-backed prefix8 planner fallback regression case",
        required: true,
        records,
        queries: vec![BenchQuery {
            query,
            expected_record_id: 173001,
        }],
    })
}

fn required_scenarios() -> Result<Vec<Scenario>, String> {
    Ok(vec![
        tiny_scenario()?,
        prepared_scenario(
            "metadata-selective",
            "required/metadata-selective",
            "metadata filters should prune most records before semantic rerank",
            200_001,
            vec![
                prepared_record(10, 200_001, 41, 1_700_100_010, 1, 0.97, [0.9, 0.1, 0.0, 0.0], &["metadata", "selective", "target"]),
                prepared_record(10, 200_002, 42, 1_700_100_011, 1, 0.99, [0.9, 0.1, 0.0, 0.0], &["metadata", "selective", "wrong-scope"]),
                prepared_record(11, 200_003, 41, 1_700_200_000, 1, 0.98, [0.9, 0.1, 0.0, 0.0], &["metadata", "selective", "wrong-time"]),
                prepared_record(11, 200_004, 41, 1_700_100_012, 1, 0.55, [0.9, 0.1, 0.0, 0.0], &["metadata", "selective", "low-confidence"]),
                prepared_record(12, 200_005, 7, 1_700_100_013, 2, 0.80, [0.0, 0.9, 0.0, 0.0], &["metadata", "distractor"]),
            ],
            RecallQuery {
                embedding: Some(vec![0.9, 0.1, 0.0, 0.0]),
                symbols: vec!["metadata".to_string(), "selective".to_string(), "target".to_string()],
                scope_id: Some(41),
                time_start: Some(1_700_100_000),
                time_end: Some(1_700_100_099),
                min_confidence: Some(0.90),
                limit: 3,
                candidate_budget: Some(16),
            },
        ),
        prepared_scenario(
            "symbol-selective",
            "required/symbol-selective",
            "symbol postings should isolate the target among near-identical vectors",
            210_001,
            vec![
                prepared_record(20, 210_001, 5, 1_700_210_001, 1, 0.96, [0.2, 0.8, 0.1, 0.0], &["symbol", "rare-route", "target"]),
                prepared_record(20, 210_002, 5, 1_700_210_002, 1, 0.99, [0.2, 0.8, 0.1, 0.0], &["symbol", "common-route"]),
                prepared_record(21, 210_003, 5, 1_700_210_003, 1, 0.98, [0.2, 0.8, 0.1, 0.0], &["symbol", "other-route"]),
                prepared_record(21, 210_004, 6, 1_700_210_004, 1, 0.95, [0.1, 0.7, 0.2, 0.0], &["symbol", "rare-route", "other-scope"]),
            ],
            RecallQuery {
                embedding: Some(vec![0.2, 0.8, 0.1, 0.0]),
                symbols: vec!["rare-route".to_string(), "target".to_string()],
                scope_id: Some(5),
                time_start: None,
                time_end: None,
                min_confidence: Some(0.90),
                limit: 3,
                candidate_budget: Some(8),
            },
        ),
        prepared_scenario(
            "broad-semantic",
            "required/broad-semantic",
            "broad semantic query with no metadata or symbol filters",
            220_001,
            vec![
                prepared_record(30, 220_001, 1, 1_700_220_001, 1, 0.92, [0.6, 0.6, 0.0, 0.0], &["semantic", "target"]),
                prepared_record(30, 220_002, 2, 1_700_220_002, 1, 0.95, [0.2, 0.9, 0.0, 0.0], &["semantic", "near"]),
                prepared_record(31, 220_003, 3, 1_700_220_003, 1, 0.95, [-0.5, 0.1, 0.0, 0.0], &["semantic", "far"]),
                prepared_record(31, 220_004, 4, 1_700_220_004, 1, 0.95, [0.0, -0.7, 0.0, 0.0], &["semantic", "far"]),
            ],
            RecallQuery {
                embedding: Some(vec![0.6, 0.6, 0.0, 0.0]),
                symbols: Vec::new(),
                scope_id: None,
                time_start: None,
                time_end: None,
                min_confidence: None,
                limit: 3,
                candidate_budget: Some(32),
            },
        ),
        prepared_scenario(
            "branch-fork",
            "required/branch-fork",
            "branch/fork fixture reserves stable records for manifest fork measurements",
            230_001,
            vec![
                prepared_record(40, 230_001, 12, 1_700_230_001, 1, 0.98, [0.0, 0.4, 0.8, 0.0], &["branch", "fork", "child-ready"]),
                prepared_record(40, 230_002, 12, 1_700_230_002, 1, 0.93, [0.0, 0.3, 0.7, 0.0], &["branch", "main"]),
                prepared_record(41, 230_003, 13, 1_700_230_003, 2, 0.91, [0.1, 0.0, 0.5, 0.2], &["branch", "sibling"]),
            ],
            RecallQuery {
                embedding: Some(vec![0.0, 0.4, 0.8, 0.0]),
                symbols: vec!["branch".to_string(), "fork".to_string()],
                scope_id: Some(12),
                time_start: None,
                time_end: None,
                min_confidence: Some(0.90),
                limit: 3,
                candidate_budget: Some(8),
            },
        ),
        prepared_scenario(
            "adversarial",
            "required/adversarial",
            "adversarial fixture with duplicate-like vectors and misleading high confidence distractors",
            240_001,
            vec![
                prepared_record(50, 240_001, 3, 1_700_240_001, 1, 0.96, [0.7, -0.1, 0.2, 0.0], &["adversarial", "target", "exact-symbol"]),
                prepared_record(50, 240_002, 3, 1_700_240_002, 1, 0.99, [0.7, -0.1, 0.2, 0.0], &["adversarial", "decoy"]),
                prepared_record(51, 240_003, 3, 1_700_240_003, 1, 0.99, [0.7, -0.1, 0.2, 0.0], &["adversarial", "exact-symbol", "wrong-token"]),
                prepared_record(51, 240_004, 4, 1_700_240_004, 1, 1.00, [0.7, -0.1, 0.2, 0.0], &["target", "exact-symbol"]),
            ],
            RecallQuery {
                embedding: Some(vec![0.7, -0.1, 0.2, 0.0]),
                symbols: vec!["adversarial".to_string(), "target".to_string(), "exact-symbol".to_string()],
                scope_id: Some(3),
                time_start: None,
                time_end: None,
                min_confidence: Some(0.90),
                limit: 3,
                candidate_budget: Some(8),
            },
        ),
        prepared_scenario(
            "fallback",
            "required/fallback",
            "fallback fixture exercises recall without an embedding vector",
            250_001,
            vec![
                prepared_record(60, 250_001, 9, 1_700_250_001, 1, 0.98, [0.1, 0.1, 0.1, 0.9], &["fallback", "metadata-only", "target"]),
                prepared_record(60, 250_002, 9, 1_700_250_002, 1, 0.90, [0.8, 0.1, 0.1, 0.0], &["fallback", "metadata-only"]),
                prepared_record(61, 250_003, 10, 1_700_250_003, 1, 0.99, [0.1, 0.1, 0.1, 0.9], &["fallback", "target"]),
            ],
            RecallQuery {
                embedding: None,
                symbols: vec!["fallback".to_string(), "metadata-only".to_string(), "target".to_string()],
                scope_id: Some(9),
                time_start: None,
                time_end: None,
                min_confidence: Some(0.95),
                limit: 3,
                candidate_budget: Some(8),
            },
        ),
    ])
}

fn prepared_scenario(
    name: &str,
    category: &'static str,
    description: &'static str,
    expected_record_id: u64,
    records: Vec<BenchRecord>,
    query: RecallQuery,
) -> Scenario {
    Scenario {
        name: name.to_string(),
        category,
        description,
        required: true,
        records,
        queries: vec![BenchQuery {
            query,
            expected_record_id,
        }],
    }
}

fn prepared_record(
    segment_id: u64,
    record_id: u64,
    scope_id: u32,
    timestamp: i64,
    source_id: u16,
    confidence: f32,
    embedding: [f32; 4],
    symbols: &[&str],
) -> BenchRecord {
    BenchRecord {
        segment_id,
        record: MemoryRecordInput {
            record_id,
            scope_id,
            timestamp,
            source_id,
            confidence,
            text: format!("prepared memory benchmark record {record_id}"),
            embedding: embedding.to_vec(),
            symbols: symbols.iter().map(|symbol| (*symbol).to_string()).collect(),
        },
    }
}

fn synthetic_scenario(
    name: &str,
    category: &'static str,
    description: &'static str,
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
        category,
        description,
        required: false,
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
    format!("{:.1}", duration_us_per_query_value(duration, queries))
}

fn duration_us_per_query_value(duration: Duration, queries: usize) -> f64 {
    duration.as_secs_f64() * 1_000_000.0 / queries.max(1) as f64
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn avg_usize(value: usize, queries: usize) -> f64 {
    value as f64 / queries.max(1) as f64
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

fn scenario_segment_count(scenario: &Scenario) -> usize {
    scenario
        .records
        .iter()
        .map(|record| record.segment_id)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
}

fn scenario_embedding_dim(scenario: &Scenario) -> usize {
    scenario
        .records
        .first()
        .map(|record| record.record.embedding.len())
        .unwrap_or(0)
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
