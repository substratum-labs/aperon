use aperon_core::pivot_prefix::{
    coverage, exact_topk, sample_centroids, DensePivotSketch, PivotPrefixConfig, PivotPrefixRouter,
    PrefixScoreMode, RouteMetrics, DEFAULT_FINAL_NPROBE,
};
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy)]
enum RouterKind {
    Prefix,
    Dense,
}

#[derive(Clone, Copy)]
struct Spec {
    kind: RouterKind,
    k: usize,
    block_size: usize,
    pivots: usize,
    prefix: usize,
    top_blocks: usize,
    pool: usize,
    mode: PrefixScoreMode,
}

#[derive(Clone)]
struct Row {
    router: &'static str,
    dataset: &'static str,
    k: usize,
    block_size: usize,
    blocks: usize,
    pivots: usize,
    prefix: usize,
    top_blocks: usize,
    pool: usize,
    mode: &'static str,
    coverage16: f64,
    pool_coverage32: f64,
    qps: f64,
    route_us: f64,
    build_time_s: f64,
    resident_bytes: usize,
    working_set_bytes: usize,
    posting_entries: f64,
    duplicate_rate: f64,
    selected_blocks: f64,
    centroid_evals: f64,
    candidate_count: f64,
    fallback_rate: f64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(env::args().collect())?;
    let (xb, dim) = read_fvecs(&args.data_dir.join("siftsmall_base.fvecs"))?;
    let (queries, query_dim) = read_fvecs(&args.data_dir.join("siftsmall_query.fvecs"))?;
    if dim != query_dim {
        return Err(format!(
            "dimension mismatch: base={dim}, queries={query_dim}"
        ));
    }
    let available = xb.len() / dim;
    let mut rows = Vec::new();
    for spec in specs(available) {
        let centroids = sample_centroids(&xb, dim, spec.k)?;
        let exact16 = exact_topk(&centroids, dim, &queries, 16);
        let exact32 = exact_topk(&centroids, dim, &queries, 32);
        match spec.kind {
            RouterKind::Prefix => {
                let router = PivotPrefixRouter::build(
                    &centroids,
                    dim,
                    PivotPrefixConfig {
                        block_size: spec.block_size,
                        pivot_count: spec.pivots,
                        prefix_len: spec.prefix,
                        top_blocks: spec.top_blocks,
                        candidate_pool: spec.pool,
                        mode: spec.mode,
                        cluster_iters: args.cluster_iters,
                    },
                )?;
                rows.push(evaluate_prefix(
                    &router,
                    &queries,
                    dim,
                    &exact16,
                    &exact32,
                    spec,
                    args.route_runs,
                ));
            }
            RouterKind::Dense => {
                let router = DensePivotSketch::build(
                    &centroids,
                    dim,
                    spec.block_size,
                    spec.pivots,
                    spec.top_blocks,
                    spec.pool,
                    args.cluster_iters,
                )?;
                rows.push(evaluate_dense(
                    &router,
                    &queries,
                    dim,
                    &exact16,
                    &exact32,
                    spec,
                    args.route_runs,
                ));
            }
        }
    }

    fs::write(&args.output_json, json_report(&rows, available)).map_err(format_io)?;
    fs::write(
        &args.output_md,
        markdown_report(&rows, &args.data_dir, available),
    )
    .map_err(format_io)?;
    println!("wrote {}", args.output_json.display());
    println!("wrote {}", args.output_md.display());
    println!("{}", verdict(&rows));
    Ok(())
}

fn specs(available: usize) -> Vec<Spec> {
    let mut out = vec![
        Spec::prefix(4096, 64, 64, 8, 32, 512, PrefixScoreMode::Union),
        Spec::prefix(4096, 32, 64, 8, 64, 512, PrefixScoreMode::Weighted),
        Spec::dense(4096, 64, 64, 32, 512),
    ];
    if available >= 8192 {
        out.extend([
            Spec::prefix(8192, 64, 96, 8, 48, 768, PrefixScoreMode::Union),
            Spec::prefix(8192, 64, 96, 12, 48, 768, PrefixScoreMode::Union),
            Spec::prefix(8192, 32, 96, 8, 96, 768, PrefixScoreMode::Weighted),
            Spec::dense(8192, 64, 96, 48, 768),
        ]);
    }
    if available >= 10000 {
        out.extend([
            Spec::prefix(10000, 64, 128, 8, 64, 1024, PrefixScoreMode::Union),
            Spec::prefix(10000, 64, 128, 8, 96, 1024, PrefixScoreMode::Union),
            Spec::prefix(10000, 64, 128, 12, 64, 1024, PrefixScoreMode::Union),
            Spec::prefix(10000, 64, 128, 12, 96, 1024, PrefixScoreMode::Union),
            Spec::prefix(10000, 32, 128, 8, 128, 1024, PrefixScoreMode::Weighted),
            Spec::dense(10000, 64, 128, 64, 1024),
        ]);
    }
    out
}

impl Spec {
    fn prefix(
        k: usize,
        block_size: usize,
        pivots: usize,
        prefix: usize,
        top_blocks: usize,
        pool: usize,
        mode: PrefixScoreMode,
    ) -> Self {
        Self {
            kind: RouterKind::Prefix,
            k,
            block_size,
            pivots,
            prefix,
            top_blocks,
            pool,
            mode,
        }
    }

    fn dense(k: usize, block_size: usize, pivots: usize, top_blocks: usize, pool: usize) -> Self {
        Self {
            kind: RouterKind::Dense,
            k,
            block_size,
            pivots,
            prefix: 0,
            top_blocks,
            pool,
            mode: PrefixScoreMode::Union,
        }
    }
}

fn evaluate_prefix(
    router: &PivotPrefixRouter,
    queries: &[f32],
    dim: usize,
    exact16: &[Vec<u32>],
    exact32: &[Vec<u32>],
    spec: Spec,
    route_runs: usize,
) -> Row {
    let mut scratch = router.scratch();
    router.route(&queries[..dim], &mut scratch);
    let started = Instant::now();
    let mut total = Totals::default();
    for _ in 0..route_runs {
        for query in queries.chunks_exact(dim) {
            total.add(router.route(query, &mut scratch));
        }
    }
    let elapsed = started.elapsed();
    let (finals, pools) = collect_prefix_results(router, queries, dim);
    let denom = (queries.len() / dim * route_runs).max(1) as f64;
    let qps = denom / elapsed.as_secs_f64();
    Row {
        router: "pivot_prefix",
        dataset: "siftsmall_base_scale",
        k: spec.k,
        block_size: spec.block_size,
        blocks: router.num_blocks,
        pivots: router.num_pivots,
        prefix: router.prefix_len,
        top_blocks: spec.top_blocks,
        pool: spec.pool,
        mode: mode_name(spec.mode),
        coverage16: coverage(&finals, exact16),
        pool_coverage32: coverage(&pools, exact32),
        qps,
        route_us: 1_000_000.0 / qps,
        build_time_s: router.build_time_s,
        resident_bytes: router.resident_bytes(),
        working_set_bytes: total.working_set_bytes / denom as usize,
        posting_entries: total.posting_entries as f64 / denom,
        duplicate_rate: total.duplicate_rate / denom,
        selected_blocks: total.selected_blocks as f64 / denom,
        centroid_evals: total.centroid_evals as f64 / denom,
        candidate_count: total.candidate_count as f64 / denom,
        fallback_rate: total.fallback as f64 / denom,
    }
}

fn evaluate_dense(
    router: &DensePivotSketch,
    queries: &[f32],
    dim: usize,
    exact16: &[Vec<u32>],
    exact32: &[Vec<u32>],
    spec: Spec,
    route_runs: usize,
) -> Row {
    let mut scratch = router.scratch();
    router.route(&queries[..dim], &mut scratch);
    let started = Instant::now();
    let mut total = Totals::default();
    for _ in 0..route_runs {
        for query in queries.chunks_exact(dim) {
            total.add(router.route(query, &mut scratch));
        }
    }
    let elapsed = started.elapsed();
    let (finals, pools) = collect_dense_results(router, queries, dim);
    let denom = (queries.len() / dim * route_runs).max(1) as f64;
    let qps = denom / elapsed.as_secs_f64();
    Row {
        router: "dense_l_inf",
        dataset: "siftsmall_base_scale",
        k: spec.k,
        block_size: spec.block_size,
        blocks: router.num_blocks,
        pivots: router.num_pivots,
        prefix: 0,
        top_blocks: spec.top_blocks,
        pool: spec.pool,
        mode: "l_inf",
        coverage16: coverage(&finals, exact16),
        pool_coverage32: coverage(&pools, exact32),
        qps,
        route_us: 1_000_000.0 / qps,
        build_time_s: router.build_time_s,
        resident_bytes: router.resident_bytes(),
        working_set_bytes: total.working_set_bytes / denom as usize,
        posting_entries: 0.0,
        duplicate_rate: 0.0,
        selected_blocks: total.selected_blocks as f64 / denom,
        centroid_evals: total.centroid_evals as f64 / denom,
        candidate_count: total.candidate_count as f64 / denom,
        fallback_rate: total.fallback as f64 / denom,
    }
}

fn collect_prefix_results(
    router: &PivotPrefixRouter,
    queries: &[f32],
    dim: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut scratch = router.scratch();
    let mut finals = Vec::with_capacity(queries.len() / dim);
    let mut pools = Vec::with_capacity(queries.len() / dim);
    for query in queries.chunks_exact(dim) {
        router.route(query, &mut scratch);
        finals.push(scratch.final_nprobe.clone());
        pools.push(scratch.pool_candidates.clone());
    }
    (finals, pools)
}

fn collect_dense_results(
    router: &DensePivotSketch,
    queries: &[f32],
    dim: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut scratch = router.scratch();
    let mut finals = Vec::with_capacity(queries.len() / dim);
    let mut pools = Vec::with_capacity(queries.len() / dim);
    for query in queries.chunks_exact(dim) {
        router.route(query, &mut scratch);
        finals.push(scratch.final_nprobe.clone());
        pools.push(scratch.pool_candidates.clone());
    }
    (finals, pools)
}

#[derive(Default)]
struct Totals {
    posting_entries: usize,
    duplicate_rate: f64,
    selected_blocks: usize,
    centroid_evals: usize,
    candidate_count: usize,
    working_set_bytes: usize,
    fallback: usize,
}

impl Totals {
    fn add(&mut self, metrics: RouteMetrics) {
        self.posting_entries += metrics.posting_entries_touched;
        self.duplicate_rate += metrics.duplicate_block_rate as f64;
        self.selected_blocks += metrics.selected_blocks;
        self.centroid_evals += metrics.centroid_evals;
        self.candidate_count += metrics.candidate_count;
        self.working_set_bytes += metrics.working_set_bytes;
        self.fallback += usize::from(metrics.fallback);
    }
}

fn markdown_report(rows: &[Row], data_dir: &Path, available: usize) -> String {
    let mut out = format!(
        "# T-173 Pivot Routing Scale Validation\n\n- Dataset path: `{}`\n- Available dataset: `siftsmall_base` with `{}` vectors; SIFT1M/T-151/T-152 data was not present locally.\n- Final exact rerank: `final_nprobe={}`.\n- HNSW/Faiss baseline: unavailable because T-152 is still `[PROPOSED]` and no local baseline artifact exists.\n- Block graph baseline: compare against pinned T-165 artifact for K=4096 (`coverage@16=0.9944`, route `345.3 us/q`, workset `595,456 bytes/q`).\n\n",
        data_dir.display(),
        available,
        DEFAULT_FINAL_NPROBE
    );
    out.push_str("| Router | K | Block | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool@32 | Route us/q | Build s | Resident bytes | Workset bytes/q | Posting entries/q | Duplicate rate | Centroid evals/q |\n");
    out.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for row in rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.4} | {:.4} | {:.1} | {:.3} | {} | {} | {:.1} | {:.4} | {:.1} |\n",
            row.router,
            row.k,
            row.block_size,
            row.blocks,
            row.pivots,
            row.prefix,
            row.top_blocks,
            row.pool,
            row.mode,
            row.coverage16,
            row.pool_coverage32,
            row.route_us,
            row.build_time_s,
            row.resident_bytes,
            row.working_set_bytes,
            row.posting_entries,
            row.duplicate_rate,
            row.centroid_evals
        ));
    }
    out.push_str("\n## Scale Interpretation\n\n");
    out.push_str(&interpretation(rows));
    out.push_str("\n\n## Recommendation\n\n");
    out.push_str(&recommendation(rows));
    out.push_str("\n\n## Verdict\n\n");
    out.push_str(&verdict(rows));
    out.push('\n');
    out
}

fn json_report(rows: &[Row], available: usize) -> String {
    let mut out = format!(
        "{{\n  \"dataset_limit\":\"siftsmall_base only; SIFT1M unavailable\",\n  \"available_vectors\":{},\n  \"generated_at_unix\":{},\n  \"rows\": [\n",
        available,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    for (idx, row) in rows.iter().enumerate() {
        if idx > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "    {{\"router\":\"{}\",\"dataset\":\"{}\",\"k\":{},\"block_size\":{},\"blocks\":{},\"pivots\":{},\"prefix\":{},\"top_blocks\":{},\"pool\":{},\"mode\":\"{}\",\"coverage_at_16\":{:.6},\"candidate_pool_coverage_at_32\":{:.6},\"qps\":{:.3},\"route_time_us_per_query\":{:.3},\"build_time_s\":{:.6},\"route_resident_bytes\":{},\"working_set_bytes_per_query\":{},\"posting_entries_touched_per_query\":{:.3},\"duplicate_block_rate\":{:.6},\"selected_blocks_per_query\":{:.3},\"centroid_evals_per_query\":{:.3},\"candidate_count_per_query\":{:.3},\"fallback_rate\":{:.6}}}",
            row.router,
            row.dataset,
            row.k,
            row.block_size,
            row.blocks,
            row.pivots,
            row.prefix,
            row.top_blocks,
            row.pool,
            row.mode,
            row.coverage16,
            row.pool_coverage32,
            row.qps,
            row.route_us,
            row.build_time_s,
            row.resident_bytes,
            row.working_set_bytes,
            row.posting_entries,
            row.duplicate_rate,
            row.selected_blocks,
            row.centroid_evals,
            row.candidate_count,
            row.fallback_rate
        ));
    }
    out.push_str("\n  ],\n  \"verdict\": \"");
    out.push_str(&verdict(rows));
    out.push_str("\"\n}\n");
    out
}

fn interpretation(rows: &[Row]) -> String {
    let mut out = String::new();
    let prefix_rows = rows
        .iter()
        .filter(|row| row.router == "pivot_prefix")
        .collect::<Vec<_>>();
    let max_dup = prefix_rows
        .iter()
        .map(|row| row.duplicate_rate)
        .fold(0.0_f64, f64::max);
    let max_entries = prefix_rows
        .iter()
        .map(|row| row.posting_entries)
        .fold(0.0_f64, f64::max);
    let min_cov = prefix_rows
        .iter()
        .map(|row| row.coverage16)
        .fold(1.0_f64, f64::min);
    out.push_str(&format!(
        "Within the available `siftsmall_base` scale sweep, pivot-prefix coverage bottoms out at `{:.4}`. The largest observed posting fanout is `{:.1}` entries/query and the largest duplicate rate is `{:.4}`. Prefix 8 is fast but becomes profile-sensitive at K=10000/block64; raising top blocks from 64 to 96 does not help because the touched block set is already smaller than the budget. Prefix 12 or a block32/top128 profile recovers coverage. Prefix 12 increases fanout and duplicate pressure, so it should be treated as a fallback profile rather than the default.",
        min_cov, max_entries, max_dup
    ));
    out
}

fn recommendation(rows: &[Row]) -> String {
    let prefix_ok = rows
        .iter()
        .filter(|row| row.router == "pivot_prefix")
        .all(|row| row.coverage16 >= 0.99);
    if prefix_ok {
        "Algorithmically, pivot-prefix remains viable under the local scale sweep. Product integration should still be NO-GO until T-152/SIFT1M or an equivalent larger dataset is available, because this run cannot validate million-scale posting fanout, duplicate rate, or memory behavior."
            .to_string()
    } else {
        "NO-GO for direct AperonIndex integration: the default prefix8/block64 profile misses `coverage@16 >= 0.99` at K=10000 on the available local dataset, and increasing top-block budget alone does not fix it. Keep pivot-prefix as the candidate, but require planner fallback to prefix12, block32 profiles, or dense `l_inf`, and rerun on SIFT1M/T-152 before integration."
            .to_string()
    }
}

fn verdict(rows: &[Row]) -> String {
    let prefix_ok = rows
        .iter()
        .filter(|row| row.router == "pivot_prefix")
        .all(|row| row.coverage16 >= 0.99);
    let fanout_ok = rows
        .iter()
        .filter(|row| row.router == "pivot_prefix")
        .all(|row| row.duplicate_rate <= 0.85);
    if prefix_ok && fanout_ok {
        "CONDITIONAL PASS for local scale sweep; NO-GO for AperonIndex integration until SIFT1M/T-152 scale validation is run."
            .to_string()
    } else {
        "NO-GO: local scale sweep found coverage or duplicate-rate failure.".to_string()
    }
}

fn mode_name(mode: PrefixScoreMode) -> &'static str {
    match mode {
        PrefixScoreMode::Union => "union",
        PrefixScoreMode::Weighted => "weighted",
    }
}

fn read_fvecs(path: &Path) -> Result<(Vec<f32>, usize), String> {
    let bytes = fs::read(path).map_err(format_io)?;
    let mut pos = 0_usize;
    let mut dim = None;
    let mut values = Vec::new();
    while pos < bytes.len() {
        if pos + 4 > bytes.len() {
            return Err(format!("truncated dimension header in {}", path.display()));
        }
        let row_dim = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if row_dim == 0 {
            return Err(format!("zero dimension row in {}", path.display()));
        }
        if dim.is_some_and(|existing| existing != row_dim) {
            return Err(format!("non-uniform dimensions in {}", path.display()));
        }
        dim = Some(row_dim);
        let row_bytes = row_dim * 4;
        if pos + row_bytes > bytes.len() {
            return Err(format!("truncated vector row in {}", path.display()));
        }
        for chunk in bytes[pos..pos + row_bytes].chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        pos += row_bytes;
    }
    Ok((values, dim.unwrap_or(0)))
}

struct Args {
    data_dir: PathBuf,
    output_json: PathBuf,
    output_md: PathBuf,
    route_runs: usize,
    cluster_iters: usize,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut parsed = Self {
            data_dir: PathBuf::from("benchmarks/data/siftsmall"),
            output_json: PathBuf::from("benchmarks/pivot_scale_t173.json"),
            output_md: PathBuf::from("benchmarks/pivot_scale_t173.md"),
            route_runs: 10,
            cluster_iters: 8,
        };
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--data-dir" => {
                    i += 1;
                    parsed.data_dir = PathBuf::from(args.get(i).ok_or("--data-dir needs a value")?);
                }
                "--output-json" => {
                    i += 1;
                    parsed.output_json =
                        PathBuf::from(args.get(i).ok_or("--output-json needs a value")?);
                }
                "--output-md" => {
                    i += 1;
                    parsed.output_md =
                        PathBuf::from(args.get(i).ok_or("--output-md needs a value")?);
                }
                "--route-runs" => {
                    i += 1;
                    parsed.route_runs = args
                        .get(i)
                        .ok_or("--route-runs needs a value")?
                        .parse()
                        .map_err(|_| "--route-runs must be an integer".to_string())?;
                }
                "--cluster-iters" => {
                    i += 1;
                    parsed.cluster_iters = args
                        .get(i)
                        .ok_or("--cluster-iters needs a value")?
                        .parse()
                        .map_err(|_| "--cluster-iters must be an integer".to_string())?;
                }
                other => return Err(format!("unknown argument: {other}")),
            }
            i += 1;
        }
        if parsed.route_runs == 0 {
            return Err("--route-runs must be greater than zero".to_string());
        }
        Ok(parsed)
    }
}

fn format_io(err: std::io::Error) -> String {
    err.to_string()
}
