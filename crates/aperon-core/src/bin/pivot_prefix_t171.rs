use aperon_core::pivot_prefix::{
    coverage, exact_topk, sample_centroids, DensePivotSketch, PivotPrefixConfig, PivotPrefixRouter,
    PrefixScoreMode, RouteMetrics, DEFAULT_FINAL_NPROBE,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    env, fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::Instant,
};

struct CountingAllocator;

static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct Spec {
    k: usize,
    block_size: usize,
    pivots: usize,
    prefix: usize,
    top_blocks: usize,
    pool: usize,
    mode: PrefixScoreMode,
}

struct Row {
    router: &'static str,
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
    pivot_evals: f64,
    posting_entries: f64,
    duplicate_rate: f64,
    selected_blocks: f64,
    centroid_evals: f64,
    candidate_count: f64,
    resident_bytes: usize,
    working_set_bytes: usize,
    build_time_s: f64,
    fallback_rate: f64,
    hot_allocations: usize,
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

    let mut rows = Vec::new();
    for spec in specs(args.no_sensitivity) {
        let centroids = sample_centroids(&xb, dim, spec.k)?;
        let exact16 = exact_topk(&centroids, dim, &queries, 16);
        let exact32 = exact_topk(&centroids, dim, &queries, 32);
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

    let dense_spec = Spec {
        k: 4096,
        block_size: 64,
        pivots: 64,
        prefix: 0,
        top_blocks: 32,
        pool: 512,
        mode: PrefixScoreMode::Union,
    };
    let dense_centroids = sample_centroids(&xb, dim, dense_spec.k)?;
    let dense_exact16 = exact_topk(&dense_centroids, dim, &queries, 16);
    let dense_exact32 = exact_topk(&dense_centroids, dim, &queries, 32);
    let dense = DensePivotSketch::build(
        &dense_centroids,
        dim,
        dense_spec.block_size,
        dense_spec.pivots,
        dense_spec.top_blocks,
        dense_spec.pool,
        args.cluster_iters,
    )?;
    rows.push(evaluate_dense(
        &dense,
        &queries,
        dim,
        &dense_exact16,
        &dense_exact32,
        dense_spec,
        args.route_runs,
    ));

    fs::write(&args.output_json, json_report(&rows)).map_err(format_io)?;
    fs::write(&args.output_md, markdown_report(&rows, &args.data_dir)).map_err(format_io)?;
    println!("wrote {}", args.output_json.display());
    println!("wrote {}", args.output_md.display());
    println!("{}", verdict(&rows));
    Ok(())
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

    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    let started = Instant::now();
    let mut total = MetricTotals::default();
    for _ in 0..route_runs {
        for query in queries.chunks_exact(dim) {
            total.add(router.route(query, &mut scratch));
        }
    }
    let elapsed = started.elapsed();
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);
    let hot_allocations = ALLOCATION_COUNT.load(Ordering::Relaxed);

    let (finals, pools) = collect_prefix_results(router, queries, dim);
    let denom = (queries.len() / dim * route_runs).max(1) as f64;
    let qps = denom / elapsed.as_secs_f64();
    Row {
        router: "T-171 pivot prefix",
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
        pivot_evals: total.pivot_evals as f64 / denom,
        posting_entries: total.posting_entries as f64 / denom,
        duplicate_rate: total.duplicate_rate / denom,
        selected_blocks: total.selected_blocks as f64 / denom,
        centroid_evals: total.centroid_evals as f64 / denom,
        candidate_count: total.candidate_count as f64 / denom,
        resident_bytes: router.resident_bytes(),
        working_set_bytes: (total.working_set_bytes / (denom as usize)).max(1),
        build_time_s: router.build_time_s,
        fallback_rate: total.fallback as f64 / denom,
        hot_allocations,
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

    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    let started = Instant::now();
    let mut total = MetricTotals::default();
    for _ in 0..route_runs {
        for query in queries.chunks_exact(dim) {
            total.add(router.route(query, &mut scratch));
        }
    }
    let elapsed = started.elapsed();
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);
    let hot_allocations = ALLOCATION_COUNT.load(Ordering::Relaxed);

    let (finals, pools) = collect_dense_results(router, queries, dim);
    let denom = (queries.len() / dim * route_runs).max(1) as f64;
    let qps = denom / elapsed.as_secs_f64();
    Row {
        router: "T-171 dense l_inf fallback",
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
        pivot_evals: total.pivot_evals as f64 / denom,
        posting_entries: 0.0,
        duplicate_rate: 0.0,
        selected_blocks: total.selected_blocks as f64 / denom,
        centroid_evals: total.centroid_evals as f64 / denom,
        candidate_count: total.candidate_count as f64 / denom,
        resident_bytes: router.resident_bytes(),
        working_set_bytes: (total.working_set_bytes / (denom as usize)).max(1),
        build_time_s: router.build_time_s,
        fallback_rate: total.fallback as f64 / denom,
        hot_allocations,
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
struct MetricTotals {
    pivot_evals: usize,
    posting_entries: usize,
    duplicate_rate: f64,
    selected_blocks: usize,
    centroid_evals: usize,
    candidate_count: usize,
    working_set_bytes: usize,
    fallback: usize,
}

impl MetricTotals {
    fn add(&mut self, metrics: RouteMetrics) {
        self.pivot_evals += metrics.pivot_evals;
        self.posting_entries += metrics.posting_entries_touched;
        self.duplicate_rate += metrics.duplicate_block_rate as f64;
        self.selected_blocks += metrics.selected_blocks;
        self.centroid_evals += metrics.centroid_evals;
        self.candidate_count += metrics.candidate_count;
        self.working_set_bytes += metrics.working_set_bytes;
        self.fallback += usize::from(metrics.fallback);
    }
}

fn specs(no_sensitivity: bool) -> Vec<Spec> {
    let mut specs = vec![
        Spec::new(1024, 32, 32, 4, 16, 256, PrefixScoreMode::Union),
        Spec::new(1024, 32, 32, 8, 16, 256, PrefixScoreMode::Union),
        Spec::new(1024, 64, 32, 8, 16, 256, PrefixScoreMode::Union),
        Spec::new(4096, 32, 64, 4, 32, 512, PrefixScoreMode::Union),
        Spec::new(4096, 32, 64, 8, 32, 512, PrefixScoreMode::Union),
        Spec::new(4096, 64, 64, 8, 32, 512, PrefixScoreMode::Union),
    ];
    if !no_sensitivity {
        specs.extend([
            Spec::new(4096, 32, 64, 8, 32, 512, PrefixScoreMode::Weighted),
            Spec::new(4096, 32, 64, 8, 16, 512, PrefixScoreMode::Weighted),
            Spec::new(4096, 32, 64, 8, 64, 512, PrefixScoreMode::Weighted),
            Spec::new(4096, 32, 64, 12, 32, 512, PrefixScoreMode::Union),
            Spec::new(4096, 32, 64, 12, 32, 512, PrefixScoreMode::Weighted),
        ]);
    }
    specs
}

impl Spec {
    fn new(
        k: usize,
        block_size: usize,
        pivots: usize,
        prefix: usize,
        top_blocks: usize,
        pool: usize,
        mode: PrefixScoreMode,
    ) -> Self {
        Self {
            k,
            block_size,
            pivots,
            prefix,
            top_blocks,
            pool,
            mode,
        }
    }
}

struct Args {
    data_dir: PathBuf,
    output_json: PathBuf,
    output_md: PathBuf,
    route_runs: usize,
    cluster_iters: usize,
    no_sensitivity: bool,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut parsed = Self {
            data_dir: PathBuf::from("benchmarks/data/siftsmall"),
            output_json: PathBuf::from("benchmarks/pivot_prefix_t171.json"),
            output_md: PathBuf::from("benchmarks/pivot_prefix_t171.md"),
            route_runs: 20,
            cluster_iters: 12,
            no_sensitivity: false,
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
                "--no-sensitivity" => parsed.no_sensitivity = true,
                "-h" | "--help" => return Err(usage()),
                other => return Err(format!("unknown argument: {other}\n{}", usage())),
            }
            i += 1;
        }
        if parsed.route_runs == 0 {
            return Err("--route-runs must be greater than zero".to_string());
        }
        Ok(parsed)
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
        match dim {
            Some(existing) if existing != row_dim => {
                return Err(format!("non-uniform dimensions in {}", path.display()));
            }
            None => dim = Some(row_dim),
            _ => {}
        }
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

fn json_report(rows: &[Row]) -> String {
    let mut out = String::from("{\n  \"rows\": [\n");
    for (idx, row) in rows.iter().enumerate() {
        if idx > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "    {{\"router\":\"{}\",\"k\":{},\"block_size\":{},\"blocks\":{},\"pivots\":{},\"prefix\":{},\"top_blocks\":{},\"pool\":{},\"posting_mode\":\"{}\",\"final_nprobe\":{},\"coverage_at_16\":{:.6},\"candidate_pool_coverage_at_32\":{:.6},\"qps\":{:.3},\"route_time_us_per_query\":{:.3},\"pivot_evals_per_query\":{:.3},\"posting_entries_touched_per_query\":{:.3},\"duplicate_block_rate\":{:.6},\"selected_blocks_per_query\":{:.3},\"centroid_evals_per_query\":{:.3},\"candidate_count_per_query\":{:.3},\"route_resident_bytes\":{},\"working_set_bytes_per_query\":{},\"build_time_s\":{:.6},\"fallback_rate\":{:.6},\"hot_query_allocations\":{}}}",
            row.router,
            row.k,
            row.block_size,
            row.blocks,
            row.pivots,
            row.prefix,
            row.top_blocks,
            row.pool,
            row.mode,
            DEFAULT_FINAL_NPROBE,
            row.coverage16,
            row.pool_coverage32,
            row.qps,
            row.route_us,
            row.pivot_evals,
            row.posting_entries,
            row.duplicate_rate,
            row.selected_blocks,
            row.centroid_evals,
            row.candidate_count,
            row.resident_bytes,
            row.working_set_bytes,
            row.build_time_s,
            row.fallback_rate,
            row.hot_allocations
        ));
    }
    out.push_str("\n  ],\n  \"verdict\": \"");
    out.push_str(&verdict(rows));
    out.push_str("\"\n}\n");
    out
}

fn markdown_report(rows: &[Row], data_dir: &Path) -> String {
    let mut out = format!(
        "# T-171 Rust Pivot-Prefix Route Kernel Prototype\n\n- Dataset path: `{}`\n- Final exact rerank: `final_nprobe={}`\n- Hot query allocation counter starts after router and scratch construction.\n\n",
        data_dir.display(),
        DEFAULT_FINAL_NPROBE
    );
    out.push_str("| Router | K | Block size | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Posting entries/q | Duplicate rate | Selected blocks/q | Centroid evals/q | Resident bytes | Workset bytes/q | Hot allocs |\n");
    out.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for row in rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.4} | {:.4} | {:.1} | {:.1} | {:.1} | {:.4} | {:.1} | {:.1} | {} | {} | {} |\n",
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
            row.qps,
            row.route_us,
            row.posting_entries,
            row.duplicate_rate,
            row.selected_blocks,
            row.centroid_evals,
            row.resident_bytes,
            row.working_set_bytes,
            row.hot_allocations
        ));
    }
    out.push_str("\n## Verdict\n\n");
    out.push_str(&verdict(rows));
    out.push('\n');
    out
}

fn verdict(rows: &[Row]) -> String {
    let primary = rows.iter().find(|row| {
        row.router == "T-171 pivot prefix"
            && row.k == 4096
            && row.block_size == 64
            && row.pivots == 64
            && row.prefix == 8
            && row.top_blocks == 32
            && row.mode == "union"
    });
    let sensitivity = rows.iter().find(|row| {
        row.router == "T-171 pivot prefix"
            && row.k == 4096
            && row.block_size == 32
            && row.pivots == 64
            && row.prefix == 8
            && row.top_blocks == 64
            && row.mode == "weighted"
    });
    let dense = rows
        .iter()
        .find(|row| row.router == "T-171 dense l_inf fallback");
    let allocations_ok = rows.iter().all(|row| row.hot_allocations == 0);
    if primary.is_some_and(|row| row.coverage16 >= 0.996 && row.pool_coverage32 >= 0.996)
        && sensitivity.is_some_and(|row| row.coverage16 >= 0.993)
        && dense.is_some_and(|row| row.coverage16 >= 0.996)
        && allocations_ok
    {
        "PASS: Rust route kernel meets the T-171 coverage and hot-query allocation gates."
            .to_string()
    } else {
        "REVIEW: one or more T-171 gates missed; inspect per-row metrics for coverage, fallback, or allocations.".to_string()
    }
}

fn mode_name(mode: PrefixScoreMode) -> &'static str {
    match mode {
        PrefixScoreMode::Union => "union",
        PrefixScoreMode::Weighted => "weighted",
    }
}

fn usage() -> String {
    "usage: cargo run -p aperon-core --bin pivot_prefix_t171 -- [--route-runs N] [--no-sensitivity]"
        .to_string()
}

fn format_io(err: std::io::Error) -> String {
    err.to_string()
}
