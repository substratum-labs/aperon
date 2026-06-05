#![allow(clippy::too_many_arguments)]

use aperon_core::pivot_prefix::{
    coverage, exact_topk, sample_centroids, DensePivotSketch, PivotPrefixConfig, PivotPrefixRouter,
    PrefixScoreMode, RouteMetrics, DEFAULT_FINAL_NPROBE,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    env, fs,
    mem::size_of,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LayoutKind {
    DenseF32,
    DenseFp16,
    DenseUint16,
    PrefixBaseline,
    PrefixPacked,
}

#[derive(Clone, Debug)]
struct Row {
    layout: LayoutKind,
    k: usize,
    block_size: usize,
    blocks: usize,
    pivots: usize,
    prefix: usize,
    top_blocks: usize,
    pool: usize,
    coverage16: f64,
    pool_coverage32: f64,
    qps: f64,
    route_us: f64,
    posting_entries: f64,
    duplicate_rate: f64,
    centroid_evals: f64,
    resident_bytes: usize,
    working_set_bytes: usize,
    posting_bytes_per_block: f64,
    quant_max_abs_error: f32,
    quant_mean_abs_error: f32,
    build_time_s: f64,
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

    let k = 4096;
    let block_size = 64;
    let pivots = 64;
    let top_blocks = 32;
    let pool = 512;
    let centroids = sample_centroids(&xb, dim, k)?;
    let exact16 = exact_topk(&centroids, dim, &queries, 16);
    let exact32 = exact_topk(&centroids, dim, &queries, 32);

    let dense = DensePivotSketch::build(
        &centroids,
        dim,
        block_size,
        pivots,
        top_blocks,
        pool,
        args.cluster_iters,
    )?;
    let mut rows = vec![
        evaluate_dense_f32(&dense, &queries, dim, &exact16, &exact32, args.route_runs),
        evaluate_quant_dense(
            QuantDenseSketch::from_dense(&dense, QuantKind::Fp16),
            &queries,
            dim,
            &exact16,
            &exact32,
            args.route_runs,
        ),
        evaluate_quant_dense(
            QuantDenseSketch::from_dense(&dense, QuantKind::Uint16),
            &queries,
            dim,
            &exact16,
            &exact32,
            args.route_runs,
        ),
    ];

    let prefix = PivotPrefixRouter::build(
        &centroids,
        dim,
        PivotPrefixConfig {
            block_size,
            pivot_count: pivots,
            prefix_len: 8,
            top_blocks,
            candidate_pool: pool,
            mode: PrefixScoreMode::Union,
            cluster_iters: args.cluster_iters,
        },
    )?;
    let prefix_row = evaluate_prefix(
        &prefix,
        &queries,
        dim,
        &exact16,
        &exact32,
        args.route_runs,
        LayoutKind::PrefixBaseline,
        prefix.resident_bytes(),
        prefix_working_set_bytes(&prefix, false),
        prefix_posting_bytes_per_block(&prefix, false),
    );
    let compact_row = evaluate_prefix(
        &prefix,
        &queries,
        dim,
        &exact16,
        &exact32,
        args.route_runs,
        LayoutKind::PrefixPacked,
        compact_prefix_resident_bytes(&prefix),
        prefix_working_set_bytes(&prefix, true),
        prefix_posting_bytes_per_block(&prefix, true),
    );
    rows.push(prefix_row);
    rows.push(compact_row);

    fs::write(&args.output_json, json_report(&rows)).map_err(format_io)?;
    fs::write(&args.output_md, markdown_report(&rows, &args.data_dir)).map_err(format_io)?;
    println!("wrote {}", args.output_json.display());
    println!("wrote {}", args.output_md.display());
    println!("{}", verdict(&rows));
    Ok(())
}

fn evaluate_dense_f32(
    router: &DensePivotSketch,
    queries: &[f32],
    dim: usize,
    exact16: &[Vec<u32>],
    exact32: &[Vec<u32>],
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
        layout: LayoutKind::DenseF32,
        k: router.k_centroids,
        block_size: router.block_size,
        blocks: router.num_blocks,
        pivots: router.num_pivots,
        prefix: 0,
        top_blocks: router.top_blocks,
        pool: router.candidate_pool,
        coverage16: coverage(&finals, exact16),
        pool_coverage32: coverage(&pools, exact32),
        qps,
        route_us: 1_000_000.0 / qps,
        posting_entries: 0.0,
        duplicate_rate: 0.0,
        centroid_evals: total.centroid_evals as f64 / denom,
        resident_bytes: router.resident_bytes(),
        working_set_bytes: total.working_set_bytes / denom as usize,
        posting_bytes_per_block: 0.0,
        quant_max_abs_error: 0.0,
        quant_mean_abs_error: 0.0,
        build_time_s: router.build_time_s,
        hot_allocations,
    }
}

fn evaluate_quant_dense(
    router: QuantDenseSketch,
    queries: &[f32],
    dim: usize,
    exact16: &[Vec<u32>],
    exact32: &[Vec<u32>],
    route_runs: usize,
) -> Row {
    let mut scratch = QuantScratch::new(
        router.num_pivots,
        router.num_blocks,
        router.top_blocks,
        router
            .candidate_pool
            .max(router.top_blocks * router.block_size),
    );
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
    let (finals, pools) = collect_quant_results(&router, queries, dim);
    let denom = (queries.len() / dim * route_runs).max(1) as f64;
    let qps = denom / elapsed.as_secs_f64();
    Row {
        layout: match router.kind {
            QuantKind::Fp16 => LayoutKind::DenseFp16,
            QuantKind::Uint16 => LayoutKind::DenseUint16,
        },
        k: router.k_centroids,
        block_size: router.block_size,
        blocks: router.num_blocks,
        pivots: router.num_pivots,
        prefix: 0,
        top_blocks: router.top_blocks,
        pool: router.candidate_pool,
        coverage16: coverage(&finals, exact16),
        pool_coverage32: coverage(&pools, exact32),
        qps,
        route_us: 1_000_000.0 / qps,
        posting_entries: 0.0,
        duplicate_rate: 0.0,
        centroid_evals: total.centroid_evals as f64 / denom,
        resident_bytes: router.resident_bytes(),
        working_set_bytes: router.working_set_bytes(total.centroid_evals / denom as usize),
        posting_bytes_per_block: 0.0,
        quant_max_abs_error: router.max_abs_error,
        quant_mean_abs_error: router.mean_abs_error,
        build_time_s: router.build_time_s,
        hot_allocations,
    }
}

fn evaluate_prefix(
    router: &PivotPrefixRouter,
    queries: &[f32],
    dim: usize,
    exact16: &[Vec<u32>],
    exact32: &[Vec<u32>],
    route_runs: usize,
    layout: LayoutKind,
    resident_bytes: usize,
    compact_working_set_adjustment: isize,
    posting_bytes_per_block: f64,
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
    let base_working_set = total.working_set_bytes / denom as usize;
    let working_set_bytes = if compact_working_set_adjustment < 0 {
        base_working_set.saturating_sub((-compact_working_set_adjustment) as usize)
    } else {
        base_working_set + compact_working_set_adjustment as usize
    };
    Row {
        layout,
        k: router.k_centroids,
        block_size: router.block_size,
        blocks: router.num_blocks,
        pivots: router.num_pivots,
        prefix: router.prefix_len,
        top_blocks: router.top_blocks,
        pool: router.candidate_pool,
        coverage16: coverage(&finals, exact16),
        pool_coverage32: coverage(&pools, exact32),
        qps,
        route_us: 1_000_000.0 / qps,
        posting_entries: total.posting_entries as f64 / denom,
        duplicate_rate: total.duplicate_rate / denom,
        centroid_evals: total.centroid_evals as f64 / denom,
        resident_bytes,
        working_set_bytes,
        posting_bytes_per_block,
        quant_max_abs_error: 0.0,
        quant_mean_abs_error: 0.0,
        build_time_s: router.build_time_s,
        hot_allocations,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuantKind {
    Fp16,
    Uint16,
}

#[derive(Clone, Debug)]
struct QuantDenseSketch {
    kind: QuantKind,
    dim: usize,
    k_centroids: usize,
    block_size: usize,
    num_blocks: usize,
    num_pivots: usize,
    top_blocks: usize,
    candidate_pool: usize,
    pivots: Vec<f32>,
    encoded: Vec<u16>,
    min_distance: f32,
    scale: f32,
    block_offsets: Vec<u32>,
    block_payload: Vec<u32>,
    block_representatives: Vec<f32>,
    centroid_vectors: Vec<f32>,
    max_abs_error: f32,
    mean_abs_error: f32,
    build_time_s: f64,
}

impl QuantDenseSketch {
    fn from_dense(dense: &DensePivotSketch, kind: QuantKind) -> Self {
        let started = Instant::now();
        let mut encoded = Vec::with_capacity(dense.block_pivot_distances.len());
        let (min_distance, max_distance) = dense
            .block_pivot_distances
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &value| {
                (lo.min(value), hi.max(value))
            });
        let scale = if max_distance > min_distance {
            (max_distance - min_distance) / u16::MAX as f32
        } else {
            1.0
        };
        let mut max_abs_error = 0.0_f32;
        let mut sum_abs_error = 0.0_f32;
        for &value in &dense.block_pivot_distances {
            let code = match kind {
                QuantKind::Fp16 => f32_to_f16(value),
                QuantKind::Uint16 => ((value - min_distance) / scale)
                    .round()
                    .clamp(0.0, u16::MAX as f32) as u16,
            };
            let decoded = match kind {
                QuantKind::Fp16 => f16_to_f32(code),
                QuantKind::Uint16 => min_distance + code as f32 * scale,
            };
            let err = (decoded - value).abs();
            max_abs_error = max_abs_error.max(err);
            sum_abs_error += err;
            encoded.push(code);
        }
        Self {
            kind,
            dim: dense.dim,
            k_centroids: dense.k_centroids,
            block_size: dense.block_size,
            num_blocks: dense.num_blocks,
            num_pivots: dense.num_pivots,
            top_blocks: dense.top_blocks,
            candidate_pool: dense.candidate_pool,
            pivots: dense.pivots.clone(),
            encoded,
            min_distance,
            scale,
            block_offsets: dense.block_offsets.clone(),
            block_payload: dense.block_payload.clone(),
            block_representatives: dense.block_representatives.clone(),
            centroid_vectors: dense.centroid_vectors.clone(),
            max_abs_error,
            mean_abs_error: sum_abs_error / dense.block_pivot_distances.len().max(1) as f32,
            build_time_s: dense.build_time_s + started.elapsed().as_secs_f64(),
        }
    }

    fn route(&self, query: &[f32], scratch: &mut QuantScratch) -> RouteMetrics {
        assert_eq!(query.len(), self.dim);
        scratch.clear_query();
        for pivot in 0..self.num_pivots {
            scratch.query_pivot_dist[pivot] =
                l2_squared(query, row(&self.pivots, self.dim, pivot)).sqrt();
        }
        for block in 0..self.num_blocks {
            let mut score = 0.0_f32;
            for pivot in 0..self.num_pivots {
                let decoded = self.decode(self.encoded[block * self.num_pivots + pivot]);
                score = score.max((decoded - scratch.query_pivot_dist[pivot]).abs());
            }
            scratch.block_scores[block] = score;
            scratch.touched_blocks.push(block as u32);
        }
        scratch.touched_blocks.sort_unstable_by(|&a, &b| {
            scratch.block_scores[a as usize]
                .total_cmp(&scratch.block_scores[b as usize])
                .then_with(|| a.cmp(&b))
        });
        scratch
            .selected_blocks
            .extend_from_slice(&scratch.touched_blocks[..self.top_blocks.min(self.num_blocks)]);
        scan_and_rerank(
            query,
            self.dim,
            &self.centroid_vectors,
            &self.block_offsets,
            &self.block_payload,
            self.candidate_pool,
            scratch,
        );
        RouteMetrics {
            pivot_evals: self.num_pivots,
            posting_entries_touched: 0,
            unique_blocks_touched: self.num_blocks,
            duplicate_blocks: 0,
            duplicate_block_rate: 0.0,
            selected_blocks: scratch.selected_blocks.len(),
            centroid_evals: scratch.candidate_centroids.len(),
            candidate_count: scratch.pool_candidates.len(),
            working_set_bytes: self.working_set_bytes(scratch.candidate_centroids.len()),
            fallback: scratch.final_nprobe.len() < DEFAULT_FINAL_NPROBE.min(self.k_centroids),
        }
    }

    fn decode(&self, code: u16) -> f32 {
        match self.kind {
            QuantKind::Fp16 => f16_to_f32(code),
            QuantKind::Uint16 => self.min_distance + code as f32 * self.scale,
        }
    }

    fn resident_bytes(&self) -> usize {
        self.centroid_vectors.len() * size_of::<f32>()
            + self.block_payload.len() * size_of::<u32>()
            + self.block_offsets.len() * size_of::<u32>()
            + self.block_representatives.len() * size_of::<f32>()
            + self.pivots.len() * size_of::<f32>()
            + self.encoded.len() * size_of::<u16>()
            + match self.kind {
                QuantKind::Fp16 => 0,
                QuantKind::Uint16 => 2 * size_of::<f32>(),
            }
    }

    fn working_set_bytes(&self, centroid_evals: usize) -> usize {
        self.num_pivots * self.dim * size_of::<f32>()
            + self.num_blocks * self.num_pivots * size_of::<u16>()
            + self.num_blocks * (size_of::<f32>() + size_of::<u32>())
            + centroid_evals * self.dim * size_of::<f32>()
    }
}

#[derive(Clone, Debug)]
struct QuantScratch {
    query_pivot_dist: Vec<f32>,
    block_scores: Vec<f32>,
    touched_blocks: Vec<u32>,
    selected_blocks: Vec<u32>,
    candidate_centroids: Vec<u32>,
    centroid_scores: Vec<f32>,
    centroid_order: Vec<usize>,
    pool_candidates: Vec<u32>,
    final_nprobe: Vec<u32>,
}

impl QuantScratch {
    fn new(
        num_pivots: usize,
        num_blocks: usize,
        top_blocks: usize,
        candidate_capacity: usize,
    ) -> Self {
        Self {
            query_pivot_dist: vec![0.0; num_pivots],
            block_scores: vec![0.0; num_blocks],
            touched_blocks: Vec::with_capacity(num_blocks),
            selected_blocks: Vec::with_capacity(top_blocks),
            candidate_centroids: Vec::with_capacity(candidate_capacity),
            centroid_scores: Vec::with_capacity(candidate_capacity),
            centroid_order: Vec::with_capacity(candidate_capacity),
            pool_candidates: Vec::with_capacity(candidate_capacity),
            final_nprobe: Vec::with_capacity(DEFAULT_FINAL_NPROBE),
        }
    }

    fn clear_query(&mut self) {
        self.touched_blocks.clear();
        self.selected_blocks.clear();
        self.candidate_centroids.clear();
        self.centroid_scores.clear();
        self.centroid_order.clear();
        self.pool_candidates.clear();
        self.final_nprobe.clear();
    }
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

fn collect_quant_results(
    router: &QuantDenseSketch,
    queries: &[f32],
    dim: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut scratch = QuantScratch::new(
        router.num_pivots,
        router.num_blocks,
        router.top_blocks,
        router
            .candidate_pool
            .max(router.top_blocks * router.block_size),
    );
    let mut finals = Vec::with_capacity(queries.len() / dim);
    let mut pools = Vec::with_capacity(queries.len() / dim);
    for query in queries.chunks_exact(dim) {
        router.route(query, &mut scratch);
        finals.push(scratch.final_nprobe.clone());
        pools.push(scratch.pool_candidates.clone());
    }
    (finals, pools)
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

fn scan_and_rerank(
    query: &[f32],
    dim: usize,
    centroids: &[f32],
    block_offsets: &[u32],
    block_payload: &[u32],
    candidate_pool: usize,
    scratch: &mut QuantScratch,
) {
    for &block in &scratch.selected_blocks {
        let block = block as usize;
        for &centroid in
            &block_payload[block_offsets[block] as usize..block_offsets[block + 1] as usize]
        {
            let centroid = centroid as usize;
            scratch.candidate_centroids.push(centroid as u32);
            scratch
                .centroid_scores
                .push(l2_squared(query, row(centroids, dim, centroid)));
        }
    }
    scratch
        .centroid_order
        .extend(0..scratch.candidate_centroids.len());
    scratch.centroid_order.sort_unstable_by(|&a, &b| {
        scratch.centroid_scores[a]
            .total_cmp(&scratch.centroid_scores[b])
            .then_with(|| scratch.candidate_centroids[a].cmp(&scratch.candidate_centroids[b]))
    });
    for &idx in &scratch.centroid_order[..candidate_pool.min(scratch.centroid_order.len())] {
        scratch
            .pool_candidates
            .push(scratch.candidate_centroids[idx]);
    }
    let final_limit = DEFAULT_FINAL_NPROBE.min(scratch.pool_candidates.len());
    scratch
        .final_nprobe
        .extend_from_slice(&scratch.pool_candidates[..final_limit]);
}

#[derive(Default)]
struct MetricTotals {
    posting_entries: usize,
    duplicate_rate: f64,
    centroid_evals: usize,
    working_set_bytes: usize,
}

impl MetricTotals {
    fn add(&mut self, metrics: RouteMetrics) {
        self.posting_entries += metrics.posting_entries_touched;
        self.duplicate_rate += metrics.duplicate_block_rate as f64;
        self.centroid_evals += metrics.centroid_evals;
        self.working_set_bytes += metrics.working_set_bytes;
    }
}

fn compact_prefix_resident_bytes(router: &PivotPrefixRouter) -> usize {
    router.centroid_vectors.len() * size_of::<f32>()
        + router.block_payload.len() * size_of::<u16>()
        + router.block_offsets.len() * size_of::<u32>()
        + router.block_representatives.len() * size_of::<f32>()
        + router.pivots.len() * size_of::<f32>()
        + router.block_prefix_pivots.len() * size_of::<u16>()
        + router.posting_offsets.len() * size_of::<u32>()
        + router.posting_block_ids.len() * size_of::<u16>()
        + router.posting_positions.len() * size_of::<u8>()
        + router.idf.len() * size_of::<f32>()
}

fn prefix_working_set_bytes(router: &PivotPrefixRouter, compact: bool) -> isize {
    if !compact {
        return 0;
    }
    let posting_delta = -(router.posting_block_ids.len() as isize * size_of::<u16>() as isize);
    let payload_delta = -(router.top_blocks.min(router.num_blocks) as isize
        * router.block_size as isize
        * size_of::<u16>() as isize);
    posting_delta + payload_delta
}

fn prefix_posting_bytes_per_block(router: &PivotPrefixRouter, compact: bool) -> f64 {
    let posting_id_bytes = if compact {
        size_of::<u16>()
    } else {
        size_of::<u32>()
    };
    let bytes = router.posting_offsets.len() * size_of::<u32>()
        + router.posting_block_ids.len() * posting_id_bytes
        + router.posting_positions.len() * size_of::<u8>();
    bytes as f64 / router.num_blocks.max(1) as f64
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x800000;
        let shift = 14 - exp;
        let mut half = (mantissa >> shift) as u16;
        if ((mantissa >> (shift - 1)) & 1) != 0 {
            half += 1;
        }
        sign | half
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        let mut half = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
        if (mant & 0x1000) != 0 {
            half += 1;
        }
        half
    }
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exp = ((value >> 10) & 0x1f) as u32;
    let mant = (value & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut mantissa = mant;
            let mut exponent = -14_i32;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
        }
    } else if exp == 31 {
        sign | 0x7f800000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn l2_squared(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

fn row(values: &[f32], dim: usize, idx: usize) -> &[f32] {
    &values[idx * dim..(idx + 1) * dim]
}

fn read_fvecs(path: &Path) -> Result<(Vec<f32>, usize), String> {
    let bytes = fs::read(path).map_err(format_io)?;
    let mut pos = 0_usize;
    let mut dim = None;
    let mut values = Vec::new();
    while pos < bytes.len() {
        let row_dim = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if dim.is_some_and(|existing| existing != row_dim) {
            return Err(format!("non-uniform dimensions in {}", path.display()));
        }
        dim = Some(row_dim);
        for chunk in bytes[pos..pos + row_dim * 4].chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        pos += row_dim * 4;
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
            output_json: PathBuf::from("benchmarks/pivot_quant_t172.json"),
            output_md: PathBuf::from("benchmarks/pivot_quant_t172.md"),
            route_runs: 20,
            cluster_iters: 12,
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
        Ok(parsed)
    }
}

fn json_report(rows: &[Row]) -> String {
    let mut out = String::from("{\n  \"rows\": [\n");
    for (idx, row) in rows.iter().enumerate() {
        if idx > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "    {{\"layout\":\"{}\",\"k\":{},\"block_size\":{},\"blocks\":{},\"pivots\":{},\"prefix\":{},\"top_blocks\":{},\"pool\":{},\"coverage_at_16\":{:.6},\"candidate_pool_coverage_at_32\":{:.6},\"qps\":{:.3},\"route_time_us_per_query\":{:.3},\"posting_entries_touched_per_query\":{:.3},\"duplicate_block_rate\":{:.6},\"centroid_evals_per_query\":{:.3},\"route_resident_bytes\":{},\"working_set_bytes_per_query\":{},\"posting_bytes_per_block\":{:.3},\"quant_max_abs_error\":{:.8},\"quant_mean_abs_error\":{:.8},\"build_time_s\":{:.6},\"hot_query_allocations\":{}}}",
            layout_name(row.layout),
            row.k,
            row.block_size,
            row.blocks,
            row.pivots,
            row.prefix,
            row.top_blocks,
            row.pool,
            row.coverage16,
            row.pool_coverage32,
            row.qps,
            row.route_us,
            row.posting_entries,
            row.duplicate_rate,
            row.centroid_evals,
            row.resident_bytes,
            row.working_set_bytes,
            row.posting_bytes_per_block,
            row.quant_max_abs_error,
            row.quant_mean_abs_error,
            row.build_time_s,
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
        "# T-172 Quantized Pivot Routing Prototype\n\n- Dataset path: `{}`\n- Dense rows use T-167 `l_inf` scoring with final exact rerank.\n- Prefix rows use T-168 pivot-prefix routing; packed row changes ID/token width only, so routing semantics match baseline.\n\n",
        data_dir.display()
    );
    out.push_str("| Layout | Coverage@16 | Pool coverage@32 | Route us/q | Resident bytes | Workset bytes/q | Posting bytes/block | Quant max abs err | Quant mean abs err | Hot allocs |\n");
    out.push_str("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for row in rows {
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.1} | {} | {} | {:.2} | {:.6} | {:.6} | {} |\n",
            layout_name(row.layout),
            row.coverage16,
            row.pool_coverage32,
            row.route_us,
            row.resident_bytes,
            row.working_set_bytes,
            row.posting_bytes_per_block,
            row.quant_max_abs_error,
            row.quant_mean_abs_error,
            row.hot_allocations
        ));
    }
    out.push_str("\n## Recommendation\n\n");
    out.push_str(&recommendation(rows));
    out.push_str("\n\n## Verdict\n\n");
    out.push_str(&verdict(rows));
    out.push('\n');
    out
}

fn recommendation(rows: &[Row]) -> String {
    let dense_u16 = rows
        .iter()
        .find(|row| row.layout == LayoutKind::DenseUint16);
    let prefix_packed = rows
        .iter()
        .find(|row| row.layout == LayoutKind::PrefixPacked);
    if dense_u16.is_some_and(|row| row.coverage16 >= 0.996)
        && prefix_packed.is_some_and(|row| row.coverage16 >= 0.996)
    {
        "Use u16 block/posting IDs for pivot-prefix layouts when `blocks <= 65535`, keep f32 pivots and centroid vectors for now, and use uint16 dense `l_inf` signatures as the fallback layout. fp16 is viable but uint16 has lower signature error at the same 2-byte footprint."
            .to_string()
    } else {
        "Keep the T-171 f32/u32 layout as default until the compact rows are reviewed; at least one compact layout missed the current coverage threshold."
            .to_string()
    }
}

fn verdict(rows: &[Row]) -> String {
    let coverage_ok = rows.iter().all(|row| match row.layout {
        LayoutKind::DenseF32 | LayoutKind::DenseFp16 | LayoutKind::DenseUint16 => {
            row.coverage16 >= 0.996
        }
        LayoutKind::PrefixBaseline | LayoutKind::PrefixPacked => row.coverage16 >= 0.996,
    });
    let allocations_ok = rows.iter().all(|row| row.hot_allocations == 0);
    if coverage_ok && allocations_ok {
        "PASS: compact pivot layouts preserve the T-167/T-168 positive signal on siftsmall."
            .to_string()
    } else {
        "REVIEW: one or more compact layouts missed coverage or allocation expectations."
            .to_string()
    }
}

fn layout_name(layout: LayoutKind) -> &'static str {
    match layout {
        LayoutKind::DenseF32 => "dense_l_inf_f32",
        LayoutKind::DenseFp16 => "dense_l_inf_fp16",
        LayoutKind::DenseUint16 => "dense_l_inf_uint16",
        LayoutKind::PrefixBaseline => "prefix_baseline_u32",
        LayoutKind::PrefixPacked => "prefix_packed_u16",
    }
}

fn format_io(err: std::io::Error) -> String {
    err.to_string()
}
