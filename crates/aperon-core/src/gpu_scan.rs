//! Portable wgpu implementation of the RFC-234-A Block-SoA scan contract.
//!
//! Vector database bytes are always uploaded into a wgpu storage buffer. This
//! is required on discrete GPUs; unified-memory systems may optimize that copy
//! internally, but the API never promises a host-pointer zero-copy binding.

use std::{
    error::Error,
    fmt,
    future::Future,
    pin::Pin,
    sync::{mpsc, Arc},
};

use wgpu::util::DeviceExt;

const BLOCK_SIZE: usize = 64;

const L2_SCAN_SHADER: &str = r#"
struct ScanParams {
    dimension: u32,
    num_vectors: u32,
    num_queries: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> db_buf: array<f32>;
@group(0) @binding(1) var<storage, read> query_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> score_buf: array<f32>;
@group(0) @binding(3) var<uniform> params: ScanParams;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let vector_idx = workgroup_id.x * 64u + local_id.x;
    let query_idx = workgroup_id.y;
    if (vector_idx >= params.num_vectors || query_idx >= params.num_queries) {
        return;
    }

    let block_idx = workgroup_id.x;
    let query_offset = query_idx * params.dimension;
    var distance: f32 = 0.0;
    for (var dimension: u32 = 0u; dimension < params.dimension; dimension = dimension + 1u) {
        let query_value = query_buf[query_offset + dimension];
        let db_value = db_buf[block_idx * params.dimension * 64u + dimension * 64u + local_id.x];
        let difference = query_value - db_value;
        distance = distance + difference * difference;
    }
    score_buf[query_idx * params.num_vectors + vector_idx] = distance;
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDistanceMetric {
    L2,
    Cosine,
    DotProduct,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GpuScanResult {
    pub vector_id: u64,
    pub score: f32,
}

pub type GpuScanFuture<E> =
    Pin<Box<dyn Future<Output = Result<Vec<Vec<GpuScanResult>>, E>> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuScanError {
    InvalidDatabase { reason: String },
    InvalidRequest { reason: String },
    UnsupportedMetric(GpuDistanceMetric),
    Backend { reason: String },
}

impl GpuScanError {
    fn new(message: impl Into<String>) -> Self {
        Self::Backend {
            reason: message.into(),
        }
    }
}

impl fmt::Display for GpuScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDatabase { reason }
            | Self::InvalidRequest { reason }
            | Self::Backend { reason } => reason.fmt(formatter),
            Self::UnsupportedMetric(metric) => {
                write!(formatter, "RFC-234-A scan does not support {metric:?}")
            }
        }
    }
}

impl Error for GpuScanError {}

pub trait GpuScanEngine: Send + Sync {
    type Error: Error + Send + Sync + 'static;

    fn init() -> impl Future<Output = Result<Self, Self::Error>> + Send
    where
        Self: Sized;

    fn bind_vector_database(
        &mut self,
        database_bytes: &[u8],
        dimension: usize,
        num_vectors: usize,
    ) -> Result<(), Self::Error>;

    fn batch_scan(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        metric: GpuDistanceMetric,
    ) -> GpuScanFuture<Self::Error>;
}

#[derive(Debug, Clone, Default)]
pub struct CpuScanEngine {
    database: Vec<f32>,
    dimension: usize,
    num_vectors: usize,
}

/// Explicit portable fallback for systems without an available wgpu adapter.
pub type CpuFallbackScanEngine = CpuScanEngine;

impl CpuScanEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GpuScanEngine for CpuScanEngine {
    type Error = GpuScanError;

    async fn init() -> Result<Self, Self::Error> {
        Ok(Self::default())
    }

    fn bind_vector_database(
        &mut self,
        database_bytes: &[u8],
        dimension: usize,
        num_vectors: usize,
    ) -> Result<(), Self::Error> {
        let database = parse_block_soa(database_bytes, dimension, num_vectors)?;
        self.database = database;
        self.dimension = dimension;
        self.num_vectors = num_vectors;
        Ok(())
    }

    fn batch_scan(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        metric: GpuDistanceMetric,
    ) -> GpuScanFuture<Self::Error> {
        let database = self.database.clone();
        let queries = queries.to_vec();
        let dimension = self.dimension;
        let num_vectors = self.num_vectors;
        Box::pin(async move {
            validate_scan(
                &database,
                dimension,
                num_vectors,
                &queries,
                num_queries,
                top_k,
                metric,
            )?;
            Ok(cpu_l2_top_k(
                &database,
                dimension,
                num_vectors,
                &queries,
                num_queries,
                top_k,
            ))
        })
    }
}

#[derive(Debug)]
struct WgpuState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

#[derive(Debug)]
pub struct WgpuScanEngine {
    state: Arc<WgpuState>,
    fallback: CpuScanEngine,
    database_buffer: Option<Arc<wgpu::Buffer>>,
}

impl GpuScanEngine for WgpuScanEngine {
    type Error = GpuScanError;

    async fn init() -> Result<Self, Self::Error> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| GpuScanError::new("no compatible wgpu adapter is available"))?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("aperon-gpu-scan-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|error| GpuScanError::new(format!("wgpu device request failed: {error}")))?;
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aperon-gpu-scan-bindings"),
            entries: &[
                storage_binding(0, true),
                storage_binding(1, true),
                storage_binding(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aperon-gpu-l2-block-soa"),
            source: wgpu::ShaderSource::Wgsl(L2_SCAN_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aperon-gpu-scan-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aperon-gpu-l2-block-soa"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            state: Arc::new(WgpuState {
                device: Arc::new(device),
                queue: Arc::new(queue),
                bind_group_layout: Arc::new(bind_group_layout),
                pipeline: Arc::new(pipeline),
            }),
            fallback: CpuScanEngine::default(),
            database_buffer: None,
        })
    }

    fn bind_vector_database(
        &mut self,
        database_bytes: &[u8],
        dimension: usize,
        num_vectors: usize,
    ) -> Result<(), Self::Error> {
        self.fallback
            .bind_vector_database(database_bytes, dimension, num_vectors)?;
        self.database_buffer = Some(Arc::new(self.state.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("aperon-gpu-block-soa-database-upload"),
                contents: database_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            },
        )));
        Ok(())
    }

    fn batch_scan(
        &self,
        queries: &[f32],
        num_queries: usize,
        top_k: usize,
        metric: GpuDistanceMetric,
    ) -> GpuScanFuture<Self::Error> {
        let state = Arc::clone(&self.state);
        let fallback = self.fallback.clone();
        let database_buffer = self.database_buffer.clone();
        let queries = queries.to_vec();
        Box::pin(async move {
            validate_scan(
                &fallback.database,
                fallback.dimension,
                fallback.num_vectors,
                &queries,
                num_queries,
                top_k,
                metric,
            )?;
            let database_buffer = database_buffer.ok_or_else(|| {
                GpuScanError::new("vector database must be bound before scanning")
            })?;
            gpu_l2_top_k(
                &state,
                &database_buffer,
                fallback.dimension,
                fallback.num_vectors,
                &queries,
                num_queries,
                top_k,
            )
        })
    }
}

fn storage_binding(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn parse_block_soa(
    database_bytes: &[u8],
    dimension: usize,
    num_vectors: usize,
) -> Result<Vec<f32>, GpuScanError> {
    if dimension == 0 {
        return Err(GpuScanError::InvalidDatabase {
            reason: "dimension must be positive".to_owned(),
        });
    }
    let blocks = num_vectors.div_ceil(BLOCK_SIZE);
    let expected_floats = blocks
        .checked_mul(dimension)
        .and_then(|value| value.checked_mul(BLOCK_SIZE))
        .ok_or_else(|| GpuScanError::InvalidDatabase {
            reason: "Block-SoA database size overflow".to_owned(),
        })?;
    let expected_bytes = expected_floats
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| GpuScanError::InvalidDatabase {
            reason: "Block-SoA database byte size overflow".to_owned(),
        })?;
    if database_bytes.len() != expected_bytes {
        return Err(GpuScanError::InvalidDatabase {
            reason: format!(
                "Block-SoA database has {} bytes; expected {expected_bytes} for {num_vectors} vectors of dimension {dimension}",
                database_bytes.len()
            ),
        });
    }
    Ok(database_bytes
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("chunk is exactly four bytes")))
        .collect())
}

fn validate_scan(
    database: &[f32],
    dimension: usize,
    num_vectors: usize,
    queries: &[f32],
    num_queries: usize,
    top_k: usize,
    metric: GpuDistanceMetric,
) -> Result<(), GpuScanError> {
    if metric != GpuDistanceMetric::L2 {
        return Err(GpuScanError::UnsupportedMetric(metric));
    }
    if database.is_empty() && num_vectors != 0 {
        return Err(GpuScanError::new(
            "vector database must be bound before scanning",
        ));
    }
    if num_queries == 0 {
        return Err(GpuScanError::new("num_queries must be positive"));
    }
    if queries.len() != num_queries.saturating_mul(dimension) {
        return Err(GpuScanError::new(format!(
            "query length mismatch: expected {}, got {}",
            num_queries.saturating_mul(dimension),
            queries.len()
        )));
    }
    if top_k == 0 || top_k > num_vectors {
        return Err(GpuScanError::new(format!(
            "top_k must be between 1 and {num_vectors}, got {top_k}"
        )));
    }
    Ok(())
}

fn cpu_l2_top_k(
    database: &[f32],
    dimension: usize,
    num_vectors: usize,
    queries: &[f32],
    num_queries: usize,
    top_k: usize,
) -> Vec<Vec<GpuScanResult>> {
    (0..num_queries)
        .map(|query_index| {
            let query = &queries[query_index * dimension..(query_index + 1) * dimension];
            let mut scores = (0..num_vectors)
                .map(|vector_id| GpuScanResult {
                    vector_id: vector_id as u64,
                    score: l2_block_soa(database, dimension, vector_id, query),
                })
                .collect::<Vec<_>>();
            scores.sort_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| left.vector_id.cmp(&right.vector_id))
            });
            scores.truncate(top_k);
            scores
        })
        .collect()
}

fn l2_block_soa(database: &[f32], dimension: usize, vector_id: usize, query: &[f32]) -> f32 {
    let block = vector_id / BLOCK_SIZE;
    let lane = vector_id % BLOCK_SIZE;
    query
        .iter()
        .enumerate()
        .map(|(coordinate, query_value)| {
            let database_value =
                database[block * dimension * BLOCK_SIZE + coordinate * BLOCK_SIZE + lane];
            let difference = query_value - database_value;
            difference * difference
        })
        .sum()
}

fn gpu_l2_top_k(
    state: &WgpuState,
    database_buffer: &wgpu::Buffer,
    dimension: usize,
    num_vectors: usize,
    queries: &[f32],
    num_queries: usize,
    top_k: usize,
) -> Result<Vec<Vec<GpuScanResult>>, GpuScanError> {
    let query_buffer = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aperon-gpu-scan-queries-upload"),
            contents: f32_bytes(queries).as_slice(),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let score_size = num_queries
        .checked_mul(num_vectors)
        .and_then(|value| value.checked_mul(size_of::<f32>()))
        .ok_or_else(|| GpuScanError::new("score buffer size overflow"))?
        as u64;
    let score_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aperon-gpu-scan-scores"),
        size: score_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aperon-gpu-scan-scores-readback"),
        size: score_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let params = [
        u32::try_from(dimension).map_err(|_| GpuScanError::new("dimension exceeds u32"))?,
        u32::try_from(num_vectors).map_err(|_| GpuScanError::new("num_vectors exceeds u32"))?,
        u32::try_from(num_queries).map_err(|_| GpuScanError::new("num_queries exceeds u32"))?,
        0,
    ];
    let parameter_buffer = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aperon-gpu-scan-params"),
            contents: u32_bytes(&params).as_slice(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("aperon-gpu-scan-bind-group"),
        layout: &state.bind_group_layout,
        entries: &[
            buffer_binding(0, database_buffer),
            buffer_binding(1, &query_buffer),
            buffer_binding(2, &score_buffer),
            buffer_binding(3, &parameter_buffer),
        ],
    });
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aperon-gpu-scan-command-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("aperon-gpu-l2-block-soa-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&state.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            num_vectors.div_ceil(BLOCK_SIZE) as u32,
            num_queries as u32,
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&score_buffer, 0, &readback_buffer, 0, score_size);
    state.queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    state.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| GpuScanError::new("wgpu readback callback was dropped"))?
        .map_err(|error| GpuScanError::new(format!("wgpu score readback failed: {error}")))?;
    let scores = readback_buffer
        .slice(..)
        .get_mapped_range()
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("chunk is exactly four bytes")))
        .collect::<Vec<_>>();
    readback_buffer.unmap();

    Ok(top_k_from_scores(&scores, num_vectors, num_queries, top_k))
}

fn buffer_binding<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn top_k_from_scores(
    scores: &[f32],
    num_vectors: usize,
    num_queries: usize,
    top_k: usize,
) -> Vec<Vec<GpuScanResult>> {
    (0..num_queries)
        .map(|query_index| {
            let mut results = scores[query_index * num_vectors..(query_index + 1) * num_vectors]
                .iter()
                .enumerate()
                .map(|(vector_id, &score)| GpuScanResult {
                    vector_id: vector_id as u64,
                    score,
                })
                .collect::<Vec<_>>();
            results.sort_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| left.vector_id.cmp(&right.vector_id))
            });
            results.truncate(top_k);
            results
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
