use aperon_core::{
    CpuFallbackScanEngine, GpuDistanceMetric, GpuScanEngine, GpuScanError, GpuScanResult,
};
use std::{
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

struct NoopWake;

impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(NoopWake));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn block_soa_bytes(vectors: &[[f32; 2]]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 64;
    let blocks = vectors.len().div_ceil(BLOCK_SIZE);
    let mut data = vec![0.0_f32; blocks * 2 * BLOCK_SIZE];
    for (vector_index, vector) in vectors.iter().enumerate() {
        let block = vector_index / BLOCK_SIZE;
        let lane = vector_index % BLOCK_SIZE;
        for (dimension, value) in vector.iter().enumerate() {
            data[block * 2 * BLOCK_SIZE + dimension * BLOCK_SIZE + lane] = *value;
        }
    }
    data.into_iter().flat_map(f32::to_le_bytes).collect()
}

#[test]
fn cpu_fallback_scans_block_soa_l2_batches_and_extracts_stable_top_k() {
    let database = block_soa_bytes(&[[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [-1.0, -1.0]]);
    let mut engine = CpuFallbackScanEngine::new();
    engine.bind_vector_database(&database, 2, 4).unwrap();

    let results =
        block_on(engine.batch_scan(&[0.0, 0.0, 2.0, 2.0], 2, 2, GpuDistanceMetric::L2)).unwrap();

    assert_eq!(
        results,
        vec![
            vec![
                GpuScanResult {
                    vector_id: 0,
                    score: 0.0,
                },
                GpuScanResult {
                    vector_id: 2,
                    score: 2.0,
                },
            ],
            vec![
                GpuScanResult {
                    vector_id: 2,
                    score: 2.0,
                },
                GpuScanResult {
                    vector_id: 1,
                    score: 5.0,
                },
            ],
        ]
    );
}

#[test]
fn cpu_fallback_rejects_non_l2_metrics_explicitly() {
    let mut engine = CpuFallbackScanEngine::new();
    engine
        .bind_vector_database(&block_soa_bytes(&[[0.0, 0.0]]), 2, 1)
        .unwrap();

    let error =
        block_on(engine.batch_scan(&[0.0, 0.0], 1, 1, GpuDistanceMetric::Cosine)).unwrap_err();

    assert_eq!(
        error,
        GpuScanError::UnsupportedMetric(GpuDistanceMetric::Cosine)
    );
}

#[test]
fn database_binding_rejects_non_block_soa_payloads() {
    let mut engine = CpuFallbackScanEngine::new();
    let error = engine.bind_vector_database(&[0; 7], 2, 1).unwrap_err();

    assert!(matches!(error, GpuScanError::InvalidDatabase { .. }));
}
