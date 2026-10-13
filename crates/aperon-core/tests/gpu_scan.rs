use aperon_core::{CpuScanEngine, GpuDistanceMetric, GpuScanEngine, GpuScanResult, WgpuScanEngine};
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
    let mut values = vec![0.0_f32; 2 * 64];
    for (lane, vector) in vectors.iter().enumerate() {
        values[lane] = vector[0];
        values[64 + lane] = vector[1];
    }
    values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>()
}

#[test]
fn cpu_fallback_scans_block_soa_l2_and_returns_sorted_top_k() {
    let mut engine = CpuScanEngine::default();
    engine
        .bind_vector_database(
            &block_soa_bytes(&[[3.0, 4.0], [1.0, 1.0], [2.0, 0.0]]),
            2,
            3,
        )
        .unwrap();

    let results = block_on(engine.batch_scan(&[0.0, 0.0], 1, 2, GpuDistanceMetric::L2)).unwrap();

    assert_eq!(
        results,
        vec![vec![
            GpuScanResult {
                vector_id: 1,
                score: 2.0,
            },
            GpuScanResult {
                vector_id: 2,
                score: 4.0,
            },
        ]]
    );
}

#[test]
fn wgpu_engine_uploads_block_soa_l2_batch_and_reads_back_top_k() {
    let mut engine = match block_on(WgpuScanEngine::init()) {
        Ok(engine) => engine,
        Err(error) => {
            assert!(error.to_string().contains("no compatible wgpu adapter"));
            return;
        }
    };
    engine
        .bind_vector_database(
            &block_soa_bytes(&[[3.0, 4.0], [1.0, 1.0], [2.0, 0.0]]),
            2,
            3,
        )
        .unwrap();

    let results =
        block_on(engine.batch_scan(&[0.0, 0.0, 2.0, 2.0], 2, 2, GpuDistanceMetric::L2)).unwrap();

    assert_eq!(
        results,
        vec![
            vec![
                GpuScanResult {
                    vector_id: 1,
                    score: 2.0,
                },
                GpuScanResult {
                    vector_id: 2,
                    score: 4.0,
                },
            ],
            vec![
                GpuScanResult {
                    vector_id: 1,
                    score: 2.0,
                },
                GpuScanResult {
                    vector_id: 2,
                    score: 4.0,
                },
            ],
        ]
    );
}
