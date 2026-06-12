use crate::layout::encode_2bit;
use crate::{AperonIndex, ScoredVector, VectorId};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

/// A Write-Ahead Buffer (WAB) record containing the original vector,
/// a compact rotated 2-bit quantization projection representation, and target grain.
#[derive(Clone, Debug)]
pub struct WabRecord {
    pub id: VectorId,
    pub original_vector: Vec<f32>,
    pub projected_2bit: Vec<u8>,
    pub scale: f32,
    pub nearest_grain: usize,
}

/// An in-memory Write-Ahead Buffer (WAB) that enables concurrent write-search workloads.
/// Stores incoming writes in a fast 2-bit packed buffer and compactions run asynchronously in the background.
pub struct WriteAheadBuffer {
    index: Arc<RwLock<AperonIndex>>,
    buffer: Arc<Mutex<Vec<WabRecord>>>,
}

impl WriteAheadBuffer {
    /// Creates a new `WriteAheadBuffer` wrapping the given `AperonIndex`.
    pub fn new(index: AperonIndex) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Accesses the underlying thread-safe shared index.
    pub fn index(&self) -> Arc<RwLock<AperonIndex>> {
        self.index.clone()
    }

    /// Inserts a vector into the WAB buffer. Projects it onto the nearest grain's local PCA
    /// basis, quantizes it to 2-bit TurboQuant, and appends it to the in-memory write buffer.
    pub fn insert(&self, id: VectorId, vector: Vec<f32>) -> Result<(), String> {
        let idx_read = self.index.read().map_err(|e| e.to_string())?;

        let nearest_grain = if idx_read.grains().len() <= 1 {
            0
        } else {
            idx_read.route(&vector, None)?[0]
        };

        let mut projected_2bit = Vec::new();
        let mut scale = 1.0_f32;
        if nearest_grain < idx_read.grains().len() {
            let grain = &idx_read.grains()[nearest_grain];
            let centered: Vec<f32> = vector
                .iter()
                .zip(grain.mean())
                .map(|(x, m)| x - m)
                .collect();
            let k = grain.local_dim();
            if k > 0 {
                // Project centered vector onto PCA projection
                let mut z = vec![0.0_f32; k];
                for (component, z_val) in z.iter_mut().enumerate() {
                    for (d, &val) in centered.iter().enumerate() {
                        *z_val += val * grain.projection()[d * k + component];
                    }
                }

                // Calculate dynamic scale factor for 2-bit quantization
                let max_abs = z
                    .iter()
                    .map(|&x| x.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(1.0);
                scale = max_abs / 3.0;
                if scale == 0.0 {
                    scale = 1.0;
                }

                // Quantize to range [-3, 3]
                let mut i8_coords = Vec::with_capacity(k);
                for &val in &z {
                    let scaled = (val / scale).round();
                    let clamped = if scaled < -3.0 {
                        -3
                    } else if scaled > 3.0 {
                        3
                    } else {
                        scaled as i8
                    };
                    i8_coords.push(clamped);
                }

                // Pack coordinates (four 2-bit values per u8 byte)
                for chunk in i8_coords.chunks(4) {
                    let mut byte = 0u8;
                    for (i, &val) in chunk.iter().enumerate() {
                        let code = encode_2bit(val) & 0b11;
                        byte |= code << (i * 2);
                    }
                    projected_2bit.push(byte);
                }
            }
        }

        let record = WabRecord {
            id,
            original_vector: vector,
            projected_2bit,
            scale,
            nearest_grain,
        };

        let mut buf = self.buffer.lock().map_err(|e| e.to_string())?;
        buf.push(record);
        Ok(())
    }

    /// Searches both the main index grains and the uncompacted WAB buffer.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredVector>, String> {
        let idx_read = self.index.read().map_err(|e| e.to_string())?;

        // 1. Search main grains using search_mode_a
        let mut results = idx_read.search_mode_a(query, top_k, 1, idx_read.rerank_factor())?;

        // 2. Search uncompacted buffer
        let buf = self.buffer.lock().map_err(|e| e.to_string())?;
        for record in buf.iter() {
            let dist = query
                .iter()
                .zip(&record.original_vector)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();
            results.push(ScoredVector {
                id: record.id,
                distance: dist as f64,
            });
        }

        // Deduplicate and sort by distance
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.dedup_by_key(|r| r.id);
        results.truncate(top_k);
        Ok(results)
    }

    /// Returns the number of uncompacted records currently in the WAB.
    pub fn buffer_len(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    /// Spawns a background compaction thread that drains the current WAB buffer,
    /// clones the active index (zero-copy copy-on-write), appends/quantizes WAB records
    /// into the main grains, rebuilds the coarse grain routers, and atomically swaps it back.
    pub fn spawn_compaction(&self) -> thread::JoinHandle<Result<(), String>> {
        let index_arc = self.index.clone();
        let buffer_arc = self.buffer.clone();

        thread::spawn(move || {
            let batch = {
                let mut buf = buffer_arc.lock().map_err(|e| e.to_string())?;
                std::mem::take(&mut *buf)
            };

            if batch.is_empty() {
                return Ok(());
            }

            let mut index_clone = {
                let idx_read = index_arc.read().map_err(|e| e.to_string())?;
                idx_read.clone()
            };

            for record in batch {
                index_clone.insert(record.id, record.original_vector)?;
            }

            let num_grains = index_clone.grains().len();
            if num_grains > 1 {
                index_clone.rebuild_n_grains(num_grains)?;
            } else {
                index_clone.rebuild_single_grain()?;
            }

            let mut idx_write = index_arc.write().map_err(|e| e.to_string())?;
            *idx_write = index_clone;

            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_concurrent_write_search_and_compaction() {
        let index = AperonIndex::with_options(4, 2, 0, 4);
        let wab = Arc::new(WriteAheadBuffer::new(index));

        // Spawn a concurrent reader thread
        let wab_read = wab.clone();
        let reader_handle = thread::spawn(move || {
            let mut queries_run = 0;
            for _ in 0..100 {
                let query = vec![1.0, 2.0, 3.0, 4.0];
                let _results = wab_read.search(&query, 5).unwrap();
                queries_run += 1;
                thread::sleep(std::time::Duration::from_millis(1));
            }
            queries_run
        });

        // Insert vectors into WAB and verify throughput
        let start = Instant::now();
        let num_writes = 5000;
        for i in 0..num_writes {
            let id = VectorId::new(100 + i);
            let vector = vec![i as f32 * 0.1, 1.0, 2.0, 3.0];
            wab.insert(id, vector).unwrap();
        }
        let elapsed = start.elapsed();
        let throughput = num_writes as f64 / elapsed.as_secs_f64();
        println!("Write throughput: {} vectors/sec", throughput);

        // Assert write performance is > 20,000 vectors per second
        assert!(
            throughput > 20000.0,
            "Throughput is too low: {}/sec",
            throughput
        );
        assert_eq!(wab.buffer_len(), num_writes as usize);

        // Run background compaction
        let compaction_handle = wab.spawn_compaction();
        compaction_handle.join().unwrap().unwrap();

        // Verify that buffer is drained and compacted into main index
        assert_eq!(wab.buffer_len(), 0);
        assert_eq!(
            wab.index.read().unwrap().stats().vectors,
            num_writes as usize
        );

        // Wait for reader thread
        let queries_run = reader_handle.join().unwrap();
        assert!(queries_run > 0);
    }
}
