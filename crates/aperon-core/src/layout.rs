/// Default block size for pointer-free structure-of-arrays storage.
pub const DEFAULT_BLOCK_SIZE: usize = 64;
pub const DUMMY_ID: u32 = u32::MAX;

/// Stable identifier for an original vector.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VectorId(u64);

impl From<u64> for VectorId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl VectorId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    pub const fn as_u32(self) -> Option<u32> {
        if self.0 <= u32::MAX as u64 {
            Some(self.0 as u32)
        } else {
            None
        }
    }
}

/// Memory-safe Block-SoA layout for quantized grain scans.
#[derive(Clone, Debug)]
pub struct BlockSoaLayout {
    local_dim: usize,
    sketch_dim: usize,
    block_size: usize,
    ids: Vec<VectorId>,
    coords: Vec<i16>,
    residuals: Vec<u16>,
    sketches: Vec<i8>,
}

impl BlockSoaLayout {
    pub fn new(dim: usize) -> Self {
        Self::with_shape(dim, 0, DEFAULT_BLOCK_SIZE)
    }

    pub fn with_shape(local_dim: usize, sketch_dim: usize, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        Self {
            local_dim,
            sketch_dim,
            block_size,
            ids: Vec::new(),
            coords: Vec::new(),
            residuals: Vec::new(),
            sketches: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.local_dim
    }

    pub fn local_dim(&self) -> usize {
        self.local_dim
    }

    pub fn sketch_dim(&self) -> usize {
        self.sketch_dim
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn push(&mut self, id: VectorId, vector: impl Into<Vec<f32>>) -> Result<(), String> {
        let vector = vector.into();
        if vector.len() != self.local_dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.local_dim,
                vector.len()
            ));
        }

        let coords = vector
            .into_iter()
            .map(|value| value as i16)
            .collect::<Vec<_>>();
        self.push_quantized(id, &coords, 0, &[])
    }

    pub fn push_quantized(
        &mut self,
        id: VectorId,
        coords: &[i16],
        residual: u16,
        sketches: &[i8],
    ) -> Result<(), String> {
        if coords.len() != self.local_dim {
            return Err(format!(
                "coordinate dimension mismatch: expected {}, got {}",
                self.local_dim,
                coords.len()
            ));
        }
        if sketches.len() != self.sketch_dim {
            return Err(format!(
                "sketch dimension mismatch: expected {}, got {}",
                self.sketch_dim,
                sketches.len()
            ));
        }

        let slot = self.ids.len() % self.block_size;
        if slot == 0 {
            self.start_block();
        }

        let block = self.ids.len() / self.block_size;
        for (k, coord) in coords.iter().enumerate() {
            let offset = self.coord_offset(block, k, slot);
            self.coords[offset] = *coord;
        }

        let residual_offset = block * self.block_size + slot;
        self.residuals[residual_offset] = residual;

        for (m, sketch) in sketches.iter().enumerate() {
            let offset = self.sketch_offset(block, m, slot);
            self.sketches[offset] = *sketch;
        }

        self.ids.push(id);
        Ok(())
    }

    pub fn id_at(&self, ordinal: usize) -> Option<VectorId> {
        self.ids.get(ordinal).copied()
    }

    pub fn ids(&self) -> &[VectorId] {
        &self.ids
    }

    pub fn block_count(&self) -> usize {
        self.ids.len().div_ceil(self.block_size)
    }

    pub fn block_len(&self, block: usize) -> usize {
        let start = block * self.block_size;
        self.ids.len().saturating_sub(start).min(self.block_size)
    }

    pub fn coord(&self, block: usize, dim: usize, lane: usize) -> i16 {
        self.coords[self.coord_offset(block, dim, lane)]
    }

    pub fn residual(&self, block: usize, lane: usize) -> u16 {
        self.residuals[block * self.block_size + lane]
    }

    pub fn sketch(&self, block: usize, dim: usize, lane: usize) -> i8 {
        self.sketches[self.sketch_offset(block, dim, lane)]
    }

    fn start_block(&mut self) {
        self.coords.extend(std::iter::repeat_n(
            i16::MAX,
            self.local_dim * self.block_size,
        ));
        self.residuals
            .extend(std::iter::repeat_n(u16::MAX, self.block_size));
        self.sketches.extend(std::iter::repeat_n(
            i8::MAX,
            self.sketch_dim * self.block_size,
        ));
    }

    fn coord_offset(&self, block: usize, dim: usize, lane: usize) -> usize {
        block * self.local_dim * self.block_size + dim * self.block_size + lane
    }

    fn sketch_offset(&self, block: usize, dim: usize, lane: usize) -> usize {
        block * self.sketch_dim * self.block_size + dim * self.block_size + lane
    }

    /// Serialise all blocks into the wire-format byte sequence expected by
    /// `binary::read_single_body`. Per block:
    ///   `local_dim * block_size` × i16 coords  (SoA, dim-major)
    ///   `block_size`             × u16 residuals
    ///   `block_size`             × u32 ids      (DUMMY_ID for padding slots)
    ///   `sketch_dim * block_size`× i8 sketches  (SoA, dim-major)
    pub fn raw_block_bytes(&self) -> Vec<u8> {
        let num_blocks = self.block_count();
        let bytes_per_block = self.block_size
            * (self.local_dim * size_of::<i16>()
                + size_of::<u16>()
                + size_of::<u32>()
                + self.sketch_dim * size_of::<i8>());
        let mut out = Vec::with_capacity(num_blocks * bytes_per_block);
        for b in 0..num_blocks {
            // coords: local_dim * block_size i16 (already stored SoA dim-major)
            let coord_start = b * self.local_dim * self.block_size;
            let coord_end = coord_start + self.local_dim * self.block_size;
            for &c in &self.coords[coord_start..coord_end] {
                out.extend_from_slice(&c.to_le_bytes());
            }
            // residuals: block_size u16
            let res_start = b * self.block_size;
            let res_end = res_start + self.block_size;
            for &r in &self.residuals[res_start..res_end] {
                out.extend_from_slice(&r.to_le_bytes());
            }
            // ids: block_size u32 (real ids then DUMMY_ID for padding)
            let real_len = self.block_len(b);
            for lane in 0..self.block_size {
                let id_u32 = if lane < real_len {
                    let ordinal = b * self.block_size + lane;
                    // Safety: ordinal < self.ids.len() because lane < real_len
                    self.ids[ordinal].as_u32().unwrap_or(DUMMY_ID)
                } else {
                    DUMMY_ID
                };
                out.extend_from_slice(&id_u32.to_le_bytes());
            }
            // sketches: sketch_dim * block_size i8 (stored SoA dim-major)
            let sk_start = b * self.sketch_dim * self.block_size;
            let sk_end = sk_start + self.sketch_dim * self.block_size;
            for &s in &self.sketches[sk_start..sk_end] {
                out.push(s as u8);
            }
        }
        out
    }

    pub fn from_raw_block_bytes(
        local_dim: usize,
        sketch_dim: usize,
        block_size: usize,
        num_vectors: usize,
        bytes: &[u8],
    ) -> Result<Self, String> {
        let bytes_per_block = block_size
            * (local_dim * size_of::<i16>()
                + size_of::<u16>()
                + size_of::<u32>()
                + sketch_dim * size_of::<i8>());
        let expected_len = num_vectors.div_ceil(block_size) * bytes_per_block;
        if bytes.len() != expected_len {
            return Err(format!(
                "block data length mismatch: expected {}, got {}",
                expected_len,
                bytes.len()
            ));
        }

        let mut layout = Self::with_shape(local_dim, sketch_dim, block_size);
        let num_blocks = num_vectors.div_ceil(block_size);
        for block in 0..num_blocks {
            let block_start = block * bytes_per_block;
            let coord_start = block_start;
            let residual_start = coord_start + local_dim * block_size * size_of::<i16>();
            let id_start = residual_start + block_size * size_of::<u16>();
            let sketch_start = id_start + block_size * size_of::<u32>();
            let real_lanes = num_vectors
                .saturating_sub(block * block_size)
                .min(block_size);

            for lane in 0..real_lanes {
                let id_offset = id_start + lane * size_of::<u32>();
                let id = u32::from_le_bytes(
                    bytes[id_offset..id_offset + size_of::<u32>()]
                        .try_into()
                        .map_err(|_| "invalid id bytes")?,
                );
                if id == DUMMY_ID {
                    return Err("dummy id found in a live vector lane".to_string());
                }

                let mut coords = Vec::with_capacity(local_dim);
                for dim in 0..local_dim {
                    let offset = coord_start + (dim * block_size + lane) * size_of::<i16>();
                    coords.push(i16::from_le_bytes(
                        bytes[offset..offset + size_of::<i16>()]
                            .try_into()
                            .map_err(|_| "invalid coord bytes")?,
                    ));
                }

                let residual_offset = residual_start + lane * size_of::<u16>();
                let residual = u16::from_le_bytes(
                    bytes[residual_offset..residual_offset + size_of::<u16>()]
                        .try_into()
                        .map_err(|_| "invalid residual bytes")?,
                );

                let mut sketches = Vec::with_capacity(sketch_dim);
                for dim in 0..sketch_dim {
                    let offset = sketch_start + dim * block_size + lane;
                    sketches.push(bytes[offset] as i8);
                }

                layout.push_quantized(
                    VectorId::new(u64::from(id)),
                    &coords,
                    residual,
                    &sketches,
                )?;
            }
        }
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_quantized_values_in_block_soa_order() {
        let mut layout = BlockSoaLayout::with_shape(2, 1, 4);
        layout
            .push_quantized(VectorId::new(7), &[10, 20], 3, &[4])
            .unwrap();
        layout
            .push_quantized(VectorId::new(8), &[11, 21], 5, &[6])
            .unwrap();

        assert_eq!(layout.block_count(), 1);
        assert_eq!(layout.coord(0, 0, 1), 11);
        assert_eq!(layout.coord(0, 1, 0), 20);
        assert_eq!(layout.residual(0, 1), 5);
        assert_eq!(layout.sketch(0, 0, 1), 6);
        assert_eq!(layout.id_at(0), Some(VectorId::new(7)));
    }
}
