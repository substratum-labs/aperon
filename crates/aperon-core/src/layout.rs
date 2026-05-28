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
