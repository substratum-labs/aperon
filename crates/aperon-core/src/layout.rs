/// Default block size for pointer-free structure-of-arrays storage.
pub const DEFAULT_BLOCK_SIZE: usize = 64;
pub const DUMMY_ID: u32 = u32::MAX;
pub const DEFAULT_RESIDUAL_BITS: u8 = 8;

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
    residual_bits: u8,
    block_size: usize,
    ids: std::sync::Arc<Vec<VectorId>>,
    coords: std::sync::Arc<Vec<i16>>,
    residuals: std::sync::Arc<Vec<u16>>,
    sketches: std::sync::Arc<Vec<u8>>,
}

impl BlockSoaLayout {
    pub fn new(dim: usize) -> Self {
        Self::with_shape(dim, 0, DEFAULT_BLOCK_SIZE)
    }

    pub fn with_shape(local_dim: usize, sketch_dim: usize, block_size: usize) -> Self {
        Self::with_shape_and_residual_bits(local_dim, sketch_dim, block_size, DEFAULT_RESIDUAL_BITS)
    }

    pub fn with_shape_and_residual_bits(
        local_dim: usize,
        sketch_dim: usize,
        block_size: usize,
        residual_bits: u8,
    ) -> Self {
        assert!(block_size > 0, "block_size must be positive");
        assert!(
            matches!(residual_bits, 1 | 2 | 8),
            "residual_bits must be 1, 2, or 8"
        );
        Self {
            local_dim,
            sketch_dim,
            residual_bits,
            block_size,
            ids: std::sync::Arc::new(Vec::new()),
            coords: std::sync::Arc::new(Vec::new()),
            residuals: std::sync::Arc::new(Vec::new()),
            sketches: std::sync::Arc::new(Vec::new()),
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

    pub fn residual_bits(&self) -> u8 {
        self.residual_bits
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

        {
            let coords_vec = std::sync::Arc::make_mut(&mut self.coords);
            let local_dim = self.local_dim;
            let block_size = self.block_size;
            for (k, coord) in coords.iter().enumerate() {
                let offset = block * local_dim * block_size + k * block_size + slot;
                coords_vec[offset] = *coord;
            }
        }

        let residual_offset = block * self.block_size + slot;
        std::sync::Arc::make_mut(&mut self.residuals)[residual_offset] = residual;

        for (m, sketch) in sketches.iter().enumerate() {
            self.set_sketch(block, m, slot, *sketch);
        }

        std::sync::Arc::make_mut(&mut self.ids).push(id);
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
        match self.residual_bits {
            8 => self.sketches[self.sketch_offset(block, dim, lane)] as i8,
            2 => decode_2bit(self.sketch_code(block, dim, lane)),
            1 => decode_1bit(self.sketch_code(block, dim, lane)),
            _ => unreachable!("validated residual_bits"),
        }
    }

    pub(crate) fn coord_block(&self, block: usize, dim: usize) -> &[i16] {
        let start = self.coord_offset(block, dim, 0);
        &self.coords[start..start + self.block_size]
    }

    #[allow(dead_code)]
    pub(crate) fn block_coords_ptr(&self, block: usize) -> *const i16 {
        let start = block * self.local_dim * self.block_size;
        unsafe { self.coords.as_ptr().add(start) }
    }

    #[allow(dead_code)]
    pub(crate) fn block_sketches_ptr(&self, block: usize) -> *const i8 {
        let start = block * self.sketch_dim * self.block_size;
        unsafe { self.sketches.as_ptr().add(start) as *const i8 }
    }

    pub(crate) fn residual_block(&self, block: usize) -> &[u16] {
        let start = block * self.block_size;
        &self.residuals[start..start + self.block_size]
    }

    pub(crate) fn sketch_block(&self, block: usize, dim: usize) -> &[i8] {
        assert_eq!(self.residual_bits, 8);
        let start = self.sketch_offset(block, dim, 0);
        let bytes = &self.sketches[start..start + self.block_size];
        // i8 and u8 have identical layout; this view keeps the SIMD scan path zero-copy.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) }
    }

    fn start_block(&mut self) {
        let coords_len = self.local_dim * self.block_size;
        let block_size = self.block_size;
        let sketch_bytes = self.sketch_bytes_per_block();

        std::sync::Arc::make_mut(&mut self.coords)
            .extend(std::iter::repeat_n(i16::MAX, coords_len));
        std::sync::Arc::make_mut(&mut self.residuals)
            .extend(std::iter::repeat_n(u16::MAX, block_size));
        std::sync::Arc::make_mut(&mut self.sketches)
            .extend(std::iter::repeat_n(0xff, sketch_bytes));
    }

    fn coord_offset(&self, block: usize, dim: usize, lane: usize) -> usize {
        block * self.local_dim * self.block_size + dim * self.block_size + lane
    }

    fn sketch_offset(&self, block: usize, dim: usize, lane: usize) -> usize {
        block * self.sketch_dim * self.block_size + dim * self.block_size + lane
    }

    fn sketch_bytes_per_block(&self) -> usize {
        packed_bytes(self.sketch_dim * self.block_size, self.residual_bits)
    }

    fn packed_sketch_offset(&self, block: usize, dim: usize, lane: usize) -> (usize, u8) {
        let bit_idx = (dim * self.block_size + lane) * self.residual_bits as usize;
        let byte = block * self.sketch_bytes_per_block() + bit_idx / 8;
        let shift = (bit_idx % 8) as u8;
        (byte, shift)
    }

    fn sketch_code(&self, block: usize, dim: usize, lane: usize) -> u8 {
        let (byte, shift) = self.packed_sketch_offset(block, dim, lane);
        let mask = (1_u8 << self.residual_bits) - 1;
        (self.sketches[byte] >> shift) & mask
    }

    fn set_sketch(&mut self, block: usize, dim: usize, lane: usize, value: i8) {
        match self.residual_bits {
            8 => {
                let offset = self.sketch_offset(block, dim, lane);
                std::sync::Arc::make_mut(&mut self.sketches)[offset] = value as u8;
            }
            2 => self.set_sketch_code(block, dim, lane, encode_2bit(value)),
            1 => self.set_sketch_code(block, dim, lane, encode_1bit(value)),
            _ => unreachable!("validated residual_bits"),
        }
    }

    fn set_sketch_code(&mut self, block: usize, dim: usize, lane: usize, code: u8) {
        let (byte, shift) = self.packed_sketch_offset(block, dim, lane);
        let mask = ((1_u8 << self.residual_bits) - 1) << shift;
        let sketches = std::sync::Arc::make_mut(&mut self.sketches);
        sketches[byte] = (sketches[byte] & !mask) | ((code << shift) & mask);
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
            * (self.local_dim * size_of::<i16>() + size_of::<u16>() + size_of::<u32>())
            + self.sketch_bytes_per_block();
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
            let sk_start = b * self.sketch_bytes_per_block();
            let sk_end = sk_start + self.sketch_bytes_per_block();
            out.extend_from_slice(&self.sketches[sk_start..sk_end]);
        }
        out
    }

    pub fn from_raw_block_bytes(
        local_dim: usize,
        sketch_dim: usize,
        residual_bits: u8,
        block_size: usize,
        num_vectors: usize,
        bytes: &[u8],
    ) -> Result<Self, String> {
        validate_residual_bits(residual_bits)?;
        let bytes_per_block = block_size
            * (local_dim * size_of::<i16>() + size_of::<u16>() + size_of::<u32>())
            + packed_bytes(sketch_dim * block_size, residual_bits);
        let expected_len = num_vectors.div_ceil(block_size) * bytes_per_block;
        if bytes.len() != expected_len {
            return Err(format!(
                "block data length mismatch: expected {}, got {}",
                expected_len,
                bytes.len()
            ));
        }

        let mut layout =
            Self::with_shape_and_residual_bits(local_dim, sketch_dim, block_size, residual_bits);
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
                    sketches.push(read_packed_sketch(
                        bytes,
                        sketch_start,
                        sketch_dim,
                        block_size,
                        residual_bits,
                        dim,
                        lane,
                    )?);
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

/// A cache-aligned structure-of-arrays block representing a subset of quantized vectors.
/// Maps directly to disk/memory-mapped files for zero-copy scans.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedBlock<const LOCAL_DIM: usize, const BLOCK_SIZE: usize> {
    pub coords: [[i16; BLOCK_SIZE]; LOCAL_DIM],
    pub residual_norms: [u16; BLOCK_SIZE],
    pub ids: [u32; BLOCK_SIZE],
}

impl<const LOCAL_DIM: usize, const BLOCK_SIZE: usize> PackedBlock<LOCAL_DIM, BLOCK_SIZE> {
    /// Casts a byte slice directly to a reference to `PackedBlock`.
    ///
    /// # Safety
    /// The caller must ensure the byte slice is aligned to 64 bytes.
    /// If alignment is incorrect, this returns an Err indicating alignment mismatch.
    pub fn from_bytes(bytes: &[u8]) -> Result<&Self, String> {
        let expected_size = std::mem::size_of::<Self>();
        if bytes.len() < expected_size {
            return Err(format!(
                "byte slice is too small: expected at least {}, got {}",
                expected_size,
                bytes.len()
            ));
        }

        let ptr = bytes.as_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<Self>()) {
            return Err(format!(
                "byte slice pointer ({:p}) is not aligned to {}",
                ptr,
                std::mem::align_of::<Self>()
            ));
        }

        // Safety: We have verified the size and alignment.
        // PackedBlock only contains primitive types (i16, u16, u32) which have no invalid bit patterns.
        unsafe { Ok(&*(ptr as *const Self)) }
    }

    /// Casts a mutable byte slice directly to a mutable reference to `PackedBlock`.
    pub fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut Self, String> {
        let expected_size = std::mem::size_of::<Self>();
        if bytes.len() < expected_size {
            return Err(format!(
                "byte slice is too small: expected at least {}, got {}",
                expected_size,
                bytes.len()
            ));
        }

        let ptr = bytes.as_mut_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<Self>()) {
            return Err(format!(
                "byte slice pointer ({:p}) is not aligned to {}",
                ptr,
                std::mem::align_of::<Self>()
            ));
        }

        // Safety: We have verified the size and alignment.
        unsafe { Ok(&mut *(ptr as *mut Self)) }
    }

    /// Reads an unaligned byte slice into a `PackedBlock` copy.
    pub fn read_unaligned(bytes: &[u8]) -> Result<Self, String> {
        let expected_size = std::mem::size_of::<Self>();
        if bytes.len() < expected_size {
            return Err(format!(
                "byte slice is too small: expected at least {}, got {}",
                expected_size,
                bytes.len()
            ));
        }
        // Safety: PackedBlock only contains plain-old-data copy types.
        unsafe { Ok(std::ptr::read_unaligned(bytes.as_ptr() as *const Self)) }
    }
}

pub fn validate_residual_bits(bits: u8) -> Result<(), String> {
    if matches!(bits, 1 | 2 | 8) {
        Ok(())
    } else {
        Err(format!("residual_bits must be 1, 2, or 8, got {bits}"))
    }
}

fn packed_bytes(values: usize, bits: u8) -> usize {
    (values * bits as usize).div_ceil(8)
}

fn encode_1bit(value: i8) -> u8 {
    if value >= 0 {
        1
    } else {
        0
    }
}

fn decode_1bit(code: u8) -> i8 {
    if code & 1 == 0 {
        -1
    } else {
        1
    }
}

pub(crate) fn encode_2bit(value: i8) -> u8 {
    match value {
        i8::MIN..=-2 => 0,
        -1 => 1,
        0 | 1 => 2,
        _ => 3,
    }
}

pub(crate) fn decode_2bit(code: u8) -> i8 {
    match code & 0b11 {
        0 => -3,
        1 => -1,
        2 => 1,
        _ => 3,
    }
}

fn read_packed_sketch(
    bytes: &[u8],
    sketch_start: usize,
    _sketch_dim: usize,
    block_size: usize,
    residual_bits: u8,
    dim: usize,
    lane: usize,
) -> Result<i8, String> {
    if residual_bits == 8 {
        return Ok(bytes[sketch_start + dim * block_size + lane] as i8);
    }
    let bit_idx = (dim * block_size + lane) * residual_bits as usize;
    let byte = sketch_start + bit_idx / 8;
    let shift = (bit_idx % 8) as u8;
    let code = bytes
        .get(byte)
        .ok_or_else(|| "invalid packed sketch bytes".to_string())?
        >> shift;
    Ok(match residual_bits {
        1 => decode_1bit(code),
        2 => decode_2bit(code),
        _ => unreachable!("validated residual_bits"),
    })
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

    #[test]
    fn packs_two_bit_residual_sketches() {
        let mut layout = BlockSoaLayout::with_shape_and_residual_bits(1, 2, 4, 2);
        layout
            .push_quantized(VectorId::new(1), &[7], 0, &[-3, 1])
            .unwrap();
        layout
            .push_quantized(VectorId::new(2), &[8], 0, &[-1, 3])
            .unwrap();

        assert_eq!(layout.sketch(0, 0, 0), -3);
        assert_eq!(layout.sketch(0, 0, 1), -1);
        assert_eq!(layout.sketch(0, 1, 0), 1);
        assert_eq!(layout.sketch(0, 1, 1), 3);
        assert_eq!(layout.raw_block_bytes().len(), 4 * (2 + 2 + 4) + 2);
    }

    #[test]
    fn zero_copy_packed_block_casting() {
        // We use a local dim of 2 and a block size of 4.
        // Size: coords = 2 * 4 * 2 = 16 bytes.
        // residual_norms = 4 * 2 = 8 bytes.
        // ids = 4 * 4 = 16 bytes.
        // Total = 40 bytes.
        // Aligned to 64 bytes, so size_of is 64.

        #[repr(C, align(64))]
        #[derive(Clone, Copy)]
        struct AlignedBuffer([u8; 64]);
        let mut buf = AlignedBuffer([0u8; 64]);

        let coords_0 = [1i16, 2, 3, 4];
        let coords_1 = [10i16, 20, 30, 40];
        let residuals = [100u16, 200, 300, 400];
        let ids = [1000u32, 2000, 3000, 4000];

        // Copy them into the buffer using native endianness (since PackedBlock zero-copy cast accesses native).
        let mut offset = 0;
        for &val in &coords_0 {
            buf.0[offset..offset + 2].copy_from_slice(&val.to_ne_bytes());
            offset += 2;
        }
        for &val in &coords_1 {
            buf.0[offset..offset + 2].copy_from_slice(&val.to_ne_bytes());
            offset += 2;
        }
        for &val in &residuals {
            buf.0[offset..offset + 2].copy_from_slice(&val.to_ne_bytes());
            offset += 2;
        }
        for &val in &ids {
            buf.0[offset..offset + 4].copy_from_slice(&val.to_ne_bytes());
            offset += 4;
        }

        // Try casting it.
        let block = PackedBlock::<2, 4>::from_bytes(&buf.0).unwrap();
        assert_eq!(block.coords[0], coords_0);
        assert_eq!(block.coords[1], coords_1);
        assert_eq!(block.residual_norms, residuals);
        assert_eq!(block.ids, ids);

        // Try mutable casting.
        let mut buf_mut = buf;
        let block_mut = PackedBlock::<2, 4>::from_bytes_mut(&mut buf_mut.0).unwrap();
        block_mut.coords[0][0] = 999;
        assert_eq!(block_mut.coords[0][0], 999);

        // Try unaligned.
        let mut unaligned_buf = [0u8; 64];
        unaligned_buf[..40].copy_from_slice(&buf.0[..40]);
        let block_unaligned = PackedBlock::<2, 4>::read_unaligned(&unaligned_buf).unwrap();
        assert_eq!(block_unaligned.coords[0], coords_0);
        assert_eq!(block_unaligned.coords[1], coords_1);
        assert_eq!(block_unaligned.residual_norms, residuals);
        assert_eq!(block_unaligned.ids, ids);

        // Verify that casting on unaligned returns an error (using an odd offset guaranteed to be unaligned for align=64).
        let shifted_buf = &unaligned_buf[1..];
        assert!(PackedBlock::<2, 4>::from_bytes(shifted_buf).is_err());
    }
}
