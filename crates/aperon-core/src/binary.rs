use std::io::{self, Read, Write};

pub const LEGACY_VERSION: u32 = 4;
pub const MIN_SUPPORTED_VERSION: u32 = 1;
const V4_FORMAT_LEGACY: u8 = 0;
const V4_FORMAT_SHARED_PQ: u8 = 1;
const V4_FORMAT_LATTICE_LEGACY: u8 = 2;
const V4_FORMAT_LATTICE_SHARED_PQ: u8 = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct RawVectors {
    pub num_vectors: u32,
    pub dimension: u32,
    pub vectors: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuerySet {
    pub num_queries: u32,
    pub dimension: u32,
    pub vectors: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacySingleGrain {
    pub num_vectors: u32,
    pub dimension: u32,
    pub local_dim: u32,
    pub block_size: u32,
    pub sketch_dim: u32,
    pub residual_bits: u8,
    pub mean: Vec<f32>,
    pub projection: Vec<f32>,
    pub proj_scales: Vec<f32>,
    pub residual_scale: f32,
    pub sketch_projection: Vec<f32>,
    pub sketch_scales: Vec<f32>,
    pub block_data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyMultiGrain {
    pub num_centroids: u32,
    pub num_vectors: u32,
    pub dimension: u32,
    pub local_dim: u32,
    pub block_size: u32,
    pub sketch_dim: u32,
    pub residual_bits: u8,
    pub centroids: Vec<f32>,
    pub grains: Vec<LegacySingleGrain>,
    pub shared: Option<LegacySharedQuantizer>,
    pub shared_grains: Vec<LegacySharedGrain>,
    pub lattice_router: Option<LegacyLatticeRouter>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacyLatticeRouter {
    pub routing_dim: u32,
    pub spacing: f32,
    pub projection: Vec<f32>,
    pub map_keys: Vec<Vec<i16>>,
    pub map_values: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacySharedQuantizer {
    pub basis_cols: u32,
    pub pq_subquantizers: u32,
    pub pq_bits: u8,
    pub opq: bool,
    pub basis: Vec<f32>,
    pub opq_rotation: Vec<f32>,
    pub pq_codebooks: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LegacySharedGrain {
    pub num_vectors: u32,
    pub local_dim: u32,
    pub block_size: u32,
    pub mean: Vec<f32>,
    pub column_indices: Vec<u8>,
    pub coord_scales: Vec<f32>,
    pub block_data: Vec<u8>,
    pub pq_codes: Vec<u8>,
    pub pq_error_scale: f32,
    pub pq_error_norms: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LegacyIndex {
    Single(LegacySingleGrain),
    Multi(LegacyMultiGrain),
}

pub fn load_raw_vectors(mut reader: impl Read) -> io::Result<RawVectors> {
    expect_magic(&mut reader, b"HNTR")?;
    expect_version(&mut reader)?;
    let num_vectors = read_u32(&mut reader)?;
    let dimension = read_u32(&mut reader)?;
    let vectors = read_f32_vec(&mut reader, num_vectors as usize * dimension as usize)?;
    Ok(RawVectors {
        num_vectors,
        dimension,
        vectors,
    })
}

pub fn load_queries(mut reader: impl Read) -> io::Result<QuerySet> {
    expect_magic(&mut reader, b"HNTQ")?;
    expect_version(&mut reader)?;
    let num_queries = read_u32(&mut reader)?;
    let dimension = read_u32(&mut reader)?;
    let vectors = read_f32_vec(&mut reader, num_queries as usize * dimension as usize)?;
    Ok(QuerySet {
        num_queries,
        dimension,
        vectors,
    })
}

pub fn load_legacy_index(mut reader: impl Read) -> io::Result<LegacyIndex> {
    let magic = read_magic(&mut reader)?;
    match &magic {
        b"HNTL" => Ok(LegacyIndex::Single(read_single_after_magic(&mut reader)?)),
        b"HNTM" => Ok(LegacyIndex::Multi(read_multi_after_magic(&mut reader)?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported legacy index magic",
        )),
    }
}

fn read_multi_after_magic(reader: &mut impl Read) -> io::Result<LegacyMultiGrain> {
    let version = read_supported_version(reader)?;
    let num_centroids = read_u32(reader)?;
    let num_vectors = read_u32(reader)?;
    let dimension = read_u32(reader)?;
    let local_dim = read_u32(reader)?;
    let block_size = read_u32(reader)?;
    let sketch_dim = read_u32(reader)?;
    let residual_bits = if version >= 2 { read_u8(reader)? } else { 8 };
    let format = if version >= 4 {
        read_u8(reader)?
    } else {
        V4_FORMAT_LEGACY
    };
    let centroids = read_f32_vec(reader, num_centroids as usize * dimension as usize)?;

    let is_shared = format == V4_FORMAT_SHARED_PQ || format == V4_FORMAT_LATTICE_SHARED_PQ;
    let has_lattice = format == V4_FORMAT_LATTICE_LEGACY || format == V4_FORMAT_LATTICE_SHARED_PQ;

    if is_shared {
        let shared = read_shared_quantizer(reader, dimension)?;
        let mut shared_grains = Vec::with_capacity(num_centroids as usize);
        for _ in 0..num_centroids {
            shared_grains.push(read_shared_grain(reader, dimension, &shared)?);
        }
        let lattice_router = if has_lattice {
            Some(read_lattice_router(reader, dimension)?)
        } else {
            None
        };
        return Ok(LegacyMultiGrain {
            num_centroids,
            num_vectors,
            dimension,
            local_dim,
            block_size,
            sketch_dim,
            residual_bits,
            centroids,
            grains: Vec::new(),
            shared: Some(shared),
            shared_grains,
            lattice_router,
        });
    }

    if format != V4_FORMAT_LEGACY && format != V4_FORMAT_LATTICE_LEGACY {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported v4 index format",
        ));
    }

    let mut grains = Vec::with_capacity(num_centroids as usize);
    for _ in 0..num_centroids {
        grains.push(read_embedded_single(
            reader,
            version,
            dimension,
            local_dim,
            block_size,
            sketch_dim,
            residual_bits,
        )?);
    }
    let lattice_router = if has_lattice {
        Some(read_lattice_router(reader, dimension)?)
    } else {
        None
    };
    Ok(LegacyMultiGrain {
        num_centroids,
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
        residual_bits,
        centroids,
        grains,
        shared: None,
        shared_grains: Vec::new(),
        lattice_router,
    })
}

fn read_single_after_magic(reader: &mut impl Read) -> io::Result<LegacySingleGrain> {
    let version = read_supported_version(reader)?;
    let num_vectors = read_u32(reader)?;
    let dimension = read_u32(reader)?;
    let local_dim = read_u32(reader)?;
    let block_size = read_u32(reader)?;
    let sketch_dim = read_u32(reader)?;
    let residual_bits = if version >= 2 { read_u8(reader)? } else { 8 };
    read_single_body(
        reader,
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
        residual_bits,
    )
}

fn read_embedded_single(
    reader: &mut impl Read,
    version: u32,
    dimension: u32,
    local_dim: u32,
    block_size: u32,
    sketch_dim: u32,
    residual_bits: u8,
) -> io::Result<LegacySingleGrain> {
    let num_vectors = read_u32(reader)?;
    let (local_dim, block_size, sketch_dim, residual_bits) = if version >= 3 {
        (
            read_u32(reader)?,
            read_u32(reader)?,
            read_u32(reader)?,
            read_u8(reader)?,
        )
    } else {
        (local_dim, block_size, sketch_dim, residual_bits)
    };
    read_single_body(
        reader,
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
        residual_bits,
    )
}

fn read_single_body(
    reader: &mut impl Read,
    num_vectors: u32,
    dimension: u32,
    local_dim: u32,
    block_size: u32,
    sketch_dim: u32,
    residual_bits: u8,
) -> io::Result<LegacySingleGrain> {
    validate_nonzero(dimension, "dimension")?;
    validate_nonzero(local_dim, "local_dim")?;
    validate_nonzero(block_size, "block_size")?;
    validate_residual_bits(residual_bits)?;
    let mean = read_f32_vec(reader, dimension as usize)?;
    let projection = read_f32_vec(reader, dimension as usize * local_dim as usize)?;
    let proj_scales = read_f32_vec(reader, local_dim as usize)?;
    let residual_scale = read_f32(reader)?;
    let (sketch_projection, sketch_scales) = if sketch_dim > 0 {
        (
            read_f32_vec(reader, dimension as usize * sketch_dim as usize)?,
            read_f32_vec(reader, sketch_dim as usize)?,
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let num_blocks = num_vectors.div_ceil(block_size);
    let block_bytes = block_size as usize
        * (local_dim as usize * size_of::<i16>() + size_of::<u16>() + size_of::<u32>())
        + packed_bytes(sketch_dim as usize * block_size as usize, residual_bits);
    let mut block_data = vec![0; num_blocks as usize * block_bytes];
    reader.read_exact(&mut block_data)?;
    Ok(LegacySingleGrain {
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
        residual_bits,
        mean,
        projection,
        proj_scales,
        residual_scale,
        sketch_projection,
        sketch_scales,
        block_data,
    })
}

fn read_magic(reader: &mut impl Read) -> io::Result<[u8; 4]> {
    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    Ok(magic)
}

fn read_shared_quantizer(
    reader: &mut impl Read,
    dimension: u32,
) -> io::Result<LegacySharedQuantizer> {
    let basis_cols = read_u32(reader)?;
    let pq_subquantizers = read_u32(reader)?;
    let pq_bits = read_u8(reader)?;
    let opq = read_u8(reader)? != 0;
    if !matches!(pq_bits, 4 | 8) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "pq_bits must be 4 or 8",
        ));
    }
    let vocabulary = 1_usize << pq_bits;
    Ok(LegacySharedQuantizer {
        basis_cols,
        pq_subquantizers,
        pq_bits,
        opq,
        basis: read_f32_vec(reader, dimension as usize * basis_cols as usize)?,
        opq_rotation: read_f32_vec(reader, dimension as usize * dimension as usize)?,
        pq_codebooks: read_f32_vec(reader, vocabulary * dimension as usize)?,
    })
}

fn read_lattice_router(reader: &mut impl Read, dimension: u32) -> io::Result<LegacyLatticeRouter> {
    let routing_dim = read_u32(reader)?;
    let spacing = read_f32(reader)?;
    let projection = read_f32_vec(reader, dimension as usize * routing_dim as usize)?;
    let num_entries = read_u32(reader)?;
    let mut map_keys = Vec::with_capacity(num_entries as usize);
    let mut map_values = Vec::with_capacity(num_entries as usize);
    for _ in 0..num_entries {
        let mut key = vec![0_i16; routing_dim as usize];
        for val in &mut key {
            let mut bytes = [0; 2];
            reader.read_exact(&mut bytes)?;
            *val = i16::from_le_bytes(bytes);
        }
        map_keys.push(key);
        let list_len = read_u32(reader)?;
        let mut val_list = Vec::with_capacity(list_len as usize);
        for _ in 0..list_len {
            val_list.push(read_u32(reader)?);
        }
        map_values.push(val_list);
    }
    Ok(LegacyLatticeRouter {
        routing_dim,
        spacing,
        projection,
        map_keys,
        map_values,
    })
}

fn write_lattice_router(writer: &mut impl Write, router: &LegacyLatticeRouter) -> io::Result<()> {
    write_u32(writer, router.routing_dim)?;
    write_f32(writer, router.spacing)?;
    write_f32_vec(writer, &router.projection)?;
    write_u32(writer, router.map_keys.len() as u32)?;
    for (key, values) in router.map_keys.iter().zip(&router.map_values) {
        for &val in key {
            writer.write_all(&val.to_le_bytes())?;
        }
        write_u32(writer, values.len() as u32)?;
        for &val in values {
            write_u32(writer, val)?;
        }
    }
    Ok(())
}

fn read_shared_grain(
    reader: &mut impl Read,
    dimension: u32,
    shared: &LegacySharedQuantizer,
) -> io::Result<LegacySharedGrain> {
    let num_vectors = read_u32(reader)?;
    let local_dim = read_u32(reader)?;
    let block_size = read_u32(reader)?;
    validate_nonzero(local_dim, "local_dim")?;
    validate_nonzero(block_size, "block_size")?;
    let mean = read_f32_vec(reader, dimension as usize)?;
    let mut column_indices = vec![0_u8; local_dim as usize];
    reader.read_exact(&mut column_indices)?;
    let coord_scales = read_f32_vec(reader, local_dim as usize)?;
    let num_blocks = num_vectors.div_ceil(block_size);
    let block_bytes = block_size as usize * (local_dim as usize + size_of::<u32>());
    let mut block_data = vec![0; num_blocks as usize * block_bytes];
    reader.read_exact(&mut block_data)?;
    let pq_code_values = num_vectors as usize * shared.pq_subquantizers as usize;
    let pq_code_bytes = if shared.pq_bits == 8 {
        pq_code_values
    } else {
        pq_code_values.div_ceil(2)
    };
    let mut pq_codes = vec![0_u8; pq_code_bytes];
    reader.read_exact(&mut pq_codes)?;
    Ok(LegacySharedGrain {
        num_vectors,
        local_dim,
        block_size,
        mean,
        column_indices,
        coord_scales,
        block_data,
        pq_codes,
        pq_error_scale: 1.0,
        pq_error_norms: Vec::new(),
    })
}

fn expect_magic(reader: &mut impl Read, expected: &[u8; 4]) -> io::Result<()> {
    let actual = read_magic(reader)?;
    if &actual == expected {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"))
    }
}

fn read_supported_version(reader: &mut impl Read) -> io::Result<u32> {
    let version = read_u32(reader)?;
    if (MIN_SUPPORTED_VERSION..=LEGACY_VERSION).contains(&version) {
        Ok(version)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported legacy version",
        ))
    }
}

fn expect_version(reader: &mut impl Read) -> io::Result<()> {
    read_supported_version(reader).map(|_| ())
}

fn validate_nonzero(value: u32, name: &str) -> io::Result<()> {
    if value == 0 {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name} must be nonzero"),
        ))
    } else {
        Ok(())
    }
}

fn validate_residual_bits(bits: u8) -> io::Result<()> {
    if matches!(bits, 1 | 2 | 8) {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "residual_bits must be 1, 2, or 8",
        ))
    }
}

fn packed_bytes(values: usize, bits: u8) -> usize {
    (values * bits as usize).div_ceil(8)
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_f32_vec(reader: &mut impl Read, count: usize) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(read_f32(reader)?);
    }
    Ok(out)
}

// ─── Writers ─────────────────────────────────────────────────────────────────

pub fn write_raw_vectors(mut writer: impl Write, raw: &RawVectors) -> io::Result<()> {
    writer.write_all(b"HNTR")?;
    write_u32(&mut writer, LEGACY_VERSION)?;
    write_u32(&mut writer, raw.num_vectors)?;
    write_u32(&mut writer, raw.dimension)?;
    write_f32_vec(&mut writer, &raw.vectors)
}

pub fn write_queries(mut writer: impl Write, qs: &QuerySet) -> io::Result<()> {
    writer.write_all(b"HNTQ")?;
    write_u32(&mut writer, LEGACY_VERSION)?;
    write_u32(&mut writer, qs.num_queries)?;
    write_u32(&mut writer, qs.dimension)?;
    write_f32_vec(&mut writer, &qs.vectors)
}

pub fn write_legacy_index(mut writer: impl Write, index: &LegacyIndex) -> io::Result<()> {
    match index {
        LegacyIndex::Single(single) => {
            writer.write_all(b"HNTL")?;
            write_u32(&mut writer, LEGACY_VERSION)?;
            write_u32(&mut writer, single.num_vectors)?;
            write_u32(&mut writer, single.dimension)?;
            write_u32(&mut writer, single.local_dim)?;
            write_u32(&mut writer, single.block_size)?;
            write_u32(&mut writer, single.sketch_dim)?;
            write_u8(&mut writer, single.residual_bits)?;
            write_single_body(&mut writer, single)
        }
        LegacyIndex::Multi(multi) => {
            writer.write_all(b"HNTM")?;
            write_u32(&mut writer, LEGACY_VERSION)?;
            write_u32(&mut writer, multi.num_centroids)?;
            write_u32(&mut writer, multi.num_vectors)?;
            write_u32(&mut writer, multi.dimension)?;
            write_u32(&mut writer, multi.local_dim)?;
            write_u32(&mut writer, multi.block_size)?;
            write_u32(&mut writer, multi.sketch_dim)?;
            write_u8(&mut writer, multi.residual_bits)?;
            let format = match (&multi.shared, &multi.lattice_router) {
                (Some(_), Some(_)) => V4_FORMAT_LATTICE_SHARED_PQ,
                (Some(_), None) => V4_FORMAT_SHARED_PQ,
                (None, Some(_)) => V4_FORMAT_LATTICE_LEGACY,
                (None, None) => V4_FORMAT_LEGACY,
            };
            write_u8(&mut writer, format)?;
            write_f32_vec(&mut writer, &multi.centroids)?;
            if let Some(shared) = &multi.shared {
                write_shared_quantizer(&mut writer, shared)?;
                for grain in &multi.shared_grains {
                    write_shared_grain(&mut writer, grain)?;
                }
            } else {
                for grain in &multi.grains {
                    write_embedded_single(&mut writer, grain)?
                }
            }
            if let Some(router) = &multi.lattice_router {
                write_lattice_router(&mut writer, router)?;
            }
            Ok(())
        }
    }
}

fn write_shared_quantizer(
    writer: &mut impl Write,
    shared: &LegacySharedQuantizer,
) -> io::Result<()> {
    write_u32(writer, shared.basis_cols)?;
    write_u32(writer, shared.pq_subquantizers)?;
    write_u8(writer, shared.pq_bits)?;
    write_u8(writer, u8::from(shared.opq))?;
    write_f32_vec(writer, &shared.basis)?;
    write_f32_vec(writer, &shared.opq_rotation)?;
    write_f32_vec(writer, &shared.pq_codebooks)
}

fn write_shared_grain(writer: &mut impl Write, grain: &LegacySharedGrain) -> io::Result<()> {
    write_u32(writer, grain.num_vectors)?;
    write_u32(writer, grain.local_dim)?;
    write_u32(writer, grain.block_size)?;
    write_f32_vec(writer, &grain.mean)?;
    writer.write_all(&grain.column_indices)?;
    write_f32_vec(writer, &grain.coord_scales)?;
    writer.write_all(&grain.block_data)?;
    writer.write_all(&grain.pq_codes)
}

/// Write a single grain body (mean, projection, scales, block_data) — no magic/version prefix.
fn write_single_body(writer: &mut impl Write, g: &LegacySingleGrain) -> io::Result<()> {
    write_f32_vec(writer, &g.mean)?;
    write_f32_vec(writer, &g.projection)?;
    write_f32_vec(writer, &g.proj_scales)?;
    write_f32(writer, g.residual_scale)?;
    if g.sketch_dim > 0 {
        write_f32_vec(writer, &g.sketch_projection)?;
        write_f32_vec(writer, &g.sketch_scales)?;
    }
    writer.write_all(&g.block_data)
}

/// Write a grain embedded inside a multi-grain index: num_vectors first, then the body.
fn write_embedded_single(writer: &mut impl Write, g: &LegacySingleGrain) -> io::Result<()> {
    write_u32(writer, g.num_vectors)?;
    write_u32(writer, g.local_dim)?;
    write_u32(writer, g.block_size)?;
    write_u32(writer, g.sketch_dim)?;
    write_u8(writer, g.residual_bits)?;
    write_single_body(writer, g)
}

fn write_u32(writer: &mut impl Write, v: u32) -> io::Result<()> {
    writer.write_all(&v.to_le_bytes())
}

fn write_u8(writer: &mut impl Write, v: u8) -> io::Result<()> {
    writer.write_all(&[v])
}

fn write_f32(writer: &mut impl Write, v: f32) -> io::Result<()> {
    writer.write_all(&v.to_le_bytes())
}

fn write_f32_vec(writer: &mut impl Write, v: &[f32]) -> io::Result<()> {
    for &x in v {
        write_f32(writer, x)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn loads_raw_vectors_wire_format() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HNTR");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let raw = load_raw_vectors(Cursor::new(bytes)).unwrap();

        assert_eq!(raw.num_vectors, 2);
        assert_eq!(raw.dimension, 3);
        assert_eq!(raw.vectors[5], 6.0);
    }

    #[test]
    fn loads_single_legacy_index_header_and_blocks() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HNTL");
        for value in [1_u32, 1, 2, 1, 2, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0_f32, 0.0, 1.0, 0.0, 1.0, 1.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(
            &[0_u8; 2 * (size_of::<i16>() + size_of::<u16>() + size_of::<u32>())],
        );

        let loaded = load_legacy_index(Cursor::new(bytes)).unwrap();

        match loaded {
            LegacyIndex::Single(single) => {
                assert_eq!(single.num_vectors, 1);
                assert_eq!(single.dimension, 2);
                assert_eq!(single.block_data.len(), 16);
            }
            LegacyIndex::Multi(_) => panic!("expected single index"),
        }
    }

    #[test]
    fn round_trips_raw_vectors() {
        let raw = RawVectors {
            num_vectors: 3,
            dimension: 2,
            vectors: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let mut buf = Vec::new();
        write_raw_vectors(&mut buf, &raw).unwrap();
        let loaded = load_raw_vectors(Cursor::new(buf)).unwrap();
        assert_eq!(loaded, raw);
    }

    fn make_single_grain(sketch_dim: u32) -> LegacySingleGrain {
        let num_vectors: u32 = 4;
        let dimension: u32 = 2;
        let local_dim: u32 = 1;
        let block_size: u32 = 2;
        // num_blocks = ceil(4/2) = 2
        // block_bytes = block_size * (local_dim * sizeof(i16) + sizeof(u16) + sizeof(u32)
        //               + sketch_dim * sizeof(i8))
        //             = 2 * (1*2 + 2 + 4 + sketch_dim*1)
        //             = 2 * (8 + sketch_dim)
        let block_bytes = block_size as usize
            * (local_dim as usize * size_of::<i16>()
                + size_of::<u16>()
                + size_of::<u32>()
                + sketch_dim as usize * size_of::<i8>());
        let num_blocks = num_vectors.div_ceil(block_size) as usize;
        let (sketch_projection, sketch_scales) = if sketch_dim > 0 {
            (
                vec![0.1_f32; dimension as usize * sketch_dim as usize],
                vec![1.0_f32; sketch_dim as usize],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        LegacySingleGrain {
            num_vectors,
            dimension,
            local_dim,
            block_size,
            sketch_dim,
            residual_bits: 8,
            mean: vec![0.0_f32; dimension as usize],
            projection: vec![1.0_f32; dimension as usize * local_dim as usize],
            proj_scales: vec![1.0_f32; local_dim as usize],
            residual_scale: 1.0,
            sketch_projection,
            sketch_scales,
            block_data: vec![0u8; num_blocks * block_bytes],
        }
    }

    #[test]
    fn round_trips_single_legacy_index() {
        let grain = make_single_grain(0);
        let index = LegacyIndex::Single(grain);
        let mut buf = Vec::new();
        write_legacy_index(&mut buf, &index).unwrap();
        let loaded = load_legacy_index(Cursor::new(buf)).unwrap();
        assert_eq!(loaded, index);
    }

    #[test]
    fn round_trips_single_with_sketch() {
        let grain = make_single_grain(2);
        let index = LegacyIndex::Single(grain);
        let mut buf = Vec::new();
        write_legacy_index(&mut buf, &index).unwrap();
        let loaded = load_legacy_index(Cursor::new(buf)).unwrap();
        assert_eq!(loaded, index);
    }
}
