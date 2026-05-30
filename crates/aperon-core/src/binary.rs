use std::io::{self, Read, Write};

pub const LEGACY_VERSION: u32 = 3;
pub const MIN_SUPPORTED_VERSION: u32 = 1;

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
    let centroids = read_f32_vec(reader, num_centroids as usize * dimension as usize)?;
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
            write_f32_vec(&mut writer, &multi.centroids)?;
            for grain in &multi.grains {
                write_embedded_single(&mut writer, grain)?
            }
            Ok(())
        }
    }
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
