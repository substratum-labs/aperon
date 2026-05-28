use std::io::{self, Read};

pub const LEGACY_VERSION: u32 = 1;

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
    expect_version(reader)?;
    let num_centroids = read_u32(reader)?;
    let num_vectors = read_u32(reader)?;
    let dimension = read_u32(reader)?;
    let local_dim = read_u32(reader)?;
    let block_size = read_u32(reader)?;
    let sketch_dim = read_u32(reader)?;
    let centroids = read_f32_vec(reader, num_centroids as usize * dimension as usize)?;
    let mut grains = Vec::with_capacity(num_centroids as usize);
    for _ in 0..num_centroids {
        grains.push(read_embedded_single(
            reader, dimension, local_dim, block_size, sketch_dim,
        )?);
    }
    Ok(LegacyMultiGrain {
        num_centroids,
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
        centroids,
        grains,
    })
}

fn read_single_after_magic(reader: &mut impl Read) -> io::Result<LegacySingleGrain> {
    expect_version(reader)?;
    let num_vectors = read_u32(reader)?;
    let dimension = read_u32(reader)?;
    let local_dim = read_u32(reader)?;
    let block_size = read_u32(reader)?;
    let sketch_dim = read_u32(reader)?;
    read_single_body(
        reader,
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
    )
}

fn read_embedded_single(
    reader: &mut impl Read,
    dimension: u32,
    local_dim: u32,
    block_size: u32,
    sketch_dim: u32,
) -> io::Result<LegacySingleGrain> {
    let num_vectors = read_u32(reader)?;
    read_single_body(
        reader,
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
    )
}

fn read_single_body(
    reader: &mut impl Read,
    num_vectors: u32,
    dimension: u32,
    local_dim: u32,
    block_size: u32,
    sketch_dim: u32,
) -> io::Result<LegacySingleGrain> {
    validate_nonzero(dimension, "dimension")?;
    validate_nonzero(local_dim, "local_dim")?;
    validate_nonzero(block_size, "block_size")?;
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
        * (local_dim as usize * size_of::<i16>()
            + size_of::<u16>()
            + size_of::<u32>()
            + sketch_dim as usize * size_of::<i8>());
    let mut block_data = vec![0; num_blocks as usize * block_bytes];
    reader.read_exact(&mut block_data)?;
    Ok(LegacySingleGrain {
        num_vectors,
        dimension,
        local_dim,
        block_size,
        sketch_dim,
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

fn expect_version(reader: &mut impl Read) -> io::Result<()> {
    let version = read_u32(reader)?;
    if version == LEGACY_VERSION {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported legacy version",
        ))
    }
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

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
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
}
