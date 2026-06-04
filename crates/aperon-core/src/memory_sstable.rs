use crate::distance::l2_squared_unchecked;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

const SEGMENT_MAGIC: &[u8; 4] = b"APMS";
const SEGMENT_VERSION: u32 = 0;
const CHECKSUM_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const CHECKSUM_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryRecordInput {
    pub record_id: u64,
    pub scope_id: u32,
    pub timestamp: i64,
    pub source_id: u16,
    pub confidence: f32,
    pub text: String,
    pub embedding: Vec<f32>,
    pub symbols: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemorySegment {
    pub dim: usize,
    pub segment_id: u64,
    pub record_ids: Vec<u64>,
    pub scope_ids: Vec<u32>,
    pub timestamps: Vec<i64>,
    pub source_ids: Vec<u16>,
    pub confidences: Vec<f32>,
    pub text_offsets: Vec<u32>,
    pub text_bytes: Vec<u8>,
    pub embeddings: Vec<f32>,
    pub symbol_terms: Vec<String>,
    pub symbol_offsets: Vec<u32>,
    pub symbol_record_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryManifest {
    pub manifest_id: u64,
    pub parent_manifest_id: Option<u64>,
    pub branch_id: u64,
    pub segment_ids: Vec<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecallQuery {
    pub embedding: Option<Vec<f32>>,
    pub symbols: Vec<String>,
    pub scope_id: Option<u32>,
    pub time_start: Option<i64>,
    pub time_end: Option<i64>,
    pub min_confidence: Option<f32>,
    pub limit: usize,
    pub candidate_budget: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MemoryHit {
    pub record_id: u64,
    pub score: f32,
    pub semantic_distance: Option<f32>,
    pub symbol_matches: usize,
    pub confidence: f32,
    pub timestamp: i64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallTrace {
    pub segment_id: u64,
    pub access_paths: Vec<&'static str>,
    pub records_total: usize,
    pub candidates_after_filters: usize,
    pub candidates_after_symbols: usize,
    pub semantic_evals: usize,
    pub returned: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecallResult {
    pub hits: Vec<MemoryHit>,
    pub trace: RecallTrace,
}

impl MemorySegment {
    pub fn build(
        segment_id: u64,
        dim: usize,
        records: Vec<MemoryRecordInput>,
    ) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be greater than zero".to_string());
        }
        let mut record_ids = Vec::with_capacity(records.len());
        let mut scope_ids = Vec::with_capacity(records.len());
        let mut timestamps = Vec::with_capacity(records.len());
        let mut source_ids = Vec::with_capacity(records.len());
        let mut confidences = Vec::with_capacity(records.len());
        let mut text_offsets = Vec::with_capacity(records.len() + 1);
        let mut text_bytes = Vec::new();
        let mut embeddings = Vec::with_capacity(records.len() * dim);
        let mut postings = BTreeMap::<String, BTreeSet<u32>>::new();

        text_offsets.push(0);
        for (local_id, record) in records.into_iter().enumerate() {
            if record.embedding.len() != dim {
                return Err(format!(
                    "record {} embedding dimension mismatch: expected {}, got {}",
                    record.record_id,
                    dim,
                    record.embedding.len()
                ));
            }
            record_ids.push(record.record_id);
            scope_ids.push(record.scope_id);
            timestamps.push(record.timestamp);
            source_ids.push(record.source_id);
            confidences.push(record.confidence);
            text_bytes.extend_from_slice(record.text.as_bytes());
            text_offsets.push(text_bytes.len() as u32);
            embeddings.extend_from_slice(&record.embedding);
            for symbol in record.symbols {
                postings
                    .entry(normalize_symbol(&symbol))
                    .or_default()
                    .insert(local_id as u32);
            }
        }

        let mut symbol_terms = Vec::with_capacity(postings.len());
        let mut symbol_offsets = Vec::with_capacity(postings.len() + 1);
        let mut symbol_record_ids = Vec::new();
        symbol_offsets.push(0);
        for (term, ids) in postings {
            symbol_terms.push(term);
            symbol_record_ids.extend(ids);
            symbol_offsets.push(symbol_record_ids.len() as u32);
        }

        Ok(Self {
            dim,
            segment_id,
            record_ids,
            scope_ids,
            timestamps,
            source_ids,
            confidences,
            text_offsets,
            text_bytes,
            embeddings,
            symbol_terms,
            symbol_offsets,
            symbol_record_ids,
        })
    }

    pub fn len(&self) -> usize {
        self.record_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.record_ids.is_empty()
    }

    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.validate_layout().map_err(invalid_data)?;

        let record_count = checked_u32(self.len(), "record count")?;
        let dim = checked_u32(self.dim, "dimension")?;
        let text_bytes_len = checked_u64(self.text_bytes.len(), "text bytes length")?;
        let embedding_count = checked_u64(self.embeddings.len(), "embedding count")?;
        let symbol_count = checked_u32(self.symbol_terms.len(), "symbol count")?;
        let symbol_record_count =
            checked_u32(self.symbol_record_ids.len(), "symbol record id count")?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(SEGMENT_MAGIC);
        write_u32(&mut bytes, SEGMENT_VERSION);
        write_u64(&mut bytes, self.segment_id);
        write_u32(&mut bytes, dim);
        write_u32(&mut bytes, record_count);
        write_u64(&mut bytes, text_bytes_len);
        write_u64(&mut bytes, embedding_count);
        write_u32(&mut bytes, symbol_count);
        write_u32(&mut bytes, symbol_record_count);

        for value in &self.record_ids {
            write_u64(&mut bytes, *value);
        }
        for value in &self.scope_ids {
            write_u32(&mut bytes, *value);
        }
        for value in &self.timestamps {
            write_i64(&mut bytes, *value);
        }
        for value in &self.source_ids {
            write_u16(&mut bytes, *value);
        }
        for value in &self.confidences {
            write_f32(&mut bytes, *value);
        }
        for value in &self.text_offsets {
            write_u32(&mut bytes, *value);
        }
        bytes.extend_from_slice(&self.text_bytes);
        for value in &self.embeddings {
            write_f32(&mut bytes, *value);
        }
        for term in &self.symbol_terms {
            let term_bytes = term.as_bytes();
            write_u32(
                &mut bytes,
                checked_u32(term_bytes.len(), "symbol term length")?,
            );
            bytes.extend_from_slice(term_bytes);
        }
        for value in &self.symbol_offsets {
            write_u32(&mut bytes, *value);
        }
        for value in &self.symbol_record_ids {
            write_u32(&mut bytes, *value);
        }

        let checksum = checksum64(&bytes);
        write_u64(&mut bytes, checksum);
        fs::write(path, bytes)
    }

    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() < 52 {
            return Err(invalid_data("segment file is too short"));
        }
        let (payload, footer) = bytes.split_at(bytes.len() - 8);
        let expected_checksum = read_footer_checksum(footer)?;
        let actual_checksum = checksum64(payload);
        if expected_checksum != actual_checksum {
            return Err(invalid_data("segment checksum mismatch"));
        }

        let mut reader = SegmentReader::new(payload);
        reader.expect_magic()?;
        let version = reader.read_u32()?;
        if version != SEGMENT_VERSION {
            return Err(invalid_data(format!(
                "unsupported memory segment version: {}",
                version
            )));
        }
        let segment_id = reader.read_u64()?;
        let dim = reader.read_u32()? as usize;
        let record_count = reader.read_u32()? as usize;
        let text_bytes_len = usize::try_from(reader.read_u64()?)
            .map_err(|_| invalid_data("text_bytes_len does not fit in usize"))?;
        let embedding_count = usize::try_from(reader.read_u64()?)
            .map_err(|_| invalid_data("embedding_count does not fit in usize"))?;
        let symbol_count = reader.read_u32()? as usize;
        let symbol_record_count = reader.read_u32()? as usize;

        let record_ids = reader.read_u64_vec(record_count)?;
        let scope_ids = reader.read_u32_vec(record_count)?;
        let timestamps = reader.read_i64_vec(record_count)?;
        let source_ids = reader.read_u16_vec(record_count)?;
        let confidences = reader.read_f32_vec(record_count)?;
        let text_offsets = reader.read_u32_vec(record_count + 1)?;
        let text_bytes = reader.read_bytes(text_bytes_len)?.to_vec();
        let embeddings = reader.read_f32_vec(embedding_count)?;

        reader.check_remaining(symbol_count, 8)?;
        let mut symbol_terms = Vec::with_capacity(symbol_count);
        for _ in 0..symbol_count {
            let len = reader.read_u32()? as usize;
            let term = std::str::from_utf8(reader.read_bytes(len)?)
                .map_err(|_| invalid_data("symbol term is not valid utf-8"))?
                .to_string();
            symbol_terms.push(term);
        }
        let symbol_offsets = reader.read_u32_vec(symbol_count + 1)?;
        let symbol_record_ids = reader.read_u32_vec(symbol_record_count)?;
        reader.expect_end()?;

        let segment = Self {
            dim,
            segment_id,
            record_ids,
            scope_ids,
            timestamps,
            source_ids,
            confidences,
            text_offsets,
            text_bytes,
            embeddings,
            symbol_terms,
            symbol_offsets,
            symbol_record_ids,
        };
        segment.validate_layout().map_err(invalid_data)?;
        Ok(segment)
    }

    pub fn recall(&self, query: &RecallQuery) -> Result<RecallResult, String> {
        if let Some(embedding) = &query.embedding {
            if embedding.len() != self.dim {
                return Err(format!(
                    "query embedding dimension mismatch: expected {}, got {}",
                    self.dim,
                    embedding.len()
                ));
            }
        }

        let limit = query.limit.max(1);
        let mut access_paths = Vec::new();
        let mut candidates = Vec::with_capacity(self.len());
        for local_id in 0..self.len() {
            if self.passes_filters(local_id, query) {
                candidates.push(local_id as u32);
            }
        }
        if query.scope_id.is_some()
            || query.time_start.is_some()
            || query.time_end.is_some()
            || query.min_confidence.is_some()
        {
            access_paths.push("column_filters");
        }
        let candidates_after_filters = candidates.len();

        if !query.symbols.is_empty() {
            access_paths.push("symbol_postings");
            let mut allowed = BTreeSet::<u32>::new();
            for symbol in &query.symbols {
                if let Some(ids) = self.symbol_postings(symbol) {
                    allowed.extend(ids.iter().copied());
                }
            }
            candidates.retain(|id| allowed.contains(id));
        }
        let candidates_after_symbols = candidates.len();

        if let Some(budget) = query.candidate_budget {
            candidates.truncate(budget);
        }

        let mut scored = Vec::with_capacity(candidates.len());
        if query.embedding.is_some() {
            access_paths.push("semantic_rerank");
        }
        let query_embedding = query.embedding.as_deref();
        for local_id in candidates {
            let local_id = local_id as usize;
            let semantic_distance = query_embedding.map(|embedding| {
                l2_squared_unchecked(embedding, self.embedding_row(local_id)).sqrt()
            });
            let symbol_matches = self.symbol_match_count(local_id, &query.symbols);
            let score = self.score(local_id, semantic_distance, symbol_matches);
            scored.push((local_id, score, semantic_distance, symbol_matches));
        }
        scored.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.record_ids[a.0].cmp(&self.record_ids[b.0]))
        });

        let hits = scored
            .iter()
            .take(limit)
            .map(
                |&(local_id, score, semantic_distance, symbol_matches)| MemoryHit {
                    record_id: self.record_ids[local_id],
                    score,
                    semantic_distance,
                    symbol_matches,
                    confidence: self.confidences[local_id],
                    timestamp: self.timestamps[local_id],
                    text: self.text(local_id).to_string(),
                },
            )
            .collect::<Vec<_>>();

        Ok(RecallResult {
            trace: RecallTrace {
                segment_id: self.segment_id,
                access_paths,
                records_total: self.len(),
                candidates_after_filters,
                candidates_after_symbols,
                semantic_evals: scored.len(),
                returned: hits.len(),
            },
            hits,
        })
    }

    pub fn text(&self, local_id: usize) -> &str {
        let start = self.text_offsets[local_id] as usize;
        let end = self.text_offsets[local_id + 1] as usize;
        std::str::from_utf8(&self.text_bytes[start..end]).unwrap_or("")
    }

    fn passes_filters(&self, local_id: usize, query: &RecallQuery) -> bool {
        if query
            .scope_id
            .is_some_and(|scope_id| self.scope_ids[local_id] != scope_id)
        {
            return false;
        }
        if query
            .time_start
            .is_some_and(|start| self.timestamps[local_id] < start)
        {
            return false;
        }
        if query
            .time_end
            .is_some_and(|end| self.timestamps[local_id] > end)
        {
            return false;
        }
        if query
            .min_confidence
            .is_some_and(|min| self.confidences[local_id] < min)
        {
            return false;
        }
        true
    }

    fn symbol_postings(&self, symbol: &str) -> Option<&[u32]> {
        let symbol = normalize_symbol(symbol);
        let pos = self.symbol_terms.binary_search(&symbol).ok()?;
        let start = self.symbol_offsets[pos] as usize;
        let end = self.symbol_offsets[pos + 1] as usize;
        Some(&self.symbol_record_ids[start..end])
    }

    fn symbol_match_count(&self, local_id: usize, symbols: &[String]) -> usize {
        symbols
            .iter()
            .filter(|symbol| {
                self.symbol_postings(symbol)
                    .is_some_and(|ids| ids.binary_search(&(local_id as u32)).is_ok())
            })
            .count()
    }

    fn embedding_row(&self, local_id: usize) -> &[f32] {
        &self.embeddings[local_id * self.dim..(local_id + 1) * self.dim]
    }

    fn score(&self, local_id: usize, semantic_distance: Option<f32>, symbol_matches: usize) -> f32 {
        let semantic = semantic_distance.map_or(0.0, |dist| -dist);
        let symbol = symbol_matches as f32 * 2.0;
        let confidence = self.confidences[local_id];
        semantic + symbol + confidence
    }

    fn validate_layout(&self) -> Result<(), String> {
        if self.dim == 0 {
            return Err("dim must be greater than zero".to_string());
        }
        let record_count = self.record_ids.len();
        if self.scope_ids.len() != record_count
            || self.timestamps.len() != record_count
            || self.source_ids.len() != record_count
            || self.confidences.len() != record_count
        {
            return Err("record column length mismatch".to_string());
        }
        if self.text_offsets.len() != record_count + 1 {
            return Err("text_offsets must have record_count + 1 entries".to_string());
        }
        if self.text_offsets.first().copied() != Some(0) {
            return Err("text_offsets must start at zero".to_string());
        }
        if self.text_offsets.last().copied().map(usize::try_from) != Some(Ok(self.text_bytes.len()))
        {
            return Err("text_offsets must end at text_bytes length".to_string());
        }
        for window in self.text_offsets.windows(2) {
            if window[0] > window[1] {
                return Err("text_offsets must be monotonic".to_string());
            }
        }
        if std::str::from_utf8(&self.text_bytes).is_err() {
            return Err("text bytes must be valid utf-8".to_string());
        }
        if self.embeddings.len() != record_count * self.dim {
            return Err("embedding column length mismatch".to_string());
        }
        if self.symbol_offsets.len() != self.symbol_terms.len() + 1 {
            return Err("symbol_offsets must have symbol_count + 1 entries".to_string());
        }
        if self.symbol_offsets.first().copied() != Some(0) {
            return Err("symbol_offsets must start at zero".to_string());
        }
        if self.symbol_offsets.last().copied().map(usize::try_from)
            != Some(Ok(self.symbol_record_ids.len()))
        {
            return Err("symbol_offsets must end at symbol_record_ids length".to_string());
        }
        for window in self.symbol_offsets.windows(2) {
            if window[0] > window[1] {
                return Err("symbol_offsets must be monotonic".to_string());
            }
        }
        for term in &self.symbol_terms {
            if term != &normalize_symbol(term) {
                return Err("symbol terms must be normalized".to_string());
            }
        }
        for window in self.symbol_terms.windows(2) {
            if window[0] >= window[1] {
                return Err("symbol terms must be sorted and unique".to_string());
            }
        }
        for &local_id in &self.symbol_record_ids {
            if local_id as usize >= record_count {
                return Err("symbol posting local id out of range".to_string());
            }
        }
        Ok(())
    }
}

fn normalize_symbol(symbol: &str) -> String {
    symbol.trim().to_ascii_lowercase()
}

fn checked_u32(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_data(format!("{} does not fit in u32", name)))
}

fn checked_u64(value: usize, name: &str) -> io::Result<u64> {
    u64::try_from(value).map_err(|_| invalid_data(format!("{} does not fit in u64", name)))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn checksum64(bytes: &[u8]) -> u64 {
    let mut checksum = CHECKSUM_OFFSET_BASIS;
    for byte in bytes {
        checksum ^= u64::from(*byte);
        checksum = checksum.wrapping_mul(CHECKSUM_PRIME);
    }
    checksum
}

fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_footer_checksum(bytes: &[u8]) -> io::Result<u64> {
    let array: [u8; 8] = bytes
        .try_into()
        .map_err(|_| invalid_data("segment footer is malformed"))?;
    Ok(u64::from_le_bytes(array))
}

struct SegmentReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> SegmentReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn expect_magic(&mut self) -> io::Result<()> {
        let magic = self.read_bytes(SEGMENT_MAGIC.len())?;
        if magic != SEGMENT_MAGIC {
            return Err(invalid_data("unsupported memory segment magic"));
        }
        Ok(())
    }

    fn expect_end(&self) -> io::Result<()> {
        if self.offset != self.bytes.len() {
            return Err(invalid_data("trailing bytes in memory segment"));
        }
        Ok(())
    }

    fn check_remaining(&self, count: usize, size_of_element: usize) -> io::Result<()> {
        let remaining = self.bytes.len().saturating_sub(self.offset);
        if count.checked_mul(size_of_element).map_or(true, |needed| needed > remaining) {
            return Err(invalid_data("unexpected end of memory segment or size mismatch"));
        }
        Ok(())
    }

    fn read_bytes(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| invalid_data("segment offset overflow"))?;
        if end > self.bytes.len() {
            return Err(invalid_data("unexpected end of memory segment"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let bytes: [u8; 2] = self.read_bytes(2)?.try_into().unwrap();
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let bytes: [u8; 8] = self.read_bytes(8)?.try_into().unwrap();
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        let bytes: [u8; 8] = self.read_bytes(8)?.try_into().unwrap();
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        Ok(f32::from_le_bytes(bytes))
    }

    fn read_u16_vec(&mut self, len: usize) -> io::Result<Vec<u16>> {
        self.check_remaining(len, 2)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u16()?);
        }
        Ok(values)
    }

    fn read_u32_vec(&mut self, len: usize) -> io::Result<Vec<u32>> {
        self.check_remaining(len, 4)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u32()?);
        }
        Ok(values)
    }

    fn read_u64_vec(&mut self, len: usize) -> io::Result<Vec<u64>> {
        self.check_remaining(len, 8)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_u64()?);
        }
        Ok(values)
    }

    fn read_i64_vec(&mut self, len: usize) -> io::Result<Vec<i64>> {
        self.check_remaining(len, 8)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_i64()?);
        }
        Ok(values)
    }

    fn read_f32_vec(&mut self, len: usize) -> io::Result<Vec<f32>> {
        self.check_remaining(len, 4)?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.read_f32()?);
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn recalls_with_symbol_scope_and_semantic_rerank() {
        let segment = MemorySegment::build(
            7,
            3,
            vec![
                record(
                    1,
                    10,
                    100,
                    "prefix8 failed at K10000",
                    [1.0, 0.0, 0.0],
                    &["T-173", "prefix8"],
                ),
                record(
                    2,
                    10,
                    110,
                    "uint16 dense fallback is stable",
                    [0.0, 1.0, 0.0],
                    &["T-172", "uint16"],
                ),
                record(
                    3,
                    11,
                    120,
                    "unrelated other project note",
                    [1.0, 0.0, 0.0],
                    &["T-173"],
                ),
            ],
        )
        .unwrap();

        let result = segment
            .recall(&RecallQuery {
                embedding: Some(vec![1.0, 0.1, 0.0]),
                symbols: vec!["prefix8".to_string()],
                scope_id: Some(10),
                limit: 3,
                ..RecallQuery::default()
            })
            .unwrap();

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].record_id, 1);
        assert_eq!(
            result.trace.access_paths,
            vec!["column_filters", "symbol_postings", "semantic_rerank"]
        );
        assert_eq!(result.trace.candidates_after_filters, 2);
        assert_eq!(result.trace.candidates_after_symbols, 1);
    }

    #[test]
    fn manifest_models_branchable_memory_views() {
        let base = MemoryManifest {
            manifest_id: 1,
            parent_manifest_id: None,
            branch_id: 42,
            segment_ids: vec![10, 11],
        };
        let branch = MemoryManifest {
            manifest_id: 2,
            parent_manifest_id: Some(base.manifest_id),
            branch_id: 43,
            segment_ids: vec![10, 11, 12],
        };
        assert_eq!(branch.parent_manifest_id, Some(1));
        assert_eq!(branch.segment_ids, vec![10, 11, 12]);
    }

    #[test]
    fn segment_file_round_trip_preserves_recall_and_layout() {
        let segment = sample_segment();
        let path = temp_segment_path("round-trip");
        segment.write(&path).unwrap();

        let loaded = MemorySegment::read(&path).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!(loaded, segment);
        assert_eq!(loaded.dim, 3);
        assert_eq!(loaded.text_offsets, segment.text_offsets);
        assert_eq!(loaded.embeddings.len(), loaded.len() * loaded.dim);
        assert_eq!(
            loaded.symbol_terms,
            vec!["prefix8", "t-172", "t-173", "uint16"]
        );
        assert_eq!(loaded.symbol_offsets, segment.symbol_offsets);
        assert_eq!(loaded.symbol_record_ids, segment.symbol_record_ids);

        let query = RecallQuery {
            embedding: Some(vec![1.0, 0.1, 0.0]),
            symbols: vec!["prefix8".to_string()],
            scope_id: Some(10),
            limit: 3,
            ..RecallQuery::default()
        };
        assert_eq!(
            loaded.recall(&query).unwrap(),
            segment.recall(&query).unwrap()
        );
    }

    #[test]
    fn segment_file_rejects_bad_magic() {
        let segment = sample_segment();
        let path = temp_segment_path("bad-magic");
        segment.write(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[0] = b'X';
        rewrite_checksum(&mut bytes);
        fs::write(&path, bytes).unwrap();

        let error = MemorySegment::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("magic"));
    }

    #[test]
    fn segment_file_rejects_bad_version() {
        let segment = sample_segment();
        let path = temp_segment_path("bad-version");
        segment.write(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        rewrite_checksum(&mut bytes);
        fs::write(&path, bytes).unwrap();

        let error = MemorySegment::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("version"));
    }

    #[test]
    fn segment_file_rejects_checksum_mismatch() {
        let segment = sample_segment();
        let path = temp_segment_path("bad-checksum");
        segment.write(&path).unwrap();

        let mut bytes = fs::read(&path).unwrap();
        bytes[16] ^= 0xff;
        fs::write(&path, bytes).unwrap();

        let error = MemorySegment::read(&path).unwrap_err();
        fs::remove_file(&path).unwrap();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checksum"));
    }

    fn sample_segment() -> MemorySegment {
        MemorySegment::build(
            7,
            3,
            vec![
                record(
                    1,
                    10,
                    100,
                    "prefix8 failed at K10000",
                    [1.0, 0.0, 0.0],
                    &["T-173", "prefix8"],
                ),
                record(
                    2,
                    10,
                    110,
                    "uint16 dense fallback is stable",
                    [0.0, 1.0, 0.0],
                    &["T-172", "uint16"],
                ),
                record(
                    3,
                    11,
                    120,
                    "unrelated other project note",
                    [1.0, 0.0, 0.0],
                    &["T-173"],
                ),
            ],
        )
        .unwrap()
    }

    fn temp_segment_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "aperon-memory-segment-{}-{}-{}.apms",
            name,
            std::process::id(),
            nanos
        ))
    }

    fn rewrite_checksum(bytes: &mut [u8]) {
        let checksum_start = bytes.len() - 8;
        let checksum = checksum64(&bytes[..checksum_start]);
        bytes[checksum_start..].copy_from_slice(&checksum.to_le_bytes());
    }

    fn record(
        record_id: u64,
        scope_id: u32,
        timestamp: i64,
        text: &str,
        embedding: [f32; 3],
        symbols: &[&str],
    ) -> MemoryRecordInput {
        MemoryRecordInput {
            record_id,
            scope_id,
            timestamp,
            source_id: 1,
            confidence: 1.0,
            text: text.to_string(),
            embedding: embedding.to_vec(),
            symbols: symbols.iter().map(|symbol| symbol.to_string()).collect(),
        }
    }
}
