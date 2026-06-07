use aperon_core::{MemoryRecordInput, MemorySegment, RecallQuery};

fn main() {
    let segment = MemorySegment::build(
        174,
        4,
        vec![
            record(
                1,
                7,
                171,
                "T-171 proved the pointerless pivot-prefix route kernel can run with zero hot-query allocations.",
                [1.0, 0.0, 0.0, 0.0],
                &["T-171", "pivot-prefix", "allocation"],
            ),
            record(
                2,
                7,
                172,
                "T-172 recommends u16 posting IDs and uint16 dense l_inf signatures for compact fallback.",
                [0.8, 0.1, 0.1, 0.0],
                &["T-172", "uint16", "layout"],
            ),
            record(
                3,
                7,
                173,
                "T-173 found prefix8 block64 misses coverage at K10000 and requires planner fallback.",
                [0.9, 0.0, 0.2, 0.0],
                &["T-173", "prefix8", "fallback"],
            ),
            record(
                4,
                8,
                174,
                "Unrelated funding memory for a different scope.",
                [0.0, 1.0, 0.0, 0.0],
                &["funding"],
            ),
        ],
    )
    .expect("segment builds");

    let result = segment
        .recall(&RecallQuery {
            embedding: Some(vec![1.0, 0.0, 0.1, 0.0]),
            symbols: vec!["prefix8".to_string()],
            scope_id: Some(7),
            limit: 5,
            ..RecallQuery::default()
        })
        .expect("recall succeeds");

    println!("Memory SSTable prototype recall");
    println!("segment_id={}", result.trace.segment_id);
    println!("access_paths={:?}", result.trace.access_paths);
    println!(
        "candidates: filters={} symbols={} vector_generator={} vector={} semantic_evals={} returned={}",
        result.trace.candidates_after_filters,
        result.trace.candidates_after_symbols,
        result.trace.vector_generator,
        result.trace.vector_candidates,
        result.trace.semantic_evals,
        result.trace.returned
    );
    for hit in result.hits {
        println!(
            "hit record_id={} score={:.3} semantic_distance={:?} symbols={} text={}",
            hit.record_id, hit.score, hit.semantic_distance, hit.symbol_matches, hit.text
        );
    }
}

fn record(
    record_id: u64,
    scope_id: u32,
    timestamp: i64,
    text: &str,
    embedding: [f32; 4],
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
        vector_id: None,
        metadata: std::collections::BTreeMap::new(),
    }
}
