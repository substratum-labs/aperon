use aperon_core::{
    binary::{load_legacy_index, load_queries, load_raw_vectors, write_legacy_index},
    l2_squared, AperonIndex, ScoredVector, VectorId,
};
use std::{
    env,
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
    process,
};

fn main() {
    if let Err(err) = run(env::args().collect()) {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let Some(command) = args.get(1).map(String::as_str) else {
        return Err(usage());
    };

    match command {
        "build" => build(&args[2..]),
        "query" => query(&args[2..]),
        "eval" => eval(&args[2..]),
        "-h" | "--help" | "help" => {
            println!("{}", usage());
            Ok(())
        }
        _ => Err(format!("unknown command: {command}\n{}", usage())),
    }
}

fn build(args: &[String]) -> Result<(), String> {
    let mut flags = Flags::new(args);
    let vectors_path = flags.required_path("--vectors")?;
    let output_path = flags.required_path("--output")?;
    let local_dim = flags.optional_nonzero_usize("--local-dim")?;
    let sketch_dim = flags.optional_usize("--sketch-dim")?.unwrap_or(0);
    let residual_bits = flags.optional_u8("--residual-bits")?.unwrap_or(8);
    let block_size = flags.optional_nonzero_usize("--block-size")?.unwrap_or(64);
    let grains = flags.optional_nonzero_usize("--grains")?.unwrap_or(1);
    let adaptive_min_local_dim = flags.optional_nonzero_usize("--adaptive-min-local-dim")?;
    let adaptive_max_local_dim = flags.optional_nonzero_usize("--adaptive-max-local-dim")?;
    let adaptive_min_sketch_dim = flags
        .optional_usize("--adaptive-min-sketch-dim")?
        .unwrap_or(0);
    let adaptive_max_sketch_dim = flags
        .optional_usize("--adaptive-max-sketch-dim")?
        .unwrap_or(0);
    let adaptive_min_residual_bits = flags
        .optional_u8("--adaptive-min-residual-bits")?
        .unwrap_or(1);
    let adaptive_max_residual_bits = flags
        .optional_u8("--adaptive-max-residual-bits")?
        .unwrap_or(2);
    let adaptive_variance_target = flags
        .optional_f32("--adaptive-variance-target")?
        .unwrap_or(0.9);
    flags.finish()?;

    let raw = load_raw_vectors(File::open(&vectors_path).map_err(format_io)?).map_err(format_io)?;
    let dim = raw.dimension as usize;
    let num_vectors = raw.num_vectors as usize;
    if dim == 0 {
        return Err("raw vector dimension must be greater than zero".to_string());
    }
    let expected_values = dim
        .checked_mul(num_vectors)
        .ok_or_else(|| "raw vector dimensions overflow usize".to_string())?;
    if raw.vectors.len() != expected_values {
        return Err(format!(
            "raw vector payload mismatch: expected {} f32 values, got {}",
            expected_values,
            raw.vectors.len()
        ));
    }

    let mut index =
        AperonIndex::with_options(dim, local_dim.unwrap_or(dim), sketch_dim, block_size);
    index.set_residual_bits(residual_bits)?;
    if adaptive_min_local_dim.is_some() || adaptive_max_local_dim.is_some() {
        let min_local_dim = adaptive_min_local_dim.ok_or_else(|| {
            "--adaptive-min-local-dim requires --adaptive-max-local-dim".to_string()
        })?;
        let max_local_dim = adaptive_max_local_dim.ok_or_else(|| {
            "--adaptive-max-local-dim requires --adaptive-min-local-dim".to_string()
        })?;
        index.enable_adaptive_quantization(
            min_local_dim,
            max_local_dim,
            adaptive_min_sketch_dim,
            adaptive_max_sketch_dim,
            adaptive_min_residual_bits,
            adaptive_max_residual_bits,
            adaptive_variance_target,
        )?;
    }
    for (idx, vector) in raw.vectors.chunks_exact(dim).enumerate() {
        index.insert(VectorId::new(idx as u64), vector.to_vec())?;
    }
    index.rebuild_n_grains(grains)?;
    let legacy = index
        .to_legacy_index()
        .ok_or_else(|| "failed to build a serializable index".to_string())?;

    let output = File::create(&output_path).map_err(format_io)?;
    write_legacy_index(BufWriter::new(output), &legacy).map_err(format_io)?;
    Ok(())
}

fn query(args: &[String]) -> Result<(), String> {
    let mut flags = Flags::new(args);
    let index_path = flags.required_path("--index")?;
    let queries_path = flags.required_path("--queries")?;
    let top_k = flags.optional_usize("--top-k")?.unwrap_or(10);
    let nprobe = flags.optional_usize("--nprobe")?;
    let rerank_factor = flags.optional_usize("--rerank-factor")?;
    flags.finish()?;

    let legacy =
        load_legacy_index(File::open(&index_path).map_err(format_io)?).map_err(format_io)?;
    let mut index = AperonIndex::from_legacy_index(legacy)?;
    if let Some(rf) = rerank_factor {
        index.set_rerank_factor(rf);
    }
    let queries = load_queries(File::open(&queries_path).map_err(format_io)?).map_err(format_io)?;
    if queries.dimension as usize != index.dim() {
        return Err(format!(
            "query dimension mismatch: index has {}, queries have {}",
            index.dim(),
            queries.dimension
        ));
    }

    let dim = queries.dimension as usize;
    let expected_values = dim
        .checked_mul(queries.num_queries as usize)
        .ok_or_else(|| "query dimensions overflow usize".to_string())?;
    if queries.vectors.len() != expected_values {
        return Err(format!(
            "query payload mismatch: expected {} f32 values, got {}",
            expected_values,
            queries.vectors.len()
        ));
    }

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    writeln!(out, "query_id,rank,vector_id,distance").map_err(format_io)?;
    for (query_id, query) in queries.vectors.chunks_exact(dim).enumerate() {
        let results = match nprobe {
            Some(nprobe) => index.search_with_nprobe(query, top_k, nprobe)?,
            None => index.search(query, top_k)?,
        };
        for (rank, scored) in results.iter().enumerate() {
            writeln!(
                out,
                "{},{},{},{}",
                query_id,
                rank,
                scored.id.get(),
                scored.distance
            )
            .map_err(format_io)?;
        }
    }
    out.flush().map_err(format_io)
}

fn eval(args: &[String]) -> Result<(), String> {
    let mut flags = Flags::new(args);
    let index_path = flags.required_path("--index")?;
    let vectors_path = flags.required_path("--vectors")?;
    let queries_path = flags.required_path("--queries")?;
    let top_k = flags.optional_usize("--top-k")?.unwrap_or(10);
    let nprobe = flags.optional_usize("--nprobe")?;
    let rerank_factor = flags.optional_usize("--rerank-factor")?;
    flags.finish()?;

    let legacy =
        load_legacy_index(File::open(&index_path).map_err(format_io)?).map_err(format_io)?;
    let mut index = AperonIndex::from_legacy_index(legacy)?;
    if let Some(rf) = rerank_factor {
        index.set_rerank_factor(rf);
    }
    let raw = load_raw_vectors(File::open(&vectors_path).map_err(format_io)?).map_err(format_io)?;
    let queries = load_queries(File::open(&queries_path).map_err(format_io)?).map_err(format_io)?;

    validate_vector_payload(
        "raw vector",
        raw.num_vectors as usize,
        raw.dimension as usize,
        raw.vectors.len(),
    )?;
    validate_vector_payload(
        "query",
        queries.num_queries as usize,
        queries.dimension as usize,
        queries.vectors.len(),
    )?;
    if raw.dimension as usize != index.dim() {
        return Err(format!(
            "raw vector dimension mismatch: index has {}, vectors have {}",
            index.dim(),
            raw.dimension
        ));
    }
    if queries.dimension as usize != index.dim() {
        return Err(format!(
            "query dimension mismatch: index has {}, queries have {}",
            index.dim(),
            queries.dimension
        ));
    }

    let dim = index.dim();
    let raw_vectors = raw.vectors.chunks_exact(dim).collect::<Vec<_>>();
    let denominator = top_k.min(raw_vectors.len());
    let mut recall_sum = 0.0_f64;

    for query in queries.vectors.chunks_exact(dim) {
        let predicted = match nprobe {
            Some(nprobe) => index.search_with_nprobe(query, top_k, nprobe)?,
            None => index.search(query, top_k)?,
        };
        let expected = brute_force_top_k(&raw_vectors, query, top_k)?;
        recall_sum += recall_at_k(&predicted, &expected, denominator);
    }

    let recall = if queries.num_queries == 0 {
        0.0
    } else {
        recall_sum / f64::from(queries.num_queries)
    };

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    writeln!(out, "queries,top_k,recall@{top_k}").map_err(format_io)?;
    writeln!(out, "{},{},{}", queries.num_queries, top_k, recall).map_err(format_io)?;
    out.flush().map_err(format_io)
}

fn validate_vector_payload(
    name: &str,
    count: usize,
    dim: usize,
    actual_values: usize,
) -> Result<(), String> {
    if dim == 0 {
        return Err(format!("{name} dimension must be greater than zero"));
    }
    let expected_values = dim
        .checked_mul(count)
        .ok_or_else(|| format!("{name} dimensions overflow usize"))?;
    if actual_values != expected_values {
        return Err(format!(
            "{name} payload mismatch: expected {} f32 values, got {}",
            expected_values, actual_values
        ));
    }
    Ok(())
}

fn brute_force_top_k(
    vectors: &[&[f32]],
    query: &[f32],
    top_k: usize,
) -> Result<Vec<VectorId>, String> {
    let mut scored = vectors
        .iter()
        .enumerate()
        .map(|(idx, vector)| {
            l2_squared(query, vector)
                .map(|distance| (VectorId::new(idx as u64), distance))
                .ok_or_else(|| "brute-force dimension mismatch".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(top_k);
    Ok(scored.into_iter().map(|(id, _)| id).collect())
}

fn recall_at_k(predicted: &[ScoredVector], expected: &[VectorId], denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    let hits = predicted
        .iter()
        .filter(|candidate| expected.contains(&candidate.id))
        .count();
    hits as f64 / denominator as f64
}

struct Flags<'a> {
    args: &'a [String],
    used: Vec<bool>,
}

impl<'a> Flags<'a> {
    fn new(args: &'a [String]) -> Self {
        Self {
            args,
            used: vec![false; args.len()],
        }
    }

    fn required_path(&mut self, name: &str) -> Result<PathBuf, String> {
        self.optional_path(name)?
            .ok_or_else(|| format!("missing required flag {name}"))
    }

    fn optional_path(&mut self, name: &str) -> Result<Option<PathBuf>, String> {
        Ok(self.optional_string(name)?.map(PathBuf::from))
    }

    fn optional_usize(&mut self, name: &str) -> Result<Option<usize>, String> {
        self.optional_string(name)?
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| format!("{name} must be an integer"))
            })
            .transpose()
    }

    fn optional_u8(&mut self, name: &str) -> Result<Option<u8>, String> {
        self.optional_string(name)?
            .map(|value| {
                value
                    .parse::<u8>()
                    .map_err(|_| format!("{name} must be an integer"))
            })
            .transpose()
    }

    fn optional_f32(&mut self, name: &str) -> Result<Option<f32>, String> {
        self.optional_string(name)?
            .map(|value| {
                value
                    .parse::<f32>()
                    .map_err(|_| format!("{name} must be a number"))
            })
            .transpose()
    }

    fn optional_nonzero_usize(&mut self, name: &str) -> Result<Option<usize>, String> {
        self.optional_usize(name)?
            .map(|value| {
                if value > 0 {
                    Ok(value)
                } else {
                    Err(format!("{name} must be greater than zero"))
                }
            })
            .transpose()
    }

    fn optional_string(&mut self, name: &str) -> Result<Option<String>, String> {
        let Some(idx) = self.args.iter().position(|arg| arg == name) else {
            return Ok(None);
        };
        self.used[idx] = true;
        let value_idx = idx + 1;
        if value_idx >= self.args.len() || self.args[value_idx].starts_with("--") {
            return Err(format!("missing value for {name}"));
        }
        self.used[value_idx] = true;
        Ok(Some(self.args[value_idx].clone()))
    }

    fn finish(&self) -> Result<(), String> {
        let extras = self
            .args
            .iter()
            .zip(&self.used)
            .filter_map(|(arg, used)| (!used).then_some(arg.as_str()))
            .collect::<Vec<_>>();
        if extras.is_empty() {
            Ok(())
        } else {
            Err(format!("unexpected argument(s): {}", extras.join(" ")))
        }
    }
}

fn format_io(err: io::Error) -> String {
    err.to_string()
}

fn usage() -> String {
    [
        "usage:",
        "  aperon build --vectors <HNTR> --output <HNTL|HNTM> [--grains N] [--local-dim N] [--sketch-dim N] [--residual-bits 1|2|8] [--block-size N] [--adaptive-min-local-dim N --adaptive-max-local-dim N]",
        "  aperon query --index <HNTL|HNTM> --queries <HNTQ> [--top-k N] [--nprobe N] [--rerank-factor N]",
        "  aperon eval --index <HNTL|HNTM> --vectors <HNTR> --queries <HNTQ> [--top-k N] [--nprobe N] [--rerank-factor N]",
    ]
    .join("\n")
}
