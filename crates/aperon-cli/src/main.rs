use aperon_core::{
    binary::{load_legacy_index, load_queries, load_raw_vectors, write_legacy_index, LegacyIndex},
    AperonIndex, VectorId,
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
    let block_size = flags.optional_nonzero_usize("--block-size")?.unwrap_or(64);
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
    for (idx, vector) in raw.vectors.chunks_exact(dim).enumerate() {
        index.insert(VectorId::new(idx as u64), vector.to_vec())?;
    }
    index.rebuild_single_grain()?;
    let single = index
        .to_legacy_single()
        .ok_or_else(|| "failed to build a serializable grain".to_string())?;

    let output = File::create(&output_path).map_err(format_io)?;
    write_legacy_index(BufWriter::new(output), &LegacyIndex::Single(single)).map_err(format_io)?;
    Ok(())
}

fn query(args: &[String]) -> Result<(), String> {
    let mut flags = Flags::new(args);
    let index_path = flags.required_path("--index")?;
    let queries_path = flags.required_path("--queries")?;
    let top_k = flags.optional_usize("--top-k")?.unwrap_or(10);
    let nprobe = flags.optional_usize("--nprobe")?;
    flags.finish()?;

    let legacy =
        load_legacy_index(File::open(&index_path).map_err(format_io)?).map_err(format_io)?;
    let index = AperonIndex::from_legacy_index(legacy)?;
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
        "  aperon build --vectors <HNTR> --output <HNTL> [--local-dim N] [--sketch-dim N] [--block-size N]",
        "  aperon query --index <HNTL|HNTM> --queries <HNTQ> [--top-k N] [--nprobe N]",
    ]
    .join("\n")
}
