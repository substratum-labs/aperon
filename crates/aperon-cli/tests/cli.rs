use aperon_core::binary::{
    load_legacy_index, write_queries, write_raw_vectors, LegacyIndex, QuerySet, RawVectors,
};
use std::{
    fs::{self, File},
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn build_writes_legacy_index() {
    let dir = TestDir::new("build_writes_legacy_index");
    let vectors = dir.path("vectors.hntr");
    let index = dir.path("index.hntl");
    write_raw_vectors(
        File::create(&vectors).unwrap(),
        &RawVectors {
            num_vectors: 3,
            dimension: 2,
            vectors: vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
        },
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_aperon"))
        .args([
            "build",
            "--vectors",
            vectors.to_str().unwrap(),
            "--output",
            index.to_str().unwrap(),
            "--local-dim",
            "2",
            "--block-size",
            "2",
        ])
        .output()
        .unwrap();

    assert_success(&output);
    let loaded = load_legacy_index(File::open(index).unwrap()).unwrap();
    match loaded {
        LegacyIndex::Single(single) => {
            assert_eq!(single.num_vectors, 3);
            assert_eq!(single.dimension, 2);
            assert_eq!(single.local_dim, 2);
            assert_eq!(single.block_size, 2);
        }
        LegacyIndex::Multi(_) => panic!("expected single-grain index"),
    }
}

#[test]
fn query_returns_nearest_ids() {
    let dir = TestDir::new("query_returns_nearest_ids");
    let vectors = dir.path("vectors.hntr");
    let index = dir.path("index.hntl");
    let queries = dir.path("queries.hntq");
    write_raw_vectors(
        File::create(&vectors).unwrap(),
        &RawVectors {
            num_vectors: 3,
            dimension: 2,
            vectors: vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0],
        },
    )
    .unwrap();
    write_queries(
        File::create(&queries).unwrap(),
        &QuerySet {
            num_queries: 1,
            dimension: 2,
            vectors: vec![9.0, 0.0],
        },
    )
    .unwrap();

    let build = Command::new(env!("CARGO_BIN_EXE_aperon"))
        .args([
            "build",
            "--vectors",
            vectors.to_str().unwrap(),
            "--output",
            index.to_str().unwrap(),
            "--local-dim",
            "2",
            "--block-size",
            "2",
        ])
        .output()
        .unwrap();
    assert_success(&build);

    let query = Command::new(env!("CARGO_BIN_EXE_aperon"))
        .args([
            "query",
            "--index",
            index.to_str().unwrap(),
            "--queries",
            queries.to_str().unwrap(),
            "--top-k",
            "2",
        ])
        .output()
        .unwrap();

    assert_success(&query);
    let stdout = String::from_utf8(query.stdout).unwrap();
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("query_id,rank,vector_id,distance"));
    assert!(lines.next().unwrap().starts_with("0,0,1,"));
    assert!(lines.next().unwrap().starts_with("0,1,0,"));
}

fn assert_success(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(name: &str) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("aperon-cli-{name}-{stamp}"));
        fs::create_dir(&path).unwrap();
        Self { path }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
