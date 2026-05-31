#!/usr/bin/env python3
"""Reproducible siftsmall benchmark harness for Aperon Mode A/Mode B."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path


TOP_K = 10
aperon = None
hnswlib = None
np = None


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    profile: str
    build_time_s: float
    latency_ms_per_query: float
    qps: float
    resident_memory_bytes: int
    index_disk_bytes: int
    cold_disk_bytes: int
    hnsw_memory_ratio: float | None
    candidate_k: int | None
    candidate_recall_at_10: float | None
    final_recall_at_10: float
    notes: str


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = raw.view(np.int32)[0]
    vectors = raw.reshape(-1, dim + 1)
    if not np.all(vectors.view(np.int32)[:, 0] == dim):
        raise ValueError(f"non-uniform vector sizes in {path}")
    return vectors[:, 1:].copy()


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.int32)
    dim = raw[0]
    vectors = raw.reshape(-1, dim + 1)
    if not np.all(vectors[:, 0] == dim):
        raise ValueError(f"non-uniform vector sizes in {path}")
    return vectors[:, 1:].copy()


def recall_against_top10(results: list[list[int]], gt_top10, result_limit: int | None = TOP_K) -> float:
    recalls = []
    for idx, retrieved in enumerate(results):
        considered = retrieved if result_limit is None else retrieved[:result_limit]
        recalls.append(len(set(considered) & set(gt_top10[idx])) / TOP_K)
    return float(np.mean(recalls)) if recalls else 0.0


def import_runtime_deps() -> None:
    global aperon, hnswlib, np
    try:
        import aperon as aperon_module
        import hnswlib as hnswlib_module
        import numpy as numpy_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "missing benchmark dependency. Run:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install maturin numpy hnswlib\n"
            "  maturin develop --release"
        ) from exc
    aperon = aperon_module
    hnswlib = hnswlib_module
    np = numpy_module


def hnsw_serialized_bytes(index) -> int:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = Path(tmp.name)
    try:
        index.save_index(str(path))
        return path.stat().st_size
    finally:
        path.unlink(missing_ok=True)


def time_queries(fn, query_count: int, runs: int) -> tuple[float, float]:
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - start
    qps = (query_count * runs) / elapsed if elapsed else 0.0
    latency_ms = (elapsed / (query_count * runs)) * 1000.0 if query_count and runs else 0.0
    return latency_ms, qps


def save_index_bytes(index):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hntm") as tmp:
        path = Path(tmp.name)
    try:
        index.save(path)
        disk_bytes = path.stat().st_size
        loaded = aperon.AperonIndex.load(path)
        return loaded, disk_bytes
    finally:
        path.unlink(missing_ok=True)


def build_hnsw(xb: np.ndarray, xq: np.ndarray, gt_top10: np.ndarray, ef_values: list[int]) -> list[BenchmarkRow]:
    index = hnswlib.Index(space="l2", dim=xb.shape[1])
    start = time.perf_counter()
    index.init_index(max_elements=len(xb), ef_construction=200, M=16)
    index.add_items(xb, np.arange(len(xb)))
    build_time = time.perf_counter() - start
    disk_bytes = hnsw_serialized_bytes(index)

    rows = []
    for ef in ef_values:
        index.set_ef(ef)

        def query_once() -> None:
            index.knn_query(xq, k=TOP_K)

        query_once()
        latency_ms, qps = time_queries(query_once, len(xq), runs=20)
        labels, _ = index.knn_query(xq, k=TOP_K)
        final_recall = recall_against_top10(labels.tolist(), gt_top10)
        rows.append(
            BenchmarkRow(
                method="HNSW",
                profile=f"M=16 ef_construction=200 ef={ef}",
                build_time_s=build_time,
                latency_ms_per_query=latency_ms,
                qps=qps,
                resident_memory_bytes=disk_bytes,
                index_disk_bytes=disk_bytes,
                cold_disk_bytes=0,
                hnsw_memory_ratio=1.0,
                candidate_k=None,
                candidate_recall_at_10=None,
                final_recall_at_10=final_recall,
                notes="serialized HNSW index bytes used as resident-memory baseline",
            )
        )
    return rows


def build_mode_a(
    xb: np.ndarray,
    xq: np.ndarray,
    gt_top10: np.ndarray,
    hnsw_bytes: int,
    grains: int,
    nprobe: int,
) -> BenchmarkRow:
    ids = np.arange(len(xb), dtype=np.uint64)
    index = aperon.AperonIndex(xb.shape[1], local_dim=32, sketch_dim=0, block_size=64)

    start = time.perf_counter()
    index.insert_many(ids, xb)
    index.rebuild_n_grains(grains)
    build_time = time.perf_counter() - start
    stats = index.stats()
    loaded, disk_bytes = save_index_bytes(index)

    def query_once() -> None:
        loaded.search_many(xq, TOP_K, nprobe)

    query_once()
    latency_ms, qps = time_queries(query_once, len(xq), runs=20)
    results = loaded.search_many(xq, TOP_K, nprobe)
    final_recall = recall_against_top10([[int(item[0]) for item in row] for row in results], gt_top10)
    resident_bytes = int(stats["encoded_bytes"])
    return BenchmarkRow(
        method="Aperon Mode A",
        profile=f"self-contained recon-only local_dim=32 grains={grains} nprobe={nprobe}",
        build_time_s=build_time,
        latency_ms_per_query=latency_ms,
        qps=qps,
        resident_memory_bytes=resident_bytes,
        index_disk_bytes=disk_bytes,
        cold_disk_bytes=0,
        hnsw_memory_ratio=resident_bytes / hnsw_bytes if hnsw_bytes else None,
        candidate_k=None,
        candidate_recall_at_10=None,
        final_recall_at_10=final_recall,
        notes="save/load recon-only search; no raw vectors attached at query time",
    )


def build_mode_b(
    xb: np.ndarray,
    xq: np.ndarray,
    gt_top10: np.ndarray,
    hnsw_bytes: int,
    raw_vector_bytes: int,
    grains: int,
    nprobe: int,
    candidate_values: list[int],
) -> list[BenchmarkRow]:
    ids = np.arange(len(xb), dtype=np.uint64)
    index = aperon.AperonIndex(
        xb.shape[1],
        local_dim=8,
        sketch_dim=8,
        block_size=64,
        residual_bits=2,
    )

    start = time.perf_counter()
    index.insert_many(ids, xb)
    index.rebuild_n_grains(grains)
    build_time = time.perf_counter() - start
    stats = index.stats()
    _, disk_bytes = save_index_bytes(index)
    index.attach_raw_vectors(ids, xb)
    resident_bytes = int(stats["encoded_bytes"])

    rows = []
    for candidate_k in candidate_values:

        def query_once() -> None:
            index.search_many_tiered(xq, TOP_K, nprobe, candidate_k)

        query_once()
        latency_ms, qps = time_queries(query_once, len(xq), runs=20)

        candidate_results = []
        final_results = []
        for query in xq:
            candidate_results.append(
                [int(item[0]) for item in index.candidates(query, nprobe, candidate_k)]
            )
            final_results.append(
                [int(item[0]) for item in index.search_tiered(query, TOP_K, nprobe, candidate_k)]
            )

        rows.append(
            BenchmarkRow(
                method="Aperon Mode B",
                profile=f"hot filter raw-rerank grains={grains} nprobe={nprobe}",
                build_time_s=build_time,
                latency_ms_per_query=latency_ms,
                qps=qps,
                resident_memory_bytes=resident_bytes,
                index_disk_bytes=disk_bytes,
                cold_disk_bytes=raw_vector_bytes,
                hnsw_memory_ratio=resident_bytes / hnsw_bytes if hnsw_bytes else None,
                candidate_k=candidate_k,
                candidate_recall_at_10=recall_against_top10(candidate_results, gt_top10, result_limit=None),
                final_recall_at_10=recall_against_top10(final_results, gt_top10),
                notes="resident bytes exclude raw vectors; cold bytes report raw siftsmall base file size",
            )
        )
    return rows


def build_mode_b_lattice(
    xb: np.ndarray,
    xq: np.ndarray,
    gt_top10: np.ndarray,
    hnsw_bytes: int,
    raw_vector_bytes: int,
    grains: int,
    nprobe: int,
    candidate_values: list[int],
    routing_dim: int = 4,
    spacing: float = 2.0,
) -> list[BenchmarkRow]:
    ids = np.arange(len(xb), dtype=np.uint64)
    index = aperon.AperonIndex(
        xb.shape[1],
        local_dim=8,
        sketch_dim=8,
        block_size=64,
        residual_bits=2,
    )

    start = time.perf_counter()
    index.insert_many(ids, xb)
    index.rebuild_n_grains(grains)
    index.enable_lattice_routing(routing_dim, spacing)
    build_time = time.perf_counter() - start
    stats = index.stats()
    loaded, disk_bytes = save_index_bytes(index)
    loaded.attach_raw_vectors(ids, xb)
    resident_bytes = int(stats["encoded_bytes"])

    rows = []
    for candidate_k in candidate_values:

        def query_once() -> None:
            loaded.search_many_tiered(xq, TOP_K, nprobe, candidate_k)

        query_once()
        latency_ms, qps = time_queries(query_once, len(xq), runs=20)

        candidate_results = []
        final_results = []
        for query in xq:
            candidate_results.append(
                [int(item[0]) for item in loaded.candidates(query, nprobe, candidate_k)]
            )
            final_results.append(
                [int(item[0]) for item in loaded.search_tiered(query, TOP_K, nprobe, candidate_k)]
            )

        rows.append(
            BenchmarkRow(
                method="Aperon Mode B (Lattice)",
                profile=f"hot filter raw-rerank grains={grains} nprobe={nprobe} r_dim={routing_dim} spacing={spacing}",
                build_time_s=build_time,
                latency_ms_per_query=latency_ms,
                qps=qps,
                resident_memory_bytes=resident_bytes,
                index_disk_bytes=disk_bytes,
                cold_disk_bytes=raw_vector_bytes,
                hnsw_memory_ratio=resident_bytes / hnsw_bytes if hnsw_bytes else None,
                candidate_k=candidate_k,
                candidate_recall_at_10=recall_against_top10(candidate_results, gt_top10, result_limit=None),
                final_recall_at_10=recall_against_top10(final_results, gt_top10),
                notes="resident bytes exclude raw vectors; O(1) lattice routing enabled",
            )
        )
    return rows


def build_mode_b_hlr(
    xb: np.ndarray,
    xq: np.ndarray,
    gt_top10: np.ndarray,
    hnsw_bytes: int,
    raw_vector_bytes: int,
    grains: int,
    nprobe: int,
    candidate_values: list[int],
    hlr_layer_configs: list[tuple[int, float]] | None = None,
) -> list[BenchmarkRow]:
    if hlr_layer_configs is None:
        hlr_layer_configs = [(2, 2.0), (2, 1.0)]
    ids = np.arange(len(xb), dtype=np.uint64)
    index = aperon.AperonIndex(
        xb.shape[1],
        local_dim=8,
        sketch_dim=8,
        block_size=64,
        residual_bits=2,
    )

    start = time.perf_counter()
    index.insert_many(ids, xb)
    index.rebuild_n_grains(grains)
    index.enable_hlr_routing(hlr_layer_configs)
    build_time = time.perf_counter() - start
    stats = index.stats()
    loaded, disk_bytes = save_index_bytes(index)
    loaded.attach_raw_vectors(ids, xb)
    resident_bytes = int(stats["encoded_bytes"])

    hlr_desc = "+".join(f"{d}d/{s:.1f}s" for d, s in hlr_layer_configs)
    rows = []
    for candidate_k in candidate_values:

        def query_once() -> None:
            loaded.search_many_tiered(xq, TOP_K, nprobe, candidate_k)

        query_once()
        latency_ms, qps = time_queries(query_once, len(xq), runs=20)

        candidate_results = []
        final_results = []
        for query in xq:
            candidate_results.append(
                [int(item[0]) for item in loaded.candidates(query, nprobe, candidate_k)]
            )
            final_results.append(
                [int(item[0]) for item in loaded.search_tiered(query, TOP_K, nprobe, candidate_k)]
            )

        rows.append(
            BenchmarkRow(
                method="Aperon Mode B (HLR)",
                profile=f"hot filter raw-rerank grains={grains} nprobe={nprobe} hlr=[{hlr_desc}]",
                build_time_s=build_time,
                latency_ms_per_query=latency_ms,
                qps=qps,
                resident_memory_bytes=resident_bytes,
                index_disk_bytes=disk_bytes,
                cold_disk_bytes=raw_vector_bytes,
                hnsw_memory_ratio=resident_bytes / hnsw_bytes if hnsw_bytes else None,
                candidate_k=candidate_k,
                candidate_recall_at_10=recall_against_top10(candidate_results, gt_top10, result_limit=None),
                final_recall_at_10=recall_against_top10(final_results, gt_top10),
                notes=f"resident bytes exclude raw vectors; HLR routing [{hlr_desc}]",
            )
        )
    return rows


def sample_centroids(xb: np.ndarray, k: int) -> np.ndarray:
    if k >= len(xb):
        return xb.copy()
    indices = np.linspace(0, len(xb) - 1, num=k, dtype=np.int64)
    return xb[indices].copy()


def exact_centroid_top10(centroids: np.ndarray, queries: np.ndarray) -> list[list[int]]:
    exact = []
    limit = min(TOP_K, len(centroids))
    for query in queries:
        diff = centroids - query
        dists = np.einsum("ij,ij->i", diff, diff)
        if limit < len(centroids):
            idx = np.argpartition(dists, limit - 1)[:limit]
            idx = idx[np.argsort(dists[idx], kind="stable")]
        else:
            idx = np.argsort(dists, kind="stable")
        exact.append([int(value) for value in idx[:limit]])
    return exact


def build_hlr_router_scale(
    xb: np.ndarray,
    xq: np.ndarray,
    hnsw_bytes: int,
    grains: int,
    nprobe: int,
    hlr_layer_configs: list[tuple[int, float]],
) -> BenchmarkRow:
    centroids = sample_centroids(xb, grains)
    start = time.perf_counter()
    router = aperon.HlrRouter(xb.shape[1], centroids, hlr_layer_configs)
    build_time = time.perf_counter() - start

    def query_once() -> None:
        router.route_many(xq, nprobe)

    query_once()
    latency_ms, qps = time_queries(query_once, len(xq), runs=200)
    routed = router.route_many(xq, nprobe)
    exact = exact_centroid_top10(centroids, xq)
    route_recall = recall_against_top10(routed, exact, result_limit=None)
    hlr_desc = "+".join(f"{d}d/{s:.1f}s" for d, s in hlr_layer_configs)
    resident_bytes = int(centroids.nbytes)

    return BenchmarkRow(
        method="Aperon HLR Router",
        profile=f"route-only centroids={len(centroids)} nprobe={nprobe} hlr=[{hlr_desc}]",
        build_time_s=build_time,
        latency_ms_per_query=latency_ms,
        qps=qps,
        resident_memory_bytes=resident_bytes,
        index_disk_bytes=0,
        cold_disk_bytes=0,
        hnsw_memory_ratio=resident_bytes / hnsw_bytes if hnsw_bytes else None,
        candidate_k=nprobe,
        candidate_recall_at_10=route_recall,
        final_recall_at_10=route_recall,
        notes="route-only HLR scale check; recall is exact-centroid top-10 coverage by routed centroids",
    )


def markdown_report(rows: list[BenchmarkRow], dataset_dir: Path, xb: np.ndarray, xq: np.ndarray) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Aperon Reproducible Benchmark (siftsmall)",
        "",
        "This report is generated by `python benchmarks/run_benchmark.py`.",
        "",
        "## Dataset",
        f"- Dataset path: `{dataset_dir}`",
        f"- Base vectors: {len(xb):,} x {xb.shape[1]} float32",
        f"- Query vectors: {len(xq):,} x {xq.shape[1]} float32",
        "- Ground truth: siftsmall top-100 neighbors, evaluated as Recall@10",
        "",
        "## Environment",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- NumPy: {np.__version__}",
        "",
        "## Results",
        "",
        "| Method | Profile | Build s | Latency ms/q | QPS | Resident bytes | Index disk bytes | Cold bytes | HNSW ratio | Candidate k | Candidate Recall@10 | Final Recall@10 | Notes |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in rows:
        ratio = "n/a" if row.hnsw_memory_ratio is None else f"{row.hnsw_memory_ratio:.4f}x"
        candidate_k = "n/a" if row.candidate_k is None else str(row.candidate_k)
        candidate_recall = (
            "n/a"
            if row.candidate_recall_at_10 is None
            else f"{row.candidate_recall_at_10:.4f}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row.method,
                    row.profile,
                    f"{row.build_time_s:.4f}",
                    f"{row.latency_ms_per_query:.4f}",
                    f"{row.qps:.1f}",
                    f"{row.resident_memory_bytes:,}",
                    f"{row.index_disk_bytes:,}",
                    f"{row.cold_disk_bytes:,}",
                    ratio,
                    candidate_k,
                    candidate_recall,
                    f"{row.final_recall_at_10:.4f}",
                    row.notes,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Metric Definitions",
            "- `Resident bytes`: bytes expected to stay in DRAM for the search structure. For HNSW this is approximated by serialized index bytes. For Aperon this is `stats()[\"encoded_bytes\"]`.",
            "- `Index disk bytes`: bytes of the serialized index file written by the benchmark.",
            "- `Cold bytes`: external payload needed for exact rerank. Mode B reports the raw `siftsmall_base.fvecs` file size; Mode A and HNSW do not require this benchmark-side cold payload.",
            "- `Candidate Recall@10`: fraction of true top-10 neighbors present in the Mode B hot-filter candidate set before exact rerank.",
            "- `Final Recall@10`: final top-10 recall after the method's ranking path: HNSW graph search, Mode A save/load recon-only search, or Mode B hot-filter plus raw rerank.",
            "- `Aperon HLR Router` rows are route-only scale checks over sampled siftsmall centroids. Their recall columns report exact-centroid top-10 coverage by the HLR routed centroid set.",
            "- HLR `residual_energy` is measured after per-layer residual centering. Entry 0 is input mean squared norm; later entries are centered residual variance after subtracting each PCA layer.",
            "",
            "## Reproduction",
            "```bash",
            "python3 -m venv .venv",
            "source .venv/bin/activate",
            "pip install maturin numpy hnswlib",
            "maturin develop --release",
            "benchmarks/run_all.sh",
            "```",
            "",
            "`benchmarks/run_all.sh` forwards arguments to `python benchmarks/run_benchmark.py`. Use `python benchmarks/run_benchmark.py --help` to change `nprobe`, `candidate_k`, or output paths. The script also writes machine-readable rows to `benchmarks/latest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parent / "data" / "siftsmall"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "README.md")
    parser.add_argument("--json-output", type=Path, default=Path(__file__).resolve().parent / "latest.json")
    parser.add_argument("--mode-a-grains", type=int, default=16)
    parser.add_argument("--mode-a-nprobe", type=int, default=8)
    parser.add_argument("--mode-b-grains", type=int, default=16)
    parser.add_argument("--mode-b-nprobe", type=int, default=16)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--hnsw-ef", type=int, nargs="+", default=[50, 100, 200])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_runtime_deps()
    base_file = args.data_dir / "siftsmall_base.fvecs"
    query_file = args.data_dir / "siftsmall_query.fvecs"
    gt_file = args.data_dir / "siftsmall_groundtruth.ivecs"
    missing = [path for path in [base_file, query_file, gt_file] if not path.exists()]
    if missing:
        raise SystemExit(f"missing siftsmall files: {', '.join(str(path) for path in missing)}")

    xb = read_fvecs(base_file)
    xq = read_fvecs(query_file)
    gt_top10 = read_ivecs(gt_file)[:, :TOP_K]

    print("Building HNSW baseline...", flush=True)
    rows = build_hnsw(xb, xq, gt_top10, args.hnsw_ef)
    hnsw_bytes = rows[0].resident_memory_bytes

    print("Building Aperon Mode A...", flush=True)
    rows.append(build_mode_a(xb, xq, gt_top10, hnsw_bytes, args.mode_a_grains, args.mode_a_nprobe))

    print("Building Aperon Mode B...", flush=True)
    rows.extend(
        build_mode_b(
            xb,
            xq,
            gt_top10,
            hnsw_bytes,
            base_file.stat().st_size,
            args.mode_b_grains,
            args.mode_b_nprobe,
            args.candidate_k,
        )
    )

    print("Building Aperon Mode B (128 Grains, Centroid baseline)...", flush=True)
    rows.extend(
        build_mode_b(
            xb,
            xq,
            gt_top10,
            hnsw_bytes,
            base_file.stat().st_size,
            grains=128,
            nprobe=16,
            candidate_values=args.candidate_k,
        )
    )

    print("Building Aperon Mode B (128 Grains, Lattice)...", flush=True)
    rows.extend(
        build_mode_b_lattice(
            xb,
            xq,
            gt_top10,
            hnsw_bytes,
            base_file.stat().st_size,
            grains=128,
            nprobe=16,
            candidate_values=args.candidate_k,
            routing_dim=5,
            spacing=100.0,
        )
    )

    print("Building Aperon Mode B (128 Grains, HLR)...", flush=True)
    rows.extend(
        build_mode_b_hlr(
            xb,
            xq,
            gt_top10,
            hnsw_bytes,
            base_file.stat().st_size,
            grains=128,
            nprobe=16,
            candidate_values=args.candidate_k,
            hlr_layer_configs=[(2, 2.0), (2, 1.0)],
        )
    )

    for hlr_grains in [128, 1024, 4096]:
        actual_grains = min(hlr_grains, len(xb))
        print(f"Building Aperon HLR router scale check ({actual_grains} centroids)...", flush=True)
        rows.append(
            build_hlr_router_scale(
                xb,
                xq,
                hnsw_bytes,
                grains=actual_grains,
                nprobe=16,
                hlr_layer_configs=[(2, 2.0), (2, 1.0)],
            )
        )

    args.output.write_text(markdown_report(rows, args.data_dir, xb, xq), encoding="utf-8")
    args.json_output.write_text(
        json.dumps([asdict(row) for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
