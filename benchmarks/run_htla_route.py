#!/usr/bin/env python3
"""Route-only HTLA benchmark over siftsmall sampled centroids."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


FINAL_NPROBE = 16


@dataclass(frozen=True)
class HtlaRow:
    dataset: str
    k: int
    levels: int
    dim: int
    beam: int
    pool: int
    coverage_at_16: float
    coverage_at_32: float
    qps: float
    exact_scan_qps: float
    route_bytes: int
    working_set_bytes_per_query: int
    build_time_s: float
    fallback_rate: float
    diagnostics: dict


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = raw.view(np.int32)[0]
    vectors = raw.reshape(-1, dim + 1)
    if not np.all(vectors.view(np.int32)[:, 0] == dim):
        raise ValueError(f"non-uniform vector sizes in {path}")
    return vectors[:, 1:].copy()


def sample_centroids(xb: np.ndarray, k: int) -> np.ndarray:
    if k >= len(xb):
        return xb.copy()
    indices = np.linspace(0, len(xb) - 1, num=k, dtype=np.int64)
    return xb[indices].copy()


def exact_topk(centroids: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    out = []
    limit = min(k, len(centroids))
    for query in queries:
        diff = centroids - query
        dists = np.einsum("ij,ij->i", diff, diff)
        if limit < len(centroids):
            idx = np.argpartition(dists, limit - 1)[:limit]
            idx = idx[np.argsort(dists[idx], kind="stable")]
        else:
            idx = np.argsort(dists, kind="stable")
        out.append([int(value) for value in idx[:limit]])
    return out


def coverage(results: list[list[int]], exact: list[list[int]]) -> float:
    values = []
    for got, want in zip(results, exact):
        denom = max(1, len(want))
        values.append(len(set(got) & set(want)) / denom)
    return float(np.mean(values)) if values else 0.0


def time_queries(fn, query_count: int, runs: int) -> tuple[float, float]:
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = time.perf_counter() - start
    qps = (query_count * runs) / elapsed if elapsed else 0.0
    return elapsed, qps


def exact_scan_once(centroids: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    return exact_topk(centroids, queries, k)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * p)
    return float(ordered[idx])


def summarize_diagnostics(diag: dict) -> dict:
    return {
        "nodes": len(diag.get("d95", [])),
        "d80_max": max(diag.get("d80", [0])),
        "d90_max": max(diag.get("d90", [0])),
        "d95_max": max(diag.get("d95", [0])),
        "d80_p50": percentile([float(v) for v in diag.get("d80", [])], 0.50),
        "d90_p50": percentile([float(v) for v in diag.get("d90", [])], 0.50),
        "d95_p50": percentile([float(v) for v in diag.get("d95", [])], 0.50),
        "residual_energy": diag.get("residual_energy", []),
        "radius_shrink_p50": percentile([float(v) for v in diag.get("radius_shrink", [])], 0.50),
        "radius_shrink_p90": percentile([float(v) for v in diag.get("radius_shrink", [])], 0.90),
        "norm_sep_p10": float(diag.get("norm_sep_p10", 0.0)),
        "norm_sep_p25": float(diag.get("norm_sep_p25", 0.0)),
    }


def run_row(aperon, xb: np.ndarray, xq: np.ndarray, k: int, levels: int, dim: int, beam: int, pool: int) -> HtlaRow:
    centroids = sample_centroids(xb, k)
    exact16 = exact_topk(centroids, xq, 16)
    exact32 = exact_topk(centroids, xq, 32)

    start = time.perf_counter()
    router = aperon.HtlaRouter(xb.shape[1], centroids, levels, dim)
    build_time = time.perf_counter() - start

    def route_once():
        return router.route_many(xq, beam, pool, FINAL_NPROBE)

    route_once()
    _, qps = time_queries(route_once, len(xq), runs=50)
    routed = route_once()
    final = [item["final_nprobe"] for item in routed]
    candidates = [item["candidates"] for item in routed]
    fallback_rate = float(np.mean([bool(item["fallback"]) for item in routed]))
    working_set = int(np.mean([int(item["working_set_bytes"]) for item in routed]))

    _, exact_qps = time_queries(lambda: exact_scan_once(centroids, xq, FINAL_NPROBE), len(xq), runs=20)
    diag = summarize_diagnostics(router.diagnostics())

    return HtlaRow(
        dataset="siftsmall",
        k=len(centroids),
        levels=levels,
        dim=dim,
        beam=beam,
        pool=pool,
        coverage_at_16=coverage(final, exact16),
        coverage_at_32=coverage(candidates, exact32),
        qps=qps,
        exact_scan_qps=exact_qps,
        route_bytes=router.resident_bytes(),
        working_set_bytes_per_query=working_set,
        build_time_s=build_time,
        fallback_rate=fallback_rate,
        diagnostics=diag,
    )


def markdown(rows: list[HtlaRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# T-160 HTLA Route-Only Benchmark",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        "",
        "| Dataset | K | Levels | Dim | Beam | Pool | Coverage@16 | Coverage@32 | QPS | Exact scan QPS | Route bytes | Working-set bytes/q | Build s | Fallback rate |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.dataset} | {row.k} | {row.levels} | {row.dim} | {row.beam} | {row.pool} | "
            f"{row.coverage_at_16:.4f} | {row.coverage_at_32:.4f} | {row.qps:.1f} | "
            f"{row.exact_scan_qps:.1f} | {row.route_bytes:,} | {row.working_set_bytes_per_query:,} | "
            f"{row.build_time_s:.4f} | {row.fallback_rate:.4f} |"
        )
    max_pool_rows = {
        128: next(row for row in rows if row.k == 128 and row.pool == 128),
        1024: next(row for row in rows if row.k == 1024 and row.pool == 256),
        4096: next(row for row in rows if row.k == 4096 and row.pool == 512),
    }
    lines.extend(
        [
            "",
            "## Acceptance Summary",
            "",
            "| K | Required pool | Coverage@16 target | Observed Coverage@16 | Result |",
            "| ---: | ---: | ---: | ---: | :--- |",
        ]
    )
    for k, row in max_pool_rows.items():
        result = "PASS" if row.coverage_at_16 >= 0.99 else "FAIL"
        lines.append(f"| {k} | {row.pool} | 0.9900 | {row.coverage_at_16:.4f} | {result} |")
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| K | Dim | d80 max | d90 max | d95 max | d80 p50 | d90 p50 | d95 p50 | radius shrink p50 | radius shrink p90 | p10(norm_sep) | p25(norm_sep) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        d = row.diagnostics
        lines.append(
            f"| {row.k} | {row.dim} | {d['d80_max']} | {d['d90_max']} | {d['d95_max']} | "
            f"{d['d80_p50']:.1f} | {d['d90_p50']:.1f} | {d['d95_p50']:.1f} | "
            f"{d['radius_shrink_p50']:.4f} | {d['radius_shrink_p90']:.4f} | "
            f"{d['norm_sep_p10']:.4f} | {d['norm_sep_p25']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Coverage@16 is measured on the final 16 centroid IDs after exact rerank of the routed pool.",
            "Coverage@32 is measured on the routed candidate pool before final exact rerank.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parent / "data" / "siftsmall"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "htla_t160.md")
    parser.add_argument("--json-output", type=Path, default=Path(__file__).resolve().parent / "htla_t160.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import aperon
    except ModuleNotFoundError as exc:
        raise SystemExit("missing aperon module. Run `maturin develop --release` first.") from exc

    xb = read_fvecs(args.data_dir / "siftsmall_base.fvecs")
    xq = read_fvecs(args.data_dir / "siftsmall_query.fvecs")
    matrix = [
        (128, 2, 8, 4, 64),
        (128, 2, 8, 8, 128),
        (1024, 3, 8, 8, 128),
        (1024, 3, 12, 8, 256),
        (4096, 4, 12, 16, 256),
        (4096, 4, 16, 16, 512),
    ]
    rows = []
    for spec in matrix:
        print(f"Running HTLA row K={spec[0]} levels={spec[1]} dim={spec[2]} beam={spec[3]} pool={spec[4]}", flush=True)
        rows.append(run_row(aperon, xb, xq, *spec))

    args.output.write_text(markdown(rows, args.data_dir), encoding="utf-8")
    args.json_output.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
