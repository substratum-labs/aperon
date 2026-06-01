#!/usr/bin/env python3
"""T-167 route-only pivot-sketch / LAESA-style block router prototype."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


FINAL_NPROBE = 16


@dataclass(frozen=True)
class PivotSketch:
    k: int
    block_size: int
    pivot_count: int
    signature: str
    offsets: np.ndarray
    payload: np.ndarray
    representatives: np.ndarray
    pivots: np.ndarray
    block_signatures: np.ndarray
    quant_min: np.ndarray | None
    quant_scale: np.ndarray | None
    build_time_s: float


@dataclass(frozen=True)
class RouteStats:
    candidates: list[int]
    pivot_evals: int
    block_rows_scored: int
    centroid_evals: int
    selected_blocks: int
    working_set_bytes: int
    fallback: bool


@dataclass(frozen=True)
class PivotRow:
    dataset: str
    k: int
    block_size: int
    blocks: int
    pivots: int
    top_blocks: int
    pool: int
    signature: str
    final_nprobe: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    qps: float
    route_time_us_per_query: float
    pivot_evals_per_query: float
    block_signature_rows_scored_per_query: float
    centroid_evals_per_query: float
    selected_blocks_per_query: float
    candidate_count_per_query: float
    route_resident_bytes: int
    working_set_bytes_per_query: int
    build_time_s: float
    fallback_rate: float


@dataclass(frozen=True)
class BaselineRow:
    router: str
    k: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    route_time_us_per_query: float
    evals_per_query: float
    candidate_count_per_query: float
    resident_bytes: int
    working_set_bytes_per_query: int
    build_time_s: float


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
    norms = np.einsum("ij,ij->i", centroids, centroids)
    for query in queries:
        dists = norms + float(np.dot(query, query)) - 2.0 * (centroids @ query)
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


def deterministic_seeds(points: np.ndarray, count: int) -> np.ndarray:
    if count >= len(points):
        return points.copy()
    center = points.mean(axis=0)
    chosen = [int(np.argmin(np.sum((points - center) ** 2, axis=1)))]
    nearest = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    while len(chosen) < count:
        nearest[np.array(chosen)] = -1.0
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        dist = np.sum((points - points[nxt]) ** 2, axis=1)
        nearest = np.minimum(nearest, dist)
    return points[np.array(chosen, dtype=np.int64)].copy()


def balanced_blocks(centroids: np.ndarray, block_size: int, iters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_count = int(math.ceil(len(centroids) / block_size))
    cap = block_size
    reps = deterministic_seeds(centroids, block_count)
    assign = np.zeros(len(centroids), dtype=np.int32)
    for _ in range(iters):
        dists = np.sum((centroids[:, None, :] - reps[None, :, :]) ** 2, axis=2)
        order = np.argsort(dists, axis=1, kind="stable")
        second = order[:, 1] if block_count > 1 else order[:, 0]
        margins = dists[np.arange(len(centroids)), second] - dists[np.arange(len(centroids)), order[:, 0]]
        point_order = np.argsort(margins, kind="stable")
        counts = np.zeros(block_count, dtype=np.int32)
        for row in point_order:
            for choice in order[row]:
                if counts[choice] < cap:
                    assign[row] = int(choice)
                    counts[choice] += 1
                    break
        for block in range(block_count):
            mask = assign == block
            if np.any(mask):
                reps[block] = centroids[mask].mean(axis=0)

    order = np.lexsort((np.arange(block_count), reps[:, 0]))
    remap = np.empty(block_count, dtype=np.int32)
    remap[order] = np.arange(block_count, dtype=np.int32)
    assign = remap[assign]

    payload: list[int] = []
    offsets = [0]
    final_reps = np.empty((block_count, centroids.shape[1]), dtype=np.float32)
    for block in range(block_count):
        members = np.flatnonzero(assign == block)
        local = centroids[members]
        center = local.mean(axis=0)
        local_dists = np.sum((local - center) ** 2, axis=1)
        member_order = members[np.argsort(local_dists, kind="stable")]
        payload.extend(int(idx) for idx in member_order.tolist())
        offsets.append(len(payload))
        final_reps[block] = center
    return np.array(offsets, dtype=np.int32), np.array(payload, dtype=np.int32), final_reps


def pivot_distances(points: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    point_norms = np.einsum("ij,ij->i", points, points)
    pivot_norms = np.einsum("ij,ij->i", pivots, pivots)
    dists = point_norms[:, None] + pivot_norms[None, :] - 2.0 * (points @ pivots.T)
    np.maximum(dists, 0.0, out=dists)
    return np.sqrt(dists, dtype=np.float32)


def quantize_uint16(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qmin = values.min(axis=0).astype(np.float32)
    qmax = values.max(axis=0).astype(np.float32)
    scale = (qmax - qmin) / 65535.0
    scale[scale == 0.0] = 1.0
    quantized = np.rint((values - qmin[None, :]) / scale[None, :])
    return quantized.clip(0, 65535).astype(np.uint16), qmin, scale.astype(np.float32)


def dequantize_uint16(values: np.ndarray, qmin: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return values.astype(np.float32) * scale[None, :] + qmin[None, :]


def build_pivot_sketch(
    centroids: np.ndarray,
    k: int,
    block_size: int,
    pivot_count: int,
    signature: str,
    cluster_iters: int,
) -> PivotSketch:
    start = time.perf_counter()
    offsets, payload, reps = balanced_blocks(centroids, block_size, cluster_iters)
    pivots = deterministic_seeds(reps, min(pivot_count, len(reps))).astype(np.float32, copy=False)
    block_signatures = pivot_distances(reps, pivots)
    quant_min = None
    quant_scale = None
    stored: np.ndarray
    if signature == "fp16":
        stored = block_signatures.astype(np.float16)
    elif signature == "uint16":
        stored, quant_min, quant_scale = quantize_uint16(block_signatures)
    else:
        stored = block_signatures.astype(np.float32, copy=False)
    return PivotSketch(
        k=k,
        block_size=block_size,
        pivot_count=len(pivots),
        signature=signature,
        offsets=offsets,
        payload=payload,
        representatives=reps,
        pivots=pivots,
        block_signatures=stored,
        quant_min=quant_min,
        quant_scale=quant_scale,
        build_time_s=time.perf_counter() - start,
    )


def final_rerank(centroids: np.ndarray, query: np.ndarray, candidates: list[int], limit: int) -> list[int]:
    if not candidates:
        return []
    idx = np.array(candidates, dtype=np.int64)
    diff = centroids[idx] - query
    dists = np.einsum("ij,ij->i", diff, diff)
    order = np.argsort(dists, kind="stable")[:limit]
    return [int(idx[pos]) for pos in order]


def route_pivot_sketch(sketch: PivotSketch, centroids: np.ndarray, query: np.ndarray, top_blocks: int, pool: int) -> RouteStats:
    qsig = pivot_distances(query.reshape(1, -1), sketch.pivots)[0]
    if sketch.signature == "uint16":
        assert sketch.quant_min is not None and sketch.quant_scale is not None
        block_sig = dequantize_uint16(sketch.block_signatures, sketch.quant_min, sketch.quant_scale)
    else:
        block_sig = sketch.block_signatures.astype(np.float32, copy=False)

    if sketch.signature == "l_inf":
        scores = np.max(np.abs(block_sig - qsig[None, :]), axis=1)
    elif sketch.signature == "l2":
        scores = np.sqrt(np.sum((block_sig - qsig[None, :]) ** 2, axis=1))
    else:
        scores = np.sum(np.abs(block_sig - qsig[None, :]), axis=1)

    limit = min(top_blocks, len(scores))
    selected = np.argpartition(scores, limit - 1)[:limit]
    selected = selected[np.argsort(scores[selected], kind="stable")]

    centroid_ids: list[int] = []
    for raw in selected.tolist():
        block = int(raw)
        start = int(sketch.offsets[block])
        end = int(sketch.offsets[block + 1])
        centroid_ids.extend(int(idx) for idx in sketch.payload[start:end].tolist())

    candidates = final_rerank(centroids, query, centroid_ids, pool)
    signature_itemsize = sketch.block_signatures.dtype.itemsize
    working_set = (
        sketch.pivot_count * centroids.shape[1] * 4
        + len(scores) * sketch.pivot_count * signature_itemsize
        + len(centroid_ids) * centroids.shape[1] * 4
        + max(len(scores), limit, 1) * 8
    )
    return RouteStats(
        candidates=candidates,
        pivot_evals=sketch.pivot_count,
        block_rows_scored=len(scores),
        centroid_evals=len(centroid_ids),
        selected_blocks=len(selected),
        working_set_bytes=int(working_set),
        fallback=len(candidates) < min(FINAL_NPROBE, len(centroids)),
    )


def route_resident_bytes(sketch: PivotSketch, centroids: np.ndarray) -> int:
    extra = 0
    if sketch.quant_min is not None and sketch.quant_scale is not None:
        extra += sketch.quant_min.nbytes + sketch.quant_scale.nbytes
    return int(
        centroids.nbytes
        + sketch.payload.nbytes
        + sketch.offsets.nbytes
        + sketch.representatives.nbytes
        + sketch.pivots.nbytes
        + sketch.block_signatures.nbytes
        + extra
    )


def evaluate(
    sketch: PivotSketch,
    centroids: np.ndarray,
    queries: np.ndarray,
    exact16: list[list[int]],
    exact32: list[list[int]],
    spec: tuple[int, int, int, int, int, str],
    route_runs: int,
) -> PivotRow:
    k, block_size, pivots, top_blocks, pool, signature = spec

    def route_once() -> list[RouteStats]:
        return [route_pivot_sketch(sketch, centroids, query, top_blocks, pool) for query in queries]

    route_once()
    start = time.perf_counter()
    for _ in range(route_runs):
        routed = route_once()
    elapsed = time.perf_counter() - start
    qps = (len(queries) * route_runs) / elapsed if elapsed else 0.0
    finals = [final_rerank(centroids, query, stat.candidates, FINAL_NPROBE) for query, stat in zip(queries, routed)]
    candidate_lists = [stat.candidates for stat in routed]
    return PivotRow(
        dataset="siftsmall",
        k=k,
        block_size=block_size,
        blocks=len(sketch.representatives),
        pivots=pivots,
        top_blocks=top_blocks,
        pool=pool,
        signature=signature,
        final_nprobe=FINAL_NPROBE,
        coverage_at_16=coverage(finals, exact16),
        candidate_pool_coverage_at_32=coverage(candidate_lists, exact32),
        qps=qps,
        route_time_us_per_query=(1_000_000.0 / qps) if qps else 0.0,
        pivot_evals_per_query=float(np.mean([stat.pivot_evals for stat in routed])),
        block_signature_rows_scored_per_query=float(np.mean([stat.block_rows_scored for stat in routed])),
        centroid_evals_per_query=float(np.mean([stat.centroid_evals for stat in routed])),
        selected_blocks_per_query=float(np.mean([stat.selected_blocks for stat in routed])),
        candidate_count_per_query=float(np.mean([len(stat.candidates) for stat in routed])),
        route_resident_bytes=route_resident_bytes(sketch, centroids),
        working_set_bytes_per_query=int(np.mean([stat.working_set_bytes for stat in routed])),
        build_time_s=sketch.build_time_s,
        fallback_rate=float(np.mean([stat.fallback for stat in routed])),
    )


def minimum_specs() -> list[tuple[int, int, int, int, int, str]]:
    return [
        (1024, 16, 16, 16, 256, "l1"),
        (1024, 32, 16, 16, 256, "l1"),
        (1024, 32, 32, 16, 256, "l1"),
        (1024, 32, 32, 16, 256, "l_inf"),
        (4096, 16, 32, 32, 512, "l1"),
        (4096, 32, 32, 32, 512, "l1"),
        (4096, 64, 32, 32, 512, "l1"),
        (4096, 64, 32, 32, 512, "l_inf"),
    ]


def sensitivity_specs() -> list[tuple[int, int, int, int, int, str]]:
    return [
        (4096, 32, 64, 32, 512, "l1"),
        (4096, 32, 32, 16, 512, "l1"),
        (4096, 32, 32, 64, 512, "l1"),
        (4096, 32, 32, 64, 512, "l_inf"),
        (4096, 32, 32, 64, 512, "l2"),
        (4096, 64, 32, 32, 512, "l2"),
        (4096, 32, 32, 32, 512, "fp16"),
        (4096, 32, 32, 32, 512, "uint16"),
    ]


def load_baselines(path: Path) -> list[BaselineRow]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
    out = []
    for row in rows:
        if path.name == "htla_t162.json":
            if row.get("router") != "pca" or row.get("spill") != 4:
                continue
            router = "T-162 overlap PCA spill x4"
            evals = float(row["child_evals_per_query"])
            resident = int(row["route_resident_bytes"])
        elif path.name == "block_graph_t165.json":
            if (row.get("k"), row.get("block_size"), row.get("beam_blocks")) not in {
                (1024, 32, 16),
                (4096, 32, 32),
            }:
                continue
            router = "T-165 block graph best practical"
            evals = float(row["block_evals_per_query"])
            resident = int(row["block_graph_resident_bytes"])
        elif path.name == "landmark_t166.json":
            if row.get("k") == 1024 and row.get("landmarks") != 64:
                continue
            if row.get("k") == 4096 and row.get("landmarks") != 256:
                continue
            router = "T-166 landmark multi-probe"
            evals = float(row["landmark_evals_per_query"])
            resident = int(row["route_resident_bytes"])
        else:
            continue
        out.append(
            BaselineRow(
                router=router,
                k=int(row["k"]),
                coverage_at_16=float(row["coverage_at_16"]),
                candidate_pool_coverage_at_32=float(row["candidate_pool_coverage_at_32"]),
                route_time_us_per_query=float(row["route_time_us_per_query"]),
                evals_per_query=evals,
                candidate_count_per_query=float(row["candidate_count_per_query"]),
                resident_bytes=resident,
                working_set_bytes_per_query=int(row["working_set_bytes_per_query"]),
                build_time_s=float(row["build_time_s"]),
            )
        )
    return sorted(out, key=lambda row: (row.k, row.router, row.route_time_us_per_query))


def best_baselines(rows: list[BaselineRow]) -> list[BaselineRow]:
    best: dict[tuple[str, int], BaselineRow] = {}
    for row in rows:
        key = (row.router, row.k)
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        row_key = (-row.coverage_at_16, row.route_time_us_per_query, -row.candidate_pool_coverage_at_32)
        current_key = (-current.coverage_at_16, current.route_time_us_per_query, -current.candidate_pool_coverage_at_32)
        if row_key < current_key:
            best[key] = row
    return sorted(best.values(), key=lambda row: (row.k, row.router, row.route_time_us_per_query))


def verdict(rows: list[PivotRow], baselines: list[BaselineRow]) -> str:
    best = {k: max((row.coverage_at_16 for row in rows if row.k == k), default=0.0) for k in (1024, 4096)}
    if best[1024] < 0.99 or best[4096] < 0.99:
        return "Negative signal: pivot-distance block sketches do not reach coverage@16 >= 0.99 at both K values under the requested pool caps."
    block_bases = [row for row in baselines if row.router.startswith("T-165")]
    faster = True
    for base in block_bases:
        best_time = min((row.route_time_us_per_query for row in rows if row.k == base.k and row.coverage_at_16 >= 0.99), default=float("inf"))
        faster = faster and best_time <= base.route_time_us_per_query
    if faster:
        return "Positive signal: pivot-sketch routing reaches coverage@16 >= 0.99 at both K values and is route-time competitive with T-165 while using dense sequential signature scans."
    return "Mixed signal: pivot-sketch routing reaches coverage@16 >= 0.99 at both K values, but it does not beat the best practical T-165 block graph route time."


def format_row(row: PivotRow) -> str:
    return (
        f"| {row.k} | {row.block_size} | {row.blocks} | {row.pivots} | {row.top_blocks} | {row.pool} | "
        f"{row.signature} | {row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | "
        f"{row.qps:.1f} | {row.route_time_us_per_query:.1f} | {row.pivot_evals_per_query:.1f} | "
        f"{row.block_signature_rows_scored_per_query:.1f} | {row.centroid_evals_per_query:.1f} | "
        f"{row.selected_blocks_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
        f"{row.route_resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} | "
        f"{row.fallback_rate:.4f} |"
    )


def markdown(rows: list[PivotRow], baselines: list[BaselineRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    minimum = set(minimum_specs())
    lines = [
        "# T-167 Pivot-Sketch / LAESA-Style Block Router Prototype",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Final exact rerank: `final_nprobe={FINAL_NPROBE}` over the bounded centroid candidate pool.",
        "- Centroids use the T-162/T-165 deterministic `linspace` sample policy.",
        "- Blocks reuse the T-165 deterministic balanced block layout shape: contiguous int32 payload plus offsets.",
        "- Routing computes query-to-pivot distances, dense-scores every block signature row with the configured signature metric (`l1`, `l_inf`, or `l2`), scans selected blocks contiguously, then exact-reranks centroids.",
        "",
        "## Minimum Matrix",
        "",
        "| K | Block size | Blocks | Pivots | Top blocks | Pool | Signature | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Block rows/q | Centroid evals/q | Selected blocks/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if (row.k, row.block_size, row.pivots, row.top_blocks, row.pool, row.signature) in minimum:
            lines.append(format_row(row))
    lines.extend(
        [
            "",
            "## Sensitivity Rows",
            "",
            "| K | Block size | Blocks | Pivots | Top blocks | Pool | Signature | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Block rows/q | Centroid evals/q | Selected blocks/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if (row.k, row.block_size, row.pivots, row.top_blocks, row.pool, row.signature) not in minimum:
            lines.append(format_row(row))

    if baselines:
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                "| Router | K | Coverage@16 | Pool coverage@32 | Route us/q | Route evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in baselines:
            lines.append(
                f"| {row.router} | {row.k} | {row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | "
                f"{row.route_time_us_per_query:.1f} | {row.evals_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
                f"{row.resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            verdict(rows, baselines),
            "",
            "## Notes",
            "",
            "- `l1` stores fp32 pivot-distance signatures and scores `sum(abs(block_sig - query_sig))`.",
            "- `l_inf` stores fp32 signatures and scores `max(abs(block_sig - query_sig))`, the LAESA/AESA-style pivot lower-bound score.",
            "- `l2` stores fp32 signatures and scores Euclidean distance between pivot-distance signatures.",
            "- `fp16` and `uint16` sensitivity rows change only the resident signature format; scoring dequantizes/casts in this Python prototype and currently uses the L1 score path.",
            "- Workset bytes estimate the dense pivot reads, dense block-signature scan, selected contiguous centroid scan, and score scratch. It is a prototype accounting model, not a hardware counter.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/data/siftsmall"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/pivot_sketch_t167.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/pivot_sketch_t167.md"))
    parser.add_argument("--route-runs", type=int, default=20)
    parser.add_argument("--cluster-iters", type=int, default=12)
    parser.add_argument("--no-sensitivity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xb = read_fvecs(args.data_dir / "siftsmall_base.fvecs")
    queries = read_fvecs(args.data_dir / "siftsmall_query.fvecs")
    specs = minimum_specs()
    if not args.no_sensitivity:
        specs.extend(sensitivity_specs())

    rows: list[PivotRow] = []
    exact_cache: dict[int, tuple[np.ndarray, list[list[int]], list[list[int]]]] = {}
    sketch_cache: dict[tuple[int, int, int, str], PivotSketch] = {}
    for spec in specs:
        k, block_size, pivots, _top_blocks, _pool, signature = spec
        if k not in exact_cache:
            centroids = sample_centroids(xb, k)
            exact_cache[k] = (centroids, exact_topk(centroids, queries, 16), exact_topk(centroids, queries, 32))
        centroids, exact16, exact32 = exact_cache[k]
        cache_key = (k, block_size, pivots, signature)
        if cache_key not in sketch_cache:
            sketch_cache[cache_key] = build_pivot_sketch(centroids, k, block_size, pivots, signature, args.cluster_iters)
        rows.append(evaluate(sketch_cache[cache_key], centroids, queries, exact16, exact32, spec, args.route_runs))

    baselines: list[BaselineRow] = []
    for path in [
        Path("benchmarks/htla_t162.json"),
        Path("benchmarks/block_graph_t165.json"),
        Path("benchmarks/landmark_t166.json"),
    ]:
        baselines.extend(load_baselines(path))
    baselines = best_baselines(baselines)

    args.output_json.write_text(
        json.dumps(
            {
                "rows": [asdict(row) for row in rows],
                "baselines": [asdict(row) for row in baselines],
                "verdict": verdict(rows, baselines),
            },
            indent=2,
        )
        + "\n"
    )
    args.output_md.write_text(markdown(rows, baselines, args.data_dir))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(verdict(rows, baselines))


if __name__ == "__main__":
    main()
