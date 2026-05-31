#!/usr/bin/env python3
"""T-165 route-only pointerless block graph prototype."""

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
class BlockGraph:
    k: int
    block_size: int
    block_m: int
    offsets: np.ndarray
    payload: np.ndarray
    representatives: np.ndarray
    neighbors: np.ndarray
    entry_blocks: np.ndarray
    build_time_s: float
    mean_neighbor_recall: float


@dataclass(frozen=True)
class RouteStats:
    candidates: list[int]
    block_evals: int
    centroid_evals: int
    selected_blocks: int
    working_set_bytes: int
    fallback: bool


@dataclass(frozen=True)
class BlockRow:
    dataset: str
    k: int
    block_size: int
    blocks: int
    block_m: int
    entry_blocks: int
    rounds: int
    beam_blocks: int
    pool: int
    final_nprobe: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    qps: float
    route_time_us_per_query: float
    block_evals_per_query: float
    centroid_evals_per_query: float
    selected_blocks_per_query: float
    candidate_count_per_query: float
    block_graph_resident_bytes: int
    working_set_bytes_per_query: int
    build_time_s: float
    fallback_rate: float
    mean_neighbor_recall: float


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
    return points[np.array(chosen)].copy()


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


def row_topk(points: np.ndarray, candidate_count: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(points)
    norms = np.einsum("ij,ij->i", points, points)
    dists = norms[None, :] + norms[:, None] - 2.0 * (points @ points.T)
    np.fill_diagonal(dists, np.inf)
    limit = min(candidate_count, count - 1)
    idx = np.argpartition(dists, limit - 1, axis=1)[:, :limit]
    local = np.take_along_axis(dists, idx, axis=1)
    order = np.argsort(local, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, axis=1)
    local = np.take_along_axis(local, order, axis=1)
    return idx.astype(np.int32, copy=False), local.astype(np.float32, copy=False)


def diversify_neighbors(points: np.ndarray, candidates: np.ndarray, candidate_dists: np.ndarray, m: int) -> np.ndarray:
    neighbors = np.empty((len(points), m), dtype=np.int32)
    for row in range(len(points)):
        selected: list[int] = []
        for cand, cand_dist in zip(candidates[row], candidate_dists[row]):
            if len(selected) >= m:
                break
            if not selected:
                selected.append(int(cand))
                continue
            chosen = points[np.array(selected, dtype=np.int64)]
            sep = np.sum((chosen - points[int(cand)]) ** 2, axis=1)
            if float(np.min(sep)) > float(cand_dist):
                selected.append(int(cand))
        if len(selected) < m:
            for cand in candidates[row]:
                value = int(cand)
                if value not in selected:
                    selected.append(value)
                if len(selected) >= m:
                    break
        neighbors[row] = np.array(selected[:m], dtype=np.int32)
    return neighbors


def choose_entry_blocks(reps: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(reps))
    center = reps.mean(axis=0)
    first = int(np.argmin(np.sum((reps - center) ** 2, axis=1)))
    chosen = [first]
    nearest = np.sum((reps - reps[first]) ** 2, axis=1)
    while len(chosen) < count:
        nearest[np.array(chosen)] = -1.0
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        dist = np.sum((reps - reps[nxt]) ** 2, axis=1)
        nearest = np.minimum(nearest, dist)
    return np.array(chosen, dtype=np.int32)


def build_block_graph(
    centroids: np.ndarray,
    k: int,
    block_size: int,
    block_m: int,
    entry_blocks: int,
    cluster_iters: int,
) -> BlockGraph:
    start = time.perf_counter()
    offsets, payload, reps = balanced_blocks(centroids, block_size, cluster_iters)
    candidate_count = min(len(reps) - 1, max(block_m, block_m * 4))
    candidates, candidate_dists = row_topk(reps, candidate_count)
    exact_m = candidates[:, :block_m].copy()
    neighbors = diversify_neighbors(reps, candidates, candidate_dists, block_m)
    entries = choose_entry_blocks(reps, entry_blocks)
    build_time = time.perf_counter() - start
    recalls = [len(set(neighbors[row].tolist()) & set(exact_m[row].tolist())) / block_m for row in range(len(reps))]
    return BlockGraph(
        k=k,
        block_size=block_size,
        block_m=block_m,
        offsets=offsets,
        payload=payload,
        representatives=reps,
        neighbors=neighbors,
        entry_blocks=entries,
        build_time_s=build_time,
        mean_neighbor_recall=float(np.mean(recalls)),
    )


def final_rerank(centroids: np.ndarray, query: np.ndarray, candidates: list[int], limit: int) -> list[int]:
    if not candidates:
        return []
    idx = np.array(candidates, dtype=np.int64)
    diff = centroids[idx] - query
    dists = np.einsum("ij,ij->i", diff, diff)
    order = np.argsort(dists, kind="stable")[:limit]
    return [int(idx[pos]) for pos in order]


def route_block_graph(
    graph: BlockGraph,
    centroids: np.ndarray,
    query: np.ndarray,
    rounds: int,
    beam_blocks: int,
    pool: int,
) -> RouteStats:
    reps = graph.representatives
    scores: dict[int, float] = {}
    expanded: set[int] = set()
    block_evals = 0

    for raw in graph.entry_blocks.tolist():
        idx = int(raw)
        diff = reps[idx] - query
        scores[idx] = float(np.dot(diff, diff))
        block_evals += 1

    frontier = sorted(scores, key=lambda idx: (scores[idx], idx))[:beam_blocks]
    for _ in range(rounds):
        next_ids: list[int] = []
        seen_next: set[int] = set()
        for idx in frontier:
            if idx in expanded:
                continue
            expanded.add(idx)
            for raw in graph.neighbors[idx].tolist():
                value = int(raw)
                if value in scores or value in seen_next:
                    continue
                seen_next.add(value)
                next_ids.append(value)
        if next_ids:
            arr = np.array(next_ids, dtype=np.int64)
            diff = reps[arr] - query
            dists = np.einsum("ij,ij->i", diff, diff)
            block_evals += len(next_ids)
            for idx, dist in zip(next_ids, dists):
                scores[int(idx)] = float(dist)
        frontier = [
            idx
            for idx in sorted(scores, key=lambda value: (scores[value], value))
            if idx not in expanded
        ][:beam_blocks]

    selected = sorted(scores, key=lambda idx: (scores[idx], idx))[:beam_blocks]
    centroid_ids: list[int] = []
    for block in selected:
        start = int(graph.offsets[block])
        end = int(graph.offsets[block + 1])
        centroid_ids.extend(int(idx) for idx in graph.payload[start:end].tolist())

    centroid_evals = len(centroid_ids)
    candidates = final_rerank(centroids, query, centroid_ids, pool)
    working_set = (
        block_evals * reps.shape[1] * 4
        + len(expanded) * graph.block_m * 4
        + centroid_evals * centroids.shape[1] * 4
        + max(len(scores), len(selected), 1) * 12
    )
    return RouteStats(
        candidates=candidates,
        block_evals=block_evals,
        centroid_evals=centroid_evals,
        selected_blocks=len(selected),
        working_set_bytes=int(working_set),
        fallback=len(candidates) < min(FINAL_NPROBE, len(centroids)),
    )


def block_graph_resident_bytes(graph: BlockGraph, centroids: np.ndarray) -> int:
    return int(
        centroids.size * 4
        + graph.payload.size * 4
        + graph.offsets.size * 4
        + graph.representatives.size * 4
        + graph.neighbors.size * 4
        + graph.entry_blocks.size * 4
    )


def evaluate_graph(
    graph: BlockGraph,
    centroids: np.ndarray,
    queries: np.ndarray,
    exact16: list[list[int]],
    exact32: list[list[int]],
    spec: tuple[int, int, int, int, int, int, int],
    route_runs: int,
) -> BlockRow:
    k, block_size, block_m, entry_blocks, rounds, beam_blocks, pool = spec

    def route_once() -> list[RouteStats]:
        return [route_block_graph(graph, centroids, query, rounds, beam_blocks, pool) for query in queries]

    route_once()
    start = time.perf_counter()
    for _ in range(route_runs):
        routed = route_once()
    elapsed = time.perf_counter() - start
    qps = (len(queries) * route_runs) / elapsed if elapsed else 0.0
    finals = [final_rerank(centroids, query, stat.candidates, FINAL_NPROBE) for query, stat in zip(queries, routed)]
    candidate_lists = [stat.candidates for stat in routed]
    return BlockRow(
        dataset="siftsmall",
        k=k,
        block_size=block_size,
        blocks=len(graph.representatives),
        block_m=block_m,
        entry_blocks=entry_blocks,
        rounds=rounds,
        beam_blocks=beam_blocks,
        pool=pool,
        final_nprobe=FINAL_NPROBE,
        coverage_at_16=coverage(finals, exact16),
        candidate_pool_coverage_at_32=coverage(candidate_lists, exact32),
        qps=qps,
        route_time_us_per_query=(1_000_000.0 / qps) if qps else 0.0,
        block_evals_per_query=float(np.mean([stat.block_evals for stat in routed])),
        centroid_evals_per_query=float(np.mean([stat.centroid_evals for stat in routed])),
        selected_blocks_per_query=float(np.mean([stat.selected_blocks for stat in routed])),
        candidate_count_per_query=float(np.mean([len(stat.candidates) for stat in routed])),
        block_graph_resident_bytes=block_graph_resident_bytes(graph, centroids),
        working_set_bytes_per_query=int(np.mean([stat.working_set_bytes for stat in routed])),
        build_time_s=graph.build_time_s,
        fallback_rate=float(np.mean([stat.fallback for stat in routed])),
        mean_neighbor_recall=graph.mean_neighbor_recall,
    )


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
            out.append(
                BaselineRow(
                    router="T-162 overlap PCA spill x4",
                    k=int(row["k"]),
                    coverage_at_16=float(row["coverage_at_16"]),
                    candidate_pool_coverage_at_32=float(row["candidate_pool_coverage_at_32"]),
                    route_time_us_per_query=float(row["route_time_us_per_query"]),
                    evals_per_query=float(row["child_evals_per_query"]),
                    candidate_count_per_query=float(row["candidate_count_per_query"]),
                    resident_bytes=int(row["route_resident_bytes"]),
                    working_set_bytes_per_query=int(row["working_set_bytes_per_query"]),
                    build_time_s=float(row["build_time_s"]),
                )
            )
        elif path.name == "graph_t164.json":
            if row.get("graph") != "diverse" or row.get("m") != 16:
                continue
            out.append(
                BaselineRow(
                    router="T-164 diverse point graph M16",
                    k=int(row["k"]),
                    coverage_at_16=float(row["coverage_at_16"]),
                    candidate_pool_coverage_at_32=float(row["candidate_pool_coverage_at_32"]),
                    route_time_us_per_query=float(row["route_time_us_per_query"]),
                    evals_per_query=float(row["edge_evals_per_query"]),
                    candidate_count_per_query=float(row["candidate_count_per_query"]),
                    resident_bytes=int(row["graph_resident_bytes"]),
                    working_set_bytes_per_query=int(row["working_set_bytes_per_query"]),
                    build_time_s=float(row["build_time_s"]),
                )
            )
    return sorted(out, key=lambda row: (row.k, row.router, row.route_time_us_per_query))


def minimum_specs() -> list[tuple[int, int, int, int, int, int, int]]:
    return [
        (1024, 16, 8, 8, 4, 16, 256),
        (1024, 32, 8, 8, 4, 16, 256),
        (1024, 64, 8, 8, 4, 16, 256),
        (4096, 16, 8, 16, 5, 32, 512),
        (4096, 32, 8, 16, 5, 32, 512),
        (4096, 64, 8, 16, 5, 32, 512),
    ]


def sensitivity_specs() -> list[tuple[int, int, int, int, int, int, int]]:
    return [
        (4096, 16, 16, 16, 5, 32, 512),
        (4096, 32, 16, 16, 5, 32, 512),
        (1024, 32, 8, 8, 3, 16, 256),
        (1024, 32, 8, 8, 5, 16, 256),
        (4096, 32, 8, 16, 5, 16, 512),
        (4096, 32, 8, 16, 5, 64, 512),
    ]


def verdict(rows: list[BlockRow], baselines: list[BaselineRow]) -> str:
    best = {
        k: max((row.coverage_at_16 for row in rows if row.k == k), default=0.0)
        for k in (1024, 4096)
    }
    if best[1024] >= 0.99 and best[4096] >= 0.99:
        faster_than_tree = True
        for base in baselines:
            if base.router.startswith("T-162"):
                best_time = min((row.route_time_us_per_query for row in rows if row.k == base.k), default=float("inf"))
                faster_than_tree = faster_than_tree and best_time <= base.route_time_us_per_query
        if faster_than_tree:
            return "Positive signal: block graph routing reaches coverage@16 >= 0.99 at both K values and is route-time competitive with the T-162 overlap PCA baseline while using regular block scans."
        return "Mixed signal: block graph routing reaches coverage@16 >= 0.99 at both K values, but route time is not clearly better than the T-162 overlap PCA baseline."
    return "Negative signal: block graph routing does not reach coverage@16 >= 0.99 at both K values under the requested pool caps; block scans improve locality but lose too many boundary cases."


def format_row(row: BlockRow) -> str:
    return (
        f"| {row.k} | {row.block_size} | {row.blocks} | {row.block_m} | {row.entry_blocks} | "
        f"{row.rounds} | {row.beam_blocks} | {row.pool} | {row.coverage_at_16:.4f} | "
        f"{row.candidate_pool_coverage_at_32:.4f} | {row.qps:.1f} | {row.route_time_us_per_query:.1f} | "
        f"{row.block_evals_per_query:.1f} | {row.centroid_evals_per_query:.1f} | {row.selected_blocks_per_query:.1f} | "
        f"{row.candidate_count_per_query:.1f} | {row.block_graph_resident_bytes:,} | "
        f"{row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} | {row.fallback_rate:.4f} | "
        f"{row.mean_neighbor_recall:.4f} |"
    )


def markdown(rows: list[BlockRow], baselines: list[BaselineRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    minimum = set(minimum_specs())
    lines = [
        "# T-165 Pointerless Block Graph Router Prototype",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Final exact rerank: `final_nprobe={FINAL_NPROBE}` over the bounded centroid candidate pool.",
        "- Centroids are deterministically sampled, clustered into near-fixed-size blocks, and stored as a contiguous payload array with int32 offsets.",
        "- Block routing uses a fixed-width `block_neighbors[B, M]` int32 matrix over block representatives; selected blocks are scanned contiguously before bounded exact centroid rerank.",
        "",
        "## Minimum Matrix",
        "",
        "| K | Block size | Blocks | Block M | Entries | Rounds | Beam blocks | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Block evals/q | Centroid evals/q | Selected blocks/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Neighbor recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        key = (row.k, row.block_size, row.block_m, row.entry_blocks, row.rounds, row.beam_blocks, row.pool)
        if key in minimum:
            lines.append(format_row(row))

    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "| K | Block size | Blocks | Block M | Entries | Rounds | Beam blocks | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Block evals/q | Centroid evals/q | Selected blocks/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Neighbor recall |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        key = (row.k, row.block_size, row.block_m, row.entry_blocks, row.rounds, row.beam_blocks, row.pool)
        if key not in minimum:
            lines.append(format_row(row))

    if baselines:
        lines.extend(
            [
                "",
                "## Baselines",
                "",
                "| K | Router | Coverage@16 | Pool coverage@32 | Route us/q | Evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |",
                "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        seen: set[tuple[str, int]] = set()
        for row in baselines:
            key = (row.router, row.k)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"| {row.k} | {row.router} | {row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | "
                f"{row.route_time_us_per_query:.1f} | {row.evals_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
                f"{row.resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} |"
            )

    lines.extend(["", "## Interpretation", "", verdict(rows, baselines)])
    return "\n".join(lines) + "\n"


def run(data_dir: Path, route_runs: int, cluster_iters: int, include_sensitivity: bool) -> tuple[list[BlockRow], list[BaselineRow]]:
    xb = read_fvecs(data_dir / "siftsmall_base.fvecs")
    xq = read_fvecs(data_dir / "siftsmall_query.fvecs")
    specs = minimum_specs()
    if include_sensitivity:
        specs = specs + sensitivity_specs()

    rows: list[BlockRow] = []
    by_k = {k: sample_centroids(xb, k) for k in sorted({spec[0] for spec in specs})}
    exact_by_k = {
        k: (exact_topk(centroids, xq, 16), exact_topk(centroids, xq, 32))
        for k, centroids in by_k.items()
    }

    graph_cache: dict[tuple[int, int, int, int], BlockGraph] = {}
    for spec in specs:
        k, block_size, block_m, entry_blocks, _, _, _ = spec
        centroids = by_k[k]
        exact16, exact32 = exact_by_k[k]
        key = (k, block_size, block_m, entry_blocks)
        if key not in graph_cache:
            graph_cache[key] = build_block_graph(centroids, k, block_size, block_m, entry_blocks, cluster_iters)
        rows.append(evaluate_graph(graph_cache[key], centroids, xq, exact16, exact32, spec, route_runs))

    benchmark_dir = data_dir.parents[1] if len(data_dir.parents) >= 2 else Path("benchmarks")
    baselines = load_baselines(benchmark_dir / "htla_t162.json") + load_baselines(benchmark_dir / "graph_t164.json")
    return rows, baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/data/siftsmall"))
    parser.add_argument("--route-runs", type=int, default=20)
    parser.add_argument("--cluster-iters", type=int, default=8)
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    rows, baselines = run(args.data_dir, args.route_runs, args.cluster_iters, not args.no_sensitivity)
    out_md = Path("benchmarks/block_graph_t165.md")
    out_json = Path("benchmarks/block_graph_t165.json")
    out_md.write_text(markdown(rows, baselines, args.data_dir))
    out_json.write_text(
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
    print(f"wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
