#!/usr/bin/env python3
"""T-169 route-only block-aware RNG / monotonic graph prototype."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from run_block_graph import (
    FINAL_NPROBE,
    balanced_blocks,
    choose_entry_blocks,
    coverage,
    exact_topk,
    final_rerank,
    read_fvecs,
    row_topk,
    sample_centroids,
)


@dataclass(frozen=True)
class RngBlockGraph:
    k: int
    block_size: int
    candidate_m: int
    final_m: int
    prune: str
    offsets: np.ndarray
    payload: np.ndarray
    representatives: np.ndarray
    neighbors: np.ndarray
    entry_blocks: np.ndarray
    build_time_s: float
    mean_neighbor_recall: float
    mean_edge_diversity: float


@dataclass(frozen=True)
class RouteStats:
    candidates: list[int]
    block_evals: int
    centroid_evals: int
    selected_blocks: int
    working_set_bytes: int
    fallback: bool


@dataclass(frozen=True)
class RngRow:
    dataset: str
    k: int
    block_size: int
    blocks: int
    candidate_m: int
    final_m: int
    entry_blocks: int
    rounds: int
    beam_blocks: int
    pool: int
    prune: str
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
    mean_edge_diversity: float


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


def prune_neighbors(
    reps: np.ndarray,
    candidates: np.ndarray,
    candidate_dists: np.ndarray,
    final_m: int,
    prune: str,
) -> np.ndarray:
    neighbors = np.empty((len(reps), final_m), dtype=np.int32)
    alpha = 1.25 if prune == "alpha" else 1.0
    for row in range(len(reps)):
        if prune == "knn":
            selected = [int(value) for value in candidates[row, :final_m].tolist()]
        else:
            selected: list[int] = []
            for cand, cand_dist in zip(candidates[row], candidate_dists[row]):
                value = int(cand)
                if len(selected) >= final_m:
                    break
                if not selected:
                    selected.append(value)
                    continue
                chosen = reps[np.array(selected, dtype=np.int64)]
                sep = np.sum((chosen - reps[value]) ** 2, axis=1)
                threshold = float(cand_dist) / alpha
                if float(np.min(sep)) >= threshold:
                    selected.append(value)
            for cand in candidates[row]:
                value = int(cand)
                if value not in selected:
                    selected.append(value)
                if len(selected) >= final_m:
                    break
        neighbors[row] = np.array(selected[:final_m], dtype=np.int32)
    return neighbors


def edge_diversity(reps: np.ndarray, neighbors: np.ndarray) -> float:
    values: list[float] = []
    for row in range(len(reps)):
        ids = neighbors[row].astype(np.int64)
        if len(ids) < 2:
            continue
        local = reps[ids]
        center_d = np.sum((local - reps[row]) ** 2, axis=1)
        pair = np.sum((local[:, None, :] - local[None, :, :]) ** 2, axis=2)
        tri = pair[np.triu_indices(len(ids), k=1)]
        denom = float(np.mean(center_d)) if np.mean(center_d) > 0 else 1.0
        values.append(float(np.mean(tri)) / denom)
    return float(np.mean(values)) if values else 0.0


def build_rng_graph(
    centroids: np.ndarray,
    k: int,
    block_size: int,
    candidate_m: int,
    final_m: int,
    entry_blocks: int,
    prune: str,
    cluster_iters: int,
) -> RngBlockGraph:
    start = time.perf_counter()
    offsets, payload, reps = balanced_blocks(centroids, block_size, cluster_iters)
    candidate_count = min(len(reps) - 1, max(candidate_m, final_m))
    candidates, candidate_dists = row_topk(reps, candidate_count)
    exact_m = candidates[:, :final_m].copy()
    neighbors = prune_neighbors(reps, candidates, candidate_dists, final_m, prune)
    entries = choose_entry_blocks(reps, entry_blocks)
    recalls = [len(set(neighbors[row].tolist()) & set(exact_m[row].tolist())) / final_m for row in range(len(reps))]
    return RngBlockGraph(
        k=k,
        block_size=block_size,
        candidate_m=candidate_m,
        final_m=final_m,
        prune=prune,
        offsets=offsets,
        payload=payload,
        representatives=reps,
        neighbors=neighbors,
        entry_blocks=entries,
        build_time_s=time.perf_counter() - start,
        mean_neighbor_recall=float(np.mean(recalls)),
        mean_edge_diversity=edge_diversity(reps, neighbors),
    )


def route_graph(
    graph: RngBlockGraph,
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
        frontier = [idx for idx in sorted(scores, key=lambda value: (scores[value], value)) if idx not in expanded][
            :beam_blocks
        ]

    selected = sorted(scores, key=lambda idx: (scores[idx], idx))[:beam_blocks]
    centroid_ids: list[int] = []
    for block in selected:
        start = int(graph.offsets[block])
        end = int(graph.offsets[block + 1])
        centroid_ids.extend(int(idx) for idx in graph.payload[start:end].tolist())

    candidates = final_rerank(centroids, query, centroid_ids, pool)
    working_set = (
        block_evals * reps.shape[1] * 4
        + len(expanded) * graph.final_m * 4
        + len(centroid_ids) * centroids.shape[1] * 4
        + max(len(scores), len(selected), 1) * 12
    )
    return RouteStats(
        candidates=candidates,
        block_evals=block_evals,
        centroid_evals=len(centroid_ids),
        selected_blocks=len(selected),
        working_set_bytes=int(working_set),
        fallback=len(candidates) < min(FINAL_NPROBE, len(centroids)),
    )


def graph_resident_bytes(graph: RngBlockGraph, centroids: np.ndarray) -> int:
    return int(
        centroids.nbytes
        + graph.payload.nbytes
        + graph.offsets.nbytes
        + graph.representatives.nbytes
        + graph.neighbors.nbytes
        + graph.entry_blocks.nbytes
    )


def evaluate(
    graph: RngBlockGraph,
    centroids: np.ndarray,
    queries: np.ndarray,
    exact16: list[list[int]],
    exact32: list[list[int]],
    spec: tuple[int, int, int, int, int, int, int, int, str],
    route_runs: int,
) -> RngRow:
    k, block_size, candidate_m, final_m, entry_blocks, rounds, beam_blocks, pool, prune = spec

    def route_once() -> list[RouteStats]:
        return [route_graph(graph, centroids, query, rounds, beam_blocks, pool) for query in queries]

    route_once()
    start = time.perf_counter()
    for _ in range(route_runs):
        routed = route_once()
    elapsed = time.perf_counter() - start
    qps = (len(queries) * route_runs) / elapsed if elapsed else 0.0
    finals = [final_rerank(centroids, query, stat.candidates, FINAL_NPROBE) for query, stat in zip(queries, routed)]
    return RngRow(
        dataset="siftsmall",
        k=k,
        block_size=block_size,
        blocks=len(graph.representatives),
        candidate_m=candidate_m,
        final_m=final_m,
        entry_blocks=entry_blocks,
        rounds=rounds,
        beam_blocks=beam_blocks,
        pool=pool,
        prune=prune,
        final_nprobe=FINAL_NPROBE,
        coverage_at_16=coverage(finals, exact16),
        candidate_pool_coverage_at_32=coverage([stat.candidates for stat in routed], exact32),
        qps=qps,
        route_time_us_per_query=(1_000_000.0 / qps) if qps else 0.0,
        block_evals_per_query=float(np.mean([stat.block_evals for stat in routed])),
        centroid_evals_per_query=float(np.mean([stat.centroid_evals for stat in routed])),
        selected_blocks_per_query=float(np.mean([stat.selected_blocks for stat in routed])),
        candidate_count_per_query=float(np.mean([len(stat.candidates) for stat in routed])),
        block_graph_resident_bytes=graph_resident_bytes(graph, centroids),
        working_set_bytes_per_query=int(np.mean([stat.working_set_bytes for stat in routed])),
        build_time_s=graph.build_time_s,
        fallback_rate=float(np.mean([stat.fallback for stat in routed])),
        mean_neighbor_recall=graph.mean_neighbor_recall,
        mean_edge_diversity=graph.mean_edge_diversity,
    )


def minimum_specs() -> list[tuple[int, int, int, int, int, int, int, int, str]]:
    return [
        (1024, 32, 32, 8, 8, 4, 16, 256, "rng"),
        (1024, 64, 32, 8, 8, 4, 16, 256, "rng"),
        (4096, 32, 32, 8, 16, 5, 32, 512, "rng"),
        (4096, 64, 32, 8, 16, 5, 32, 512, "rng"),
        (4096, 32, 64, 8, 16, 5, 32, 512, "rng"),
        (4096, 32, 64, 16, 16, 5, 32, 512, "rng"),
    ]


def sensitivity_specs() -> list[tuple[int, int, int, int, int, int, int, int, str]]:
    return [
        (4096, 32, 64, 8, 16, 5, 16, 512, "rng"),
        (4096, 32, 64, 8, 16, 5, 64, 512, "rng"),
        (4096, 32, 64, 8, 16, 5, 32, 512, "alpha"),
        (1024, 32, 32, 8, 8, 4, 16, 256, "knn"),
        (4096, 32, 32, 8, 16, 5, 32, 512, "knn"),
        (4096, 32, 64, 8, 16, 5, 32, 512, "knn"),
    ]


def load_t165(path: Path) -> list[BaselineRow]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
    out: list[BaselineRow] = []
    seen: set[tuple[int, int, int, int]] = set()
    for row in rows:
        if (row.get("k"), row.get("block_size"), row.get("beam_blocks")) not in {
            (1024, 32, 16),
            (4096, 32, 16),
            (4096, 32, 32),
            (4096, 32, 64),
            (4096, 64, 32),
        }:
            continue
        key = (int(row["k"]), int(row["block_size"]), int(row["block_m"]), int(row["beam_blocks"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            BaselineRow(
                router=f"T-165 block graph bs{int(row['block_size'])} beam{int(row['beam_blocks'])}",
                k=int(row["k"]),
                coverage_at_16=float(row["coverage_at_16"]),
                candidate_pool_coverage_at_32=float(row["candidate_pool_coverage_at_32"]),
                route_time_us_per_query=float(row["route_time_us_per_query"]),
                evals_per_query=float(row["block_evals_per_query"]),
                candidate_count_per_query=float(row["candidate_count_per_query"]),
                resident_bytes=int(row["block_graph_resident_bytes"]),
                working_set_bytes_per_query=int(row["working_set_bytes_per_query"]),
                build_time_s=float(row["build_time_s"]),
            )
        )
    return sorted(out, key=lambda row: (row.k, row.coverage_at_16, row.route_time_us_per_query))


def verdict(rows: list[RngRow], baselines: list[BaselineRow]) -> str:
    same_beam32 = [
        row for row in rows if row.k == 4096 and row.block_size == 32 and row.beam_blocks == 32 and row.prune in {"rng", "alpha"}
    ]
    best_4096_32 = max((row.coverage_at_16 for row in same_beam32), default=0.0)
    best_pool_4096_32 = max((row.candidate_pool_coverage_at_32 for row in same_beam32), default=0.0)
    beam16 = max((row.coverage_at_16 for row in rows if row.k == 4096 and row.block_size == 32 and row.beam_blocks == 16), default=0.0)
    t165_beam32 = 0.9944
    t165_pool_beam32 = 0.9947
    t165_beam16 = 0.9563
    improves_beam32 = best_4096_32 > t165_beam32 and best_pool_4096_32 >= t165_pool_beam32
    improves_beam16 = beam16 > t165_beam16
    if improves_beam32 and improves_beam16:
        return "Positive signal: RNG/monotonic edges improve T-165 block graph coverage and K=4096 beam-16 sensitivity while retaining fixed-width pointerless adjacency."
    if improves_beam32 or improves_beam16:
        return "Mixed signal: RNG/monotonic edges improve K=4096 beam-16 sensitivity, but same-beam coverage does not beat T-165 enough to clearly replace it."
    return "Negative signal: RNG/monotonic edge pruning does not improve T-165 coverage or beam sensitivity enough to justify the extra graph construction complexity."


def format_row(row: RngRow) -> str:
    return (
        f"| {row.k} | {row.block_size} | {row.blocks} | {row.candidate_m} | {row.final_m} | "
        f"{row.entry_blocks} | {row.rounds} | {row.beam_blocks} | {row.pool} | {row.prune} | "
        f"{row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | {row.qps:.1f} | "
        f"{row.route_time_us_per_query:.1f} | {row.block_evals_per_query:.1f} | {row.centroid_evals_per_query:.1f} | "
        f"{row.selected_blocks_per_query:.1f} | {row.candidate_count_per_query:.1f} | {row.block_graph_resident_bytes:,} | "
        f"{row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} | {row.fallback_rate:.4f} | "
        f"{row.mean_neighbor_recall:.4f} | {row.mean_edge_diversity:.4f} |"
    )


def markdown(rows: list[RngRow], baselines: list[BaselineRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    minimum = set(minimum_specs())
    header = (
        "| K | Block size | Blocks | Candidate M | Final M | Entries | Rounds | Beam blocks | Pool | Prune | "
        "Coverage@16 | Pool coverage@32 | QPS | Route us/q | Block evals/q | Centroid evals/q | "
        "Selected blocks/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Neighbor recall | Edge diversity |"
    )
    sep = (
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [
        "# T-169 Block-Aware Monotonic / RNG Graph Router Prototype",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Final exact rerank: `final_nprobe={FINAL_NPROBE}` over the bounded centroid candidate pool.",
        "- Blocks reuse the T-165 deterministic balanced block layout.",
        "- `rng` greedily keeps candidates whose distance to already selected neighbors is at least the candidate distance from the source block; deterministic kNN fill preserves fixed width.",
        "- `alpha` is a more permissive RNG variant using `candidate_distance / 1.25` as the separation threshold.",
        "",
        "## Minimum Matrix",
        "",
        header,
        sep,
    ]
    for row in rows:
        if (row.k, row.block_size, row.candidate_m, row.final_m, row.entry_blocks, row.rounds, row.beam_blocks, row.pool, row.prune) in minimum:
            lines.append(format_row(row))
    lines.extend(["", "## Sensitivity Rows", "", header, sep])
    for row in rows:
        if (row.k, row.block_size, row.candidate_m, row.final_m, row.entry_blocks, row.rounds, row.beam_blocks, row.pool, row.prune) not in minimum:
            lines.append(format_row(row))
    if baselines:
        lines.extend(
            [
                "",
                "## T-165 Baselines",
                "",
                "| Router | K | Coverage@16 | Pool coverage@32 | Route us/q | Block evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in baselines:
            lines.append(
                f"| {row.router} | {row.k} | {row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | "
                f"{row.route_time_us_per_query:.1f} | {row.evals_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
                f"{row.resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} |"
            )
    lines.extend(["", "## Verdict", "", verdict(rows, baselines)])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/data/siftsmall"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/block_rng_t169.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/block_rng_t169.md"))
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

    rows: list[RngRow] = []
    exact_cache: dict[int, tuple[np.ndarray, list[list[int]], list[list[int]]]] = {}
    graph_cache: dict[tuple[int, int, int, int, int, str], RngBlockGraph] = {}
    for spec in specs:
        k, block_size, candidate_m, final_m, entry_blocks, _rounds, _beam_blocks, _pool, prune = spec
        if k not in exact_cache:
            centroids = sample_centroids(xb, k)
            exact_cache[k] = (centroids, exact_topk(centroids, queries, 16), exact_topk(centroids, queries, 32))
        centroids, exact16, exact32 = exact_cache[k]
        cache_key = (k, block_size, candidate_m, final_m, entry_blocks, prune)
        if cache_key not in graph_cache:
            graph_cache[cache_key] = build_rng_graph(
                centroids, k, block_size, candidate_m, final_m, entry_blocks, prune, args.cluster_iters
            )
        rows.append(evaluate(graph_cache[cache_key], centroids, queries, exact16, exact32, spec, args.route_runs))

    baselines = load_t165(Path("benchmarks/block_graph_t165.json"))
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
