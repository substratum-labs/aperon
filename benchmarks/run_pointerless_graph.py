#!/usr/bin/env python3
"""T-164 route-only pointerless fixed-M centroid graph prototype."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


FINAL_NPROBE = 16
GraphKind = Literal["knn", "diverse"]


@dataclass(frozen=True)
class Graph:
    kind: GraphKind
    k: int
    m: int
    neighbors: np.ndarray
    entry_points: np.ndarray
    build_time_s: float
    mean_neighbor_recall: float


@dataclass(frozen=True)
class RouteStats:
    candidates: list[int]
    edge_evals: int
    duplicate_edges: int
    working_set_bytes: int
    fallback: bool


@dataclass(frozen=True)
class GraphRow:
    dataset: str
    graph: GraphKind
    k: int
    m: int
    entry_points: int
    rounds: int
    beam: int
    pool: int
    final_nprobe: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    qps: float
    route_time_us_per_query: float
    edge_evals_per_query: float
    candidate_count_per_query: float
    graph_resident_bytes: int
    working_set_bytes_per_query: int
    build_time_s: float
    fallback_rate: float
    duplicate_expansion_rate: float
    mean_neighbor_recall: float


@dataclass(frozen=True)
class BaselineRow:
    router: str
    k: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    route_time_us_per_query: float
    edge_evals_per_query: float
    candidate_count_per_query: float
    route_resident_bytes: int
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


def final_rerank(centroids: np.ndarray, query: np.ndarray, candidates: list[int], limit: int) -> list[int]:
    if not candidates:
        return []
    idx = np.array(candidates, dtype=np.int64)
    diff = centroids[idx] - query
    dists = np.einsum("ij,ij->i", diff, diff)
    order = np.argsort(dists, kind="stable")[:limit]
    return [int(idx[pos]) for pos in order]


def row_topk(
    centroids: np.ndarray,
    k: int,
    candidate_count: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    norms = np.einsum("ij,ij->i", centroids, centroids)
    out_idx = np.empty((len(centroids), candidate_count), dtype=np.int32)
    out_dist = np.empty((len(centroids), candidate_count), dtype=np.float32)
    limit = min(candidate_count + 1, len(centroids))
    for start in range(0, len(centroids), block_size):
        end = min(start + block_size, len(centroids))
        dists = norms[None, :] + norms[start:end, None] - 2.0 * (centroids[start:end] @ centroids.T)
        rows = np.arange(end - start)
        dists[rows, np.arange(start, end)] = np.inf
        idx = np.argpartition(dists, limit - 1, axis=1)[:, :limit]
        local = np.take_along_axis(dists, idx, axis=1)
        order = np.argsort(local, axis=1, kind="stable")
        idx = np.take_along_axis(idx, order, axis=1)[:, :candidate_count]
        local = np.take_along_axis(local, order, axis=1)[:, :candidate_count]
        out_idx[start:end] = idx.astype(np.int32, copy=False)
        out_dist[start:end] = local.astype(np.float32, copy=False)
    return out_idx, out_dist


def choose_entry_points(centroids: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(centroids))
    center = centroids.mean(axis=0)
    first = int(np.argmin(np.sum((centroids - center) ** 2, axis=1)))
    chosen = [first]
    nearest = np.sum((centroids - centroids[first]) ** 2, axis=1)
    while len(chosen) < count:
        nearest[np.array(chosen)] = -1.0
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        dist = np.sum((centroids - centroids[nxt]) ** 2, axis=1)
        nearest = np.minimum(nearest, dist)
    return np.array(chosen, dtype=np.int32)


def diversify_neighbors(
    centroids: np.ndarray,
    candidates: np.ndarray,
    candidate_dists: np.ndarray,
    m: int,
) -> np.ndarray:
    neighbors = np.empty((len(centroids), m), dtype=np.int32)
    for row in range(len(centroids)):
        selected: list[int] = []
        for cand, cand_dist in zip(candidates[row], candidate_dists[row]):
            if len(selected) >= m:
                break
            if not selected:
                selected.append(int(cand))
                continue
            chosen = centroids[np.array(selected, dtype=np.int64)]
            sep = np.sum((chosen - centroids[int(cand)]) ** 2, axis=1)
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


def build_graph(
    centroids: np.ndarray,
    k: int,
    m: int,
    entry_count: int,
    kind: GraphKind,
    block_size: int,
) -> Graph:
    start = time.perf_counter()
    candidate_count = min(len(centroids) - 1, max(m, m * 4))
    candidates, candidate_dists = row_topk(centroids, k, candidate_count, block_size)
    exact_m = candidates[:, :m].copy()
    if kind == "knn":
        neighbors = exact_m
    else:
        neighbors = diversify_neighbors(centroids, candidates, candidate_dists, m)
    entry_points = choose_entry_points(centroids, entry_count)
    build_time = time.perf_counter() - start
    recalls = [len(set(neighbors[row].tolist()) & set(exact_m[row].tolist())) / m for row in range(k)]
    return Graph(
        kind=kind,
        k=k,
        m=m,
        neighbors=neighbors.astype(np.int32, copy=False),
        entry_points=entry_points,
        build_time_s=build_time,
        mean_neighbor_recall=float(np.mean(recalls)),
    )


def route_graph(
    graph: Graph,
    centroids: np.ndarray,
    query: np.ndarray,
    rounds: int,
    beam: int,
    pool: int,
) -> RouteStats:
    scores: dict[int, float] = {}
    expanded: set[int] = set()
    duplicate_edges = 0
    edge_evals = 0

    for idx in graph.entry_points.tolist():
        diff = centroids[idx] - query
        scores[int(idx)] = float(np.dot(diff, diff))

    frontier = sorted(scores, key=lambda idx: (scores[idx], idx))[:beam]
    for _ in range(rounds):
        if not frontier:
            break
        next_ids: list[int] = []
        seen_next: set[int] = set()
        for idx in frontier:
            if idx in expanded:
                duplicate_edges += graph.m
                continue
            expanded.add(idx)
            row = graph.neighbors[idx]
            edge_evals += len(row)
            for raw in row.tolist():
                value = int(raw)
                if value in scores or value in seen_next:
                    duplicate_edges += 1
                    continue
                seen_next.add(value)
                next_ids.append(value)
        if next_ids:
            next_arr = np.array(next_ids, dtype=np.int64)
            diff = centroids[next_arr] - query
            dists = np.einsum("ij,ij->i", diff, diff)
            for idx, dist in zip(next_ids, dists):
                scores[int(idx)] = float(dist)
        frontier = [
            idx
            for idx in sorted(scores, key=lambda value: (scores[value], value))
            if idx not in expanded
        ][:beam]

    ordered = sorted(scores, key=lambda idx: (scores[idx], idx))[:pool]
    scored_count = len(scores)
    working_set = (
        len(expanded) * graph.m * 4
        + scored_count * centroids.shape[1] * 4
        + max(scored_count, beam) * 12
    )
    return RouteStats(
        candidates=[int(idx) for idx in ordered],
        edge_evals=edge_evals,
        duplicate_edges=duplicate_edges,
        working_set_bytes=int(working_set),
        fallback=len(ordered) < min(FINAL_NPROBE, len(centroids)),
    )


def evaluate_graph(
    graph: Graph,
    centroids: np.ndarray,
    queries: np.ndarray,
    exact16: list[list[int]],
    exact32: list[list[int]],
    spec: tuple[int, int, int, int, int],
    route_runs: int,
) -> GraphRow:
    k, entry_count, rounds, beam, pool = spec

    def route_once() -> list[RouteStats]:
        return [route_graph(graph, centroids, query, rounds, beam, pool) for query in queries]

    route_once()
    start = time.perf_counter()
    for _ in range(route_runs):
        routed = route_once()
    elapsed = time.perf_counter() - start
    qps = (len(queries) * route_runs) / elapsed if elapsed else 0.0
    finals = [final_rerank(centroids, query, stat.candidates, FINAL_NPROBE) for query, stat in zip(queries, routed)]
    candidate_lists = [stat.candidates for stat in routed]
    edge_total = sum(stat.edge_evals for stat in routed)
    dup_total = sum(stat.duplicate_edges for stat in routed)

    return GraphRow(
        dataset="siftsmall",
        graph=graph.kind,
        k=k,
        m=graph.m,
        entry_points=entry_count,
        rounds=rounds,
        beam=beam,
        pool=pool,
        final_nprobe=FINAL_NPROBE,
        coverage_at_16=coverage(finals, exact16),
        candidate_pool_coverage_at_32=coverage(candidate_lists, exact32),
        qps=qps,
        route_time_us_per_query=(1_000_000.0 / qps) if qps else 0.0,
        edge_evals_per_query=float(np.mean([stat.edge_evals for stat in routed])),
        candidate_count_per_query=float(np.mean([len(stat.candidates) for stat in routed])),
        graph_resident_bytes=int(centroids.size * 4 + graph.neighbors.size * 4 + graph.entry_points.size * 4),
        working_set_bytes_per_query=int(np.mean([stat.working_set_bytes for stat in routed])),
        build_time_s=graph.build_time_s,
        fallback_rate=float(np.mean([stat.fallback for stat in routed])),
        duplicate_expansion_rate=float(dup_total / max(1, edge_total + dup_total)),
        mean_neighbor_recall=graph.mean_neighbor_recall,
    )


def load_baselines(path: Path) -> list[BaselineRow]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
    out = []
    for row in rows:
        if row.get("router") != "pca" or row.get("spill") != 4:
            continue
        out.append(
            BaselineRow(
                router="T-162 overlap PCA spill x4",
                k=int(row["k"]),
                coverage_at_16=float(row["coverage_at_16"]),
                candidate_pool_coverage_at_32=float(row["candidate_pool_coverage_at_32"]),
                route_time_us_per_query=float(row["route_time_us_per_query"]),
                edge_evals_per_query=float(row["child_evals_per_query"]),
                candidate_count_per_query=float(row["candidate_count_per_query"]),
                route_resident_bytes=int(row["route_resident_bytes"]),
                working_set_bytes_per_query=int(row["working_set_bytes_per_query"]),
                build_time_s=float(row["build_time_s"]),
            )
        )
    return sorted(out, key=lambda row: row.k)


def minimum_specs() -> list[tuple[int, int, int, int, int, int]]:
    return [
        (1024, 8, 8, 4, 32, 256),
        (1024, 16, 8, 4, 32, 256),
        (1024, 32, 8, 4, 32, 256),
        (4096, 8, 16, 5, 64, 512),
        (4096, 16, 16, 5, 64, 512),
        (4096, 32, 16, 5, 64, 512),
    ]


def sensitivity_specs() -> list[tuple[int, int, int, int, int, int]]:
    return [
        (1024, 16, 8, 3, 32, 256),
        (1024, 16, 8, 5, 32, 256),
        (1024, 16, 8, 4, 16, 256),
        (1024, 16, 8, 4, 64, 256),
        (4096, 16, 16, 4, 64, 512),
        (4096, 16, 16, 6, 64, 512),
        (4096, 16, 16, 5, 32, 512),
        (4096, 16, 16, 5, 128, 512),
    ]


def verdict(rows: list[GraphRow], baselines: list[BaselineRow]) -> str:
    best = {
        k: max((row.coverage_at_16 for row in rows if row.k == k), default=0.0)
        for k in (1024, 4096)
    }
    comparable = []
    for base in baselines:
        graph_best = min((row.route_time_us_per_query for row in rows if row.k == base.k), default=float("inf"))
        comparable.append(graph_best <= base.route_time_us_per_query)
    if best[1024] >= 0.99 and best[4096] >= 0.99 and all(comparable):
        return "Positive signal: fixed-M pointerless graph routing reaches coverage@16 >= 0.99 at both K values and is route-time competitive with the T-162 overlap PCA baseline."
    if best[1024] >= 0.99 and best[4096] >= 0.99:
        return "Mixed signal: fixed-M pointerless graph routing reaches the coverage target, but route time or workset is not clearly better than the T-162 overlap PCA baseline."
    return "Negative signal: fixed-M pointerless graph routing does not reach coverage@16 >= 0.99 at both K values under the requested pool caps; graph routing needs stronger entry/multiprobe logic or a block-level design before a Rust pass."


def markdown(rows: list[GraphRow], baselines: list[BaselineRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    minimum = {(k, m, e, r, b, p) for k, m, e, r, b, p in minimum_specs()}
    lines = [
        "# T-164 Pointerless Fixed-M Graph Router Prototype",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Final exact rerank: `final_nprobe={FINAL_NPROBE}` over the bounded centroid candidate pool.",
        "- Route structure uses fixed-width `neighbors[K, M]` int32 arrays plus fixed entry-point arrays.",
        "- `knn` uses exact centroid KNN rows; `diverse` greedily prunes a 4M exact candidate list and fills back to fixed width.",
        "",
        "## Minimum Matrix",
        "",
        "| K | Graph | M | Entries | Rounds | Beam | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Edge evals/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Duplicate rate | Neighbor recall |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        key = (row.k, row.m, row.entry_points, row.rounds, row.beam, row.pool)
        if key not in minimum:
            continue
        lines.append(format_row(row))

    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "| K | Graph | M | Entries | Rounds | Beam | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Edge evals/q | Candidates/q | Graph bytes | Workset bytes/q | Build s | Fallback | Duplicate rate | Neighbor recall |",
            "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        key = (row.k, row.m, row.entry_points, row.rounds, row.beam, row.pool)
        if key in minimum:
            continue
        lines.append(format_row(row))

    if baselines:
        lines.extend(
            [
                "",
                "## T-162 Baseline",
                "",
                "| K | Router | Coverage@16 | Pool coverage@32 | Route us/q | Evals/q | Candidates/q | Route bytes | Workset bytes/q | Build s |",
                "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in baselines:
            lines.append(
                f"| {row.k} | {row.router} | {row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | "
                f"{row.route_time_us_per_query:.1f} | {row.edge_evals_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
                f"{row.route_resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} |"
            )

    lines.extend(["", "## Interpretation", "", verdict(rows, baselines)])
    return "\n".join(lines) + "\n"


def format_row(row: GraphRow) -> str:
    return (
        f"| {row.k} | {row.graph} | {row.m} | {row.entry_points} | {row.rounds} | {row.beam} | {row.pool} | "
        f"{row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | {row.qps:.1f} | "
        f"{row.route_time_us_per_query:.1f} | {row.edge_evals_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
        f"{row.graph_resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} | "
        f"{row.fallback_rate:.4f} | {row.duplicate_expansion_rate:.4f} | {row.mean_neighbor_recall:.4f} |"
    )


def run(data_dir: Path, route_runs: int, block_size: int, include_sensitivity: bool) -> tuple[list[GraphRow], list[BaselineRow]]:
    xb = read_fvecs(data_dir / "siftsmall_base.fvecs")
    xq = read_fvecs(data_dir / "siftsmall_query.fvecs")
    specs = minimum_specs()
    if include_sensitivity:
        specs = specs + sensitivity_specs()

    rows: list[GraphRow] = []
    by_k = {k: sample_centroids(xb, k) for k in sorted({spec[0] for spec in specs})}
    exact_by_k = {
        k: (exact_topk(centroids, xq, 16), exact_topk(centroids, xq, 32))
        for k, centroids in by_k.items()
    }

    graph_cache: dict[tuple[int, int, int, GraphKind], Graph] = {}
    for k, m, entry_count, rounds, beam, pool in specs:
        centroids = by_k[k]
        exact16, exact32 = exact_by_k[k]
        for kind in ("knn", "diverse"):
            key = (k, m, entry_count, kind)
            if key not in graph_cache:
                graph_cache[key] = build_graph(centroids, k, m, entry_count, kind, block_size)
            rows.append(
                evaluate_graph(
                    graph_cache[key],
                    centroids,
                    xq,
                    exact16,
                    exact32,
                    (k, entry_count, rounds, beam, pool),
                    route_runs,
                )
            )

    benchmark_dir = data_dir.parents[1] if len(data_dir.parents) >= 2 else Path("benchmarks")
    baselines = load_baselines(benchmark_dir / "htla_t162.json")
    return rows, baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/data/siftsmall"))
    parser.add_argument("--route-runs", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    rows, baselines = run(args.data_dir, args.route_runs, args.block_size, not args.no_sensitivity)
    out_md = Path("benchmarks/graph_t164.md")
    out_json = Path("benchmarks/graph_t164.json")
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
