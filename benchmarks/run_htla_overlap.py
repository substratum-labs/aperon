#!/usr/bin/env python3
"""T-162 route-only overlap/spill HTLA prototype."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


FINAL_NPROBE = 16


@dataclass
class Node:
    indices: np.ndarray
    centroid: np.ndarray
    children: list[int]
    depth: int
    chart_mean: np.ndarray | None = None
    chart_basis: np.ndarray | None = None
    child_coords: np.ndarray | None = None


@dataclass(frozen=True)
class RouteStats:
    candidates: list[int]
    child_evals: int
    working_set_bytes: int
    fallback: bool


@dataclass(frozen=True)
class OverlapRow:
    dataset: str
    router: Literal["exact", "pca"]
    k: int
    levels: int
    chart_dim: int
    beam: int
    pool: int
    spill: int
    final_nprobe: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    qps: float
    route_time_us_per_query: float
    child_evals_per_query: float
    candidate_count_per_query: float
    route_resident_bytes: int
    working_set_bytes_per_query: int
    build_time_s: float
    fallback_rate: float
    nodes: int
    leaves: int
    max_depth: int
    d80_p50: float
    d90_p50: float
    d95_p50: float
    d95_max: int
    pca_neighbor_recall: float
    morton_neighbor_recall: float


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


def fanout_for(count: int, remaining_levels: int) -> int:
    if remaining_levels <= 2:
        return max(1, count)
    internal_hops = remaining_levels - 1
    fanout = math.ceil(count ** (1.0 / internal_hops))
    return max(2, min(fanout, count))


def deterministic_seeds(points: np.ndarray, fanout: int) -> np.ndarray:
    if fanout >= len(points):
        return points.copy()
    center = points.mean(axis=0)
    seeds = [int(np.argmin(np.sum((points - center) ** 2, axis=1)))]
    while len(seeds) < fanout:
        chosen = points[np.array(seeds)]
        nearest = np.min(np.sum((points[:, None, :] - chosen[None, :, :]) ** 2, axis=2), axis=1)
        nearest[np.array(seeds)] = -1.0
        seeds.append(int(np.argmax(nearest)))
    return points[np.array(seeds)].copy()


def balanced_kmeans_groups(points: np.ndarray, indices: np.ndarray, fanout: int, iters: int = 12) -> list[np.ndarray]:
    if fanout >= len(indices):
        return [np.array([idx], dtype=np.int64) for idx in indices]
    centers = deterministic_seeds(points, fanout)
    assign = np.zeros(len(indices), dtype=np.int64)
    cap = math.ceil(len(indices) / fanout)
    for _ in range(iters):
        dists = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        order = np.argsort(dists, axis=1, kind="stable")
        margins = dists[np.arange(len(points)), order[:, 1]] - dists[np.arange(len(points)), order[:, 0]]
        point_order = np.argsort(margins, kind="stable")
        counts = np.zeros(fanout, dtype=np.int64)
        for row in point_order:
            for choice in order[row]:
                if counts[choice] < cap:
                    assign[row] = choice
                    counts[choice] += 1
                    break
        for c in range(fanout):
            mask = assign == c
            if np.any(mask):
                centers[c] = points[mask].mean(axis=0)
    return [indices[assign == c] for c in range(fanout) if np.any(assign == c)]


def pca(points: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean
    if len(points) <= 1 or dim <= 0:
        return mean, np.zeros((points.shape[1], 0), dtype=np.float32), np.zeros(0, dtype=np.float32)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    used = min(dim, vt.shape[0])
    basis = vt[:used].T.astype(np.float32, copy=False)
    values = ((singular[:used] ** 2) / max(1, len(points) - 1)).astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), basis, values


def attach_charts(nodes: list[Node], chart_dim: int) -> None:
    for node in nodes:
        if not node.children:
            continue
        child_centroids = np.stack([nodes[child].centroid for child in node.children])
        mean, basis, _ = pca(child_centroids, min(chart_dim, child_centroids.shape[1], len(node.children)))
        node.chart_mean = mean
        node.chart_basis = basis
        node.child_coords = (child_centroids - mean) @ basis


def build_tree(centroids: np.ndarray, levels: int, chart_dim: int) -> list[Node]:
    nodes: list[Node] = []

    def add(indices: np.ndarray, depth: int) -> int:
        node_idx = len(nodes)
        points = centroids[indices]
        nodes.append(Node(indices=indices, centroid=points.mean(axis=0), children=[], depth=depth))
        if len(indices) <= 1 or depth + 1 >= levels:
            return node_idx
        fanout = fanout_for(len(indices), levels - depth)
        groups = balanced_kmeans_groups(points, indices, fanout)
        child_ids = [add(group, depth + 1) for group in groups]
        child_ids.sort(key=lambda child: (tuple(nodes[child].centroid.tolist()), int(nodes[child].indices[0])))
        nodes[node_idx].children = child_ids
        return node_idx

    add(np.arange(len(centroids), dtype=np.int64), 0)
    attach_charts(nodes, chart_dim)
    return nodes


def final_rerank(centroids: np.ndarray, query: np.ndarray, candidates: list[int], limit: int) -> list[int]:
    scored = [(float(np.sum((centroids[idx] - query) ** 2)), int(idx)) for idx in candidates]
    scored.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, idx in scored[:limit]]


def route_tree(
    nodes: list[Node],
    centroids: np.ndarray,
    query: np.ndarray,
    beam: int,
    pool: int,
    spill: int,
    router: Literal["exact", "pca"],
) -> RouteStats:
    frontier = [0]
    candidates: list[int] = []
    child_evals = 0
    working_set = 0
    effective_beam = max(1, beam * spill)

    while frontier and len(candidates) < pool:
        scored_children: list[tuple[float, int]] = []
        leaf_scored: list[tuple[float, int]] = []
        for node_idx in frontier:
            node = nodes[node_idx]
            if not node.children:
                local = node.indices.tolist()
                local.sort(key=lambda idx: (float(np.sum((centroids[idx] - query) ** 2)), int(idx)))
                candidates.extend(int(idx) for idx in local[: max(0, pool - len(candidates))])
                working_set += len(local) * centroids.shape[1] * 4
                continue

            child_evals += len(node.children)
            working_set += len(node.children) * centroids.shape[1] * 4
            if router == "pca":
                assert node.chart_mean is not None and node.chart_basis is not None and node.child_coords is not None
                qcoord = (query - node.chart_mean) @ node.chart_basis
                diffs = node.child_coords - qcoord
                scores = np.einsum("ij,ij->i", diffs, diffs)
                # The projected distance preserves ordering within one parent.
                # Add the query residual constant so scores from different
                # parent-local charts are comparable when frontiers merge.
                qcenter = query - node.chart_mean
                residual = max(0.0, float(np.dot(qcenter, qcenter) - np.dot(qcoord, qcoord)))
                scores = scores + residual
                working_set += node.child_coords.size * 4 + qcoord.size * 4
            else:
                child_centroids = np.stack([nodes[child].centroid for child in node.children])
                diffs = child_centroids - query
                scores = np.einsum("ij,ij->i", diffs, diffs)

            for pos, child_idx in enumerate(node.children):
                score = float(scores[pos])
                child = nodes[child_idx]
                if child.children:
                    scored_children.append((score, child_idx))
                else:
                    for idx in child.indices.tolist():
                        leaf_scored.append((score, int(idx)))

        if leaf_scored:
            leaf_scored.sort(key=lambda item: (item[0], item[1]))
            for _, idx in leaf_scored[: max(0, pool - len(candidates))]:
                candidates.append(idx)
        if not scored_children:
            break
        scored_children.sort(key=lambda item: (item[0], item[1]))
        frontier = [child for _, child in scored_children[:effective_beam]]

    seen: set[int] = set()
    deduped = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    limited = deduped[:pool]
    return RouteStats(
        candidates=limited,
        child_evals=child_evals,
        working_set_bytes=working_set,
        fallback=len(limited) < min(FINAL_NPROBE, len(centroids)),
    )


def energy_dim(values: np.ndarray, target: float) -> int:
    total = float(np.sum(values))
    if total <= 0.0:
        return 0
    acc = 0.0
    for idx, value in enumerate(values):
        acc += float(value)
        if acc / total >= target:
            return idx + 1
    return len(values)


def morton_u128(coords: np.ndarray) -> int:
    dims = min(16, len(coords))
    if dims == 0:
        return 0
    shifted = np.clip(np.rint(coords), -32768, 32767).astype(np.int32) + 32768
    bits = min(16, 128 // dims)
    key = 0
    for bit in range(bits - 1, -1, -1):
        for value in shifted[:dims]:
            key = (key << 1) | ((int(value) >> bit) & 1)
    return key


def tree_diagnostics(nodes: list[Node], chart_dim: int) -> tuple[dict, float, float]:
    d80: list[int] = []
    d90: list[int] = []
    d95: list[int] = []
    pca_recalls: list[float] = []
    morton_recalls: list[float] = []

    for node in nodes:
        if len(node.children) < 3:
            continue
        child_centroids = np.stack([nodes[child].centroid for child in node.children])
        mean, basis, values = pca(child_centroids, min(chart_dim, child_centroids.shape[1], len(node.children)))
        d80.append(energy_dim(values, 0.80))
        d90.append(energy_dim(values, 0.90))
        d95.append(energy_dim(values, 0.95))
        coords = (child_centroids - mean) @ basis
        keys = [morton_u128(row) for row in coords]
        m = min(8, len(node.children) - 1)
        for i in range(len(node.children)):
            exact_d = np.sum((child_centroids - child_centroids[i]) ** 2, axis=1)
            exact_order = [idx for idx in np.argsort(exact_d, kind="stable").tolist() if idx != i][:m]
            pca_d = np.sum((coords - coords[i]) ** 2, axis=1)
            pca_order = [idx for idx in np.argsort(pca_d, kind="stable").tolist() if idx != i][:m]
            key_order = sorted(
                [idx for idx in range(len(keys)) if idx != i],
                key=lambda idx: (abs(keys[idx] - keys[i]), idx),
            )[:m]
            want = set(exact_order)
            pca_recalls.append(len(want & set(pca_order)) / m)
            morton_recalls.append(len(want & set(key_order)) / m)

    summary = {
        "d80_p50": float(np.median(d80)) if d80 else 0.0,
        "d90_p50": float(np.median(d90)) if d90 else 0.0,
        "d95_p50": float(np.median(d95)) if d95 else 0.0,
        "d95_max": int(max(d95)) if d95 else 0,
    }
    return (
        summary,
        float(np.mean(pca_recalls)) if pca_recalls else 0.0,
        float(np.mean(morton_recalls)) if morton_recalls else 0.0,
    )


def route_resident_bytes(nodes: list[Node], vector_dim: int, include_charts: bool) -> int:
    total = 0
    for node in nodes:
        total += node.centroid.size * 4
        total += node.indices.size * 8
        total += len(node.children) * 8
        if include_charts and node.chart_mean is not None:
            total += node.chart_mean.size * 4
        if include_charts and node.chart_basis is not None:
            total += node.chart_basis.size * 4
        if include_charts and node.child_coords is not None:
            total += node.child_coords.size * 4
    total += len(nodes) * (vector_dim * 4 + 48)
    return int(total)


def evaluate_router(
    centroids: np.ndarray,
    queries: np.ndarray,
    nodes: list[Node],
    exact16: list[list[int]],
    exact32: list[list[int]],
    spec: tuple[int, int, int, int, int, int],
    router: Literal["exact", "pca"],
    build_time: float,
    diag: dict,
    pca_neighbor_recall: float,
    morton_neighbor_recall: float,
    route_runs: int,
) -> OverlapRow:
    k, levels, chart_dim, beam, pool, spill = spec

    def route_once() -> list[RouteStats]:
        return [route_tree(nodes, centroids, query, beam, pool, spill, router) for query in queries]

    route_once()
    start = time.perf_counter()
    for _ in range(route_runs):
        routed = route_once()
    elapsed = time.perf_counter() - start
    qps = (len(queries) * route_runs) / elapsed if elapsed else 0.0
    finals = [final_rerank(centroids, query, stat.candidates, FINAL_NPROBE) for query, stat in zip(queries, routed)]
    candidate_lists = [stat.candidates for stat in routed]
    leaves = sum(1 for node in nodes if not node.children)

    return OverlapRow(
        dataset="siftsmall",
        router=router,
        k=k,
        levels=levels,
        chart_dim=chart_dim,
        beam=beam,
        pool=pool,
        spill=spill,
        final_nprobe=FINAL_NPROBE,
        coverage_at_16=coverage(finals, exact16),
        candidate_pool_coverage_at_32=coverage(candidate_lists, exact32),
        qps=qps,
        route_time_us_per_query=(1_000_000.0 / qps) if qps else 0.0,
        child_evals_per_query=float(np.mean([stat.child_evals for stat in routed])),
        candidate_count_per_query=float(np.mean([len(stat.candidates) for stat in routed])),
        route_resident_bytes=route_resident_bytes(nodes, centroids.shape[1], include_charts=router == "pca"),
        working_set_bytes_per_query=int(np.mean([stat.working_set_bytes for stat in routed])),
        build_time_s=build_time,
        fallback_rate=float(np.mean([stat.fallback for stat in routed])),
        nodes=len(nodes),
        leaves=leaves,
        max_depth=max(node.depth for node in nodes),
        d80_p50=diag["d80_p50"],
        d90_p50=diag["d90_p50"],
        d95_p50=diag["d95_p50"],
        d95_max=diag["d95_max"],
        pca_neighbor_recall=pca_neighbor_recall,
        morton_neighbor_recall=morton_neighbor_recall,
    )


def evaluate_spec(
    xb: np.ndarray,
    xq: np.ndarray,
    spec: tuple[int, int, int, int, int, int],
    route_runs: int,
) -> list[OverlapRow]:
    k, levels, chart_dim, _, _, _ = spec
    centroids = sample_centroids(xb, k)
    exact16 = exact_topk(centroids, xq, 16)
    exact32 = exact_topk(centroids, xq, 32)
    start = time.perf_counter()
    nodes = build_tree(centroids, levels, chart_dim)
    build_time = time.perf_counter() - start
    diag, pca_neighbor_recall, morton_neighbor_recall = tree_diagnostics(nodes, chart_dim)
    return [
        evaluate_router(
            centroids,
            xq,
            nodes,
            exact16,
            exact32,
            spec,
            router,
            build_time,
            diag,
            pca_neighbor_recall,
            morton_neighbor_recall,
            route_runs,
        )
        for router in ("exact", "pca")
    ]


def verdict(rows: list[OverlapRow]) -> str:
    pca_rows = [row for row in rows if row.router == "pca"]
    best_1024 = max((row.coverage_at_16 for row in pca_rows if row.k == 1024 and row.spill <= 4), default=0.0)
    best_4096 = max((row.coverage_at_16 for row in pca_rows if row.k == 4096 and row.spill <= 4), default=0.0)
    exact_best = {
        k: max((row.coverage_at_16 for row in rows if row.router == "exact" and row.k == k), default=0.0)
        for k in (1024, 4096)
    }
    if best_1024 >= 0.99 and best_4096 >= 0.99:
        return "Positive signal: PCA child-distance overlap routing reaches coverage@16 >= 0.99 at K=1024 and K=4096 with spill <= x4 under the requested pool caps. This remains worth a Rust implementation pass."
    if all(value >= 0.99 for value in exact_best.values()):
        return "Negative signal: exact child-distance overlap routing passes, but PCA child-distance routing does not reach the requested high-K coverage. Local chart distance is not yet a sufficient replacement for exact child comparisons."
    return "Negative signal: overlap routing did not reach the requested high-K coverage under the pool caps; the pool/beam story needs another design pass before Rust implementation."


def markdown(rows: list[OverlapRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# T-162 HTLA Overlap / Spill Router Prototype",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Final exact rerank: `final_nprobe={FINAL_NPROBE}` over the bounded candidate pool",
        "- PCA routing uses parent-local chart distance plus the query residual constant for cross-parent score comparability.",
        "- Morton/key ordering is reported only as a local-preservation diagnostic.",
        "",
        "## Route Results",
        "",
        "| K | Levels | Chart dim | Router | Beam | Pool | Spill | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Child evals/q | Candidates/q | Route bytes | Workset bytes/q | Build s | Fallback |",
        "| ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.k} | {row.levels} | {row.chart_dim} | {row.router} | {row.beam} | {row.pool} | x{row.spill} | "
            f"{row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | {row.qps:.1f} | "
            f"{row.route_time_us_per_query:.1f} | {row.child_evals_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
            f"{row.route_resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} | "
            f"{row.fallback_rate:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Spill Sensitivity",
            "",
            "| K | Router | Spill x1 | Spill x2 | Spill x4 |",
            "| ---: | :--- | ---: | ---: | ---: |",
        ]
    )
    for k in sorted({row.k for row in rows}):
        for router in ("exact", "pca"):
            by_spill = {row.spill: row for row in rows if row.k == k and row.router == router}
            lines.append(
                f"| {k} | {router} | "
                f"{by_spill[1].coverage_at_16:.4f} | {by_spill[2].coverage_at_16:.4f} | {by_spill[4].coverage_at_16:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Local PCA Diagnostics",
            "",
            "| K | Chart dim | Nodes | Leaves | Max depth | d80 p50 | d90 p50 | d95 p50 | d95 max | PCA neighbor recall | Morton/key neighbor recall |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.k, row.chart_dim)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {row.k} | {row.chart_dim} | {row.nodes} | {row.leaves} | {row.max_depth} | "
            f"{row.d80_p50:.1f} | {row.d90_p50:.1f} | {row.d95_p50:.1f} | {row.d95_max} | "
            f"{row.pca_neighbor_recall:.4f} | {row.morton_neighbor_recall:.4f} |"
        )

    lines.extend(["", "## Interpretation", "", verdict(rows), ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parent / "data" / "siftsmall"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "htla_t162.md")
    parser.add_argument("--json-output", type=Path, default=Path(__file__).resolve().parent / "htla_t162.json")
    parser.add_argument("--route-runs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xb = read_fvecs(args.data_dir / "siftsmall_base.fvecs")
    xq = read_fvecs(args.data_dir / "siftsmall_query.fvecs")
    matrix = [
        (1024, 3, 12, 8, 256, 1),
        (1024, 3, 12, 8, 256, 2),
        (1024, 3, 12, 8, 256, 4),
        (4096, 4, 16, 16, 512, 1),
        (4096, 4, 16, 16, 512, 2),
        (4096, 4, 16, 16, 512, 4),
    ]
    rows: list[OverlapRow] = []
    for spec in matrix:
        print(
            f"Running T-162 row K={spec[0]} levels={spec[1]} chart_dim={spec[2]} "
            f"beam={spec[3]} pool={spec[4]} spill=x{spec[5]}",
            flush=True,
        )
        rows.extend(evaluate_spec(xb, xq, spec, args.route_runs))

    args.output.write_text(markdown(rows, args.data_dir), encoding="utf-8")
    args.json_output.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
