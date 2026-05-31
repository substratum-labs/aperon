#!/usr/bin/env python3
"""T-161 local-structure diagnostics for HTLA routing."""

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


@dataclass
class Node:
    indices: np.ndarray
    centroid: np.ndarray
    children: list[int]
    depth: int


@dataclass(frozen=True)
class OracleRow:
    dataset: str
    k: int
    levels: int
    dim: int
    beam: int
    pool: int
    coverage_at_16: float
    coverage_at_32: float
    fallback_rate: float
    spill2_coverage_at_16: float
    spill4_coverage_at_16: float
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
        dists = np.sum((centroids - query) ** 2, axis=1)
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
    seeds = [int(np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))]
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


def build_tree(centroids: np.ndarray, levels: int) -> list[Node]:
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
    return nodes


def route_exact_tree(
    nodes: list[Node],
    centroids: np.ndarray,
    query: np.ndarray,
    beam: int,
    pool: int,
) -> tuple[list[int], bool]:
    frontier = [0]
    candidates: list[int] = []
    while frontier and len(candidates) < pool:
        scored_children = []
        leaf_scored = []
        for node_idx in frontier:
            node = nodes[node_idx]
            if not node.children:
                local = node.indices.tolist()
                local.sort(key=lambda idx: (float(np.sum((centroids[idx] - query) ** 2)), int(idx)))
                candidates.extend(int(idx) for idx in local[: max(0, pool - len(candidates))])
                continue
            for child_idx in node.children:
                dist = float(np.sum((nodes[child_idx].centroid - query) ** 2))
                if not nodes[child_idx].children:
                    for idx in nodes[child_idx].indices.tolist():
                        exact_dist = float(np.sum((centroids[idx] - query) ** 2))
                        leaf_scored.append((exact_dist, int(idx)))
                else:
                    scored_children.append((dist, child_idx))
        if leaf_scored:
            leaf_scored.sort(key=lambda item: (item[0], item[1]))
            for _, idx in leaf_scored[: max(0, pool - len(candidates))]:
                candidates.append(idx)
        if not scored_children:
            break
        scored_children.sort(key=lambda item: (item[0], item[1]))
        frontier = [child for _, child in scored_children[:beam]]
    seen = set()
    deduped = []
    for idx in candidates:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped[:pool], len(deduped) < min(FINAL_NPROBE, len(centroids))


def final_rerank(centroids: np.ndarray, query: np.ndarray, candidates: list[int], limit: int) -> list[int]:
    scored = [(float(np.sum((centroids[idx] - query) ** 2)), int(idx)) for idx in candidates]
    scored.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, idx in scored[:limit]]


def pca(points: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    centered = points - mean
    if len(points) <= 1:
        return mean, np.eye(points.shape[1], dim, dtype=np.float32), np.zeros(dim, dtype=np.float32)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:dim].T
    values = (singular[:dim] ** 2) / max(1, len(points) - 1)
    if basis.shape[1] < dim:
        pad = np.eye(points.shape[1], dim - basis.shape[1], dtype=np.float32)
        basis = np.concatenate([basis, pad], axis=1)
        values = np.concatenate([values, np.zeros(dim - len(values), dtype=np.float32)])
    return mean, basis, values


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


def tree_diagnostics(nodes: list[Node], centroids: np.ndarray, chart_dim: int) -> tuple[dict, float, float]:
    d80: list[int] = []
    d90: list[int] = []
    d95: list[int] = []
    pca_recalls: list[float] = []
    morton_recalls: list[float] = []

    for node in nodes:
        if len(node.children) < 3:
            continue
        child_centroids = np.stack([nodes[child].centroid for child in node.children])
        mean, basis, values = pca(child_centroids, min(chart_dim, child_centroids.shape[1]))
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


def evaluate_row(xb: np.ndarray, xq: np.ndarray, k: int, levels: int, dim: int, beam: int, pool: int) -> OracleRow:
    centroids = sample_centroids(xb, k)
    nodes = build_tree(centroids, levels)
    exact16 = exact_topk(centroids, xq, 16)
    exact32 = exact_topk(centroids, xq, 32)

    def run(beam_factor: int) -> tuple[float, float, float]:
        finals = []
        candidates = []
        fallbacks = []
        for query in xq:
            cand, fallback = route_exact_tree(nodes, centroids, query, beam * beam_factor, pool)
            candidates.append(cand)
            finals.append(final_rerank(centroids, query, cand, FINAL_NPROBE))
            fallbacks.append(fallback)
        return coverage(finals, exact16), coverage(candidates, exact32), float(np.mean(fallbacks))

    cov16, cov32, fallback = run(1)
    spill2, _, _ = run(2)
    spill4, _, _ = run(4)
    diag, pca_recall, morton_recall = tree_diagnostics(nodes, centroids, dim)
    leaves = sum(1 for node in nodes if not node.children)

    return OracleRow(
        dataset="siftsmall",
        k=len(centroids),
        levels=levels,
        dim=dim,
        beam=beam,
        pool=pool,
        coverage_at_16=cov16,
        coverage_at_32=cov32,
        fallback_rate=fallback,
        spill2_coverage_at_16=spill2,
        spill4_coverage_at_16=spill4,
        nodes=len(nodes),
        leaves=leaves,
        max_depth=max(node.depth for node in nodes),
        d80_p50=diag["d80_p50"],
        d90_p50=diag["d90_p50"],
        d95_p50=diag["d95_p50"],
        d95_max=diag["d95_max"],
        pca_neighbor_recall=pca_recall,
        morton_neighbor_recall=morton_recall,
    )


def interpretation(rows: list[OracleRow]) -> str:
    max_rows = {
        128: next(row for row in rows if row.k == 128 and row.pool == 128),
        1024: next(row for row in rows if row.k == 1024 and row.pool == 256),
        4096: next(row for row in rows if row.k == 4096 and row.pool == 512),
    }
    high_k = [row for k, row in max_rows.items() if k >= 1024]
    if all(row.coverage_at_16 < 0.99 for row in high_k):
        if any(row.spill2_coverage_at_16 >= 0.99 or row.spill4_coverage_at_16 >= 0.99 for row in high_k):
            return "Exact child-distance tree misses the original high-K beam budgets, but spill/overlap largely recovers coverage. The first failure is boundary/beam sensitivity in the hierarchy; Morton/key routing is an additional loss channel, not the only problem."
        return "Exact child-distance tree also fails at high K under the T-160 pool budgets; the first failure is hierarchy/beam/pool viability, before PCA or Morton routing."
    if any(row.pca_neighbor_recall < 0.8 for row in max_rows.values()):
        return "Exact tree is viable for at least one high-K row, but local PCA loses child-neighbor ordering."
    if any(row.morton_neighbor_recall < row.pca_neighbor_recall - 0.1 for row in max_rows.values()):
        return "Local PCA mostly preserves neighbors, but Morton/key ordering introduces a larger loss."
    return "Oracle results do not isolate a single dominant failure; inspect spill sensitivity and per-node diagnostics."


def markdown(rows: list[OracleRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# T-161 HTLA Local Structure Sanity Check",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        "",
        "## Exact-Tree Oracle",
        "",
        "| K | Levels | Dim | Beam | Pool | Coverage@16 | Coverage@32 | Fallback | Spill x2 Coverage@16 | Spill x4 Coverage@16 | Nodes | Leaves | Max depth |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.k} | {row.levels} | {row.dim} | {row.beam} | {row.pool} | "
            f"{row.coverage_at_16:.4f} | {row.coverage_at_32:.4f} | {row.fallback_rate:.4f} | "
            f"{row.spill2_coverage_at_16:.4f} | {row.spill4_coverage_at_16:.4f} | "
            f"{row.nodes} | {row.leaves} | {row.max_depth} |"
        )
    lines.extend(
        [
            "",
            "## Local Preservation",
            "",
            "| K | Dim | d80 p50 | d90 p50 | d95 p50 | d95 max | PCA neighbor recall | Morton/key neighbor recall |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.k} | {row.dim} | {row.d80_p50:.1f} | {row.d90_p50:.1f} | "
            f"{row.d95_p50:.1f} | {row.d95_max} | {row.pca_neighbor_recall:.4f} | "
            f"{row.morton_neighbor_recall:.4f} |"
        )
    lines.extend(["", "## Interpretation", "", interpretation(rows), ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parent / "data" / "siftsmall"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "htla_t161.md")
    parser.add_argument("--json-output", type=Path, default=Path(__file__).resolve().parent / "htla_t161.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        print(f"Running oracle row K={spec[0]} levels={spec[1]} dim={spec[2]} beam={spec[3]} pool={spec[4]}", flush=True)
        rows.append(evaluate_row(xb, xq, *spec))
    args.output.write_text(markdown(rows, args.data_dir), encoding="utf-8")
    args.json_output.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {args.json_output}", flush=True)


if __name__ == "__main__":
    main()
