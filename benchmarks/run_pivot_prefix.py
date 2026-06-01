#!/usr/bin/env python3
"""T-168 route-only permutation-prefix pivot posting router prototype."""

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
class PrefixRouter:
    k: int
    block_size: int
    pivot_count: int
    prefix_len: int
    mode: str
    offsets: np.ndarray
    payload: np.ndarray
    representatives: np.ndarray
    pivots: np.ndarray
    block_prefixes: np.ndarray
    posting_offsets: np.ndarray
    posting_payload: np.ndarray
    posting_positions: np.ndarray
    idf: np.ndarray
    build_time_s: float


@dataclass(frozen=True)
class RouteStats:
    candidates: list[int]
    pivot_evals: int
    posting_entries_touched: int
    duplicate_block_rate: float
    selected_blocks: int
    centroid_evals: int
    candidate_count: int
    working_set_bytes: int
    fallback: bool


@dataclass(frozen=True)
class PrefixRow:
    dataset: str
    k: int
    block_size: int
    blocks: int
    pivots: int
    prefix: int
    top_blocks: int
    pool: int
    posting_mode: str
    final_nprobe: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    qps: float
    route_time_us_per_query: float
    pivot_evals_per_query: float
    posting_entries_touched_per_query: float
    duplicate_block_rate: float
    selected_blocks_per_query: float
    centroid_evals_per_query: float
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


def build_prefix_router(
    centroids: np.ndarray,
    k: int,
    block_size: int,
    pivot_count: int,
    prefix_len: int,
    mode: str,
    cluster_iters: int,
) -> PrefixRouter:
    start = time.perf_counter()
    offsets, payload, reps = balanced_blocks(centroids, block_size, cluster_iters)
    pivots = deterministic_seeds(reps, min(pivot_count, len(reps))).astype(np.float32, copy=False)
    dists = pivot_distances(reps, pivots)
    prefix = np.argsort(dists, axis=1, kind="stable")[:, : min(prefix_len, len(pivots))].astype(np.int32)

    buckets: list[list[tuple[int, int]]] = [[] for _ in range(len(pivots))]
    for block in range(len(reps)):
        for pos, pivot_id in enumerate(prefix[block].tolist()):
            buckets[int(pivot_id)].append((block, pos))

    posting_payload: list[int] = []
    posting_positions: list[int] = []
    posting_offsets = [0]
    df = np.zeros(len(pivots), dtype=np.int32)
    for bucket in buckets:
        bucket.sort(key=lambda item: (item[1], item[0]))
        df[len(posting_offsets) - 1] = len(bucket)
        for block, pos in bucket:
            posting_payload.append(int(block))
            posting_positions.append(int(pos))
        posting_offsets.append(len(posting_payload))

    idf = np.log((len(reps) + 1.0) / (df.astype(np.float32) + 1.0)) + 1.0
    return PrefixRouter(
        k=k,
        block_size=block_size,
        pivot_count=len(pivots),
        prefix_len=prefix.shape[1],
        mode=mode,
        offsets=offsets,
        payload=payload,
        representatives=reps,
        pivots=pivots,
        block_prefixes=prefix,
        posting_offsets=np.array(posting_offsets, dtype=np.int32),
        posting_payload=np.array(posting_payload, dtype=np.int32),
        posting_positions=np.array(posting_positions, dtype=np.int16),
        idf=idf.astype(np.float32),
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


def route_prefix(router: PrefixRouter, centroids: np.ndarray, query: np.ndarray, top_blocks: int, pool: int) -> RouteStats:
    qdists = pivot_distances(query.reshape(1, -1), router.pivots)[0]
    qprefix = np.argsort(qdists, kind="stable")[: router.prefix_len]
    scores: dict[int, float] = {}
    entries = 0
    duplicates = 0
    seen_entries: set[int] = set()

    for qpos, pivot_id in enumerate(qprefix.tolist()):
        start = int(router.posting_offsets[int(pivot_id)])
        end = int(router.posting_offsets[int(pivot_id) + 1])
        blocks = router.posting_payload[start:end]
        positions = router.posting_positions[start:end]
        entries += len(blocks)
        for block, bpos in zip(blocks.tolist(), positions.tolist()):
            if int(block) in seen_entries:
                duplicates += 1
            seen_entries.add(int(block))
            if router.mode == "weighted":
                weight = float(router.idf[int(pivot_id)]) / float((qpos + 1) * (int(bpos) + 1))
            else:
                weight = 1.0 / float(qpos + int(bpos) + 1)
            scores[int(block)] = scores.get(int(block), 0.0) + weight

    if not scores:
        selected = np.arange(min(top_blocks, len(router.representatives)), dtype=np.int64)
    else:
        scored = np.array(list(scores), dtype=np.int64)
        values = np.array([scores[int(block)] for block in scored], dtype=np.float32)
        order = np.lexsort((scored, -values))
        selected = scored[order[: min(top_blocks, len(scored))]]

    centroid_ids: list[int] = []
    for raw in selected.tolist():
        block = int(raw)
        start = int(router.offsets[block])
        end = int(router.offsets[block + 1])
        centroid_ids.extend(int(idx) for idx in router.payload[start:end].tolist())

    candidates = final_rerank(centroids, query, centroid_ids, pool)
    workset = (
        router.pivot_count * centroids.shape[1] * 4
        + entries * (4 + 2)
        + len(scores) * 12
        + len(centroid_ids) * centroids.shape[1] * 4
    )
    return RouteStats(
        candidates=candidates,
        pivot_evals=router.pivot_count,
        posting_entries_touched=entries,
        duplicate_block_rate=(duplicates / entries) if entries else 0.0,
        selected_blocks=len(selected),
        centroid_evals=len(centroid_ids),
        candidate_count=len(candidates),
        working_set_bytes=int(workset),
        fallback=len(candidates) < min(FINAL_NPROBE, len(centroids)),
    )


def route_resident_bytes(router: PrefixRouter, centroids: np.ndarray) -> int:
    return int(
        centroids.nbytes
        + router.payload.nbytes
        + router.offsets.nbytes
        + router.representatives.nbytes
        + router.pivots.nbytes
        + router.block_prefixes.nbytes
        + router.posting_offsets.nbytes
        + router.posting_payload.nbytes
        + router.posting_positions.nbytes
        + router.idf.nbytes
    )


def evaluate(
    router: PrefixRouter,
    centroids: np.ndarray,
    queries: np.ndarray,
    exact16: list[list[int]],
    exact32: list[list[int]],
    spec: tuple[int, int, int, int, int, int, str],
    route_runs: int,
) -> PrefixRow:
    k, block_size, pivots, prefix, top_blocks, pool, mode = spec

    def route_once() -> list[RouteStats]:
        return [route_prefix(router, centroids, query, top_blocks, pool) for query in queries]

    route_once()
    start = time.perf_counter()
    for _ in range(route_runs):
        routed = route_once()
    elapsed = time.perf_counter() - start
    qps = (len(queries) * route_runs) / elapsed if elapsed else 0.0
    finals = [final_rerank(centroids, query, stat.candidates, FINAL_NPROBE) for query, stat in zip(queries, routed)]
    return PrefixRow(
        dataset="siftsmall",
        k=k,
        block_size=block_size,
        blocks=len(router.representatives),
        pivots=pivots,
        prefix=prefix,
        top_blocks=top_blocks,
        pool=pool,
        posting_mode=mode,
        final_nprobe=FINAL_NPROBE,
        coverage_at_16=coverage(finals, exact16),
        candidate_pool_coverage_at_32=coverage([stat.candidates for stat in routed], exact32),
        qps=qps,
        route_time_us_per_query=(1_000_000.0 / qps) if qps else 0.0,
        pivot_evals_per_query=float(np.mean([stat.pivot_evals for stat in routed])),
        posting_entries_touched_per_query=float(np.mean([stat.posting_entries_touched for stat in routed])),
        duplicate_block_rate=float(np.mean([stat.duplicate_block_rate for stat in routed])),
        selected_blocks_per_query=float(np.mean([stat.selected_blocks for stat in routed])),
        centroid_evals_per_query=float(np.mean([stat.centroid_evals for stat in routed])),
        candidate_count_per_query=float(np.mean([stat.candidate_count for stat in routed])),
        route_resident_bytes=route_resident_bytes(router, centroids),
        working_set_bytes_per_query=int(np.mean([stat.working_set_bytes for stat in routed])),
        build_time_s=router.build_time_s,
        fallback_rate=float(np.mean([stat.fallback for stat in routed])),
    )


def minimum_specs() -> list[tuple[int, int, int, int, int, int, str]]:
    return [
        (1024, 32, 32, 4, 16, 256, "union"),
        (1024, 32, 32, 8, 16, 256, "union"),
        (1024, 64, 32, 8, 16, 256, "union"),
        (4096, 32, 64, 4, 32, 512, "union"),
        (4096, 32, 64, 8, 32, 512, "union"),
        (4096, 64, 64, 8, 32, 512, "union"),
    ]


def sensitivity_specs() -> list[tuple[int, int, int, int, int, int, str]]:
    return [
        (4096, 32, 64, 8, 32, 512, "weighted"),
        (4096, 32, 64, 8, 16, 512, "weighted"),
        (4096, 32, 64, 8, 64, 512, "weighted"),
        (4096, 32, 64, 12, 32, 512, "union"),
        (4096, 32, 64, 12, 32, 512, "weighted"),
    ]


def load_baselines(path: Path) -> list[BaselineRow]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    rows = raw["rows"] if isinstance(raw, dict) and "rows" in raw else raw
    out = []
    for row in rows:
        if path.name == "block_graph_t165.json":
            if (row.get("k"), row.get("block_size"), row.get("beam_blocks")) not in {(1024, 32, 16), (4096, 32, 32)}:
                continue
            router = "T-165 block graph best practical"
            evals = float(row["block_evals_per_query"])
            resident = int(row["block_graph_resident_bytes"])
        elif path.name == "pivot_sketch_t167.json":
            if row.get("signature") != "l_inf":
                continue
            router = "T-167 pivot sketch l_inf"
            evals = float(row["block_signature_rows_scored_per_query"])
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
    best: dict[tuple[str, int], BaselineRow] = {}
    for row in out:
        key = (row.router, row.k)
        current = best.get(key)
        if current is None or (-row.coverage_at_16, row.route_time_us_per_query) < (
            -current.coverage_at_16,
            current.route_time_us_per_query,
        ):
            best[key] = row
    return sorted(best.values(), key=lambda row: (row.k, row.router))


def verdict(rows: list[PrefixRow], baselines: list[BaselineRow]) -> str:
    best = {k: max((row.coverage_at_16 for row in rows if row.k == k), default=0.0) for k in (1024, 4096)}
    if best[1024] < 0.99 or best[4096] < 0.99:
        return "Negative signal: permutation-prefix postings do not reach coverage@16 >= 0.99 at both K values under the requested pool caps."
    t167 = [row for row in baselines if row.router.startswith("T-167")]
    bytes_ok = True
    for base in t167:
        best_workset = min(
            (row.working_set_bytes_per_query for row in rows if row.k == base.k and row.coverage_at_16 >= 0.99),
            default=10**18,
        )
        bytes_ok = bytes_ok and best_workset < base.working_set_bytes_per_query
    if bytes_ok:
        return "Positive signal: permutation-prefix postings reach coverage@16 >= 0.99 and reduce route working-set bytes versus T-167 dense pivot scans."
    return "Mixed signal: permutation-prefix postings reach coverage@16 >= 0.99, but do not clearly reduce route working-set bytes versus T-167."


def format_row(row: PrefixRow) -> str:
    return (
        f"| {row.k} | {row.block_size} | {row.blocks} | {row.pivots} | {row.prefix} | {row.top_blocks} | "
        f"{row.pool} | {row.posting_mode} | {row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | "
        f"{row.qps:.1f} | {row.route_time_us_per_query:.1f} | {row.pivot_evals_per_query:.1f} | "
        f"{row.posting_entries_touched_per_query:.1f} | {row.duplicate_block_rate:.4f} | "
        f"{row.selected_blocks_per_query:.1f} | {row.centroid_evals_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
        f"{row.route_resident_bytes:,} | {row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} | {row.fallback_rate:.4f} |"
    )


def markdown(rows: list[PrefixRow], baselines: list[BaselineRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    minimum = set(minimum_specs())
    lines = [
        "# T-168 Permutation-Prefix Pivot Posting Router Prototype",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Final exact rerank: `final_nprobe={FINAL_NPROBE}` over the bounded centroid candidate pool.",
        "- Blocks reuse the T-165 deterministic balanced block layout.",
        "- Routing computes query pivot prefix, unions pivot postings, scores candidate blocks by overlap or weighted overlap, scans selected blocks contiguously, then exact-reranks centroids.",
        "",
        "## Minimum Matrix",
        "",
        "| K | Block size | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Posting entries/q | Duplicate block rate | Selected blocks/q | Centroid evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if (row.k, row.block_size, row.pivots, row.prefix, row.top_blocks, row.pool, row.posting_mode) in minimum:
            lines.append(format_row(row))
    lines.extend(
        [
            "",
            "## Sensitivity Rows",
            "",
            "| K | Block size | Blocks | Pivots | Prefix | Top blocks | Pool | Mode | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Pivot evals/q | Posting entries/q | Duplicate block rate | Selected blocks/q | Centroid evals/q | Candidates/q | Resident bytes | Workset bytes/q | Build s | Fallback |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if (row.k, row.block_size, row.pivots, row.prefix, row.top_blocks, row.pool, row.posting_mode) not in minimum:
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
    lines.extend(["", "## Verdict", "", verdict(rows, baselines)])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmarks/data/siftsmall"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/pivot_prefix_t168.json"))
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/pivot_prefix_t168.md"))
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

    rows: list[PrefixRow] = []
    exact_cache: dict[int, tuple[np.ndarray, list[list[int]], list[list[int]]]] = {}
    router_cache: dict[tuple[int, int, int, int, str], PrefixRouter] = {}
    for spec in specs:
        k, block_size, pivots, prefix, _top_blocks, _pool, mode = spec
        if k not in exact_cache:
            centroids = sample_centroids(xb, k)
            exact_cache[k] = (centroids, exact_topk(centroids, queries, 16), exact_topk(centroids, queries, 32))
        centroids, exact16, exact32 = exact_cache[k]
        cache_key = (k, block_size, pivots, prefix, mode)
        if cache_key not in router_cache:
            router_cache[cache_key] = build_prefix_router(centroids, k, block_size, pivots, prefix, mode, args.cluster_iters)
        rows.append(evaluate(router_cache[cache_key], centroids, queries, exact16, exact32, spec, args.route_runs))

    baselines: list[BaselineRow] = []
    for path in [Path("benchmarks/block_graph_t165.json"), Path("benchmarks/pivot_sketch_t167.json")]:
        baselines.extend(load_baselines(path))
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
