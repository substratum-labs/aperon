#!/usr/bin/env python3
"""T-166 route-only landmark multi-probe router prototype."""

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
class LandmarkRouter:
    k: int
    landmarks: np.ndarray
    assignment: int
    posting_cap: int
    offsets: np.ndarray
    payload: np.ndarray
    build_time_s: float
    assigned_entries: int
    retained_entries: int


@dataclass(frozen=True)
class RouteStats:
    candidates: list[int]
    landmark_evals: int
    posting_entries_touched: int
    duplicate_rate: float
    working_set_bytes: int
    fallback: bool


@dataclass(frozen=True)
class LandmarkRow:
    dataset: str
    k: int
    landmarks: int
    assignment: str
    probes: int
    posting_cap: int
    pool: int
    final_nprobe: int
    coverage_at_16: float
    candidate_pool_coverage_at_32: float
    qps: float
    route_time_us_per_query: float
    landmark_evals_per_query: float
    posting_entries_touched_per_query: float
    candidate_count_per_query: float
    duplicate_candidate_rate: float
    route_resident_bytes: int
    working_set_bytes_per_query: int
    build_time_s: float
    fallback_rate: float
    assigned_entries: int
    retained_entries: int
    retained_entry_rate: float


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


def deterministic_seeds(centroids: np.ndarray, count: int) -> np.ndarray:
    if count >= len(centroids):
        return centroids.copy()
    center = centroids.mean(axis=0)
    chosen = [int(np.argmin(np.sum((centroids - center) ** 2, axis=1)))]
    nearest = np.sum((centroids - centroids[chosen[0]]) ** 2, axis=1)
    while len(chosen) < count:
        nearest[np.array(chosen)] = -1.0
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        dist = np.sum((centroids - centroids[nxt]) ** 2, axis=1)
        nearest = np.minimum(nearest, dist)
    return centroids[np.array(chosen, dtype=np.int64)].copy()


def balanced_landmarks(centroids: np.ndarray, count: int, iters: int = 12) -> np.ndarray:
    if count >= len(centroids):
        return centroids.copy()
    centers = deterministic_seeds(centroids, count)
    assign = np.zeros(len(centroids), dtype=np.int32)
    cap = int(np.ceil(len(centroids) / count))
    for _ in range(iters):
        dists = np.sum((centroids[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        order = np.argsort(dists, axis=1, kind="stable")
        second = order[:, 1] if count > 1 else order[:, 0]
        margins = dists[np.arange(len(centroids)), second] - dists[np.arange(len(centroids)), order[:, 0]]
        point_order = np.argsort(margins, kind="stable")
        counts = np.zeros(count, dtype=np.int32)
        for row in point_order:
            for choice in order[row]:
                if counts[choice] < cap:
                    assign[row] = int(choice)
                    counts[choice] += 1
                    break
        for landmark_id in range(count):
            mask = assign == landmark_id
            if np.any(mask):
                centers[landmark_id] = centroids[mask].mean(axis=0)

    order = np.lexsort((np.arange(count), centers[:, 0]))
    return centers[order].astype(np.float32, copy=False)


def build_landmark_router(
    centroids: np.ndarray,
    k: int,
    landmark_count: int,
    assignment: int,
    posting_cap: int,
) -> LandmarkRouter:
    start = time.perf_counter()
    landmarks = balanced_landmarks(centroids, landmark_count)
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids)
    landmark_norms = np.einsum("ij,ij->i", landmarks, landmarks)
    dists = centroid_norms[:, None] + landmark_norms[None, :] - 2.0 * (centroids @ landmarks.T)
    assign_count = min(assignment, landmark_count)
    nearest = np.argpartition(dists, assign_count - 1, axis=1)[:, :assign_count]
    local = np.take_along_axis(dists, nearest, axis=1)
    order = np.argsort(local, axis=1, kind="stable")
    nearest = np.take_along_axis(nearest, order, axis=1)
    local = np.take_along_axis(local, order, axis=1)

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(landmark_count)]
    for centroid_id in range(len(centroids)):
        for pos in range(assign_count):
            buckets[int(nearest[centroid_id, pos])].append((float(local[centroid_id, pos]), centroid_id))

    payload: list[int] = []
    offsets = [0]
    assigned_entries = 0
    for bucket in buckets:
        assigned_entries += len(bucket)
        bucket.sort(key=lambda item: (item[0], item[1]))
        payload.extend(idx for _, idx in bucket[:posting_cap])
        offsets.append(len(payload))

    return LandmarkRouter(
        k=k,
        landmarks=landmarks.astype(np.float32, copy=False),
        assignment=assignment,
        posting_cap=posting_cap,
        offsets=np.array(offsets, dtype=np.int32),
        payload=np.array(payload, dtype=np.int32),
        build_time_s=time.perf_counter() - start,
        assigned_entries=assigned_entries,
        retained_entries=len(payload),
    )


def final_rerank(centroids: np.ndarray, query: np.ndarray, candidates: list[int], limit: int) -> list[int]:
    if not candidates:
        return []
    idx = np.array(candidates, dtype=np.int64)
    diff = centroids[idx] - query
    dists = np.einsum("ij,ij->i", diff, diff)
    order = np.argsort(dists, kind="stable")[:limit]
    return [int(idx[pos]) for pos in order]


def route_landmarks(router: LandmarkRouter, centroids: np.ndarray, query: np.ndarray, probes: int, pool: int) -> RouteStats:
    landmark_diff = router.landmarks - query
    landmark_dists = np.einsum("ij,ij->i", landmark_diff, landmark_diff)
    probe_count = min(probes, len(router.landmarks))
    if probe_count < len(router.landmarks):
        probe_ids = np.argpartition(landmark_dists, probe_count - 1)[:probe_count]
        probe_ids = probe_ids[np.argsort(landmark_dists[probe_ids], kind="stable")]
    else:
        probe_ids = np.argsort(landmark_dists, kind="stable")

    raw: list[int] = []
    for landmark_id in probe_ids.tolist():
        start = int(router.offsets[landmark_id])
        end = int(router.offsets[landmark_id + 1])
        raw.extend(int(idx) for idx in router.payload[start:end].tolist())

    seen: set[int] = set()
    unique: list[int] = []
    for idx in raw:
        if idx in seen:
            continue
        seen.add(idx)
        unique.append(idx)

    candidates = final_rerank(centroids, query, unique, pool)
    duplicate_rate = 0.0 if not raw else 1.0 - (len(unique) / len(raw))
    working_set = (
        router.landmarks.size * 4
        + len(raw) * 4
        + len(unique) * (centroids.shape[1] * 4 + 4)
        + max(len(raw), 1) * 4
    )
    return RouteStats(
        candidates=candidates,
        landmark_evals=len(router.landmarks),
        posting_entries_touched=len(raw),
        duplicate_rate=float(duplicate_rate),
        working_set_bytes=int(working_set),
        fallback=len(candidates) < min(FINAL_NPROBE, len(centroids)),
    )


def route_resident_bytes(router: LandmarkRouter) -> int:
    return int(router.landmarks.size * 4 + router.offsets.size * 4 + router.payload.size * 4)


def evaluate_router(
    router: LandmarkRouter,
    centroids: np.ndarray,
    queries: np.ndarray,
    exact16: list[list[int]],
    exact32: list[list[int]],
    spec: tuple[int, int, int, int, int],
    route_runs: int,
) -> LandmarkRow:
    k, landmark_count, assignment, probes, posting_cap, pool = spec

    def route_once() -> list[RouteStats]:
        return [route_landmarks(router, centroids, query, probes, pool) for query in queries]

    route_once()
    start = time.perf_counter()
    for _ in range(route_runs):
        routed = route_once()
    elapsed = time.perf_counter() - start
    qps = (len(queries) * route_runs) / elapsed if elapsed else 0.0
    finals = [final_rerank(centroids, query, stat.candidates, FINAL_NPROBE) for query, stat in zip(queries, routed)]
    candidate_lists = [stat.candidates for stat in routed]
    return LandmarkRow(
        dataset="siftsmall",
        k=k,
        landmarks=landmark_count,
        assignment=f"top-{assignment}",
        probes=probes,
        posting_cap=posting_cap,
        pool=pool,
        final_nprobe=FINAL_NPROBE,
        coverage_at_16=coverage(finals, exact16),
        candidate_pool_coverage_at_32=coverage(candidate_lists, exact32),
        qps=qps,
        route_time_us_per_query=(1_000_000.0 / qps) if qps else 0.0,
        landmark_evals_per_query=float(np.mean([stat.landmark_evals for stat in routed])),
        posting_entries_touched_per_query=float(np.mean([stat.posting_entries_touched for stat in routed])),
        candidate_count_per_query=float(np.mean([len(stat.candidates) for stat in routed])),
        duplicate_candidate_rate=float(np.mean([stat.duplicate_rate for stat in routed])),
        route_resident_bytes=route_resident_bytes(router),
        working_set_bytes_per_query=int(np.mean([stat.working_set_bytes for stat in routed])),
        build_time_s=router.build_time_s,
        fallback_rate=float(np.mean([stat.fallback for stat in routed])),
        assigned_entries=router.assigned_entries,
        retained_entries=router.retained_entries,
        retained_entry_rate=router.retained_entries / max(1, router.assigned_entries),
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
        elif path.name == "block_graph_t165.json":
            if row.get("block_size") != 32 or row.get("block_m") != 8:
                continue
            expected_beam = 16 if int(row["k"]) == 1024 else 32
            if row.get("beam_blocks") != expected_beam:
                continue
            out.append(
                BaselineRow(
                    router="T-165 block graph size32 M8",
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
    return sorted(out, key=lambda row: (row.k, row.router, row.route_time_us_per_query))


def minimum_specs() -> list[tuple[int, int, int, int, int, int]]:
    return [
        (1024, 64, 1, 4, 64, 256),
        (1024, 64, 2, 4, 64, 256),
        (1024, 128, 1, 4, 64, 256),
        (1024, 128, 2, 4, 64, 256),
        (4096, 128, 1, 8, 128, 512),
        (4096, 128, 2, 8, 128, 512),
        (4096, 256, 1, 8, 128, 512),
        (4096, 256, 2, 8, 128, 512),
    ]


def sensitivity_specs() -> list[tuple[int, int, int, int, int, int]]:
    return [
        (4096, 128, 1, 4, 128, 512),
        (4096, 128, 1, 16, 128, 512),
        (4096, 128, 2, 4, 128, 512),
        (4096, 128, 2, 16, 128, 512),
        (4096, 256, 1, 4, 128, 512),
        (4096, 256, 1, 16, 128, 512),
        (4096, 256, 2, 4, 128, 512),
        (4096, 256, 2, 16, 128, 512),
        (4096, 256, 2, 8, 64, 512),
        (4096, 256, 2, 8, 256, 512),
    ]


def verdict(rows: list[LandmarkRow], baselines: list[BaselineRow]) -> str:
    best = {k: max((row.coverage_at_16 for row in rows if row.k == k), default=0.0) for k in (1024, 4096)}
    broad_scan = any(row.posting_entries_touched_per_query >= row.k * 0.5 for row in rows if row.coverage_at_16 >= 0.99)
    if best[1024] >= 0.99 and best[4096] >= 0.99 and not broad_scan:
        return "Positive signal: landmark multi-probe routing reaches coverage@16 >= 0.99 at both K values without touching large fractions of K, so it should remain a serious pointerless routing candidate."
    if best[1024] >= 0.99 and best[4096] >= 0.99:
        return "Mixed signal: landmark multi-probe routing reaches coverage@16 >= 0.99, but the passing rows rely on broad postings scans that weaken the route-substrate case."
    baseline_4096 = max((row.coverage_at_16 for row in baselines if row.k == 4096), default=0.0)
    if baseline_4096 >= 0.99 and best[4096] < 0.99:
        return "Negative signal: landmark postings stay simple and regular, but they miss high-K boundary cases under the requested pool caps while T-162/T-165 baselines clear coverage@16 >= 0.99."
    return "Mixed signal: landmark postings are regular and cheap to scan, but the current deterministic assignment does not clearly dominate the tree or block baselines."


def format_row(row: LandmarkRow) -> str:
    return (
        f"| {row.k} | {row.landmarks} | {row.assignment} | {row.probes} | {row.posting_cap} | {row.pool} | "
        f"{row.coverage_at_16:.4f} | {row.candidate_pool_coverage_at_32:.4f} | {row.qps:.1f} | "
        f"{row.route_time_us_per_query:.1f} | {row.landmark_evals_per_query:.1f} | "
        f"{row.posting_entries_touched_per_query:.1f} | {row.candidate_count_per_query:.1f} | "
        f"{row.duplicate_candidate_rate:.4f} | {row.route_resident_bytes:,} | "
        f"{row.working_set_bytes_per_query:,} | {row.build_time_s:.4f} | {row.fallback_rate:.4f} | "
        f"{row.retained_entries:,} | {row.retained_entry_rate:.4f} |"
    )


def markdown(rows: list[LandmarkRow], baselines: list[BaselineRow], data_dir: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    minimum = set(minimum_specs())
    lines = [
        "# T-166 Landmark Multi-Probe Router Prototype",
        "",
        f"- Dataset path: `{data_dir}`",
        f"- Generated at: {generated_at}",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Final exact rerank: `final_nprobe={FINAL_NPROBE}` over the bounded candidate pool.",
        "- Centroids use the same deterministic linspace sampling policy as T-162.",
        "- Landmark sets use deterministic balanced k-means representatives over sampled centroids.",
        "- Postings are fixed-cap int32 lists per landmark; `top-2` assignment stores overlapping centroid IDs and duplicate rate is measured after top-R probe union.",
        "",
        "## Minimum Matrix",
        "",
        "| K | Landmarks | Assignment | Probes | Posting cap | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Landmark evals/q | Posting entries/q | Candidates/q | Duplicate rate | Route bytes | Workset bytes/q | Build s | Fallback | Retained entries | Retained rate |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        key = (row.k, row.landmarks, int(row.assignment.split("-")[1]), row.probes, row.posting_cap, row.pool)
        if key in minimum:
            lines.append(format_row(row))

    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "| K | Landmarks | Assignment | Probes | Posting cap | Pool | Coverage@16 | Pool coverage@32 | QPS | Route us/q | Landmark evals/q | Posting entries/q | Candidates/q | Duplicate rate | Route bytes | Workset bytes/q | Build s | Fallback | Retained entries | Retained rate |",
            "| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        key = (row.k, row.landmarks, int(row.assignment.split("-")[1]), row.probes, row.posting_cap, row.pool)
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


def run(data_dir: Path, route_runs: int, include_sensitivity: bool) -> tuple[list[LandmarkRow], list[BaselineRow]]:
    xb = read_fvecs(data_dir / "siftsmall_base.fvecs")
    xq = read_fvecs(data_dir / "siftsmall_query.fvecs")
    specs = minimum_specs()
    if include_sensitivity:
        specs = specs + sensitivity_specs()

    rows: list[LandmarkRow] = []
    by_k = {k: sample_centroids(xb, k) for k in sorted({spec[0] for spec in specs})}
    exact_by_k = {
        k: (exact_topk(centroids, xq, 16), exact_topk(centroids, xq, 32))
        for k, centroids in by_k.items()
    }
    router_cache: dict[tuple[int, int, int, int], LandmarkRouter] = {}
    for spec in specs:
        k, landmark_count, assignment, _, posting_cap, _ = spec
        centroids = by_k[k]
        exact16, exact32 = exact_by_k[k]
        key = (k, landmark_count, assignment, posting_cap)
        if key not in router_cache:
            router_cache[key] = build_landmark_router(centroids, k, landmark_count, assignment, posting_cap)
        rows.append(evaluate_router(router_cache[key], centroids, xq, exact16, exact32, spec, route_runs))

    benchmark_dir = data_dir.parents[1] if len(data_dir.parents) >= 2 else Path("benchmarks")
    baselines = load_baselines(benchmark_dir / "htla_t162.json") + load_baselines(benchmark_dir / "block_graph_t165.json")
    return rows, baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parent / "data" / "siftsmall"
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "landmark_t166.md")
    parser.add_argument("--json-output", type=Path, default=Path(__file__).resolve().parent / "landmark_t166.json")
    parser.add_argument("--route-runs", type=int, default=20)
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    rows, baselines = run(args.data_dir, args.route_runs, not args.no_sensitivity)
    args.output.write_text(markdown(rows, baselines, args.data_dir))
    args.json_output.write_text(
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
    print(f"wrote {args.output} and {args.json_output}")


if __name__ == "__main__":
    main()
