#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Set styling
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

BENCH_DIR = Path("/Users/yong/projects/aperon/benchmarks")
ARTIFACTS_DIR = Path("/Users/yong/.gemini/antigravity-cli/brain/25abe1a0-36e1-4506-a57f-9580a12f7e91")

def plot_scenario(summary_path, out_png_name, title):
    if not summary_path.exists():
        print(f"Warning: summary file not found at {summary_path}, skipping.")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    # Filter for the relevant paths
    target_paths = {
        "memory-sstable-array": "Array Scan (Flat)",
        "memory-sstable-pivot": "Pivot Prefix Index",
        "memory-sstable-htla": "HTLA Tangent Index",
        "memory-sstable-planner": "Adaptive Planner"
    }

    labels = []
    latencies = []
    working_sets = []
    recalls = []

    for row in data["rows"]:
        path_name = row["path"]
        if path_name in target_paths:
            labels.append(target_paths[path_name])
            latencies.append(row["latency_us_per_query"])
            
            # Convert working set to KB
            ws_bytes = row["working_set_bytes_per_query"]
            ws_kb = (ws_bytes / 1024.0) if ws_bytes is not None else 0.0
            working_sets.append(ws_kb)
            
            recalls.append(row["top_k_recall"] if row["top_k_recall"] is not None else 0.0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # 1. Latency Bar Chart
    axes[0].bar(labels, latencies, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axes[0].set_title("Query Latency (Lower is Better)")
    axes[0].set_ylabel("Latency (microseconds / query)")
    for i, v in enumerate(latencies):
        axes[0].text(i, v + (max(latencies) * 0.02), f"{v:.1f} us", ha='center', fontweight='bold')

    # 2. Working Set Size Bar Chart
    axes[1].bar(labels, working_sets, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axes[1].set_title("Query Working-Set Size (Lower is Better)")
    axes[1].set_ylabel("Memory Overhead (KB / query)")
    for i, v in enumerate(working_sets):
        axes[1].text(i, v + (max(working_sets) * 0.02), f"{v:.1f} KB", ha='center', fontweight='bold')

    # 3. Recall Bar Chart
    axes[2].bar(labels, recalls, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axes[2].set_title("Recall@10 Accuracy (Higher is Better)")
    axes[2].set_ylabel("Recall")
    axes[2].set_ylim(0.0, 1.1)
    for i, v in enumerate(recalls):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    out_png = BENCH_DIR / out_png_name
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_png}")

    # Copy to artifacts directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = ARTIFACTS_DIR / out_png_name
    shutil.copy(out_png, dest_path)
    print(f"Copied plot to artifacts directory: {dest_path}")

def main():
    # Plot Scenario 1: Synthetic Custom (100k records)
    plot_scenario(
        Path("/Users/yong/projects/aperon/target/memory-sstable-bench/synthetic-custom/summary.json"),
        "sstable_bench_comparison.png",
        "SSTable Search Candidate Generation Path Comparison (Synthetic N = 100,000)"
    )

    # Plot Scenario 2: Real Agent Memory (17k records)
    plot_scenario(
        Path("/Users/yong/projects/aperon/target/memory-sstable-bench/agent-memory/summary.json"),
        "sstable_bench_agent_memory.png",
        "SSTable Search Candidate Generation Path Comparison (Real Agent Memory N = 17,485)"
    )

if __name__ == "__main__":
    main()
