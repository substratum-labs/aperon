#!/usr/bin/env python3
"""
Aperon MemorySpace Crash Recovery & Flush Demo

This script demonstrates:
1. Opening a MemorySpace from a manifest file.
2. Inserting new records (which are written to the Write-Ahead Log (WAL) and MemTable).
3. Deleting records (creating tombstones).
4. Simulating a crash (dropping/releasing the space without flushing).
5. Recovering the state from the WAL upon reopen.
6. Performing a manual Flush to compile the MemTable into an immutable SSTable segment.
"""

from __future__ import annotations
import shutil
import sys
import tempfile
from pathlib import Path

# Try importing from the built extension
try:
    from aperon import MemorySpace, RecallQuery, MemoryManifestFile
except ImportError:
    print("Error: Could not import 'aperon'. Please run 'maturin develop' first to install the bindings.")
    sys.exit(1)


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)


def main() -> None:
    # Set up a clean temporary directory for the demo
    temp_dir = Path(tempfile.mkdtemp(prefix="aperon-crash-demo-"))
    manifest_path = temp_dir / "main.apmf"
    print(f"Created temporary workspace at: {temp_dir}")

    try:
        # Step 1: Create an initial empty manifest file
        print_header("Step 1: Initializing Manifest File")
        manifest = MemoryManifestFile("main", [], None)
        manifest.write(manifest_path)
        print(f"Initial manifest written to: {manifest_path}")

        # Step 2: Open the MemorySpace
        print_header("Step 2: Opening MemorySpace")
        space = MemorySpace.open(manifest_path)
        print("MemorySpace successfully opened!")
        print(f"Active WAL location: {temp_dir / 'wal_active.apmw'}")

        # Step 3: Inserting records (WAL + MemTable write path)
        print_header("Step 3: Inserting Records")
        record_1 = {
            "record_id": 101,
            "scope_id": 1,
            "timestamp": 1000,
            "source_id": 42,
            "confidence": 0.99,
            "text": "Aperon's MemTable is fully functional.",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "symbols": ["memtable", "rust", "python"]
        }
        record_2 = {
            "record_id": 102,
            "scope_id": 1,
            "timestamp": 1005,
            "source_id": 42,
            "confidence": 0.85,
            "text": "This record will be deleted prior to crash.",
            "embedding": [0.0, 1.0, 0.0, 0.0],
            "symbols": ["temp", "delete"]
        }
        
        print(f"Inserting Record 101: '{record_1['text']}'")
        space.insert(record_1)
        
        print(f"Inserting Record 102: '{record_2['text']}'")
        space.insert(record_2)

        # Step 4: Perform recall query
        print_header("Step 4: Recall Query Before Deletion")
        query = RecallQuery(
            embedding=[1.0, 0.0, 0.0, 0.0],
            symbols=["python"],
            limit=5
        )
        result = space.recall(query)
        print(f"Recall hits returned: {len(result['hits'])}")
        for hit in result["hits"]:
            print(f" - Hit: record_id={hit['record_id']}, score={hit['score']:.4f}, text='{hit['text']}'")

        # Step 5: Delete Record 102
        print_header("Step 5: Deleting Record 102")
        print("Deleting Record 102 (writing tombstone to WAL and updating MemTable)")
        space.delete(102)

        # Recall all records to verify deletion
        all_query = RecallQuery(limit=5)
        result = space.recall(all_query)
        print(f"Active records in space: {[h['record_id'] for h in result['hits']]}")

        # Check WAL file exists
        wal_file = temp_dir / "wal_active.apmw"
        print(f"Current WAL file size: {wal_file.stat().st_size} bytes")

        # Step 6: Simulate a crash
        print_header("Step 6: Simulating Crash / Shutdown")
        print("Dropping MemorySpace instance without calling flush().")
        del space
        print("MemorySpace destroyed (simulating un-flushed crash/shutdown).")

        # Step 7: Recover from Crash (Re-open Space)
        print_header("Step 7: Recovering MemorySpace State")
        print("Re-opening MemorySpace from manifest. Replaying active WAL...")
        space = MemorySpace.open(manifest_path)
        print("MemorySpace successfully recovered and re-opened!")

        # Recall after recovery to show WAL replay worked
        result = space.recall(all_query)
        print(f"Recovered active records in space: {[h['record_id'] for h in result['hits']]}")
        assert len(result["hits"]) == 1
        assert result["hits"][0]["record_id"] == 101
        print("Verification SUCCESS: Record 101 is recovered, deleted Record 102 is omitted!")

        # Step 8: Manually Flush MemTable to Segment
        print_header("Step 8: Flushing MemTable to SSTable Segment")
        print("Flushing MemTable...")
        space.flush()
        print("Flush completed successfully!")

        # Verify filesystem changes
        segments = list(temp_dir.glob("*.apms"))
        print(f"Created segment files on disk: {[s.name for s in segments]}")
        print(f"Manifest file updated: {manifest_path.name}")
        
        # Recall query still works on segment
        result = space.recall(all_query)
        print(f"Recall hits after flush: {[h['record_id'] for h in result['hits']]}")
        assert len(result["hits"]) == 1
        assert result["hits"][0]["record_id"] == 101
        print("Verification SUCCESS: Memory Space recall still functional after flush!")

    finally:
        # Cleanup
        print_header("Step 9: Cleaning up workspace")
        shutil.rmtree(temp_dir)
        print("Temporary workspace cleaned up.")


if __name__ == "__main__":
    main()
