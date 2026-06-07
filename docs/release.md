# Release Checklist

This document details the step-by-step verification and release process for publishing new versions of the Aperon Rust crate and Python library.

---

## 1. Pre-Release Local Validation

Before pushing version tags, ensure all local tests, style checks, and builds pass cleanly:

### 1.1 Code Quality Checks
Run formatting and clippy lints across the entire workspace (Rust core, CLI, and Python bindings):
```bash
# Verify formatting
cargo fmt --all --check

# Run lints with warnings treated as errors
cargo clippy --workspace --all-targets -- -D warnings
```

### 1.2 Test Suite Execution
Execute the entire cargo workspace test suite:
```bash
cargo test --workspace
```

### 1.3 CLI Smoke Tests
Run the end-to-end memory SSTable lifecycle script:
```bash
./scripts/run_memory_sstable_mvp.sh
```
Verify that the `memory_sstable_demo` successfully builds a space from a JSONL file, recalls queries, and forks a child manifest.

Verify the command-line index compiler on a synthetic dataset:
```bash
# Generate toy data
python examples/generate_toy.py --out target/toy

# Build an index
cargo run -p aperon-cli -- build --vectors target/toy/vectors.hntr --output target/toy/index.hntm --grains 4

# Query the index
cargo run -p aperon-cli -- query --index target/toy/index.hntm --queries target/toy/queries.hntq --top-k 5

# Evaluate recall
cargo run -p aperon-cli -- eval --index target/toy/index.hntm --vectors target/toy/vectors.hntr --queries target/toy/queries.hntq --top-k 5
```

### 1.4 Python Smoke Tests
Verify Python integration by building the extension locally and running the crash-recovery demo:
```bash
# Build & install in the local virtualenv
maturin develop

# Run the python recovery and flush demo
python examples/crash_recovery_demo.py
```

---

## 2. Versioning and Changelog

Aperon follows Semantic Versioning (`MAJOR.MINOR.PATCH`).

1. **Update `Cargo.toml`**: Increment the version in the workspace `Cargo.toml`:
   ```toml
   [workspace.package]
   version = "0.1.1" # Update this
   ```
2. **Update `pyproject.toml`**: Increment the package metadata version:
   ```toml
   [project]
   version = "0.1.1" # Update this
   ```
3. **Update `CHANGELOG.md`**: Summarize key enhancements, bug fixes, performance improvements, and any API deprecations under a new release header.

---

## 3. Rust Crate Publishing

Publish the Rust library package `aperon-core` and CLI utility `aperon-cli` to [crates.io](https://crates.io):

1. Log in to crates.io via cargo (if not already logged in):
   ```bash
   cargo login <your-api-token>
   ```
2. Perform a dry-run check:
   ```bash
   cargo publish --package aperon-core --dry-run
   cargo publish --package aperon-cli --dry-run
   ```
3. Publish packages in dependency order:
   ```bash
   # 1. Publish core library first
   cargo publish --package aperon-core
   
   # 2. Publish the CLI compiler
   cargo publish --package aperon-cli
   ```

---

## 4. Python Wheel Publishing

We use GitHub Actions to build and release cross-platform Python wheels:

1. **Create and Push Git Tag**: Tag the release commit and push to GitHub:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```
2. **Trigger Workflow**: Run the **Python Release** workflow in GitHub Actions (manually triggerable via `workflow_dispatch` on the Actions tab).
3. **Review TestPyPI**: Verify upload success and installability from the TestPyPI repository:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aperon
   ```
4. **Publish to PyPI**: Once verified, trigger the production release step to publish wheels to [PyPI](https://pypi.org/project/aperon/).

---

## 5. Post-Release Verification

After packages are live on crates.io and PyPI:

1. Create a clean virtual environment and install `aperon`:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install aperon
   ```
2. Run the crash recovery demo against the installed wheel:
   ```bash
   python examples/crash_recovery_demo.py
   ```
3. In a temporary directory, install `aperon-cli` via cargo and verify:
   ```bash
   cargo install aperon-cli
   aperon --help
   ```
4. Draft the GitHub release notes on the tag, summarizing the changelog and linking to the PyPI and crates.io packages.
