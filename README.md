# trace-data-pipeline-duckdb

A high-performance, optimized workflow for processing TRACE and FISD bond data locally — built on top of [Alexander-M-Dickerson/trace-data-pipeline](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/tree/main).

---

## Overview

This project extends the original TRACE data pipeline by introducing a new processing architecture designed for speed, scalability, and efficiency. While the original pipeline lays out the core logic, this version replaces key bottlenecks with modern tools — most notably **DuckDB**, **ThreadPoolExecutor**, and **Numba** — to significantly reduce runtime when working with large datasets locally.

---

## Key Features

### 🔄 Automatic Data Download & Update
- Downloads TRACE and FISD data from scratch on first run.
- On subsequent runs, **automatically detects existing data and updates only the missing records** up to the latest available date — no manual intervention needed.

### ⚡ Stage 0: Optimized Data Ingestion

| Component | Description |
|---|---|
| **DuckDB Storage** | Raw data is read and stored in DuckDB, an in-process analytical database that supports fast multi-threaded reads — ideal for large financial datasets. |
| **ThreadPoolExecutor** | Parallel processing of data chunks from DuckDB using Python's `concurrent.futures.ThreadPoolExecutor`, reducing wall-clock time significantly. |
| **`numba_cores` Module** | A custom module that rewrites performance-critical `while` and `for` loops using [Numba](https://numba.readthedocs.io/) JIT compilation, achieving near-C-speed execution for numeric operations. |

### 📋 Stage 1: Original Pipeline Logic
Stage 1 follows the same structure as the [original pipeline](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/tree/main), ensuring compatibility and reproducibility with existing research workflows.

---

## Getting Started

### Prerequisites
```bash
pip install duckdb numba pandas tqdm
```

### Run
```bash
python main.py
```

---

## Project Structure

```
trace-data-pipeline-duckdb/
├── main.py               # Entry point
├── numba_cores.py        # Numba-optimized loop utilities
├── stage0/               # DuckDB ingestion & parallel processing
├── stage1/               # Original pipeline logic
└── README.md
```

---

## Credits

This project builds upon the excellent foundational work by [Alexander M. Dickerson](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/tree/main). All credit for the original pipeline design goes to the original authors.

## Contact

If you have any suggestions or feedback, feel free to reach out! 📧 [jun.he@nhh.no](mailto:jun.he@nhh.no)

