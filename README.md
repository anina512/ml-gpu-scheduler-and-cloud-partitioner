# Greedy GPU Scheduler & Divide-and-Conquer Cloud Partitioner

This repository contains two complete algorithmic solutions developed for an Analysis of Algorithms project:

1. **A Greedy GPU Model Scheduler** — selects the maximum number of AI models that can be trained under a fixed GPU-hour budget.
2. **A Divide-and-Conquer Cloud Load Balancer** — routes client requests to distributed servers using a kd-tree spatial index with near–optimal cost.

Both implementations include reproducible experiments, benchmarks, figures, and analysis.
This work is based on the full report included in the project, covering proofs, complexity analysis, and validation.

---

## 📌 Project Structure

```
.
├── greedy_gpu_scheduler/
│   ├── greedy_gpu_budget_experiments.py
│   ├── data_generators.py
│   ├── plots/
│   └── README_gpu.md
│
├── cloud_load_balancer/
│   ├── kdtree.py
│   ├── load_balancer.py
│   ├── generators.py
│   ├── benchmark.py
│   ├── main.py
│   └── plots/
│
├── out/               # Generated figures, JSON logs, runtime plots
├── report/            # Full LaTeX report & PDF
└── README.md          # (this file)
```

---

# 1. ✅ Greedy GPU Scheduler

### Problem
Given:
- a GPU-hour budget **B**,
- a set of models with individual training times **t₁ … tₙ**,
- and equal value for each completed model,

**Goal:** Maximize the number of models completed within the budget.

### Solution
A greedy algorithm that:
1. Sorts models by training time (ascending),
2. Selects models until the budget is exhausted.

### Why Greedy Is Optimal
This is a *uniform-profit knapsack*.
Selecting the shortest jobs first ensures the largest possible count.
Formal proof and correctness theorem are provided in the report.

### Complexity
- **Time:** `O(n log n)`
- **Space:** `O(n)`

### Experiments
Experiments validate:
- Zero optimality gap compared to brute force,
- Exact match to theoretical `O(n log n)` scaling,
- 40–50% better throughput than random scheduling.

Run:

```bash
python greedy_gpu_scheduler/greedy_gpu_budget_experiments.py
```

---

# 2. ✅ Divide-and-Conquer Cloud Load Balancer

### Problem
Route client requests `(lat, lon, size)` to a set of distributed servers with:
- geographic coordinates,
- load and capacity,
- liveness status.

**Goal:** Minimize combined cost of distance and load pressure.

### Solution
A **kd-tree spatial index**:
- recursively partitions servers by median splits,
- performs nearest-neighbor search in `O(log n)` expected time,
- evaluates a small candidate set for feasibility and cost.

### Complexity
- **Build:** `O(n log n)`
- **Query:** `O(log n)`
- **Brute force:** `O(n)` per query

Run:

```bash
python -m cloud_load_balancer.main
```

---

# ✅ Installation

```bash
git clone https://github.com/anina512/ml-gpu-scheduler-and-cloud-partitioner
cd ml-gpu-scheduler-and-cloud-partitioner
pip install -r requirements.txt
```

Python ≥ 3.8 recommended.

---

# ✅ Reproducing All Experiments

```bash
python -m cloud_load_balancer.main
python greedy_gpu_scheduler/greedy_gpu_budget_experiments.py
```

Outputs appear in `out/`.

---

# ✅ Citation

```
Manne, G. C., & Pillai, A. (2025).
Greedy Optimization and Divide-and-Conquer Spatial Partitioning:
Two Real-World Algorithmic Solutions for GPU Scheduling and Cloud Load Balancing.
University of Florida.
```

---

# ✅ LLM Use Disclosure
Portions of this repository’s documentation—including text formatting and structural editing—were supported by a large language model (LLM).  
All algorithms, implementations, experiments, and analyses were created and validated by the project authors.

---

# ✅ Contributors
- **Ganesh Chowdary Manne** — University of Florida  
- **Anina Pillai** — University of Florida
