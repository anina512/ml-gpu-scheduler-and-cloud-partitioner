
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy GPU-Budget Selection: Full Experiments (Polished Figures)
----------------------------------------------------------------
Implements:
  - Greedy algorithm: sort by duration ascending, take while feasible.
  - Brute-force verification on small instances (optimality gap).
  - Runtime scaling vs n with n log n normalization.
  - Quality vs random-order baseline.
  - Figure generation for LaTeX inclusion (saves PNG and PDF).

Usage:
  python greedy_gpu_budget_experiments.py
  python greedy_gpu_budget_experiments.py --help
"""

import argparse
import math
import random
import time
from itertools import combinations
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Global style for camera-ready figs
plt.rcParams.update({
    'figure.dpi': 200,     # higher DPI for clearer PNGs
    'font.size': 11,       # consistent font size for ACM/IEEE
    'axes.grid': True,     # light grid aids readability
    'grid.linestyle': ':',
})

# Core Algorithms 

def greedy_max_count(durations: List[int], budget: int) -> Tuple[int, List[int]]:
    """
    Greedy algorithm: sort jobs by ascending duration and take while feasible.

    Parameters
    ----------
    durations : list of positive ints
        GPU-hours for each model.
    budget : int
        Total GPU-hour budget.

    Returns
    -------
    (count, chosen_indices)
        Maximum number of models selected by greedy and their indices.
    """
    order = sorted(range(len(durations)), key=lambda i: durations[i])
    total = 0
    chosen = []
    for i in order:
        if total + durations[i] <= budget:
            chosen.append(i)
            total += durations[i]
        else:
            break
    return len(chosen), chosen


def brute_force_max_count(durations: List[int], budget: int) -> Tuple[int, List[int]]:
    """
    Brute force search: returns true optimum on small instances.

    Tries subsets in decreasing size; stops once a feasible subset is found.
    Intended only for small n (e.g., n<=20) due to 2^n scaling.
    """
    n = len(durations)
    # Try larger cardinalities first to short-circuit early on success
    for k in range(n, -1, -1):
        for comb in combinations(range(n), k):
            s = sum(durations[i] for i in comb)
            if s <= budget:
                return k, list(comb)
    return 0, []


# Experiments

def validate_optimality(trials=200, n=18, dur_low=1, dur_high=100, budget_factor=0.37, seed=7):
    """
    Runs many random small instances and compares greedy vs brute-force optimum.
    Returns the list of optimality gaps (opt - greedy) for counts.
    """
    random.seed(seed)
    gaps = []
    for _ in range(trials):
        durations = [random.randint(dur_low, dur_high) for _ in range(n)]
        budget = int(budget_factor * sum(durations))
        g_cnt, _ = greedy_max_count(durations, budget)
        b_cnt, _ = brute_force_max_count(durations, budget)
        gaps.append(b_cnt - g_cnt)
    return gaps


def measure_runtime(ns, trials=5, dur_high=10_000, seed=11):
    """
    Measures wall-clock time for greedy over increasing n (min of several trials).
    Returns arrays: ns, times, and n*log n normalizer.
    """
    random.seed(seed)
    times = []
    norm = []
    for n in ns:
        durations = [random.randint(1, dur_high) for _ in range(n)]
        budget = sum(durations) // 2  # budget ~ half total time
        # Warmup
        greedy_max_count(durations, budget)
        tmin = float('inf')
        for _ in range(trials):
            t0 = time.perf_counter()
            greedy_max_count(durations, budget)
            t1 = time.perf_counter()
            tmin = min(tmin, t1 - t0)
        times.append(tmin)
        norm.append(n * math.log(max(n, 2)))
    return np.array(times), np.array(norm)


def random_baseline_quality(trials=200, n=200, dur_low=1, dur_high=1000, budget_factor=0.30, seed=23):
    """
    Compares greedy solution quality vs a random-order baseline.
    Returns two arrays: greedy_counts, random_counts
    """
    random.seed(seed)
    greedy_counts = []
    random_counts = []
    for _ in range(trials):
        durations = [random.randint(dur_low, dur_high) for _ in range(n)]
        budget = int(budget_factor * sum(durations))
        g_cnt, _ = greedy_max_count(durations, budget)
        # Random order baseline
        idx = list(range(n))
        random.shuffle(idx)
        total = 0
        cnt = 0
        for i in idx:
            if total + durations[i] <= budget:
                total += durations[i]
                cnt += 1
        greedy_counts.append(g_cnt)
        random_counts.append(cnt)
    return np.array(greedy_counts), np.array(random_counts)


# Save utilities 

def save_png_and_pdf(out_path_png: str):
    """
    Saves current Matplotlib figure to both PNG and PDF.
    The PDF path is out_path_png with the '.png' suffix replaced by '.pdf'.
    """
    plt.savefig(out_path_png, bbox_inches='tight')
    if out_path_png.lower().endswith('.png'):
        out_path_pdf = out_path_png[:-4] + '.pdf'
    else:
        out_path_pdf = out_path_png + '.pdf'
    plt.savefig(out_path_pdf, bbox_inches='tight')


# Plots 

def plot_optimality_gaps(gaps, out="greedy_opt_gap.png"):
    plt.figure()
    import numpy as np
    plt.hist(gaps, bins=np.arange(min(gaps)-0.5, max(gaps)+1.5, 1), edgecolor='black')
    plt.title("Greedy vs Optimal: Count Gap (Optimal - Greedy)")
    plt.xlabel("Gap")
    plt.ylabel("Frequency")
    save_png_and_pdf(out)
    plt.close()


def plot_runtime(ns, times, out="greedy_runtime.png"):
    plt.figure()
    plt.plot(ns, times, marker='o')
    plt.title("Greedy Selection Runtime vs n")
    plt.xlabel("n (number of models)")
    plt.ylabel("time (seconds)")
    save_png_and_pdf(out)
    plt.close()


def plot_nlogn_normalized(ns, times, norm, out="greedy_nlogn_normalized.png"):
    plt.figure()
    plt.plot(ns, times / norm, marker='o')
    plt.title("Runtime / (n log n) vs n")
    plt.xlabel("n")
    plt.ylabel("seconds / (n log n)")
    # Keep y-axis labels in plain (non-scientific) notation for readability
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='y')
    save_png_and_pdf(out)
    plt.close()


def plot_quality(greedy_counts, random_counts, out="greedy_quality.png"):
    plt.figure()
    plt.plot(sorted(random_counts), label="random baseline")
    plt.plot(sorted(greedy_counts), label="greedy")
    plt.title("Solution Quality across trials (higher is better)")
    plt.xlabel("trial (sorted by count)")
    plt.ylabel("# models trained within budget")
    plt.legend()
    save_png_and_pdf(out)
    plt.close()


# Main 

def main():
    parser = argparse.ArgumentParser(description="Greedy GPU-Budget Selection Experiments (Polished)")
    parser.add_argument("--opt_trials", type=int, default=200, help="Trials for optimality check")
    parser.add_argument("--n_small", type=int, default=18, help="n for brute-force validation")
    parser.add_argument("--budget_factor_small", type=float, default=0.37, help="Budget factor for small instances")
    parser.add_argument("--runtime_ns", type=str, default="1000,2000,5000,10000,20000,40000,80000",
                        help="Comma-separated n values for runtime scaling")
    parser.add_argument("--quality_trials", type=int, default=200, help="Trials for quality comparison")
    parser.add_argument("--quality_n", type=int, default=200, help="n for quality comparison")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to save figures")
    args = parser.parse_args()

    # 1) Optimality validation (small n) 
    gaps = validate_optimality(
        trials=args.opt_trials,
        n=args.n_small,
        budget_factor=args.budget_factor_small
    )
    plot_optimality_gaps(gaps, out=f"{args.out_dir}/greedy_opt_gap.png")
    zero_gap = all(g == 0 for g in gaps)

    # 2) Runtime scaling 
    ns = [int(x) for x in args.runtime_ns.split(",") if x.strip()]
    times, norm = measure_runtime(ns)
    plot_runtime(ns, times, out=f"{args.out_dir}/greedy_runtime.png")
    plot_nlogn_normalized(ns, times, norm, out=f"{args.out_dir}/greedy_nlogn_normalized.png")

    # 3) Quality vs random baseline 
    gq, rq = random_baseline_quality(
        trials=args.quality_trials,
        n=args.quality_n
    )
    plot_quality(gq, rq, out=f"{args.out_dir}/greedy_quality.png")

    # Console summary
    print("==== Greedy GPU-Budget Selection: Summary ====")
    print(f"Optimality trials           : {args.opt_trials} (n={args.n_small})")
    print(f"Optimality gap all zero?    : {zero_gap}")
    print(f"Runtime sizes tested        : {ns}")
    print(f"Quality trials (n={args.quality_n}): {args.quality_trials}")
    print(f"Figures written to          : {args.out_dir}")
    print("Files:")
    print(f" - {args.out_dir}/greedy_opt_gap.png and .pdf")
    print(f" - {args.out_dir}/greedy_runtime.png and .pdf")
    print(f" - {args.out_dir}/greedy_nlogn_normalized.png and .pdf")
    print(f" - {args.out_dir}/greedy_quality.png and .pdf")


if __name__ == "__main__":
    main()
