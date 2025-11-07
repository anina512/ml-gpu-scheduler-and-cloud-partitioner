
from __future__ import annotations
import json
from pathlib import Path
from cloud_load_balancer.benchmark import exp_lookup_scaling, exp_failure_tolerance, exp_diurnal_latency, exp_build_time_vs_n, exp_query_time_normalized, exp_optimality_gap_hist, exp_tree_depth_vs_logn

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

def main():
    report = {}

    print("Running Experiment A (lookup scaling)...")
    pA, dataA = exp_lookup_scaling(str(OUT))
    report["expA_plot"] = str(pA)
    report["expA_data"] = [{"servers": s, "kd_time": kd, "brute_time": br}
                           for (s, kd, br) in dataA]

    print("Running Experiment B (failure tolerance)...")
    pB, dataB = exp_failure_tolerance(str(OUT))
    report["expB_plot"] = str(pB)
    report["expB_data"] = [{"frac_down": f, "success_rate": sr} for (f, sr) in dataB]

    print("Running Experiment C (diurnal latency)...")
    pC, dataC = exp_diurnal_latency(str(OUT))
    report["expC_plot"] = str(pC)
    report["expC_data"] = [{"hour": h, "mean_distance_proxy": v} for (h, v) in dataC]
    print("Running Experiment D (build time vs n)...")
    (pD1, pD2), dataD = exp_build_time_vs_n(str(OUT))
    report["expD_plots"] = [str(pD1), str(pD2)]
    report["expD_data"] = [{"servers": n, "build_time": t, "normalized": norm} for (n, t, norm) in dataD]

    print("Running Experiment E (per-query time normalized)...")
    pE, dataE = exp_query_time_normalized(str(OUT))
    report["expE_plot"] = str(pE)
    report["expE_data"] = [{"servers": n, "kd_per_over_log": kd_n, "br_per_over_n": br_n}
                           for (n, kd_n, br_n) in dataE]

    print("Running Experiment F (optimality gap histogram)...")
    pF, gaps = exp_optimality_gap_hist(str(OUT))
    report["expF_plot"] = str(pF)
    report["expF_summary"] = {"min": float(min(gaps)), "max": float(max(gaps)),
                              "mean": float(sum(gaps)/len(gaps))}

    print("Running Experiment G (tree depth vs log n)...")
    pG, dataG = exp_tree_depth_vs_logn(str(OUT))
    report["expG_plot"] = str(pG)
    report["expG_data"] = [{"servers": n, "avg_depth": d, "log2n": l} for (n, d, l) in dataG]


    # Save JSON report
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Done. Report at {report_path}")

if __name__ == "__main__":
    main()
