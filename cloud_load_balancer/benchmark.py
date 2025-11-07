
from __future__ import annotations
from typing import List, Tuple
import time
import math
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cloud_load_balancer.load_balancer import Server, KDRouter, brute_route
from cloud_load_balancer.generators import gen_servers, inject_failures, gen_requests_hotspots

def to_servers(tuples: List[Tuple[float, float, int]]) -> List[Server]:
    return [Server(i, lat, lon, cap) for i, (lat, lon, cap) in enumerate(tuples)]

def reset_load(servers: List[Server]):
    for s in servers:
        s.load = 0

def exp_lookup_scaling(outdir: str):
    ns = [1000, 3000, 6000, 12000]
    kd_times = []
    brute_times = []

    for n in ns:
        base = to_servers(gen_servers(n, seed=1))
        kd = KDRouter(base, alpha=1.0, beta=0.6, replicas=5)

        queries = gen_requests_hotspots(2000, hour_utc=12, seed=2)
        t0 = time.time()
        for lat, lon, sz in queries:
            kd.route(lat, lon, sz)
        kd_times.append(time.time() - t0)

        reset_load(base)
        t0 = time.time()
        for lat, lon, sz in queries:
            brute_route(base, lat, lon, sz)
        brute_times.append(time.time() - t0)

    plt.figure()
    plt.plot(ns, kd_times, marker="o", label="KD-tree (D&C)")
    plt.plot(ns, brute_times, marker="o", label="Brute force")
    plt.xlabel("Number of servers")
    plt.ylabel("Routing time (s)")
    plt.title("Experiment A: Lookup scaling (KD vs Brute)")
    plt.legend()
    path = f"{outdir}/expA_lookup_scaling.png"
    plt.savefig(path, bbox_inches="tight")
    return path, list(zip(ns, kd_times, brute_times))

def exp_failure_tolerance(outdir: str):
    n = 4000
    base = to_servers(gen_servers(n, seed=3))
    kd = KDRouter(base, alpha=1.0, beta=0.6, replicas=7)

    fracs = [0.0, 0.02, 0.05, 0.1, 0.2]
    success_rates = []

    for f in fracs:
        # reset state
        for s in base:
            s.alive = True
            s.load = 0
        inject_failures([s.alive for s in base], f, seed=4)  # temp list
        # need to actually set alive flags deterministically
        # simplify: randomly mark fraction down again
        down = set(random.sample(range(n), int(n*f)))
        for i, s in enumerate(base):
            s.alive = (i not in down)

        kd.rebuild()

        reqs = gen_requests_hotspots(3000, hour_utc=18, seed=5)
        routed = 0
        for lat, lon, sz in reqs:
            if kd.route(lat, lon, sz):
                routed += 1
        success_rates.append(routed / len(reqs))

    plt.figure()
    plt.plot([100*x for x in fracs], success_rates, marker="o")
    plt.xlabel("Servers down (%)")
    plt.ylabel("Success rate")
    plt.title("Experiment B: Failure tolerance (replicated KD routing)")
    path = f"{outdir}/expB_failure_tolerance.png"
    plt.savefig(path, bbox_inches="tight")
    return path, list(zip(fracs, success_rates))

def exp_diurnal_latency(outdir: str):
    n = 5000
    base = to_servers(gen_servers(n, seed=6))
    kd = KDRouter(base, alpha=1.0, beta=0.6, replicas=7)

    hours = list(range(24))
    mean_latency = []

    for h in hours:
        # release loads each hour to mimic request completion
        for s in base:
            s.load = max(0, int(s.load * 0.4))

        reqs = gen_requests_hotspots(1500, hour_utc=h, seed=7)
        dist_sum = 0.0
        routed = 0
        for lat, lon, sz in reqs:
            picked = kd.route(lat, lon, sz)
            if picked:
                # km as a proxy for network latency
                dkm = math.hypot(lat - picked.lat, lon - picked.lon)  # cheap proxy for speed
                dist_sum += dkm
                routed += 1
        mean_latency.append(dist_sum / max(1, routed))

    plt.figure()
    plt.plot(hours, mean_latency, marker="o")
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Mean geo distance proxy")
    plt.title("Experiment C: Diurnal traffic latency proxy")
    path = f"{outdir}/expC_diurnal_latency.png"
    plt.savefig(path, bbox_inches="tight")
    return path, list(zip(hours, mean_latency))


def _cost(alpha, beta, dist_km, load, cap):
    load_frac = (load / cap) if cap > 0 else 1.0
    return alpha * dist_km + beta * load_frac * 1000.0

def exp_build_time_vs_n(outdir: str):
    """
    Measure KD-tree build time vs n and compare with n log n by normalizing.
    Outputs two plots:
      - build_time_vs_n.png
      - build_time_over_nlogn.png (should be ~flat if O(n log n))
    """
    import time, math, matplotlib.pyplot as plt
    ns = [800, 2000, 4000, 8000]
    build_times = []
    for n in ns:
        base = to_servers(gen_servers(n, seed=10))
        t0 = time.time()
        kd = KDRouter(base, alpha=1.0, beta=0.6, replicas=5)
        build_times.append(time.time() - t0)

    plt.figure()
    plt.plot(ns, build_times, marker="o")
    plt.xlabel("n (servers)")
    plt.ylabel("Build time (s)")
    plt.title("KD-tree build time vs n")
    p1 = f"{outdir}/build_time_vs_n.png"
    plt.savefig(p1, bbox_inches="tight")

    plt.figure()
    norm = [t / (n * math.log(max(2, n))) for t, n in zip(build_times, ns)]
    plt.plot(ns, norm, marker="o")
    plt.xlabel("n (servers)")
    plt.ylabel("Build time / (n log n)")
    plt.title("Normalized build time ~ O(n log n)")
    p2 = f"{outdir}/build_time_over_nlogn.png"
    plt.savefig(p2, bbox_inches="tight")
    return (p1, p2), list(zip(ns, build_times, norm))

def exp_query_time_normalized(outdir: str):
    import time, math, matplotlib.pyplot as plt
    ns = [800, 1500, 2500]
    per_kd = []
    per_br = []
    Q = 600
    for n in ns:
        baseA = to_servers(gen_servers(n, seed=21))
        baseB = [Server(s.id, s.lat, s.lon, s.capacity) for s in baseA]
        kd = KDRouter(baseA, alpha=1.0, beta=0.6, replicas=5)
        queries = gen_requests_hotspots(Q, hour_utc=9, seed=22)

        t0 = time.time()
        for lat, lon, sz in queries:
            kd.route(lat, lon, sz)
        kd_time = (time.time() - t0) / Q
        per_kd.append(kd_time / math.log(max(2, n)))

        for s in baseB: s.load = 0
        t0 = time.time()
        for lat, lon, sz in queries:
            brute_route(baseB, lat, lon, sz)
        br_time = (time.time() - t0) / Q
        per_br.append(br_time / n)

    plt.figure()
    plt.plot(ns, per_kd, marker="o", label="KD per-query / log n")
    plt.plot(ns, per_br, marker="o", label="Brute per-query / n")
    plt.xlabel("n (servers)")
    plt.ylabel("Normalized per-query time")
    plt.title("Per-query time normalized by theory")
    plt.legend()
    p = f"{outdir}/query_time_normalized.png"
    plt.savefig(p, bbox_inches="tight")
    return p, list(zip(ns, per_kd, per_br))

def exp_optimality_gap_hist(outdir: str):
    import matplotlib.pyplot as plt
    from cloud_load_balancer.kdtree import haversine
    n = 1500
    baseA = to_servers(gen_servers(n, seed=31))
    baseB = [Server(s.id, s.lat, s.lon, s.capacity) for s in baseA]
    kd = KDRouter(baseA, alpha=1.0, beta=0.6, replicas=7)
    gaps = []
    for hour in range(0, 24, 8):
        reqs = gen_requests_hotspots(80, hour_utc=hour, seed=32+hour)
        for lat, lon, sz in reqs:
            loadsB = [s.load for s in baseB]
            kds = kd.route(lat, lon, sz)
            for s, l in zip(baseB, loadsB): s.load = l
            brs = brute_route(baseB, lat, lon, sz)
            if kds and brs:
                kd_cost = _cost(1.0, 0.6, haversine(lat, lon, kds.lat, kds.lon), kds.load, kds.capacity)
                br_cost = _cost(1.0, 0.6, haversine(lat, lon, brs.lat, brs.lon), brs.load, brs.capacity)
                gaps.append(kd_cost - br_cost)
            elif (kds is None) and (brs is None):
                gaps.append(0.0)
            else:
                gaps.append(0.0 if kds else 1.0)

    plt.figure()
    plt.hist(gaps, bins=20)
    plt.xlabel("Cost gap (KD - Brute)")
    plt.ylabel("Frequency")
    plt.title("Optimality gap: KD vs Brute (≈0)")
    p = f"{outdir}/optimality_gap_hist.png"
    plt.savefig(p, bbox_inches="tight")
    return p, gaps

def exp_tree_depth_vs_logn(outdir: str):
    """
    Compute average leaf depth of the KD-tree and compare to log2 n.
    """
    import math, matplotlib.pyplot as plt
    from cloud_load_balancer.kdtree import KDNode
    ns = [800, 2000, 4000, 8000]
    avg_depths = []; logs = []
    for n in ns:
        base = to_servers(gen_servers(n, seed=41))
        kd = KDRouter(base)
        # traverse to compute depths of leaves
        depths = []
        stack = [(kd.root, 0)]
        while stack:
            node, d = stack.pop()
            if node is None: continue
            if (node.left is None) and (node.right is None):
                depths.append(d)
            else:
                stack.append((node.left, d+1))
                stack.append((node.right, d+1))
        avg_depths.append(sum(depths)/max(1, len(depths)))
        logs.append(math.log2(n))

    plt.figure()
    plt.plot(ns, avg_depths, marker="o", label="Avg leaf depth (KD)")
    plt.plot(ns, logs, marker="o", label="log2 n")
    plt.xlabel("n (servers)")
    plt.ylabel("Depth")
    plt.title("KD-tree average leaf depth vs log2 n")
    plt.legend()
    p = f"{outdir}/tree_depth_vs_logn.png"
    plt.savefig(p, bbox_inches="tight")
    return p, list(zip(ns, avg_depths, logs))
