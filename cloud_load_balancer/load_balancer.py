
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from cloud_load_balancer.kdtree import build, k_nearest, haversine

@dataclass
class Server:
    id: int
    lat: float
    lon: float
    capacity: int
    load: int = 0
    alive: bool = True

    @property
    def headroom(self) -> int:
        return max(0, self.capacity - self.load)

    def admit(self, k: int) -> bool:
        if not self.alive:
            return False
        if self.load + k <= self.capacity:
            self.load += k
            return True
        return False

class KDRouter:
    """
    Divide & Conquer router built on a KD-tree over (lat, lon).
    Cost = alpha * distance_km + beta * (load/capacity) * 1000
    """
    def __init__(self, servers: List[Server], alpha: float = 1.0, beta: float = 0.6, replicas: int = 5):
        self.servers = servers
        self.alpha = alpha
        self.beta = beta
        self.k = max(1, replicas)
        pts = [((s.lat, s.lon), s) for s in servers]
        self.root = build(pts)

    def rebuild(self):
        pts = [((s.lat, s.lon), s) for s in self.servers]
        self.root = build(pts)

    def route(self, lat: float, lon: float, size: int = 1) -> Optional[Server]:
        candidates = k_nearest(self.root, (lat, lon), self.k)
        best = None
        best_cost = float("inf")
        for dist_km, s in candidates:
            if not s.alive or s.headroom < size:
                continue
            load_frac = (s.load / s.capacity) if s.capacity > 0 else 1.0
            cost = self.alpha * dist_km + self.beta * load_frac * 1000.0
            if cost < best_cost:
                best_cost = cost
                best = s
        if best and best.admit(size):
            return best
        # fallback: any alive candidate with capacity
        for _, s in candidates:
            if s.alive and s.admit(size):
                return s
        return None

def brute_route(servers: List[Server], lat: float, lon: float, size: int = 1,
                alpha: float = 1.0, beta: float = 0.6) -> Optional[Server]:
    best = None
    best_cost = float("inf")
    for s in servers:
        if not s.alive or s.headroom < size:
            continue
        d = haversine(lat, lon, s.lat, s.lon)
        load_frac = (s.load / s.capacity) if s.capacity > 0 else 1.0
        cost = alpha * d + beta * load_frac * 1000.0
        if cost < best_cost:
            best_cost = cost
            best = s
    if best and best.admit(size):
        return best
    # else: nearest alive with capacity
    alive = [s for s in servers if s.alive and s.headroom >= size]
    if not alive:
        return None
    alive.sort(key=lambda s: haversine(lat, lon, s.lat, s.lon))
    alive[0].admit(size)
    return alive[0]
