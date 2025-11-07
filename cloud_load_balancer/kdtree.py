
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

EARTH_R = 6371.0  # km

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R * c

@dataclass
class KDNode:
    point: Tuple[float, float]     # (lat, lon)
    payload: object                # arbitrary attached object (e.g., Server)
    axis: int                      # 0 for lat, 1 for lon
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None

def _coord(point: Tuple[float, float], axis: int) -> float:
    return point[axis]

def build(points: List[Tuple[Tuple[float, float], object]], depth: int = 0) -> Optional[KDNode]:
    if not points:
        return None
    axis = depth % 2
    points.sort(key=lambda it: _coord(it[0], axis))
    mid = len(points) // 2
    node = KDNode(points[mid][0], points[mid][1], axis)
    node.left  = build(points[:mid], depth+1)
    node.right = build(points[mid+1:], depth+1)
    return node

def k_nearest(root: Optional[KDNode], target: Tuple[float, float], k: int) -> List[Tuple[float, object]]:
    # max-heap simulated via sorted list (descending by distance)
    best: List[Tuple[float, object]] = []

    def visit(node: Optional[KDNode]):
        nonlocal best
        if node is None:
            return
        d = haversine(target[0], target[1], node.point[0], node.point[1])
        best.append((d, node.payload))
        best.sort(key=lambda x: -x[0])
        if len(best) > k:
            best.pop(0)

        axis = node.axis
        t_coord = target[axis]
        n_coord = node.point[axis]
        first, second = (node.left, node.right) if t_coord < n_coord else (node.right, node.left)
        visit(first)

        # bound to decide if we should cross back over the splitting plane
        deg_gap = abs(t_coord - n_coord)
        if axis == 0:
            plane_dist = 111.0 * deg_gap
        else:
            plane_dist = 111.0 * math.cos(math.radians(target[0])) * deg_gap

        worst_best = best[0][0] if best else float("inf")
        if len(best) < k or plane_dist <= worst_best:
            visit(second)

    visit(root)
    return sorted(best, key=lambda x: x[0])[:k]
