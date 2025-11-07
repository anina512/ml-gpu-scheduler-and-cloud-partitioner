
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import math

@dataclass
class Region:
    lat: float
    lon: float
    s_lat: float     # stddev for lat
    s_lon: float     # stddev for lon

REGIONS = [
    Region(37.0,  -95.0, 10.0, 20.0),  # North America
    Region(50.0,   10.0,  8.0, 20.0),  # Europe
    Region(22.0,   78.0, 12.0, 30.0),  # India
    Region(35.0,  105.0, 10.0, 25.0),  # China
    Region(-23.0, 133.0,  8.0, 25.0),  # Australia
    Region(-15.0, -55.0, 10.0, 30.0),  # South America
]

def gen_servers(n: int, seed: int = 0) -> List[Tuple[float, float, int]]:
    """
    Returns list of (lat, lon, capacity).
    """
    random.seed(seed)
    servers = []
    for i in range(n):
        r = REGIONS[i % len(REGIONS)]
        lat = random.gauss(r.lat, r.s_lat)
        lon = random.gauss(r.lon, r.s_lon)
        capacity = random.randint(300, 2000)
        servers.append((lat, lon, capacity))
    return servers

def inject_failures(alive: List[bool], frac_down: float, seed: int = 1):
    random.seed(seed)
    n = len(alive)
    k = int(n * frac_down)
    indices = list(range(n))
    random.shuffle(indices)
    for i in indices[:k]:
        alive[i] = False

# Diurnal traffic weights per region (rough proxy): values over 24 hours.
# We'll create region-specific demand waves shifted by timezone.
def diurnal_profile(hour_local: int) -> float:
    # peak during daytime (8-20), trough at night
    # cosine-shaped curve normalized to [0.2, 1.0]
    base = 0.6 + 0.4 * math.cos((hour_local-14) * math.pi / 12)  # peak ~14:00
    return max(0.2, base)

def hour_to_local(hour_utc: int, lon: float) -> int:
    # crude mapping: 15 degrees per hour
    offset = int(lon / 15.0)
    h = (hour_utc + offset) % 24
    return h

def gen_requests_hotspots(m: int, hour_utc: int, seed: int = 0) -> List[Tuple[float, float, int]]:
    """
    Generate (lat, lon, size) requests with regional diurnal variation.
    """
    random.seed(seed + hour_utc * 1000)
    hotspots = [
        (40.7, -74.0),   # NYC
        (51.5,  -0.12),  # London
        (28.6,  77.2),   # Delhi
        (35.7, 139.7),   # Tokyo
        (-33.9, 151.2),  # Sydney
        (-23.5, -46.6),  # São Paulo
    ]
    weights = []
    for lat, lon in hotspots:
        hl = hour_to_local(hour_utc, lon)
        weights.append(diurnal_profile(hl))
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    reqs = []
    for _ in range(m):
        idx = random.choices(range(len(hotspots)), weights=probs, k=1)[0]
        cx, cy = hotspots[idx]
        lat = random.gauss(cx, 2.5)
        lon = random.gauss(cy, 2.5)
        size = 1 if random.random() < 0.9 else random.randint(2, 6)
        reqs.append((lat, lon, size))
    return reqs
