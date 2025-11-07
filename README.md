# Cloud Load Balancing via Divide & Conquer (KD-Tree)

This project implements a geographically-aware cloud load balancer using a 
Divide-and-Conquer KD-Tree spatial partitioning algorithm.

✅ Build KD-tree over server regions  
✅ O(log n) nearest-region routing  
✅ Compare against brute-force O(n) routing  
✅ Experiments included (runtime vs N)  

### Included Algorithms
- **D&C KD-tree** for splitting the world region into subregions.
- **Nearest-neighbor routing** for load balancing.
- **Benchmark suite** to validate asymptotic performance.

### Run Demo
```bash
cd cloud-load-balancer
python main.py
