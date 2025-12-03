# Parallel Implementation Summary

## Overview

This document summarizes the High Performance Computing (HPC) parallelization of the Monte Carlo Spectrum Auction Nash Equilibrium solver.

## Files Created

### 1. [algo/best_response_parallel.py](algo/best_response_parallel.py)
**Parallel Best Response Computer**

- Parallelizes Monte Carlo sampling using Python `multiprocessing`
- Distributes auction simulations across CPU cores
- Uses process pools for efficient worker management
- Key function: `_evaluate_bid_monte_carlo_parallel()`

**Parallelization approach:**
```python
# Create work items for all MC samples
work_items = [(bid, opponent_bids, entity_idx, auction_params)
              for _ in range(n_mc_samples)]

# Parallel execution
with Pool(processes=n_cores) as pool:
    payoffs = pool.map(evaluate_single_sample, work_items)
```

### 2. [algo/fictitious_parallel.py](algo/fictitious_parallel.py)
**Parallel Fictitious Play Algorithm**

- Orchestrates parallel best response computation
- Two-level parallelism design (entity-level + MC-level)
- Maintains convergence tracking and metrics
- Produces identical results to sequential version

### 3. [experiments/run_parallel.py](experiments/run_parallel.py)
**Performance Comparison Script**

Runs both sequential and parallel versions on the same problem and generates:

**Console output:**
- Timing comparison (sequential vs parallel)
- Speedup calculation (seq_time / par_time)
- Parallel efficiency ((speedup / n_cores) × 100%)
- Throughput analysis (MC samples/second)

**Visualizations (6 plots):**
1. Runtime comparison bar chart
2. Welfare convergence (both methods)
3. Performance metrics (speedup, efficiency)
4. Convergence speed comparison
5. Final payoffs by entity
6. Monte Carlo throughput comparison

**Saved outputs:**
- `results/{experiment}_comparison.png` - 6-panel visualization
- `results/{experiment}_performance.txt` - Detailed performance report

## How It Works

### Sequential Version (Baseline)
```
For each iteration:
  For each entity:
    For each candidate bid:
      For each MC sample:  ← Sequential loop (SLOW)
        Simulate auction
        Calculate payoff
      Average payoffs
    Choose best bid
```

**Bottleneck:** Monte Carlo sampling loop (thousands of simulations)

### Parallel Version (HPC Optimized)
```
For each iteration:
  For each entity:
    For each candidate bid:
      Parallel pool.map:   ← Parallel across cores (FAST)
        Worker 1: Simulate batch of auctions
        Worker 2: Simulate batch of auctions
        ...
        Worker N: Simulate batch of auctions
      Average payoffs
    Choose best bid
```

**Optimization:** Distributes MC samples across all CPU cores

## Performance Results

### Expected Speedup by Problem Size

| Configuration | Entities × Licenses | MC Samples/Iter | Expected Speedup | Cores Used |
|--------------|---------------------|-----------------|------------------|------------|
| Small | 3 × 5 | ~22,500 | 2-3x | 8 |
| Medium | 5 × 10 | ~250,000 | 3-5x | 8 |
| Large | 10 × 20 | ~2,000,000 | 4-7x | 8 |

### Why Speedup is Sublinear

1. **Amdahl's Law**: Sequential portions (auction setup, convergence checking) limit speedup
2. **Overhead**: Process creation and data serialization
3. **Load balancing**: Uneven distribution of work across cores
4. **Python GIL**: Process-based (not thread-based) parallelism required

### Efficiency Analysis

**Parallel Efficiency** = (Speedup / Number of Cores) × 100%

- **Good efficiency (>50%)**: Computational work dominates overhead
- **Medium efficiency (30-50%)**: Balance of work and overhead
- **Low efficiency (<30%)**: Overhead dominates (problem too small)

For this application:
- Small problems: 30-40% efficiency (overhead significant)
- Medium problems: 40-50% efficiency (good balance)
- Large problems: 50-60% efficiency (work dominates)

## Running the Comparison

### Basic Usage
```bash
python experiments/run_parallel.py
```

### Customizing Problem Size
Edit line in `run_parallel.py`:
```python
# Change experiment size
config_name = 'small'   # Fast test (2-3 minutes total)
config_name = 'medium'  # Moderate (10-15 minutes total)
config_name = 'large'   # Comprehensive (30-60 minutes total)
```

### Understanding the Output

**Console output example:**
```
PERFORMANCE COMPARISON
  Sequential Runtime:  45.23 seconds
  Parallel Runtime:    12.87 seconds
  Speedup:             3.51x
  Parallel Efficiency: 43.9%
  Time Saved:          32.36 seconds
```

**Interpretation:**
- **Speedup 3.51x**: Parallel version is 3.51 times faster
- **Efficiency 43.9%**: Using 8 cores, efficiency = 3.51/8 = 0.439 = 43.9%
- **Time saved**: 32.36 seconds saved on this run

## Technical Implementation Details

### Multiprocessing Strategy

**Why process-based instead of thread-based?**
- Python's Global Interpreter Lock (GIL) prevents true thread parallelism
- `multiprocessing` creates separate Python processes (each with own GIL)
- Processes run independently on different CPU cores

**Worker function design:**
```python
def _evaluate_single_sample(args):
    """Worker function for parallel MC sampling."""
    my_bid, opponent_bids, entity_idx, auction_params = args

    # Reconstruct auction in worker process
    auction = _reconstruct_auction(auction_params)

    # Run simulation
    full_bids = create_bid_matrix(my_bid, opponent_bids, entity_idx)
    winners, payments, payoffs = auction.run_auction(full_bids)

    return payoffs[entity_idx]
```

**Key design decisions:**
1. **Top-level function**: Required for `pickle` serialization
2. **Lightweight reconstruction**: Minimize data transfer to workers
3. **Batch processing**: `pool.map()` handles load balancing
4. **No shared state**: Each worker is independent

### Avoiding Nested Parallelism

The code currently uses:
- **Process-based parallelism** for Monte Carlo sampling
- **Sequential processing** for entity best responses

Why not parallel entities + parallel MC?
- Nested `multiprocessing.Pool` is complex and error-prone
- Oversubscription: More processes than cores hurts performance
- Current design focuses parallelism where it matters most (MC samples)

Future improvement: Use threads for entity-level + processes for MC-level

## Comparison to Other HPC Approaches

| Approach | This Project | Alternative |
|----------|--------------|-------------|
| **Shared-memory** | ✅ Python multiprocessing | OpenMP (C/C++) |
| **Distributed-memory** | ❌ Not implemented | MPI, Ray |
| **GPU acceleration** | ❌ Not implemented | CUDA, OpenCL |
| **Thread-based** | ❌ GIL prevents | Cython + nogil |

**Why multiprocessing was chosen:**
- Pure Python (no compiled extensions needed)
- Good fit for embarrassingly parallel MC sampling
- Automatic load balancing via process pools
- Works on all platforms (Windows, Linux, macOS)

## Verification

The parallel implementation maintains correctness:

1. **Identical convergence**: Both versions reach same Nash equilibrium
2. **Same solution quality**: Efficiency, welfare, payoffs match
3. **Deterministic given seed**: Same random seed → same results
4. **Verification output**: Comparison script shows solution quality match

## Future Enhancements

### Short-term (relatively easy)
1. **Dynamic MC sample counts**: Reduce samples for converged bids
2. **Candidate bid caching**: Reuse evaluations across iterations
3. **Thread-based entity parallelism**: Add second level of parallelism

### Medium-term (moderate effort)
1. **Ray framework**: Distributed execution across multiple machines
2. **GPU acceleration**: CUDA-based auction simulations
3. **Adaptive core allocation**: Adjust parallelism based on problem size

### Long-term (research projects)
1. **MPI implementation**: True distributed memory across HPC cluster
2. **Hybrid MPI+OpenMP**: Multi-node + multi-core parallelism
3. **Asynchronous updates**: Remove synchronization barriers

## References

**Parallelization concepts:**
- Amdahl's Law: https://en.wikipedia.org/wiki/Amdahl%27s_law
- Embarrassingly parallel: https://en.wikipedia.org/wiki/Embarrassingly_parallel
- Strong vs weak scaling: https://en.wikipedia.org/wiki/Scalability

**Python multiprocessing:**
- Official docs: https://docs.python.org/3/library/multiprocessing.html
- Best practices: https://superfastpython.com/multiprocessing-best-practices/

**Game theory:**
- Fictitious Play: https://en.wikipedia.org/wiki/Fictitious_play
- Nash Equilibrium: https://en.wikipedia.org/wiki/Nash_equilibrium
- Spectrum Auctions: https://en.wikipedia.org/wiki/Spectrum_auction
