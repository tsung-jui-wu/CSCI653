# Implementation Guide

This document provides technical details about the code structure, implementation, and how to extend the project.

## Project Structure

```
CSCI653/
├── config/
│   └── config.py                    # Experiment configurations (small/medium/large)
├── environment/
│   ├── auction.py                   # Spectrum auction logic (bid validation, winner determination)
│   └── valuation.py                 # Valuation & budget generation (uniform/structured/geographic)
├── algo/
│   ├── fictitious.py                # Sequential Fictitious Play algorithm
│   ├── fictitious_parallel.py       # Parallel Fictitious Play (multiprocessing)
│   ├── best_response.py             # Sequential best response with Monte Carlo
│   └── best_response_parallel.py    # Parallel best response with multiprocessing
├── experiments/
│   ├── run_sequential.py            # Run sequential experiments
│   └── run_parallel.py              # Run parallel experiments with performance comparison
├── results/                         # Output directory for plots and performance data
└── utils/                           # Utility functions
```

## Code Components

### Configuration ([config/config.py](config/config.py))
- `AuctionConfig`: Auction parameters (entities, licenses, budgets, complementarities)
- `AlgorithmConfig`: Fictitious Play parameters (iterations, MC samples, convergence)
- `ParallelConfig`: MPI and performance settings
- Predefined experiments: `get_small_experiment()`, `get_medium_experiment()`, `get_large_experiment()`

### Auction Environment ([environment/auction.py](environment/auction.py))
- `SpectrumAuction`: Manages sealed-bid auction execution
  - Validates bids against budget constraints
  - Determines winners (highest bidder per license)
  - Calculates payments and payoffs (value - payment)
  - Handles complementarities through custom valuation functions

### Valuation Generation ([environment/valuation.py](environment/valuation.py))
- `ValuationGenerator`: Creates entity valuations and budgets
  - **Uniform**: Random independent values
  - **Structured**: Entities have different preferences (realistic)
  - **Geographic**: Licenses organized by region × frequency
  - Adds complementarity bonuses for synergy groups

### Best Response ([algo/best_response.py](algo/best_response.py))
- `BestResponseComputer`: Computes optimal bid for one entity
  - Generates candidate bids (uniform, value-proportional, top-k focus, random)
  - Evaluates candidates via Monte Carlo simulation
  - Samples opponent bids from historical distribution
  - Returns bid with highest expected payoff

### Fictitious Play ([algo/fictitious.py](algo/fictitious.py))
- `FictitiousPlay`: Main algorithm orchestration (sequential)
  - Runs iterative learning process
  - Tracks bid history and convergence metrics
  - Computes efficiency vs optimal welfare
  - Verifies Nash equilibrium (exploitability check)

### Parallel Implementation (HPC)

#### Parallel Best Response ([algo/best_response_parallel.py](algo/best_response_parallel.py))
- `ParallelBestResponseComputer`: Parallel version using Python multiprocessing
  - Distributes Monte Carlo samples across CPU cores
  - Uses process pools for parallel auction simulations
  - Maintains same candidate generation strategy as sequential version
  - Achieves significant speedup on multi-core systems

#### Parallel Fictitious Play ([algo/fictitious_parallel.py](algo/fictitious_parallel.py))
- `ParallelFictitiousPlay`: Parallel algorithm orchestration
  - Two-level parallelism:
    1. **Entity-level**: Multiple entities compute best responses simultaneously
    2. **MC-level**: Each entity's Monte Carlo sampling is parallelized
  - Automatically detects and utilizes available CPU cores
  - Maintains identical convergence properties as sequential version

## HPC Implementation Details

### Parallelization Strategy

**Two-Level Parallelism:**

1. **Monte Carlo Level (Primary Parallelization)**
   - Each best response computation requires evaluating multiple candidate bids
   - Each candidate requires simulating 500-2000 Monte Carlo samples
   - Each MC sample is an independent auction simulation
   - **Solution**: Distribute MC samples across process pool using `multiprocessing.Pool`
   - **Speedup**: Near-linear for MC sampling portion (embarrassingly parallel)

2. **Entity Level (Secondary Parallelization)**
   - Multiple entities can compute best responses simultaneously
   - Each entity's computation is independent given bid history
   - **Current implementation**: Sequential entity processing to avoid nested multiprocessing complexity
   - **Future work**: Thread-based entity parallelism or MPI for distributed memory

### Technology Stack

- **Python multiprocessing**: Shared-memory parallelism for multi-core CPUs
- **Process pools**: Efficient worker management and load balancing
- **NumPy**: Vectorized operations for efficient array computations
- **Comparison to OpenMP**: Similar shared-memory model, but process-based (Python GIL workaround)

### Performance Characteristics

| Problem Size | MC Samples/Iteration | Expected Speedup | Parallel Overhead |
|--------------|---------------------|------------------|-------------------|
| Small (3×5) | 22,500 | 2-3x | ~15-20% |
| Medium (5×10) | 250,000 | 3-5x | ~10-15% |
| Large (10×20) | 2,000,000 | 4-7x | ~5-10% |

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

## Customizing Experiments

### Edit Configuration

Edit [config/config.py](config/config.py) or use the custom experiment function:

```python
from config.config import get_custom_experiment

config = get_custom_experiment(
    n_entities=7,
    n_licenses=15,
    budget_tightness=0.5,
    max_iterations=800,
    valuation_type='geographic'
)
```

### Change Experiment Size

Modify the experiment size in [run_parallel.py](experiments/run_parallel.py):
```python
# Change 'small' to 'medium' or 'large' for bigger experiments
config_name = 'small'  # 'small', 'medium', or 'large'
```

## Extending the Implementation

### Adding New Valuation Models

1. Create new method in `ValuationGenerator` class
2. Add case in `generate()` method
3. Update `AuctionConfig` with new parameters

### Adding New Auction Formats

1. Extend `SpectrumAuction` class
2. Implement new winner determination rule
3. Implement new payment calculation
4. Update tests and documentation

### Adding New Best Response Strategies

1. Add method to `BestResponseComputer` class
2. Update `_generate_candidate_bids()` method
3. Test convergence properties

### GPU Acceleration (Future Work)

Potential approaches:
1. **CuPy**: Drop-in NumPy replacement for CUDA
2. **Numba CUDA**: JIT compilation of Python to GPU kernels
3. **PyTorch**: Batch auction simulations as tensor operations

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
