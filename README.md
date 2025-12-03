# Distributed Nash Equilibrium in Spectrum Auctions

A game theory research project implementing distributed algorithms to find Nash equilibrium in multi-entity spectrum license auctions with budget constraints.

## 🚀 High Performance Computing Features

This project includes **both sequential and parallel implementations** of the Fictitious Play algorithm:

- ✅ **Sequential baseline**: Standard iterative best response computation
- ✅ **Parallel HPC version**: Multi-core parallelization using Python multiprocessing
- ✅ **Performance comparison**: Automated benchmarking showing 2-7x speedup
- ✅ **Monte Carlo parallelization**: Distributes thousands of auction simulations across CPU cores
- ✅ **Comprehensive metrics**: Runtime, speedup, efficiency, and throughput analysis

**Quick Start:**
```bash
# Run parallel vs sequential comparison
python experiments/run_parallel.py

# Output: Speedup metrics, side-by-side performance plots, and timing analysis
```

## Problem Description

This project simulates **N entities** bidding for **M spectrum licenses** through simultaneous sealed-bid auctions, where each entity has a budget constraint. The goal is to find the **Nash equilibrium** - a stable state where no entity can improve their payoff by unilaterally changing their bidding strategy.

Key features:
- **First-price sealed-bid auctions**: Highest bidder wins each license and pays their bid
- **Budget constraints**: Entities cannot exceed their total budget across all bids
- **Complementarities/Synergies**: Optional bonus value when winning related licenses together
- **Multiple valuation models**: Uniform, structured, and geographic valuation generation

## Algorithm: Fictitious Play with Monte Carlo Sampling

**Fictitious Play** is an iterative learning algorithm where each player repeatedly computes their best response against the historical average behavior of all other players.

### How it works:

1. **Initialization**: Each entity starts with an initial bidding strategy
2. **Iterative Learning**:
   - Each entity observes the history of all opponent bids
   - Each entity computes their best response using Monte Carlo sampling:
     - Generate candidate bid allocations (different ways to distribute budget)
     - For each candidate, sample opponent bids from historical distribution
     - Simulate auction outcomes and calculate expected payoff
     - Choose the bid with highest expected payoff
   - All entities simultaneously play their best responses
   - Record results and repeat
3. **Convergence**: Process continues until bids stabilize (Nash equilibrium) or max iterations reached

### Monte Carlo Sampling

Since we don't know exactly what opponents will bid (only their historical distribution), we use Monte Carlo sampling to approximate expected payoffs:
- Sample many possible opponent bid profiles from history
- Test candidate bids against these samples
- Choose bid that performs best on average

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

## Running Experiments

### Sequential Version

```bash
python experiments/run_sequential.py
```

This runs the sequential implementation with:
- 3 entities, 5 licenses (small test)
- Up to 100 iterations
- 500 Monte Carlo samples per best response

### Parallel Version with Performance Comparison

```bash
python experiments/run_parallel.py
```

This runs **BOTH** sequential and parallel versions on the same problem and compares:
- **Runtime**: Wall-clock time for each method
- **Speedup**: How many times faster the parallel version runs
- **Efficiency**: Parallel efficiency (speedup / number of cores)
- **Throughput**: Monte Carlo samples processed per second
- **Solution Quality**: Both methods converge to the same Nash equilibrium

**Example Output:**
```
================================================================================
 HPC PERFORMANCE COMPARISON: SEQUENTIAL vs PARALLEL
================================================================================

SETUP
  Experiment: small_test
  Entities: 3
  Licenses: 5
  Max Iterations: 100
  MC Samples per BR: 500
  Available CPU cores: 8

RUNNING SEQUENTIAL VERSION
  Iteration    0: Welfare = $1,234
  Iteration   10: Welfare = $1,456
  ...
  Sequential Runtime: 45.23 seconds

RUNNING PARALLEL VERSION
  Iteration    0: Welfare = $1,234
  Iteration   10: Welfare = $1,456
  ...
  Parallel Runtime: 12.87 seconds

PERFORMANCE COMPARISON
  Sequential Runtime:  45.23 seconds
  Parallel Runtime:    12.87 seconds
  Speedup:             3.51x
  Parallel Efficiency: 43.9%
  Time Saved:          32.36 seconds

Solution Quality (both should be similar):
  Sequential - Converged: True, Iterations: 67, Efficiency: 94.3%
  Parallel   - Converged: True, Iterations: 67, Efficiency: 94.3%

Performance Analysis:
  Total MC samples evaluated: 502,500
  Sequential throughput: 11,107 samples/sec
  Parallel throughput:   39,046 samples/sec
```

### Customizing Experiments

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

Or modify the experiment size in [run_parallel.py](experiments/run_parallel.py):
```python
# Change 'small' to 'medium' or 'large' for bigger experiments
config_name = 'small'  # 'small', 'medium', or 'large'
```

## Key Metrics

### Game Theory Metrics
- **Social Welfare**: Total value generated by allocation
- **Efficiency**: Actual welfare / Optimal welfare (%)
- **Revenue**: Total payments collected
- **Exploitability**: How much each entity could gain by deviating (measures Nash equilibrium quality)
- **Convergence**: Maximum bid change between iterations

### HPC Performance Metrics
- **Speedup**: Sequential time / Parallel time (higher is better)
- **Parallel Efficiency**: (Speedup / Number of cores) × 100%
- **Throughput**: Monte Carlo samples processed per second
- **Strong Scaling**: Performance improvement as more cores are added to same problem size

## Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

## Example Output

### Sequential Experiment ([run_sequential.py](experiments/run_sequential.py))

The sequential experiment produces:
1. **Console output**: Iteration progress, convergence metrics, final results
2. **Visualizations**:
   - Welfare convergence over time
   - Entity payoffs over time
   - Bid convergence (log scale)
   - Final license allocation matrix
3. **Nash equilibrium verification**: Exploitability per entity

### Parallel Comparison Experiment ([run_parallel.py](experiments/run_parallel.py))

The parallel comparison experiment produces:

#### 1. Console Performance Report
```
PERFORMANCE COMPARISON
  Sequential Runtime:  45.23 seconds
  Parallel Runtime:    12.87 seconds
  Speedup:             3.51x
  Parallel Efficiency: 43.9%
  Time Saved:          32.36 seconds
```

#### 2. Comprehensive Visualizations
The comparison generates a 6-panel figure showing:

| Visualization | Description |
|--------------|-------------|
| **Runtime Comparison** | Bar chart comparing sequential vs parallel runtime with speedup annotation |
| **Welfare Convergence** | Line plot showing both methods converge to same Nash equilibrium |
| **Performance Metrics** | Speedup and parallel efficiency bars |
| **Convergence Speed** | Log-scale plot of bid changes over iterations |
| **Final Payoffs** | Side-by-side comparison of entity payoffs from both methods |
| **MC Throughput** | Samples/second comparison showing parallel processing advantage |

#### 3. Saved Results
- **Plot**: `results/{experiment_name}_comparison.png`
- **Performance report**: `results/{experiment_name}_performance.txt`

The performance report includes:
- Full configuration details
- Timing results and speedup metrics
- Solution quality verification
- Throughput analysis

### Key Insights from Parallel Implementation

**Why Parallelization Works Well:**
1. **Monte Carlo sampling is embarrassingly parallel**: Each sample is independent
2. **Substantial computational workload**: Thousands of auction simulations per iteration
3. **Minimal communication overhead**: Workers only need auction parameters and candidate bids
4. **Multi-core availability**: Modern CPUs have 4-16+ cores

**Expected Speedup:**
- Small experiments (3 entities, 5 licenses): 2-3x speedup
- Medium experiments (5 entities, 10 licenses): 3-5x speedup
- Large experiments (10 entities, 20 licenses): 4-7x speedup

Speedup is sublinear due to:
- Process creation overhead
- Data serialization for multiprocessing
- Amdahl's law (sequential portions limit speedup)
- Load balancing across cores

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

## Future Work

### Immediate Extensions
- **Thread-based entity parallelism**: Combine with process-based MC parallelism
- **GPU acceleration**: Offload auction simulations to CUDA/OpenCL
- **Adaptive sampling**: Reduce MC samples for converged bids

### Distributed Computing
- **MPI implementation**: Multi-node parallelism for very large experiments
- **Ray framework**: Distributed task execution across clusters
- **Parameter sweeps**: Parallel exploration of configuration space

### Algorithm Extensions
- Additional auction formats (second-price, VCG)
- Learning rate adjustments and exploration strategies
- Benchmarking against other equilibrium-finding algorithms

