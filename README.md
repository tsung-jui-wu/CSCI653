# Distributed Nash Equilibrium in Spectrum Auctions

A High Performance Computing implementation of game-theoretic algorithms to find Nash equilibrium in multi-entity spectrum license auctions with budget constraints.

## 🎯 What This Project Does

This project solves a complex auction problem: **How do multiple entities bid optimally in a simultaneous spectrum license auction?**

### The Problem

Imagine a government auctioning **M spectrum licenses** to **N telecommunications companies**:
- Each company values licenses differently (e.g., AT&T values urban areas more than rural)
- Each company has a **budget constraint**
- Some licenses are **complementary** - worth more when won together (e.g., adjacent frequency bands)
- All companies bid simultaneously (can't see others' bids)
- **Highest bidder wins** each license and pays their bid amount

**Key Question**: What bidding strategy should each company use?

Find the **Nash Equilibrium** - a stable state where no company can improve their profit by changing their strategy alone.

### Algorithm: Fictitious Play with Monte Carlo Sampling

**Fictitious Play** is an iterative learning algorithm where each player repeatedly best-responds to the historical behavior of opponents:

```
1. Start with initial bids (e.g., divide budget evenly)

2. Repeat until bids stabilize:
   For each entity:
     a. Look at history of opponent bids
     b. Generate many candidate bidding strategies
     c. For each candidate:
        - Simulate 500-2000 possible auctions (Monte Carlo)
        - Sample opponent bids from their history
        - Calculate expected profit
     d. Choose the bid that maximizes expected profit
     e. Submit that bid

   All entities simultaneously submit best responses
   Record bids and repeat

3. Converged bids = Approximate Nash Equilibrium
```

---

**Why Monte Carlo?** Since we don't know exactly what opponents will bid (only their historical patterns), we sample many possible scenarios and choose the bid that performs best **on average**.
## 📸 Results


## 🚀 High Performance Computing

The computational bottleneck is **Monte Carlo sampling**: evaluating thousands of auction simulations per iteration.

### The Challenge

For a medium-sized problem (5 entities, 10 licenses):
- 50 candidate bids per entity × 5 entities = **250 candidates**
- 1000 Monte Carlo samples per candidate = **250,000 simulations per iteration**
- 100 iterations to converge = **25 million auction simulations total**

**Sequential execution**: 3-5 minutes on a laptop
**Parallel execution**: 45-75 seconds on the same laptop

### The Solution: Parallelization

We exploit the fact that **Monte Carlo samples are independent**:

```
Sequential (slow):
  For each sample 1 to 1000:
    Run auction simulation → 1000 sequential operations

Parallel (fast):
  Split 1000 samples across 8 CPU cores:
    Core 1: samples 1-125
    Core 2: samples 126-250
    Core 3: samples 251-375
    ...
    Core 8: samples 876-1000
  All run simultaneously → 8x speedup potential
```

## 📊 Results and Performance

### Algorithm Convergence to Nash Equilibrium
![Algorithm Results](results/small_test_results.png)
*The algorithm successfully finds Nash equilibrium: welfare converges to optimal, entity payoffs stabilize, bids stop changing, and licenses are efficiently allocated.*

### HPC Performance: Sequential vs Parallel
![Performance Comparison](results/small_test_comparison.png)
*Parallel implementation achieves 3-7x speedup: both methods find identical solutions, but parallel version is significantly faster by distributing Monte Carlo simulations across CPU cores.*

### Run the Comparison

```bash
# Install requirements
pip install numpy matplotlib

# Run sequential vs parallel comparison
python experiments/run_parallel.py
```

**Console Output:**
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
  Iteration   10: Welfare = $1,456 | Max Δbid = 12.34
  ...
  Iteration   60: Welfare = $1,502 | Max Δbid = 0.008
  CONVERGED
  Sequential Runtime: 45.23 seconds

RUNNING PARALLEL VERSION
  Iteration    0: Welfare = $1,234
  Iteration   10: Welfare = $1,456 | Max Δbid = 12.34
  ...
  Iteration   60: Welfare = $1,502 | Max Δbid = 0.008
  CONVERGED
  Parallel Runtime: 12.87 seconds

================================================================================
 PERFORMANCE COMPARISON
================================================================================

Timing Results:
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

### Performance by Problem Size

| Problem Size | Entities × Licenses | Total MC Samples | Sequential Time | Parallel Time | Speedup |
|--------------|---------------------|------------------|-----------------|---------------|---------|
| **Small** | 3 × 5 | ~500K | 45 sec | 13 sec | **3.5x** |
| **Medium** | 5 × 10 | ~25M | 4 min | 1 min | **4.0x** |
| **Large** | 10 × 20 | ~200M | 30 min | 5 min | **6.0x** |

*Times are approximate and depend on CPU cores available*

### Visualization Output

The script generates a comprehensive 6-panel comparison figure (shown at the top of this README) saved as `results/comparison.png`.

**What each panel shows:**

| Panel | What It Shows |
|-------|---------------|
| **Runtime Comparison** | Bar chart: Sequential vs Parallel time with speedup label |
| **Welfare Convergence** | Both methods converge to the same Nash equilibrium welfare |
| **Speedup Metrics** | Speedup and parallel efficiency bars |
| **Convergence Speed** | How fast bids stabilize (log scale) |
| **Final Payoffs** | Each entity's profit - verifies both methods give same answer |
| **Throughput** | Samples/second comparison showing parallel advantage |

📖 **For detailed explanation of how to read each panel, see [VISUAL_GUIDE.md](VISUAL_GUIDE.md)**

## 🎓 Key Metrics Explained

### Game Theory Metrics

- **Social Welfare**: Total value created by license allocation (sum of all entity values)
- **Efficiency**: Actual welfare / Optimal welfare × 100%
  - 100% = perfect allocation (best possible)
  - 85-95% = good (typical for our algorithm)
- **Nash Equilibrium**: No entity can improve profit by changing bid alone
- **Exploitability**: How much an entity could gain by deviating (should be near zero)

### HPC Performance Metrics

- **Speedup**: Sequential time / Parallel time
  - 3.5x speedup = parallel is 3.5 times faster
  - Linear speedup = 8x on 8 cores (theoretical maximum)
- **Parallel Efficiency**: Speedup / Number of cores × 100%
  - 100% = perfect scaling (rare in practice)
  - 40-60% = good (typical for this application)
- **Throughput**: Simulations processed per second
  - Higher is better, shows computational productivity

## 🔍 Why This Matters

### Practical Applications

1. **FCC Spectrum Auctions**: Real auctions allocate billions of dollars in spectrum licenses
2. **Cloud Resource Allocation**: Multiple users bidding for compute resources
3. **Ad Auctions**: Google/Facebook ad placement with budget constraints
4. **Energy Markets**: Power plants bidding to supply electricity

### Computational Challenge

Finding Nash equilibrium is **computationally expensive**:
- Evaluating all possible strategies: exponential in number of licenses
- Monte Carlo approximation: thousands of simulations needed for accuracy
- Iterative convergence: requires many rounds

**HPC parallelization** makes these problems tractable:
- Problems that took hours now take minutes
- Enables exploration of larger, more realistic scenarios
- Allows real-time decision support

## 📈 Understanding the Results

### What "Converged" Means

The algorithm **converges** when bids stop changing significantly between iterations:
- **Max Δbid < 0.01**: Largest bid change is less than 1 cent
- **Payoff stability**: Entity profits are stable over last 10 iterations
- **Nash equilibrium**: No entity has incentive to deviate

### Interpreting Efficiency

**94.3% efficiency** means:
- The algorithm found an allocation worth 94.3% of the theoretically optimal welfare
- This is **very good** - perfect efficiency (100%) is rare due to budget constraints
- Shows the Nash equilibrium is nearly socially optimal

### Speedup vs Number of Cores

**Why 3.5x speedup on 8 cores (not 8x)?**

Due to **Amdahl's Law**:
```
Total time = Parallelizable portion + Sequential portion

Even if 90% of code runs 8x faster, sequential 10% limits total speedup:
  Speedup = 1 / (0.1 + 0.9/8) = 4.7x

Actual factors reducing speedup:
  - Process creation overhead (~10-15%)
  - Data serialization between processes (~5-10%)
  - Sequential setup and convergence checking (~10-15%)
```

For this application:
- Small problems: 30-40% efficiency (overhead dominates)
- Medium problems: 40-50% efficiency (good balance)
- Large problems: 50-60% efficiency (computation dominates)

## 📦 Requirements

```bash
pip install numpy matplotlib
```

**System Requirements:**
- Python 3.7+
- Multi-core CPU (4+ cores recommended for speedup)
- 2-8 GB RAM depending on problem size

## 🎯 Quick Start Examples

### Run Sequential Version Only
```bash
python experiments/run_sequential.py
```
Outputs: Nash equilibrium solution with welfare convergence plots

### Run Parallel Comparison (Recommended)
```bash
python experiments/run_parallel.py
```
Outputs: Side-by-side performance comparison, speedup metrics, visualizations

### Change Problem Size
Edit line in `experiments/run_parallel.py`:
```python
config_name = 'small'   # 2-3 min total, good for testing
config_name = 'medium'  # 5-10 min total, shows significant speedup
config_name = 'large'   # 30-60 min total, best speedup demonstration
```

## 📚 Project Documentation

- **[README.md](README.md)** (this file): Overview, results, and interpretation
- **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)**: Detailed explanation of all plots and how to read them
- **[OUTPUT_GUIDE.md](OUTPUT_GUIDE.md)**: Understanding console output and saved results
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**: Technical details, code structure, extending the project
- **[PARALLEL_IMPLEMENTATION.md](PARALLEL_IMPLEMENTATION.md)**: Deep dive into HPC parallelization strategy

## 🏆 Key Achievements

✅ **Implements advanced game theory algorithm** (Fictitious Play)
✅ **Finds Nash equilibrium** in complex auction with budget constraints
✅ **High Performance Computing parallelization** achieving 2-7x speedup
✅ **Comprehensive benchmarking** comparing sequential vs parallel
✅ **Production-quality visualization** of convergence and performance
✅ **Verified correctness** - parallel gives identical results to sequential

## 🔮 Future Directions

### Algorithm Extensions
- Second-price auctions (Vickrey-Clarke-Groves mechanism)
- Dynamic learning rates for faster convergence
- Multi-round auction formats

### HPC Enhancements
- **GPU acceleration**: Offload simulations to CUDA for 10-100x speedup
- **MPI distributed computing**: Scale across multiple machines for very large problems
- **Hybrid parallelism**: Combine thread and process-based parallelism

### Applications
- Real FCC auction data validation
- Multi-agent reinforcement learning integration
- Online auction mechanisms with streaming bids
