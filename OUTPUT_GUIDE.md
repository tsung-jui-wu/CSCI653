# Output Guide: Understanding Your Results

This guide explains how to interpret the output from the parallel comparison experiments.

## Console Output Breakdown

### Section 1: Setup Information

```
SETUP
  Experiment: small_test
  Entities: 3
  Licenses: 5
  Max Iterations: 100
  MC Samples per BR: 500
  Available CPU cores: 8
```

**What this means:**
- **3 entities**: Three companies bidding
- **5 licenses**: Five spectrum licenses available
- **100 iterations max**: Algorithm will run at most 100 rounds
- **500 MC samples**: Each bid evaluation uses 500 simulations
- **8 CPU cores**: Your computer has 8 cores available for parallelization

### Section 2: Algorithm Convergence

```
RUNNING SEQUENTIAL VERSION
  Iteration    0: Welfare = $1,234
  Iteration   10: Welfare = $1,456 | Max Δbid = 12.34
  Iteration   20: Welfare = $1,489 | Max Δbid = 5.67
  Iteration   30: Welfare = $1,498 | Max Δbid = 2.13
  Iteration   40: Welfare = $1,501 | Max Δbid = 0.45
  Iteration   50: Welfare = $1,502 | Max Δbid = 0.08
  Iteration   60: Welfare = $1,502 | Max Δbid = 0.008
  CONVERGED
```

**What to look for:**
- **Welfare increasing**: Good! Algorithm is finding better allocations
- **Welfare plateaus**: Algorithm approaching Nash equilibrium
- **Max Δbid decreasing**: Bids are stabilizing
  - Early: 12.34 (bids changing a lot)
  - Middle: 2.13 (starting to stabilize)
  - Late: 0.008 (converged - changes < 0.01)
- **CONVERGED**: Algorithm successfully found Nash equilibrium

**Red flags:**
- ❌ Welfare decreasing: Algorithm may have issues
- ❌ Welfare oscillating wildly: Convergence problems
- ❌ Never converges (hits max iterations): May need more iterations or algorithm tuning

### Section 3: Performance Comparison

```
PERFORMANCE COMPARISON
  Sequential Runtime:  45.23 seconds
  Parallel Runtime:    12.87 seconds
  Speedup:             3.51x
  Parallel Efficiency: 43.9%
  Time Saved:          32.36 seconds
```

**Interpreting speedup:**
- **3.51x speedup**: Parallel version is 3.51 times faster
- **Good speedup**: 2-4x on 8 cores for small/medium problems
- **Great speedup**: 5-7x on 8 cores for large problems

**Interpreting efficiency:**
- **43.9% efficiency**: Using 43.9% of theoretical maximum (8x on 8 cores)
- **30-40%**: Acceptable for small problems (overhead significant)
- **40-50%**: Good for medium problems
- **50-60%**: Excellent for large problems
- **<30%**: Problem too small to benefit from parallelization
- **>70%**: Exceptional! (rare for this type of problem)

### Section 4: Solution Quality Verification

```
Solution Quality (both should be similar):
  Sequential - Converged: True, Iterations: 67, Efficiency: 94.3%
  Parallel   - Converged: True, Iterations: 67, Efficiency: 94.3%
```

**Critical check:**
- ✅ **Both converged**: Both methods successfully found equilibrium
- ✅ **Same iterations**: Both took same number of rounds (deterministic)
- ✅ **Same efficiency**: Both found same solution quality (94.3%)

**If they differ:**
- ⚠️ Different efficiency: Possible randomness in MC sampling (minor differences OK)
- ❌ One didn't converge: Implementation issue - report as bug
- ❌ Very different iterations: Algorithm divergence - needs investigation

**What is "Efficiency"?**
- **94.3%**: Algorithm found allocation worth 94.3% of theoretical optimum
- **90-95%**: Excellent (typical for constrained auctions)
- **85-90%**: Good
- **<85%**: Algorithm may need tuning or more iterations

### Section 5: Throughput Analysis

```
Performance Analysis:
  Total MC samples evaluated: 502,500
  Sequential throughput: 11,107 samples/sec
  Parallel throughput:   39,046 samples/sec
```

**Understanding throughput:**
- **502,500 total samples**: How much computation was done
- **39,046 samples/sec (parallel)**: Computational productivity
- **Ratio**: 39,046 / 11,107 = 3.51x speedup (matches above)

**Typical throughput values:**
- **Sequential**: 5,000-15,000 samples/sec (depends on CPU)
- **Parallel (8 cores)**: 20,000-60,000 samples/sec

## Visualization Output

The comparison script generates `results/{experiment_name}_comparison.png` with 6 panels:

### Panel 1: Runtime Comparison
```
Bar chart with two bars:
  - Red bar: Sequential (45.23s)
  - Blue bar: Parallel (12.87s)
  - Yellow box: "Speedup: 3.51x"
```

**Reading it:**
- Shorter bar = faster
- Speedup annotation shows benefit
- Large difference = good parallelization

### Panel 2: Welfare Convergence
```
Line plot with two overlapping lines:
  - Red solid: Sequential welfare over iterations
  - Blue dashed: Parallel welfare over iterations
  - Black dotted: Optimal welfare (theoretical maximum)
```

**Reading it:**
- Both lines should overlap perfectly (same convergence path)
- Both should approach the black dotted line
- Gap between final welfare and optimal shows constraint penalty

### Panel 3: Performance Metrics
```
Two bars:
  - Green: Speedup (3.51x)
  - Pink: Efficiency (43.9%)
```

**Reading it:**
- Speedup > 1.0 is good (parallel faster than sequential)
- Efficiency shows how well cores are utilized

### Panel 4: Convergence Speed
```
Log-scale plot showing Max Δbid over time:
  - Red: Sequential convergence
  - Blue: Parallel convergence
  - Red dotted: Threshold (0.01)
```

**Reading it:**
- Both lines should decrease exponentially
- Crossing threshold = convergence achieved
- Lines should overlap (same convergence behavior)

### Panel 5: Final Payoffs
```
Side-by-side bars for each entity:
  - Red: Sequential payoffs
  - Blue: Parallel payoffs
```

**Reading it:**
- Bars should match (same solution)
- Shows how profit is distributed across entities
- All non-negative = budget constraints satisfied

### Panel 6: MC Throughput
```
Bar chart:
  - Red: Sequential samples/sec (11,107)
  - Blue: Parallel samples/sec (39,046)
```

**Reading it:**
- Parallel bar should be much higher
- Ratio equals speedup
- Shows computational productivity improvement

## Saved Text Report

File: `results/{experiment_name}_performance.txt`

Contains all the above information in text format for easy reference and archiving.

## Common Issues and What They Mean

### Issue: Speedup < 2x on 8 cores

**Possible causes:**
1. Problem size too small (overhead dominates)
2. CPU thermal throttling (reduce background tasks)
3. Other programs using CPU cores

**Solutions:**
- Try larger problem size (medium or large)
- Close other programs
- Check CPU temperature and cooling

### Issue: Sequential and parallel give different efficiencies

**If difference < 2%:** Normal due to Monte Carlo randomness
**If difference > 5%:** Potential issue

**Solutions:**
- Set same random seed for both
- Increase MC samples for stability
- Check for implementation bugs

### Issue: Algorithm doesn't converge

**Symptoms:**
- Hits max iterations without CONVERGED message
- Max Δbid stays high (> 0.1)
- Welfare oscillates

**Solutions:**
- Increase max_iterations in config
- Reduce convergence_threshold (more lenient)
- Increase n_mc_samples (more accurate)
- Check for bug in best response computation

### Issue: Very low welfare efficiency (< 70%)

**Possible causes:**
1. Budget constraints too tight
2. Poor candidate bid generation
3. Insufficient MC samples
4. Premature convergence

**Solutions:**
- Increase budget_tightness in config
- Increase n_candidate_bids
- Increase n_mc_samples
- Lower convergence_threshold

## Best Practices for Experiments

### For Testing (Quick validation)
```python
config_name = 'small'
# 3 entities, 5 licenses, 500 MC samples
# Takes: 2-3 minutes total
# Purpose: Verify code works, decent speedup
```

### For Demonstration (Show speedup)
```python
config_name = 'medium'
# 5 entities, 10 licenses, 1000 MC samples
# Takes: 5-10 minutes total
# Purpose: Clear speedup demonstration
```

### For Research (Realistic scenarios)
```python
config_name = 'large'
# 10 entities, 20 licenses, 2000 MC samples
# Takes: 30-60 minutes total
# Purpose: Best speedup, publication-quality results
```

## Interpreting Results for Reports

### When reporting speedup:
```
"The parallel implementation achieved a {speedup}x speedup on {cores}
CPU cores, corresponding to a parallel efficiency of {efficiency}%.
This demonstrates effective utilization of HPC techniques for Monte
Carlo-based game theory algorithms."

Example:
"The parallel implementation achieved a 3.51x speedup on 8 CPU cores,
corresponding to a parallel efficiency of 43.9%. This demonstrates
effective utilization of HPC techniques for Monte Carlo-based game
theory algorithms."
```

### When reporting solution quality:
```
"Both sequential and parallel implementations converged to the same
Nash equilibrium after {iterations} iterations, achieving {efficiency}%
of the optimal social welfare. This verifies that parallelization
maintains solution correctness while improving performance."

Example:
"Both sequential and parallel implementations converged to the same
Nash equilibrium after 67 iterations, achieving 94.3% of the optimal
social welfare. This verifies that parallelization maintains solution
correctness while improving performance."
```

### When reporting throughput:
```
"The parallel implementation processed {parallel_throughput} Monte Carlo
samples per second, a {ratio}x improvement over the sequential baseline
of {sequential_throughput} samples per second. This enabled evaluation
of {total_samples} total samples in {parallel_time} seconds."

Example:
"The parallel implementation processed 39,046 Monte Carlo samples per
second, a 3.51x improvement over the sequential baseline of 11,107
samples per second. This enabled evaluation of 502,500 total samples
in 12.87 seconds."
```
