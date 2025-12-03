# Visual Guide: Understanding the Result Plots

This guide explains how to read and interpret each visualization produced by the experiments.

## Sequential Algorithm Results

![Sequential Results](results/small_test_results.png)

This 4-panel figure shows the Nash equilibrium solution found by the algorithm.

### Panel 1: Welfare Convergence (Top Left)

**What it shows:**
- **Blue line**: Social welfare over iterations (total value created)
- **Red dashed line**: Optimal welfare (theoretical maximum)
- **X-axis**: Iteration number
- **Y-axis**: Social welfare in dollars

**How to read it:**
- ✅ Line should increase and plateau (converging to equilibrium)
- ✅ Should approach (but may not reach) the red dashed line
- ✅ Gap between blue and red shows cost of budget constraints

**Good signs:**
- Steady upward trend early on
- Smooth plateau at the end
- Getting close to optimal (red line)

**Red flags:**
- ❌ Oscillating wildly (convergence issues)
- ❌ Decreasing trend (algorithm problem)
- ❌ Very large gap from optimal (poor solution quality)

### Panel 2: Entity Payoffs Over Time (Top Right)

**What it shows:**
- **Colored lines**: Each entity's profit over iterations
- **X-axis**: Iteration number
- **Y-axis**: Payoff in dollars

**How to read it:**
- Each entity learns to bid optimally over time
- Lines should stabilize (Nash equilibrium reached)
- Different colors = different entities

**Good signs:**
- All lines converge to stable values
- All payoffs non-negative (budget constraints met)
- Fair distribution across entities

**Red flags:**
- ❌ Lines keep changing dramatically (not converging)
- ❌ Negative payoffs (budget constraint violation - bug!)
- ❌ One entity gets everything (potential monopoly issue)

### Panel 3: Bid Convergence (Bottom Left)

**What it shows:**
- **Blue line**: Maximum bid change between iterations (log scale)
- **Red dashed line**: Convergence threshold (default: 0.01)
- **X-axis**: Iteration number (after min_iterations)
- **Y-axis**: Maximum bid change (log scale)

**How to read it:**
- Measures how much bids are changing
- Log scale: drops exponentially as algorithm converges
- Crosses red line = algorithm has converged

**Good signs:**
- Exponential decay (straight line on log plot)
- Crosses threshold quickly
- Stays below threshold once crossed

**Red flags:**
- ❌ Not decreasing (not learning)
- ❌ Oscillating around threshold (unstable)
- ❌ Never crosses threshold (needs more iterations)

### Panel 4: Final Allocation (Bottom Right)

**What it shows:**
- **Heatmap**: Which entity won which license
- **Rows**: Entities (0, 1, 2, ...)
- **Columns**: Licenses (0, 1, 2, ...)
- **White squares**: Entity won that license
- **Blue squares**: Entity lost that license
- **Red "W" markers**: Winners

**How to read it:**
- Each column should have exactly one white square (one winner per license)
- Each row shows which licenses an entity won
- Distribution shows how licenses were allocated

**Good signs:**
- Each license has exactly one winner (one W per column)
- Licenses distributed somewhat evenly (unless valuations are very skewed)
- Pattern makes sense given entity preferences

**Red flags:**
- ❌ Column with no winner (license not allocated - bug!)
- ❌ Column with multiple winners (tie not resolved - bug!)
- ❌ One entity won everything (unless they value everything highly)

---

## Parallel vs Sequential Comparison

![Comparison Results](results/small_test_comparison.png)

This 6-panel figure compares performance and validates correctness.

### Panel 1: Runtime Comparison (Top Left)

**What it shows:**
- **Red bar**: Sequential runtime in seconds
- **Blue bar**: Parallel runtime in seconds
- **Yellow box**: Speedup (seq_time / par_time)

**How to read it:**
- Shorter bar = faster
- Speedup > 1.0 = parallel is faster
- Speedup annotation shows benefit

**Good results:**
- Small problems: 2-3x speedup
- Medium problems: 3-5x speedup
- Large problems: 5-7x speedup

**Interpreting speedup:**
- 1x = no benefit from parallelization
- 2x = parallel is twice as fast
- 8x = theoretical maximum on 8 cores (rare in practice)

### Panel 2: Welfare Convergence (Top Middle)

**What it shows:**
- **Red solid line**: Sequential welfare
- **Blue dashed line**: Parallel welfare
- **Black dotted line**: Optimal welfare

**How to read it:**
- Both lines should **overlap perfectly**
- Both should converge to same final value
- Both should approach optimal (black line)

**Critical check:**
- ✅ Lines overlap = both methods find same solution
- ✅ Same final welfare = correctness verified
- ❌ Lines diverge = implementation bug!

### Panel 3: Performance Metrics (Top Right)

**What it shows:**
- **Green bar**: Speedup (parallel faster than sequential)
- **Pink bar**: Parallel efficiency percentage

**How to read it:**
- **Speedup**: Higher is better (theoretical max = number of cores)
- **Efficiency**: (Speedup / Cores) × 100%

**Interpreting efficiency:**
- 100% = perfect scaling (theoretical, rarely achieved)
- 50-60% = excellent for this application
- 40-50% = good
- 30-40% = acceptable for small problems
- <30% = problem too small, overhead dominates

### Panel 4: Convergence Speed (Bottom Left)

**What it shows:**
- **Red line**: Sequential convergence (max bid change)
- **Blue dashed line**: Parallel convergence
- **Red dotted line**: Convergence threshold

**How to read it:**
- Both lines should **overlap perfectly**
- Both should cross threshold at same iteration
- Log scale shows exponential decay

**Critical check:**
- ✅ Lines overlap = same convergence behavior
- ✅ Cross at same point = identical algorithm behavior
- ❌ Different convergence = randomness or bug

### Panel 5: Final Payoffs (Bottom Middle)

**What it shows:**
- **Red bars**: Sequential final payoffs
- **Blue bars**: Parallel final payoffs
- One pair of bars per entity

**How to read it:**
- Bars should **match exactly** (same height)
- Shows profit distribution across entities
- All bars should be non-negative

**Critical check:**
- ✅ Bars match = both found same solution
- ✅ All positive = budgets satisfied
- ❌ Different heights = different solutions (bug!)

### Panel 6: MC Throughput (Bottom Right)

**What it shows:**
- **Red bar**: Sequential samples per second
- **Blue bar**: Parallel samples per second

**How to read it:**
- Parallel bar should be much taller
- Ratio of heights = speedup
- Shows computational productivity

**Typical values:**
- Sequential: 5,000-15,000 samples/sec
- Parallel (8 cores): 20,000-60,000 samples/sec
- Depends on CPU speed and problem complexity

---

## Common Patterns and What They Mean

### Pattern: Both methods converge identically
**Meaning**: ✅ Parallel implementation is correct
**What to report**: "Parallel implementation maintains solution correctness"

### Pattern: Parallel has higher throughput
**Meaning**: ✅ Parallelization is working
**What to report**: "HPC techniques improve computational efficiency"

### Pattern: Speedup < theoretical maximum
**Meaning**: ✅ Normal due to Amdahl's Law
**What to report**: "Achieved Xx speedup with Y% efficiency"

### Pattern: Welfare reaches ~95% of optimal
**Meaning**: ✅ Algorithm works well despite constraints
**What to report**: "Algorithm achieves high efficiency (95%)"

### Pattern: Welfare reaches only ~70% of optimal
**Meaning**: ⚠️ Constraints are very tight or algorithm needs tuning
**What to check**: Budget tightness, MC samples, convergence settings

### Pattern: Sequential and parallel give different results
**Meaning**: ❌ Bug in implementation
**What to do**: Check random seeds, verify both use same algorithm

### Pattern: No speedup (1x or less)
**Meaning**: ❌ Problem too small or parallelization not working
**What to do**: Try larger problem, check if multiprocessing is enabled

---

## Using These Plots in Reports

### For Academic Papers

**Algorithm convergence plot:**
> "Figure 1 shows the convergence of Fictitious Play to Nash equilibrium.
> Social welfare converges to $1,502, achieving 94.3% of the optimal welfare
> of $1,592. The algorithm converged after 67 iterations with all entity
> payoffs stabilizing as shown in the top-right panel."

**Performance comparison plot:**
> "Figure 2 demonstrates the performance improvement from HPC parallelization.
> The parallel implementation achieved a 3.51× speedup on 8 CPU cores,
> reducing runtime from 45.23s to 12.87s. Parallel efficiency of 43.9%
> indicates effective utilization of computational resources. Critically,
> both implementations found identical solutions (welfare converged to same
> value, payoffs matched exactly), verifying correctness."

### For Presentations

**Slide 1: Algorithm works**
- Show welfare convergence plot
- Highlight that it reaches near-optimal (94%)
- Point out stable convergence

**Slide 2: Parallelization speeds it up**
- Show runtime comparison bars
- Emphasize speedup number in yellow box
- Note time saved

**Slide 3: Correctness maintained**
- Show overlapping welfare lines
- Show matching payoff bars
- State "Identical solutions, 3.5× faster"

### For README/Documentation

Use the plots at the top of the README to immediately show:
1. The algorithm works (Nash equilibrium found)
2. Parallelization works (significant speedup)
3. Both verified together (same solutions)

This gives readers instant confidence in the project without reading details.

---

## Troubleshooting Bad Plots

### Welfare Not Converging
**Symptoms**: Line keeps oscillating, never plateaus
**Fix**: Increase max_iterations, increase MC samples, check learning rate

### Different Sequential/Parallel Results
**Symptoms**: Lines don't overlap in comparison plots
**Fix**: Set same random seed, verify identical configs, check for bugs

### No Speedup
**Symptoms**: Bars almost same height, speedup near 1.0
**Fix**: Use larger problem size, close other programs, check CPU usage

### Low Efficiency (<70% optimal)
**Symptoms**: Large gap between welfare and optimal line
**Fix**: Increase budget tightness, more MC samples, more candidate bids

### Negative Payoffs
**Symptoms**: Lines go below zero in payoff plot
**Fix**: Critical bug! Check budget constraint validation in auction code
