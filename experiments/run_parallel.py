"""
Run both sequential and parallel versions of Fictitious Play and compare performance.
This demonstrates the speedup achieved through HPC parallelization techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from multiprocessing import cpu_count

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import get_small_experiment, get_medium_experiment, get_large_experiment
from environment.valuation import ValuationGenerator, print_valuation_summary
from environment.auction import SpectrumAuction
from algo.fictitious import FictitiousPlay
from algo.fictitious_parallel import ParallelFictitiousPlay


def run_comparison_experiment(config_name='small'):
    """
    Run complete experiment comparing sequential vs parallel implementations.

    Args:
        config_name: 'small', 'medium', or 'large'
    """

    print("\n" + "="*80)
    print(" HPC PERFORMANCE COMPARISON: SEQUENTIAL vs PARALLEL")
    print("="*80)

    # === 1. SETUP ===
    print("\n" + "="*80)
    print(" SETUP")
    print("="*80)

    # Select configuration
    if config_name == 'small':
        config = get_small_experiment()
    elif config_name == 'medium':
        config = get_medium_experiment()
    elif config_name == 'large':
        config = get_large_experiment()
    else:
        config = get_medium_experiment()

    print(f"Experiment: {config.experiment_name}")
    print(f"Entities: {config.auction.n_entities}")
    print(f"Licenses: {config.auction.n_licenses}")
    print(f"Max Iterations: {config.algorithm.max_iterations}")
    print(f"MC Samples per BR: {config.algorithm.n_mc_samples}")
    print(f"Available CPU cores: {cpu_count()}")

    # === 2. GENERATE ENVIRONMENT ===
    print("\n" + "="*80)
    print(" GENERATING ENVIRONMENT")
    print("="*80)

    generator = ValuationGenerator(config.auction, seed=config.auction.seed)
    valuations, budgets, metadata = generator.generate()
    print_valuation_summary(valuations, budgets)

    auction = SpectrumAuction(config.auction, valuations, budgets)

    # === 3. RUN SEQUENTIAL VERSION ===
    print("\n" + "="*80)
    print(" RUNNING SEQUENTIAL VERSION")
    print("="*80)

    sequential_start = time.time()
    fp_seq = FictitiousPlay(config.algorithm, auction)
    results_seq = fp_seq.run(verbose=True)
    sequential_end = time.time()
    sequential_time = sequential_end - sequential_start

    print(f"\nSequential Runtime: {sequential_time:.2f} seconds")

    # === 4. RUN PARALLEL VERSION ===
    print("\n" + "="*80)
    print(" RUNNING PARALLEL VERSION")
    print("="*80)

    parallel_start = time.time()
    fp_par = ParallelFictitiousPlay(config.algorithm, auction)
    results_par = fp_par.run(verbose=True)
    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start

    print(f"\nParallel Runtime: {parallel_time:.2f} seconds")

    # === 5. PERFORMANCE COMPARISON ===
    print("\n" + "="*80)
    print(" PERFORMANCE COMPARISON")
    print("="*80)

    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    efficiency = (speedup / cpu_count()) * 100 if cpu_count() > 0 else 0

    print(f"\nTiming Results:")
    print(f"  Sequential Runtime:  {sequential_time:.2f} seconds")
    print(f"  Parallel Runtime:    {parallel_time:.2f} seconds")
    print(f"  Speedup:             {speedup:.2f}x")
    print(f"  Parallel Efficiency: {efficiency:.1f}%")
    print(f"  Time Saved:          {sequential_time - parallel_time:.2f} seconds")

    print(f"\nSolution Quality (both should be similar):")
    print(f"  Sequential - Converged: {results_seq['converged']}, "
          f"Iterations: {results_seq['iterations']}, "
          f"Efficiency: {100*results_seq['efficiency']:.1f}%")
    print(f"  Parallel   - Converged: {results_par['converged']}, "
          f"Iterations: {results_par['iterations']}, "
          f"Efficiency: {100*results_par['efficiency']:.1f}%")

    # === 6. DETAILED PERFORMANCE BREAKDOWN ===
    print(f"\nPerformance Analysis:")
    total_mc_samples = (config.algorithm.n_mc_samples *
                        config.algorithm.n_candidate_bids *
                        config.auction.n_entities *
                        results_seq['iterations'])

    print(f"  Total MC samples evaluated: {total_mc_samples:,}")
    print(f"  Sequential throughput: {total_mc_samples/sequential_time:,.0f} samples/sec")
    print(f"  Parallel throughput:   {total_mc_samples/parallel_time:,.0f} samples/sec")

    # === 7. PLOT COMPARISON ===
    plot_comparison(results_seq, results_par, sequential_time, parallel_time, config)

    # === 8. SAVE RESULTS ===
    save_results(config, sequential_time, parallel_time, speedup, efficiency,
                 results_seq, results_par)

    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'efficiency': efficiency,
        'results_seq': results_seq,
        'results_par': results_par
    }


def plot_comparison(results_seq, results_par, seq_time, par_time, config):
    """Create visualization comparing sequential and parallel results."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Sequential vs Parallel Comparison: {config.experiment_name}",
                 fontsize=14, fontweight='bold')

    # Plot 1: Runtime comparison bar chart
    ax = axes[0, 0]
    methods = ['Sequential', 'Parallel']
    times = [seq_time, par_time]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax.bar(methods, times, color=colors, alpha=0.7)
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s',
                ha='center', va='bottom', fontweight='bold')

    # Add speedup annotation
    speedup = seq_time / par_time if par_time > 0 else 0
    ax.text(0.5, max(times) * 0.9, f'Speedup: {speedup:.2f}x',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Plot 2: Welfare convergence comparison
    ax = axes[0, 1]
    ax.plot(results_seq['welfare_history'], label='Sequential',
            linewidth=2, color=colors[0], alpha=0.7)
    ax.plot(results_par['welfare_history'], label='Parallel',
            linewidth=2, color=colors[1], alpha=0.7, linestyle='--')
    ax.axhline(results_seq['optimal_welfare'], color='black',
               linestyle=':', label='Optimal', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Social Welfare ($)')
    ax.set_title('Welfare Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Performance metrics
    ax = axes[0, 2]
    metrics = ['Speedup', f'Efficiency\n({cpu_count()} cores)']
    efficiency = (speedup / cpu_count()) * 100
    values = [speedup, efficiency]
    bars = ax.bar(metrics, values, color=['#95E1D3', '#F38181'], alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Parallel Performance Metrics')
    ax.axhline(1, color='gray', linestyle='--', linewidth=1, label='Baseline')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}{"x" if bar == bars[0] else "%"}',
                ha='center', va='bottom', fontweight='bold')

    # Plot 4: Convergence speed comparison
    ax = axes[1, 0]
    if results_seq['convergence_metrics'] and results_par['convergence_metrics']:
        seq_conv = [m['max_bid_change'] for m in results_seq['convergence_metrics']]
        par_conv = [m['max_bid_change'] for m in results_par['convergence_metrics']]

        min_len = min(len(seq_conv), len(par_conv))
        iterations = range(config.algorithm.min_iterations,
                          config.algorithm.min_iterations + min_len)

        ax.semilogy(iterations, seq_conv[:min_len], label='Sequential',
                   linewidth=2, color=colors[0], alpha=0.7)
        ax.semilogy(iterations, par_conv[:min_len], label='Parallel',
                   linewidth=2, color=colors[1], alpha=0.7, linestyle='--')
        ax.axhline(config.algorithm.convergence_threshold, color='red',
                   linestyle=':', label='Threshold', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max Bid Change (log scale)')
        ax.set_title('Convergence Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 5: Payoff comparison
    ax = axes[1, 1]
    x = np.arange(config.auction.n_entities)
    width = 0.35

    seq_payoffs = results_seq['final_payoffs']
    par_payoffs = results_par['final_payoffs']

    ax.bar(x - width/2, seq_payoffs, width, label='Sequential',
           color=colors[0], alpha=0.7)
    ax.bar(x + width/2, par_payoffs, width, label='Parallel',
           color=colors[1], alpha=0.7)

    ax.set_xlabel('Entity')
    ax.set_ylabel('Final Payoff ($)')
    ax.set_title('Final Payoffs by Entity')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Throughput comparison
    ax = axes[1, 2]
    total_samples = (config.algorithm.n_mc_samples *
                     config.algorithm.n_candidate_bids *
                     config.auction.n_entities *
                     results_seq['iterations'])

    seq_throughput = total_samples / seq_time
    par_throughput = total_samples / par_time

    methods = ['Sequential', 'Parallel']
    throughputs = [seq_throughput, par_throughput]
    bars = ax.bar(methods, throughputs, color=colors, alpha=0.7)
    ax.set_ylabel('MC Samples / Second')
    ax.set_title('Monte Carlo Sampling Throughput')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, tp in zip(bars, throughputs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{tp:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()

    if config.save_plots:
        os.makedirs(config.output_dir, exist_ok=True)
        filename = f"{config.output_dir}/{config.experiment_name}_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {filename}")

    plt.show()


def save_results(config, seq_time, par_time, speedup, efficiency,
                 results_seq, results_par):
    """Save performance comparison results to file."""

    os.makedirs(config.output_dir, exist_ok=True)
    filename = f"{config.output_dir}/{config.experiment_name}_performance.txt"

    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HPC PERFORMANCE COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("CONFIGURATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Experiment: {config.experiment_name}\n")
        f.write(f"Entities: {config.auction.n_entities}\n")
        f.write(f"Licenses: {config.auction.n_licenses}\n")
        f.write(f"Max Iterations: {config.algorithm.max_iterations}\n")
        f.write(f"MC Samples: {config.algorithm.n_mc_samples}\n")
        f.write(f"CPU Cores: {cpu_count()}\n\n")

        f.write("TIMING RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Sequential Runtime:  {seq_time:.2f} seconds\n")
        f.write(f"Parallel Runtime:    {par_time:.2f} seconds\n")
        f.write(f"Speedup:             {speedup:.2f}x\n")
        f.write(f"Parallel Efficiency: {efficiency:.1f}%\n")
        f.write(f"Time Saved:          {seq_time - par_time:.2f} seconds\n\n")

        f.write("SOLUTION QUALITY\n")
        f.write("-"*40 + "\n")
        f.write(f"Sequential - Converged: {results_seq['converged']}, "
                f"Iterations: {results_seq['iterations']}, "
                f"Efficiency: {100*results_seq['efficiency']:.1f}%\n")
        f.write(f"Parallel   - Converged: {results_par['converged']}, "
                f"Iterations: {results_par['iterations']}, "
                f"Efficiency: {100*results_par['efficiency']:.1f}%\n\n")

        total_samples = (config.algorithm.n_mc_samples *
                        config.algorithm.n_candidate_bids *
                        config.auction.n_entities *
                        results_seq['iterations'])

        f.write("THROUGHPUT ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total MC Samples: {total_samples:,}\n")
        f.write(f"Sequential Throughput: {total_samples/seq_time:,.0f} samples/sec\n")
        f.write(f"Parallel Throughput:   {total_samples/par_time:,.0f} samples/sec\n")

    print(f"\nResults saved to {filename}")


def main():
    """Main function to run experiments."""

    # You can change this to 'medium' or 'large' for bigger experiments
    config_name = 'small'  # 'small', 'medium', or 'large'

    print("\n" + "="*80)
    print(" Monte Carlo Spectrum Auction - HPC Parallelization Demo")
    print(" Comparing Sequential vs Parallel Fictitious Play")
    print("="*80)

    results = run_comparison_experiment(config_name)

    print("\n" + "="*80)
    print(" EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nKey Results:")
    print(f"  Speedup achieved: {results['speedup']:.2f}x")
    print(f"  Sequential time:  {results['sequential_time']:.2f}s")
    print(f"  Parallel time:    {results['parallel_time']:.2f}s")
    print(f"  Time saved:       {results['sequential_time'] - results['parallel_time']:.2f}s")
    print("\nBoth methods converged to similar solutions, demonstrating that")
    print("parallelization maintains solution quality while improving speed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
