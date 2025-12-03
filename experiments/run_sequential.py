"""
Complete example: Run fictitious play and analyze results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import get_small_experiment, get_medium_experiment
from environment.valuation import ValuationGenerator, print_valuation_summary
from environment.auction import SpectrumAuction
from algo.fictitious import FictitiousPlay, verify_nash_equilibrium


def run_complete_experiment():
    """Run complete experiment with analysis."""
    
    # === 1. SETUP ===
    print("\n" + "="*70)
    print(" SETUP")
    print("="*70)
    
    # config = get_medium_experiment()
    config = get_small_experiment()
    print(f"Experiment: {config.experiment_name}")
    print(f"Entities: {config.auction.n_entities}")
    print(f"Licenses: {config.auction.n_licenses}")
    
    # === 2. GENERATE ENVIRONMENT ===
    print("\n" + "="*70)
    print(" GENERATING ENVIRONMENT")
    print("="*70)
    
    generator = ValuationGenerator(config.auction, seed=config.auction.seed)
    valuations, budgets, metadata = generator.generate()
    print_valuation_summary(valuations, budgets)
    
    auction = SpectrumAuction(config.auction, valuations, budgets)
    
    # === 3. RUN FICTITIOUS PLAY ===
    fp = FictitiousPlay(config.algorithm, auction)
    results = fp.run(verbose=True)
    
    # === 4. VERIFY EQUILIBRIUM ===
    print("\n" + "="*70)
    print(" VERIFYING NASH EQUILIBRIUM")
    print("="*70)
    
    is_nash, exploitability = verify_nash_equilibrium(
        auction, results['final_bids'], epsilon=10.0
    )
    
    print(f"Is Nash equilibrium: {is_nash}")
    print(f"Exploitability per entity:")
    for i, exploit in enumerate(exploitability):
        print(f"  Entity {i}: ${exploit:.2f}")
    
    # === 5. PLOT RESULTS ===
    plot_results(results, config)
    
    return results


def plot_results(results: dict, config):
    """Create visualization of results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Fictitious Play Results: {config.experiment_name}", fontsize=14)
    
    # Plot 1: Welfare over time
    ax = axes[0, 0]
    welfare_history = results['welfare_history']
    ax.plot(welfare_history, label='Actual Welfare', linewidth=2)
    ax.axhline(results['optimal_welfare'], color='r', linestyle='--', 
               label='Optimal Welfare')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Social Welfare ($)')
    ax.set_title('Welfare Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Payoffs over time
    ax = axes[0, 1]
    payoff_history = results['payoff_history']
    for entity_idx in range(payoff_history.shape[1]):
        ax.plot(payoff_history[:, entity_idx], label=f'Entity {entity_idx}', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Payoff ($)')
    ax.set_title('Entity Payoffs Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Bid convergence (max change)
    ax = axes[1, 0]
    if results['convergence_metrics']:
        max_changes = [m['max_bid_change'] for m in results['convergence_metrics']]
        iterations = range(config.algorithm.min_iterations, 
                          config.algorithm.min_iterations + len(max_changes))
        ax.semilogy(iterations, max_changes, linewidth=2)
        ax.axhline(config.algorithm.convergence_threshold, color='r', 
                   linestyle='--', label='Threshold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max Bid Change (log scale)')
        ax.set_title('Convergence Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Final allocation
    ax = axes[1, 1]
    winners = results['final_winners']
    im = ax.imshow(winners, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('License')
    ax.set_ylabel('Entity')
    ax.set_title('Final Allocation (White = Won)')
    plt.colorbar(im, ax=ax)
    
    # Add numbers to show which entity won each license
    for license_idx in range(winners.shape[1]):
        winner = np.argmax(winners[:, license_idx])
        ax.text(license_idx, winner, 'W', ha='center', va='center', 
               color='red', fontweight='bold')
    
    plt.tight_layout()
    
    if config.save_plots:
        import os
        os.makedirs(config.output_dir, exist_ok=True)
        plt.savefig(f"{config.output_dir}/{config.experiment_name}_results.png", 
                   dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {config.output_dir}/{config.experiment_name}_results.png")
    
    plt.show()


if __name__ == "__main__":
    results = run_complete_experiment()
    
    print("\n" + "="*70)
    print(" EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"Final efficiency: {100*results['efficiency']:.1f}%")
    print(f"Converged: {results['converged']}")
    print(f"Runtime: {results['runtime']:.2f} seconds")
    print("="*70 + "\n")