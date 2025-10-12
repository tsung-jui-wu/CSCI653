"""
Fictitious Play algorithm for computing Nash equilibrium.
Each entity iteratively best-responds to opponents' historical play.
"""

import numpy as np
from typing import List, Dict, Tuple
import time

from algo.best_response import BestResponseComputer


class FictitiousPlay:
    """
    Fictitious Play learning algorithm.
    
    Process:
    1. All entities start with initial bids
    2. In each iteration:
       - Each entity observes history of opponent bids
       - Each entity computes best response
       - All entities play their best responses
       - Record this round's bids
    3. Repeat until convergence or max iterations
    """
    
    def __init__(self, config, auction):
        """
        Initialize Fictitious Play.
        
        Args:
            config: AlgorithmConfig object
            auction: SpectrumAuction environment
        """
        self.config = config
        self.auction = auction
        
        self.n_entities = auction.n_entities
        self.n_licenses = auction.n_licenses
        
        # Create best response computers for each entity
        self.br_computers = {
            entity_idx: BestResponseComputer(config, auction, entity_idx)
            for entity_idx in range(self.n_entities)
        }
        
        # Storage
        self.bid_history = []
        self.payoff_history = []
        self.welfare_history = []
        self.convergence_metrics = []
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Run fictitious play until convergence.
        
        Args:
            verbose: Print progress
            
        Returns:
            results: Dictionary with final results and metrics
        """
        start_time = time.time()
        
        if verbose:
            print("\n" + "="*70)
            print("RUNNING FICTITIOUS PLAY")
            print("="*70)
            print(f"Max iterations: {self.config.max_iterations}")
            print(f"Convergence threshold: {self.config.convergence_threshold}")
            print(f"MC samples per best response: {self.config.n_mc_samples}")
        
        converged = False
        iteration = 0
        
        while iteration < self.config.max_iterations and not converged:
            # Each entity computes best response
            current_bids = np.zeros((self.n_entities, self.n_licenses))
            
            for entity_idx in range(self.n_entities):
                br_computer = self.br_computers[entity_idx]
                best_bid = br_computer.compute_best_response(self.bid_history)
                current_bids[entity_idx] = best_bid
            
            # Run auction with these bids
            winners, payments, payoffs = self.auction.run_auction(current_bids)
            welfare = self.auction.compute_social_welfare(winners)
            
            # Store results
            self.bid_history.append(current_bids.copy())
            self.payoff_history.append(payoffs.copy())
            self.welfare_history.append(welfare)
            
            # Check convergence
            if iteration >= self.config.min_iterations:
                converged, metrics = self._check_convergence()
                self.convergence_metrics.append(metrics)
            
            # Print progress
            if verbose and (iteration % 10 == 0 or converged):
                self._print_progress(iteration, welfare, metrics if converged else None)
            
            iteration += 1
        
        end_time = time.time()
        
        # Final results
        results = self._compile_results(converged, iteration, end_time - start_time)
        
        if verbose:
            self._print_final_results(results)
        
        return results
    
    def _check_convergence(self) -> Tuple[bool, Dict]:
        """
        Check if fictitious play has converged.
        
        Convergence criteria:
        1. Bids haven't changed much recently
        2. Payoffs are stable
        
        Returns:
            converged: True if converged
            metrics: Convergence metrics
        """
        if len(self.bid_history) < 2:
            return False, {}
        
        # Compare last two rounds
        recent_bids = self.bid_history[-1]
        previous_bids = self.bid_history[-2]
        
        # Maximum bid change across all entities and licenses
        max_bid_change = np.max(np.abs(recent_bids - previous_bids))
        
        # Average bid change
        avg_bid_change = np.mean(np.abs(recent_bids - previous_bids))
        
        # Payoff stability (last 10 iterations)
        if len(self.payoff_history) >= 10:
            recent_payoffs = np.array(self.payoff_history[-10:])
            payoff_std = np.std(recent_payoffs, axis=0)
            avg_payoff_std = np.mean(payoff_std)
        else:
            avg_payoff_std = np.inf
        
        metrics = {
            'max_bid_change': max_bid_change,
            'avg_bid_change': avg_bid_change,
            'avg_payoff_std': avg_payoff_std
        }
        
        # Converged if bid changes are small
        converged = max_bid_change < self.config.convergence_threshold
        
        return converged, metrics
    
    def _print_progress(self, iteration: int, welfare: float, metrics: Dict = None):
        """Print progress update."""
        msg = f"Iteration {iteration:4d}: Welfare = ${welfare:,.0f}"
        
        if metrics:
            msg += f" | Max Δbid = {metrics['max_bid_change']:.4f}"
        
        print(msg)
    
    def _compile_results(self, converged: bool, iterations: int, 
                         runtime: float) -> Dict:
        """
        Compile final results.
        
        Returns:
            results: Dictionary with all results and analysis
        """
        final_bids = self.bid_history[-1]
        winners, payments, payoffs = self.auction.run_auction(final_bids)
        final_welfare = self.auction.compute_social_welfare(winners)
        
        # Compute optimal welfare for comparison
        from environment.valuation import ValuationGenerator
        generator = ValuationGenerator(self.auction.config)
        optimal_allocation, optimal_welfare = generator.compute_optimal_allocation(
            self.auction.base_valuations
        )
        
        efficiency = final_welfare / optimal_welfare if optimal_welfare > 0 else 0
        
        results = {
            # Convergence info
            'converged': converged,
            'iterations': iterations,
            'runtime': runtime,
            
            # Final outcome
            'final_bids': final_bids,
            'final_winners': winners,
            'final_payments': payments,
            'final_payoffs': payoffs,
            'final_welfare': final_welfare,
            
            # Benchmarks
            'optimal_welfare': optimal_welfare,
            'efficiency': efficiency,
            
            # History
            'bid_history': np.array(self.bid_history),
            'payoff_history': np.array(self.payoff_history),
            'welfare_history': np.array(self.welfare_history),
            'convergence_metrics': self.convergence_metrics,
            
            # Revenue
            'total_revenue': np.sum(payments),
            'avg_price_per_license': np.sum(payments) / self.n_licenses
        }
        
        return results
    
    def _print_final_results(self, results: Dict):
        """Print summary of final results."""
        print("\n" + "="*70)
        print("FICTITIOUS PLAY RESULTS")
        print("="*70)
        
        print(f"\nConvergence:")
        print(f"  Status: {'CONVERGED' if results['converged'] else 'MAX ITERATIONS'}")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Runtime: {results['runtime']:.2f} seconds")
        
        print(f"\nOutcome:")
        print(f"  Final welfare: ${results['final_welfare']:,.0f}")
        print(f"  Optimal welfare: ${results['optimal_welfare']:,.0f}")
        print(f"  Efficiency: {100*results['efficiency']:.1f}%")
        print(f"  Total revenue: ${results['total_revenue']:,.0f}")
        print(f"  Avg price/license: ${results['avg_price_per_license']:,.0f}")
        
        print(f"\nPer-entity payoffs:")
        for i, payoff in enumerate(results['final_payoffs']):
            n_won = np.sum(results['final_winners'][i])
            paid = results['final_payments'][i]
            print(f"  Entity {i}: ${payoff:,.0f} (won {n_won:.0f} licenses, paid ${paid:,.0f})")
        
        print("="*70 + "\n")


# === Verification Functions ===

def verify_nash_equilibrium(auction, final_bids: np.ndarray, 
                            epsilon: float = 0.01) -> Tuple[bool, np.ndarray]:
    """
    Verify if final bids constitute an approximate Nash equilibrium.
    
    Check: Can any entity improve payoff by >epsilon by changing bid?
    
    Args:
        auction: SpectrumAuction environment
        final_bids: Final bid matrix to verify
        epsilon: Tolerance for approximate equilibrium
        
    Returns:
        is_nash: True if approximate Nash equilibrium
        exploitability: How much each entity could gain by deviating
    """
    n_entities = auction.n_entities
    exploitability = np.zeros(n_entities)
    
    # Get baseline payoffs
    _, _, baseline_payoffs = auction.run_auction(final_bids)
    
    for entity_idx in range(n_entities):
        # Try different bids for this entity
        best_alternative_payoff = baseline_payoffs[entity_idx]
        
        # Test some alternative bids
        br_computer = BestResponseComputer(None, auction, entity_idx)
        br_computer.config = type('obj', (object,), {
            'n_candidate_bids': 20,
            'init_strategy': 'uniform'
        })()
        
        alternative_bids = br_computer._generate_candidate_bids()
        
        for alt_bid in alternative_bids:
            test_bids = final_bids.copy()
            test_bids[entity_idx] = alt_bid
            
            _, _, test_payoffs = auction.run_auction(test_bids)
            
            if test_payoffs[entity_idx] > best_alternative_payoff:
                best_alternative_payoff = test_payoffs[entity_idx]
        
        # Exploitability = how much better they could do
        exploitability[entity_idx] = max(0, best_alternative_payoff - baseline_payoffs[entity_idx])
    
    # Is Nash if no one can improve by more than epsilon
    is_nash = np.all(exploitability < epsilon)
    
    return is_nash, exploitability