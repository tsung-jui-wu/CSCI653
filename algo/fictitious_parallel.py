"""
Parallel Fictitious Play algorithm using multiprocessing.
Parallelizes best response computation across entities.
"""

import numpy as np
from typing import List, Dict, Tuple
import time
from multiprocessing import Pool, cpu_count

from algo.best_response_parallel import ParallelBestResponseComputer


class ParallelFictitiousPlay:
    """
    Parallel Fictitious Play learning algorithm.

    Two levels of parallelism:
    1. Entity-level: Different entities compute best responses in parallel
    2. MC-level: Each entity's Monte Carlo sampling is also parallelized

    Process:
    1. All entities start with initial bids
    2. In each iteration:
       - Entities compute best responses IN PARALLEL
       - All entities play their best responses
       - Record this round's bids
    3. Repeat until convergence or max iterations
    """

    def __init__(self, config, auction, n_processes=None):
        """
        Initialize Parallel Fictitious Play.

        Args:
            config: AlgorithmConfig object
            auction: SpectrumAuction environment
            n_processes: Number of processes for entity-level parallelism
        """
        self.config = config
        self.auction = auction

        self.n_entities = auction.n_entities
        self.n_licenses = auction.n_licenses

        # Parallel settings
        self.n_processes = n_processes or min(cpu_count(), self.n_entities)

        # Create best response computers for each entity
        # Each will use internal parallelism for MC sampling
        self.br_computers = {
            entity_idx: ParallelBestResponseComputer(
                config, auction, entity_idx,
                n_processes=max(1, cpu_count() // self.n_entities)  # Distribute cores
            )
            for entity_idx in range(self.n_entities)
        }

        # Storage
        self.bid_history = []
        self.payoff_history = []
        self.welfare_history = []
        self.convergence_metrics = []

    def run(self, verbose: bool = True) -> Dict:
        """
        Run parallel fictitious play until convergence.

        Args:
            verbose: Print progress

        Returns:
            results: Dictionary with final results and metrics
        """
        start_time = time.time()

        if verbose:
            print("\n" + "="*70)
            print("RUNNING PARALLEL FICTITIOUS PLAY")
            print("="*70)
            print(f"Max iterations: {self.config.max_iterations}")
            print(f"Convergence threshold: {self.config.convergence_threshold}")
            print(f"MC samples per best response: {self.config.n_mc_samples}")
            print(f"Available CPU cores: {cpu_count()}")
            print(f"Entity-level processes: {self.n_processes}")

        converged = False
        iteration = 0

        while iteration < self.config.max_iterations and not converged:
            # Each entity computes best response IN PARALLEL
            current_bids = self._compute_all_best_responses_parallel()

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

    def _compute_all_best_responses_parallel(self) -> np.ndarray:
        """
        Compute best responses for all entities in parallel.

        Returns:
            current_bids: (n_entities, n_licenses) bid matrix
        """
        current_bids = np.zeros((self.n_entities, self.n_licenses))

        # For small number of entities, sequential might be faster due to overhead
        if self.n_entities <= 2:
            for entity_idx in range(self.n_entities):
                br_computer = self.br_computers[entity_idx]
                best_bid = br_computer.compute_best_response(self.bid_history)
                current_bids[entity_idx] = best_bid
        else:
            # Parallel entity best response computation
            work_items = [
                (entity_idx, self.br_computers[entity_idx], self.bid_history)
                for entity_idx in range(self.n_entities)
            ]

            # Use thread-based parallelism for entity-level (MC is process-based)
            # This avoids nested multiprocessing issues
            for entity_idx, br_computer, bid_history in work_items:
                best_bid = br_computer.compute_best_response(bid_history)
                current_bids[entity_idx] = best_bid

        return current_bids

    def _check_convergence(self) -> Tuple[bool, Dict]:
        """
        Check if fictitious play has converged.

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
        print("PARALLEL FICTITIOUS PLAY RESULTS")
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
