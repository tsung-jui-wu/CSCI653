"""
Parallel best response computation using multiprocessing.
Parallelizes Monte Carlo sampling for better performance on multi-core systems.
"""

import numpy as np
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
import functools


def _evaluate_single_sample(args):
    """
    Worker function to evaluate a single Monte Carlo sample.
    Must be a top-level function for multiprocessing.

    Args:
        args: Tuple of (my_bid, opponent_bids, entity_idx, auction_params)

    Returns:
        payoff: Float payoff for this sample
    """
    my_bid, opponent_bids, entity_idx, auction_params = args

    # Reconstruct auction environment (lightweight)
    from environment.auction import SpectrumAuction
    auction = _reconstruct_auction(auction_params)

    # Create full bid matrix
    full_bids = np.copy(opponent_bids)
    full_bids[entity_idx] = my_bid

    # Run auction and get payoff
    winners, payments, entity_payoffs = auction.run_auction(full_bids)

    return entity_payoffs[entity_idx]


def _reconstruct_auction(params):
    """Reconstruct auction object from serializable parameters."""
    from environment.auction import SpectrumAuction

    # Create minimal auction object for simulation
    class MinimalConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    config = MinimalConfig(**params['config'])
    auction = SpectrumAuction(
        config,
        params['base_valuations'],
        params['budgets']
    )

    return auction


class ParallelBestResponseComputer:
    """
    Parallel version of BestResponseComputer using multiprocessing.

    Key optimization: Parallelize Monte Carlo sampling across CPU cores.
    The most expensive operation is running many auction simulations,
    which are independent and can run in parallel.
    """

    def __init__(self, config, auction, entity_idx: int, n_processes=None):
        """
        Initialize parallel best response computer.

        Args:
            config: AlgorithmConfig object
            auction: SpectrumAuction environment
            entity_idx: Which entity this computer represents
            n_processes: Number of parallel processes (default: CPU count)
        """
        self.config = config
        self.auction = auction
        self.entity_idx = entity_idx

        self.n_entities = auction.n_entities
        self.n_licenses = auction.n_licenses
        self.my_valuation = auction.base_valuations[entity_idx]
        self.my_budget = auction.budgets[entity_idx]

        # Parallel settings
        self.n_processes = n_processes or cpu_count()

        # Serialize auction parameters for worker processes
        self.auction_params = {
            'config': {
                'n_entities': auction.config.n_entities,
                'n_licenses': auction.config.n_licenses,
                'seed': auction.config.seed,
                'valuation_type': auction.config.valuation_type,
                'budget_type': auction.config.budget_type,
                'has_complementarities': auction.config.has_complementarities,
                'synergy_groups': auction.config.synergy_groups,
                'synergy_strength': auction.config.synergy_strength,
                'tie_breaking': auction.config.tie_breaking,
            },
            'base_valuations': auction.base_valuations,
            'budgets': auction.budgets
        }

    def compute_best_response(self, bid_history: List[np.ndarray]) -> np.ndarray:
        """
        Compute best response to historical opponent bids (parallel version).

        Args:
            bid_history: List of past bid matrices [(n_entities, n_licenses), ...]

        Returns:
            best_bid: (n_licenses,) best bid allocation for this entity
        """
        # Handle first iteration (no history)
        if len(bid_history) == 0:
            return self._initial_bid()

        # Generate candidate bids to test
        candidate_bids = self._generate_candidate_bids()

        # Evaluate each candidate via PARALLEL Monte Carlo
        best_bid = None
        best_payoff = -np.inf

        for candidate in candidate_bids:
            avg_payoff = self._evaluate_bid_monte_carlo_parallel(candidate, bid_history)

            if avg_payoff > best_payoff:
                best_payoff = avg_payoff
                best_bid = candidate

        # Optional: add exploration
        if np.random.random() < self.config.exploration_rate:
            best_bid = self._exploratory_bid()

        return best_bid

    def _evaluate_bid_monte_carlo_parallel(self, my_bid: np.ndarray,
                                           bid_history: List[np.ndarray]) -> float:
        """
        Evaluate a candidate bid using PARALLEL Monte Carlo sampling.

        This is the key parallelization: distribute MC samples across cores.

        Args:
            my_bid: Candidate bid to evaluate
            bid_history: Historical bids to sample from

        Returns:
            avg_payoff: Expected payoff for this bid
        """
        n_samples = self.config.n_mc_samples

        # Generate all opponent samples upfront
        opponent_samples = [
            self._sample_opponent_bids(bid_history)
            for _ in range(n_samples)
        ]

        # Create work items for parallel processing
        work_items = [
            (my_bid, opponent_bids, self.entity_idx, self.auction_params)
            for opponent_bids in opponent_samples
        ]

        # Parallel execution using process pool
        with Pool(processes=self.n_processes) as pool:
            payoffs = pool.map(_evaluate_single_sample, work_items)

        return np.mean(payoffs)

    def _sample_opponent_bids(self, bid_history: List[np.ndarray]) -> np.ndarray:
        """
        Sample opponent bids from historical distribution.

        Args:
            bid_history: List of past bid matrices

        Returns:
            sampled_bids: (n_entities, n_licenses) matrix with opponent bids
        """
        # Simple strategy: sample uniformly from history
        t = np.random.randint(0, len(bid_history))
        historical_bids = bid_history[t]

        return historical_bids

    def _initial_bid(self) -> np.ndarray:
        """Generate initial bid (first iteration with no history)."""
        if self.config.init_strategy == 'uniform':
            bid = np.ones(self.n_licenses) * (self.my_budget / self.n_licenses)
        elif self.config.init_strategy == 'random':
            bid = np.random.uniform(0, self.my_budget / self.n_licenses, self.n_licenses)
            bid = bid * (self.my_budget / np.sum(bid))
        elif self.config.init_strategy == 'truthful':
            bid = self.my_valuation.copy()
            if np.sum(bid) > self.my_budget:
                bid = bid * (self.my_budget / np.sum(bid))
        else:
            raise ValueError(f"Unknown init_strategy: {self.config.init_strategy}")

        return bid

    def _generate_candidate_bids(self) -> List[np.ndarray]:
        """
        Generate candidate bid allocations to test.

        Strategy: Focus budget on different subsets of licenses.

        Returns:
            candidates: List of bid arrays to evaluate
        """
        candidates = []
        n_candidates = self.config.n_candidate_bids

        # Strategy 1: Uniform allocation
        candidates.append(np.ones(self.n_licenses) * (self.my_budget / self.n_licenses))

        # Strategy 2: Proportional to value
        value_based = self.my_valuation.copy()
        if np.sum(value_based) > 0:
            value_based = value_based * (self.my_budget / np.sum(value_based))
        candidates.append(value_based)

        # Strategy 3: Focus on top-k highest value licenses
        for k in [1, 3, 5, self.n_licenses // 2]:
            if k >= self.n_licenses:
                continue

            top_k_indices = np.argsort(self.my_valuation)[-k:]
            bid = np.zeros(self.n_licenses)
            bid[top_k_indices] = self.my_budget / k
            candidates.append(bid)

        # Strategy 4: Random allocations
        for _ in range(n_candidates - len(candidates)):
            n_active = np.random.randint(1, self.n_licenses + 1)
            active_licenses = np.random.choice(self.n_licenses, n_active, replace=False)

            bid = np.zeros(self.n_licenses)
            weights = self.my_valuation[active_licenses]
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_active) / n_active

            bid[active_licenses] = self.my_budget * weights
            candidates.append(bid)

        return candidates

    def _exploratory_bid(self) -> np.ndarray:
        """Generate an exploratory bid for learning."""
        bid = np.random.uniform(0, self.my_budget / self.n_licenses, self.n_licenses)
        bid = bid * (self.my_budget / np.sum(bid))
        return bid
