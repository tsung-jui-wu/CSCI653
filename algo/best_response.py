"""
Best response computation for fictitious play.
Uses Monte Carlo sampling to approximate optimal bids given opponent behavior.
"""

import numpy as np
from typing import List, Callable, Tuple


class BestResponseComputer:
    """
    Computes approximately optimal bids using Monte Carlo sampling.
    
    The key challenge: we don't know exactly what opponents will bid,
    only their historical distribution. So we:
    1. Sample many possible opponent bid profiles
    2. Test our candidate bids against these samples
    3. Choose the bid that performs best on average
    """
    
    def __init__(self, config, auction, entity_idx: int):
        """
        Initialize best response computer for one entity.
        
        Args:
            config: AlgorithmConfig object
            auction: SpectrumAuction environment
            entity_idx: Which entity this computer represents
        """
        self.config = config
        self.auction = auction
        self.entity_idx = entity_idx
        
        self.n_entities = auction.n_entities
        self.n_licenses = auction.n_licenses
        self.my_valuation = auction.base_valuations[entity_idx]
        self.my_budget = auction.budgets[entity_idx]
    
    def compute_best_response(self, bid_history: List[np.ndarray]) -> np.ndarray:
        """
        Compute best response to historical opponent bids.
        
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
        
        # Evaluate each candidate via Monte Carlo
        best_bid = None
        best_payoff = -np.inf
        
        for candidate in candidate_bids:
            avg_payoff = self._evaluate_bid_monte_carlo(candidate, bid_history)
            
            if avg_payoff > best_payoff:
                best_payoff = avg_payoff
                best_bid = candidate
        
        # Optional: add exploration
        if np.random.random() < self.config.exploration_rate:
            best_bid = self._exploratory_bid()
        
        return best_bid
    
    def _initial_bid(self) -> np.ndarray:
        """
        Generate initial bid (first iteration with no history).
        
        Returns:
            bid: Initial bid allocation
        """
        if self.config.init_strategy == 'uniform':
            # Spread budget uniformly
            bid = np.ones(self.n_licenses) * (self.my_budget / self.n_licenses)
        
        elif self.config.init_strategy == 'random':
            # Random allocation respecting budget
            bid = np.random.uniform(0, self.my_budget / self.n_licenses, self.n_licenses)
            bid = bid * (self.my_budget / np.sum(bid))  # Normalize
        
        elif self.config.init_strategy == 'truthful':
            # Bid proportional to value
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
            # Random subset of licenses to bid on
            n_active = np.random.randint(1, self.n_licenses + 1)
            active_licenses = np.random.choice(self.n_licenses, n_active, replace=False)
            
            bid = np.zeros(self.n_licenses)
            # Allocate budget to active licenses (weighted by value)
            weights = self.my_valuation[active_licenses]
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_active) / n_active
            
            bid[active_licenses] = self.my_budget * weights
            candidates.append(bid)
        
        return candidates
    
    def _evaluate_bid_monte_carlo(self, my_bid: np.ndarray, 
                                   bid_history: List[np.ndarray]) -> float:
        """
        Evaluate a candidate bid using Monte Carlo sampling.
        
        Args:
            my_bid: Candidate bid to evaluate
            bid_history: Historical bids to sample from
            
        Returns:
            avg_payoff: Expected payoff for this bid
        """
        payoffs = []
        
        for _ in range(self.config.n_mc_samples):
            # Sample opponent bids from history
            opponent_bids = self._sample_opponent_bids(bid_history)
            
            # Create full bid matrix
            full_bids = np.copy(opponent_bids)
            full_bids[self.entity_idx] = my_bid
            
            # Run auction and get my payoff
            winners, payments, entity_payoffs = self.auction.run_auction(full_bids)
            payoffs.append(entity_payoffs[self.entity_idx])
        
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
        
        # Could also do: weighted sampling (recent bids more likely)
        # Or: sample from empirical distribution per entity
        
        return historical_bids
    
    def _exploratory_bid(self) -> np.ndarray:
        """
        Generate an exploratory bid for learning.
        
        Returns:
            bid: Random bid allocation
        """
        bid = np.random.uniform(0, self.my_budget / self.n_licenses, self.n_licenses)
        bid = bid * (self.my_budget / np.sum(bid))
        return bid


# === Utility Functions ===

def compute_empirical_strategy(bid_history: List[np.ndarray], 
                               entity_idx: int) -> np.ndarray:
    """
    Compute empirical average strategy for an entity.
    
    This is what fictitious play assumes opponents are playing.
    
    Args:
        bid_history: List of historical bid matrices
        entity_idx: Which entity
        
    Returns:
        avg_bid: (n_licenses,) average bid across history
    """
    if len(bid_history) == 0:
        return None
    
    entity_bids = [bids[entity_idx] for bids in bid_history]
    avg_bid = np.mean(entity_bids, axis=0)
    
    return avg_bid


def compute_regret(actual_payoffs: List[float], 
                   best_possible_payoffs: List[float]) -> float:
    """
    Compute regret: how much better could we have done?
    
    Args:
        actual_payoffs: What we actually got
        best_possible_payoffs: What we could have got in hindsight
        
    Returns:
        regret: Average regret per iteration
    """
    actual = np.array(actual_payoffs)
    best = np.array(best_possible_payoffs)
    
    regret = np.mean(best - actual)
    return regret