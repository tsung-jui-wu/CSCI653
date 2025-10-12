"""
Core auction environment that runs sealed-bid auctions.
Handles bid submission, winner determination, and payoff calculation.
"""

import numpy as np
from typing import Tuple, Dict


class SpectrumAuction:
    """
    Sealed-bid spectrum auction environment.
    
    In each round:
    1. Entities submit bids for each license
    2. Highest bidder wins each license
    3. Winners pay their bids
    4. Payoffs = value - payment for won licenses
    
    This is a first-price sealed-bid auction format.
    """
    
    def __init__(self, config, valuations: np.ndarray, budgets: np.ndarray,
                 valuation_functions=None):
        """
        Initialize the auction environment.
        
        Args:
            config: AuctionConfig object
            valuations: (n_entities, n_licenses) base valuations
            budgets: (n_entities,) budget constraints
            valuation_functions: Optional dict of entity_idx -> valuation_function
                                for handling complementarities
        """
        self.config = config
        self.n_entities = config.n_entities
        self.n_licenses = config.n_licenses
        
        self.base_valuations = valuations
        self.budgets = budgets
        self.valuation_functions = valuation_functions
        
        # Validation
        assert valuations.shape == (self.n_entities, self.n_licenses)
        assert budgets.shape == (self.n_entities,)
    
    def run_auction(self, bids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute a sealed-bid auction given all bids.
        
        Args:
            bids: (n_entities, n_licenses) matrix of bid amounts
            
        Returns:
            winners: (n_entities, n_licenses) binary matrix (1 = won)
            payments: (n_entities,) total payment per entity
            payoffs: (n_entities,) net payoff per entity
        """
        # Validate bids
        if not self._validate_bids(bids):
            raise ValueError("Invalid bids: some entity exceeded budget")
        
        # Determine winners (highest bidder wins each license)
        winners = self._determine_winners(bids)
        
        # Calculate payments (winners pay their bid)
        payments = self._calculate_payments(bids, winners)
        
        # Calculate payoffs (value - payment)
        payoffs = self._calculate_payoffs(winners, payments)
        
        return winners, payments, payoffs
    
    def _validate_bids(self, bids: np.ndarray) -> bool:
        """
        Check if all bids respect budget constraints.
        
        Args:
            bids: Bid matrix to validate
            
        Returns:
            valid: True if all bids valid
        """
        # Check shape
        if bids.shape != (self.n_entities, self.n_licenses):
            return False
        
        # Check non-negative
        if np.any(bids < 0):
            return False
        
        # Check budget constraints
        total_bids = np.sum(bids, axis=1)
        if np.any(total_bids > self.budgets + 1e-6):  # Small tolerance for float errors
            return False
        
        return True
    
    def _determine_winners(self, bids: np.ndarray) -> np.ndarray:
        """
        Determine auction winners.
        Highest bidder wins each license.
        
        Args:
            bids: (n_entities, n_licenses) bid matrix
            
        Returns:
            winners: (n_entities, n_licenses) binary allocation matrix
        """
        winners = np.zeros_like(bids)
        
        for license_idx in range(self.n_licenses):
            license_bids = bids[:, license_idx]
            
            # Find highest bidder
            max_bid = np.max(license_bids)
            
            # Handle ties according to config
            if self.config.tie_breaking == 'random':
                # Among highest bidders, pick one randomly
                highest_bidders = np.where(license_bids == max_bid)[0]
                winner_idx = np.random.choice(highest_bidders)
            elif self.config.tie_breaking == 'first':
                # First entity with highest bid wins
                winner_idx = np.argmax(license_bids)
            elif self.config.tie_breaking == 'highest_id':
                # Highest ID with highest bid wins
                highest_bidders = np.where(license_bids == max_bid)[0]
                winner_idx = highest_bidders[-1]
            else:
                raise ValueError(f"Unknown tie_breaking: {self.config.tie_breaking}")
            
            winners[winner_idx, license_idx] = 1
        
        return winners
    
    def _calculate_payments(self, bids: np.ndarray, winners: np.ndarray) -> np.ndarray:
        """
        Calculate how much each entity pays.
        In first-price auction, winners pay their bid.
        
        Args:
            bids: Bid matrix
            winners: Winner allocation matrix
            
        Returns:
            payments: (n_entities,) total payment per entity
        """
        payments = np.sum(winners * bids, axis=1)
        return payments
    
    def _calculate_payoffs(self, winners: np.ndarray, payments: np.ndarray) -> np.ndarray:
        """
        Calculate entity payoffs (utility).
        Payoff = value of won licenses - payment
        
        Args:
            winners: Winner allocation matrix
            payments: Payment amounts
            
        Returns:
            payoffs: (n_entities,) net payoff per entity
        """
        payoffs = np.zeros(self.n_entities)
        
        for entity_idx in range(self.n_entities):
            # Get allocation for this entity
            allocation = winners[entity_idx]
            
            # Calculate value (with or without complementarities)
            if self.valuation_functions and entity_idx in self.valuation_functions:
                # Use complementary valuation function
                value = self.valuation_functions[entity_idx](entity_idx, allocation)
            else:
                # Simple additive valuation
                value = np.sum(self.base_valuations[entity_idx] * allocation)
            
            # Payoff = value - payment
            payoffs[entity_idx] = value - payments[entity_idx]
        
        return payoffs
    
    def compute_social_welfare(self, winners: np.ndarray) -> float:
        """
        Compute total social welfare (sum of all entity values).
        
        Args:
            winners: Allocation matrix
            
        Returns:
            welfare: Total value generated
        """
        welfare = 0.0
        
        for entity_idx in range(self.n_entities):
            allocation = winners[entity_idx]
            
            if self.valuation_functions and entity_idx in self.valuation_functions:
                value = self.valuation_functions[entity_idx](entity_idx, allocation)
            else:
                value = np.sum(self.base_valuations[entity_idx] * allocation)
            
            welfare += value
        
        return welfare
    
    def get_truthful_bids(self) -> np.ndarray:
        """
        Generate truthful bids (bid your value, scaled to budget).
        This is a baseline strategy for comparison.
        
        Returns:
            bids: (n_entities, n_licenses) truthful bid matrix
        """
        bids = np.copy(self.base_valuations)
        
        # Scale down if total exceeds budget
        for entity_idx in range(self.n_entities):
            total_value = np.sum(bids[entity_idx])
            if total_value > self.budgets[entity_idx]:
                # Scale proportionally to fit budget
                scale_factor = self.budgets[entity_idx] / total_value
                bids[entity_idx] *= scale_factor
        
        return bids


# === Utility Functions ===

def print_auction_results(winners: np.ndarray, payments: np.ndarray, 
                         payoffs: np.ndarray, welfare: float):
    """Print a summary of auction outcomes."""
    print("\n" + "="*60)
    print("AUCTION RESULTS")
    print("="*60)
    
    n_entities = winners.shape[0]
    
    for entity_idx in range(n_entities):
        licenses_won = np.where(winners[entity_idx] == 1)[0]
        n_won = len(licenses_won)
        
        print(f"\nEntity {entity_idx}:")
        print(f"  Licenses won: {n_won} {list(licenses_won) if n_won <= 5 else '...'}")
        print(f"  Payment: ${payments[entity_idx]:,.2f}")
        print(f"  Payoff: ${payoffs[entity_idx]:,.2f}")
    
    print(f"\nTotal social welfare: ${welfare:,.2f}")
    print(f"Total revenue: ${np.sum(payments):,.2f}")
    print("="*60 + "\n")