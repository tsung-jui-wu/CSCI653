"""
Generate valuations and budgets for auction entities.
Supports multiple generation strategies for different experimental scenarios.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


class ValuationGenerator:
    """
    Generates entity valuations and budgets for spectrum licenses.
    
    This class creates the private values that each entity has for licenses,
    which drives their bidding behavior. Different generation methods allow
    for testing various market structures.
    """
    
    def __init__(self, config, seed: Optional[int] = None):
        """
        Initialize the valuation generator.
        
        Args:
            config: AuctionConfig object with parameters
            seed: Random seed for reproducibility
        """
        self.config = config
        if seed is not None:
            np.random.seed(seed)
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate valuations and budgets based on config.
        
        Returns:
            valuations: (n_entities, n_licenses) array of values
            budgets: (n_entities,) array of budget constraints
            metadata: Dictionary with additional information
        """
        if self.config.valuation_type == 'uniform':
            valuations = self._generate_uniform()
        elif self.config.valuation_type == 'structured':
            valuations = self._generate_structured()
        elif self.config.valuation_type == 'geographic':
            valuations = self._generate_geographic()
        else:
            raise ValueError(f"Unknown valuation_type: {self.config.valuation_type}")
        
        budgets = self._generate_budgets(valuations)
        
        metadata = {
            'valuation_type': self.config.valuation_type,
            'total_value': np.sum(valuations),
            'avg_value_per_license': np.mean(valuations, axis=0),
            'avg_value_per_entity': np.mean(valuations, axis=1)
        }
        
        return valuations, budgets, metadata
    
    def _generate_uniform(self) -> np.ndarray:
        """
        Generate uniform random valuations.
        Each entity-license pair gets an independent random value.
        
        Use case: Baseline experiment, no structure
        """
        valuations = np.random.uniform(
            low=self.config.min_valuation,
            high=self.config.max_valuation,
            size=(self.config.n_entities, self.config.n_licenses)
        )
        return valuations
    
    def _generate_structured(self) -> np.ndarray:
        """
        Generate structured valuations with entity preferences.
        Each entity has different preferences for different licenses.
        
        Use case: Realistic scenario where entities specialize
        """
        valuations = np.zeros((self.config.n_entities, self.config.n_licenses))
        
        for entity_idx in range(self.config.n_entities):
            # Each entity has a base value (how much capital they have)
            base_value = np.random.uniform(
                self.config.min_valuation * 2,
                self.config.max_valuation * 0.8
            )
            
            # Random preference for each license (0.5 to 1.5 multiplier)
            # Some licenses are worth more to this entity, others less
            preferences = np.random.uniform(0.5, 1.5, self.config.n_licenses)
            
            valuations[entity_idx] = base_value * preferences
        
        return valuations
    
    def _generate_geographic(self) -> np.ndarray:
        """
        Generate valuations with geographic structure.
        Licenses = regions × frequencies.
        Entities value different regions differently.
        
        Use case: Spectrum auctions where licenses have geographic structure
        """
        n_regions = self.config.n_regions or 5
        n_frequencies = self.config.n_frequencies or 2
        
        assert self.config.n_licenses == n_regions * n_frequencies, \
            f"n_licenses must equal n_regions * n_frequencies"
        
        valuations = np.zeros((self.config.n_entities, self.config.n_licenses))
        
        for entity_idx in range(self.config.n_entities):
            # Each entity values regions differently
            # (e.g., AT&T values urban areas more, rural carrier values rural)
            region_values = np.random.uniform(
                self.config.min_valuation,
                self.config.max_valuation,
                n_regions
            )
            
            # Frequency multipliers (all entities value frequencies similarly)
            freq_multipliers = np.random.uniform(0.8, 1.2, n_frequencies)
            
            # Combine region and frequency values
            license_idx = 0
            for region_idx in range(n_regions):
                for freq_idx in range(n_frequencies):
                    valuations[entity_idx, license_idx] = \
                        region_values[region_idx] * freq_multipliers[freq_idx]
                    license_idx += 1
        
        return valuations
    
    def _generate_budgets(self, valuations: np.ndarray) -> np.ndarray:
        """
        Generate budget constraints for entities.
        
        Args:
            valuations: Entity valuations to base budgets on
            
        Returns:
            budgets: (n_entities,) array of budget limits
        """
        if self.config.budget_type == 'proportional':
            # Budget = fraction of total valuation
            # (Can't afford to bid full value on everything)
            total_valuations = np.sum(valuations, axis=1)
            budgets = self.config.budget_tightness * total_valuations
            
        elif self.config.budget_type == 'uniform':
            # All entities have same budget
            budgets = np.ones(self.config.n_entities) * self.config.min_budget
            
        elif self.config.budget_type == 'tight':
            # Very tight budgets (can only bid on few licenses)
            total_valuations = np.sum(valuations, axis=1)
            budgets = 0.3 * total_valuations
            
        else:
            raise ValueError(f"Unknown budget_type: {self.config.budget_type}")
        
        return budgets
    
    def compute_optimal_allocation(self, valuations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the optimal allocation that maximizes social welfare.
        This ignores budgets and strategic behavior - just assigns each license
        to the entity that values it most.
        
        This serves as a benchmark for auction efficiency.
        
        Args:
            valuations: (n_entities, n_licenses) value matrix
            
        Returns:
            allocation: (n_entities, n_licenses) binary matrix (1 = winner)
            welfare: Total social welfare achieved
        """
        allocation = np.zeros_like(valuations)
        
        # For each license, assign to entity with highest valuation
        for license_idx in range(self.config.n_licenses):
            best_entity = np.argmax(valuations[:, license_idx])
            allocation[best_entity, license_idx] = 1
        
        welfare = np.sum(allocation * valuations)
        
        return allocation, welfare
    
    def add_complementarities(self, base_valuations: np.ndarray) -> callable:
        """
        Create a valuation function that includes complementarities.
        
        When an entity wins multiple licenses in a synergy group,
        they get a bonus value.
        
        Args:
            base_valuations: Base values without synergies
            
        Returns:
            valuation_function: Function that takes allocation and returns total value
        """
        if not self.config.has_complementarities:
            # No complementarities - just return sum of base values
            def simple_valuation(entity_idx, allocation):
                return np.sum(base_valuations[entity_idx] * allocation)
            return simple_valuation
        
        synergy_groups = self.config.synergy_groups
        synergy_strength = self.config.synergy_strength
        
        def complementary_valuation(entity_idx, allocation):
            """
            Compute entity's value for a given allocation including synergies.
            
            Args:
                entity_idx: Which entity
                allocation: Binary array indicating which licenses won
                
            Returns:
                total_value: Base value + synergy bonuses
            """
            # Start with base value
            base_value = np.sum(base_valuations[entity_idx] * allocation)
            
            # Add synergy bonuses
            synergy_bonus = 0.0
            for group in synergy_groups:
                # How many licenses in this group did entity win?
                licenses_in_group = [lic for lic in group if allocation[lic] == 1]
                n_won = len(licenses_in_group)
                
                # Bonus increases with number won
                # e.g., winning all 3 in group gives bigger bonus than just 2
                if n_won >= 2:
                    group_base_value = np.sum(base_valuations[entity_idx, group])
                    synergy_bonus += synergy_strength * group_base_value * (n_won / len(group))
            
            return base_value + synergy_bonus
        
        return complementary_valuation


# === Utility Functions ===

def print_valuation_summary(valuations: np.ndarray, budgets: np.ndarray):
    """Print a nice summary of generated valuations and budgets."""
    print("\n" + "="*60)
    print("VALUATION SUMMARY")
    print("="*60)
    print(f"Number of entities: {valuations.shape[0]}")
    print(f"Number of licenses: {valuations.shape[1]}")
    print(f"\nValuations range: [{np.min(valuations):.2f}, {np.max(valuations):.2f}]")
    print(f"Average valuation: {np.mean(valuations):.2f}")
    print(f"\nBudgets range: [{np.min(budgets):.2f}, {np.max(budgets):.2f}]")
    print(f"Average budget: {np.mean(budgets):.2f}")
    
    print(f"\nPer-entity summary:")
    for i in range(valuations.shape[0]):
        total_val = np.sum(valuations[i])
        budget_pct = 100 * budgets[i] / total_val
        print(f"  Entity {i}: Total value = ${total_val:,.0f}, "
              f"Budget = ${budgets[i]:,.0f} ({budget_pct:.1f}%)")
    print("="*60 + "\n")