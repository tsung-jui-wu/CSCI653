"""
Configuration file for spectrum auction experiments.
Modify parameters here to run different experiments without changing core code.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AuctionConfig:
    """
    Configuration for the auction environment.
    All parameters that define the auction setup.
    """
    
    # === Basic Setup ===
    n_entities: int = 5              # Number of bidders
    n_licenses: int = 10             # Number of spectrum licenses
    seed: int = 42                   # Random seed for reproducibility
    
    # === Valuation Generation ===
    valuation_type: str = 'uniform'  # 'uniform', 'structured', 'geographic'
    min_valuation: float = 100.0     # Minimum value for licenses
    max_valuation: float = 1000.0    # Maximum value for licenses
    
    # For 'geographic' type
    n_regions: Optional[int] = 5     # Number of geographic regions
    n_frequencies: Optional[int] = 2 # Number of frequency bands
    
    # === Budget Configuration ===
    budget_type: str = 'proportional'  # 'proportional', 'uniform', 'tight'
    budget_tightness: float = 0.7      # Fraction of total valuation (for proportional)
    min_budget: float = 500.0          # For uniform budget type
    max_budget: float = 5000.0         # For uniform budget type
    
    # === Complementarities (Synergies) ===
    has_complementarities: bool = False
    synergy_groups: Optional[list] = None  # List of license groups, e.g., [[0,1,2], [3,4]]
    synergy_strength: float = 0.3          # Bonus as fraction of base value
    
    # === Auction Rules ===
    tie_breaking: str = 'random'     # 'random', 'first', 'highest_id'
    

@dataclass
class AlgorithmConfig:
    """
    Configuration for the Fictitious Play algorithm.
    Parameters that control the learning process.
    """
    
    # === Fictitious Play Parameters ===
    max_iterations: int = 1000          # Maximum FP iterations
    convergence_threshold: float = 0.01 # Stop if bid changes < this
    min_iterations: int = 10            # Minimum iterations before checking convergence
    
    # === Monte Carlo Sampling ===
    n_mc_samples: int = 1000           # Number of MC samples for best response
    n_candidate_bids: int = 50         # Number of candidate bid allocations to test
    
    # === Best Response Strategy ===
    exploration_rate: float = 0.1      # Probability of random exploration
    bid_discretization: int = 20       # Number of discrete bid levels
    
    # === Initialization ===
    init_strategy: str = 'uniform'     # 'uniform', 'random', 'truthful'
    

@dataclass
class ParallelConfig:
    """
    Configuration for parallel execution.
    MPI and performance settings.
    """
    
    # === MPI Settings ===
    use_mpi: bool = True               # Whether to use MPI
    verbose: bool = True               # Print progress
    
    # === Performance ===
    profile: bool = False              # Profile computation time
    save_history: bool = True          # Save bid history
    checkpoint_interval: int = 100     # Save checkpoint every N iterations
    

@dataclass
class ExperimentConfig:
    """
    Complete configuration for an experiment.
    Combines all config components.
    """
    
    auction: AuctionConfig
    algorithm: AlgorithmConfig
    parallel: ParallelConfig
    
    # === Output Settings ===
    experiment_name: str = "default"
    output_dir: str = "./results"
    save_plots: bool = True
    save_data: bool = True
    
    
# === Predefined Experiment Configurations ===

def get_small_experiment():
    """Small experiment for quick testing (2-3 minutes)."""
    return ExperimentConfig(
        auction=AuctionConfig(
            n_entities=3,
            n_licenses=5,
            valuation_type='uniform',
            budget_tightness=0.7
        ),
        algorithm=AlgorithmConfig(
            max_iterations=100,
            n_mc_samples=500
        ),
        parallel=ParallelConfig(
            verbose=True
        ),
        experiment_name="small_test"
    )


def get_medium_experiment():
    """Medium experiment with complementarities."""
    return ExperimentConfig(
        auction=AuctionConfig(
            n_entities=5,
            n_licenses=10,
            valuation_type='structured',
            budget_tightness=0.7,
            has_complementarities=True,
            synergy_groups=[[0, 1, 2], [3, 4], [5, 6, 7]],
            synergy_strength=0.3
        ),
        algorithm=AlgorithmConfig(
            max_iterations=500,
            n_mc_samples=1000,
            n_candidate_bids=50
        ),
        parallel=ParallelConfig(
            verbose=True,
            save_history=True
        ),
        experiment_name="medium_with_synergies"
    )


def get_large_experiment():
    """Large-scale experiment for scalability testing."""
    return ExperimentConfig(
        auction=AuctionConfig(
            n_entities=10,
            n_licenses=20,
            valuation_type='geographic',
            n_regions=10,
            n_frequencies=2,
            budget_tightness=0.6
        ),
        algorithm=AlgorithmConfig(
            max_iterations=1000,
            n_mc_samples=2000,
            n_candidate_bids=100
        ),
        parallel=ParallelConfig(
            verbose=True,
            profile=True,
            checkpoint_interval=100
        ),
        experiment_name="large_scale"
    )


def get_custom_experiment(**kwargs):
    """
    Create a custom experiment by overriding default parameters.
    
    Example:
        config = get_custom_experiment(
            n_entities=7,
            n_licenses=15,
            budget_tightness=0.5,
            max_iterations=800
        )
    """
    # Start with default medium config
    config = get_medium_experiment()
    
    # Override auction parameters
    for key in ['n_entities', 'n_licenses', 'valuation_type', 
                'budget_tightness', 'has_complementarities']:
        if key in kwargs:
            setattr(config.auction, key, kwargs[key])
    
    # Override algorithm parameters
    for key in ['max_iterations', 'n_mc_samples', 'convergence_threshold']:
        if key in kwargs:
            setattr(config.algorithm, key, kwargs[key])
    
    # Override parallel parameters
    for key in ['use_mpi', 'verbose', 'save_history']:
        if key in kwargs:
            setattr(config.parallel, key, kwargs[key])
    
    if 'experiment_name' in kwargs:
        config.experiment_name = kwargs['experiment_name']
    
    return config