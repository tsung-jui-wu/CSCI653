# Distributed Nash Equilibrium in Spectrum Auctions

A game theory research project implementing distributed algorithms to find Nash equilibrium in multi-entity spectrum license auctions with budget constraints.

## Problem Description

This project simulates **N entities** bidding for **M spectrum licenses** through simultaneous sealed-bid auctions, where each entity has a budget constraint. The goal is to find the **Nash equilibrium** - a stable state where no entity can improve their payoff by unilaterally changing their bidding strategy.

Key features:
- **First-price sealed-bid auctions**: Highest bidder wins each license and pays their bid
- **Budget constraints**: Entities cannot exceed their total budget across all bids
- **Complementarities/Synergies**: Optional bonus value when winning related licenses together
- **Multiple valuation models**: Uniform, structured, and geographic valuation generation

## Algorithm: Fictitious Play with Monte Carlo Sampling

**Fictitious Play** is an iterative learning algorithm where each player repeatedly computes their best response against the historical average behavior of all other players.

### How it works:

1. **Initialization**: Each entity starts with an initial bidding strategy
2. **Iterative Learning**:
   - Each entity observes the history of all opponent bids
   - Each entity computes their best response using Monte Carlo sampling:
     - Generate candidate bid allocations (different ways to distribute budget)
     - For each candidate, sample opponent bids from historical distribution
     - Simulate auction outcomes and calculate expected payoff
     - Choose the bid with highest expected payoff
   - All entities simultaneously play their best responses
   - Record results and repeat
3. **Convergence**: Process continues until bids stabilize (Nash equilibrium) or max iterations reached

### Monte Carlo Sampling

Since we don't know exactly what opponents will bid (only their historical distribution), we use Monte Carlo sampling to approximate expected payoffs:
- Sample many possible opponent bid profiles from history
- Test candidate bids against these samples
- Choose bid that performs best on average

## Project Structure

```
FinalProject/
├── config/
│   └── config.py              # Experiment configurations (small/medium/large)
├── environment/
│   ├── auction.py             # Spectrum auction logic (bid validation, winner determination)
│   └── valuation.py           # Valuation & budget generation (uniform/structured/geographic)
├── algo/
│   ├── fictitious.py          # Fictitious Play algorithm implementation
│   └── best_response.py       # Best response computation with Monte Carlo sampling
├── experiments/
│   └── run_sequential.py      # Run experiments and visualize results
├── parallel/                  # (For future MPI-based parallel implementation)
└── utils/                     # Utility functions
```

## Code Components

### Configuration ([config/config.py](config/config.py))
- `AuctionConfig`: Auction parameters (entities, licenses, budgets, complementarities)
- `AlgorithmConfig`: Fictitious Play parameters (iterations, MC samples, convergence)
- `ParallelConfig`: MPI and performance settings
- Predefined experiments: `get_small_experiment()`, `get_medium_experiment()`, `get_large_experiment()`

### Auction Environment ([environment/auction.py](environment/auction.py))
- `SpectrumAuction`: Manages sealed-bid auction execution
  - Validates bids against budget constraints
  - Determines winners (highest bidder per license)
  - Calculates payments and payoffs (value - payment)
  - Handles complementarities through custom valuation functions

### Valuation Generation ([environment/valuation.py](environment/valuation.py))
- `ValuationGenerator`: Creates entity valuations and budgets
  - **Uniform**: Random independent values
  - **Structured**: Entities have different preferences (realistic)
  - **Geographic**: Licenses organized by region × frequency
  - Adds complementarity bonuses for synergy groups

### Best Response ([algo/best_response.py](algo/best_response.py))
- `BestResponseComputer`: Computes optimal bid for one entity
  - Generates candidate bids (uniform, value-proportional, top-k focus, random)
  - Evaluates candidates via Monte Carlo simulation
  - Samples opponent bids from historical distribution
  - Returns bid with highest expected payoff

### Fictitious Play ([algo/fictitious.py](algo/fictitious.py))
- `FictitiousPlay`: Main algorithm orchestration
  - Runs iterative learning process
  - Tracks bid history and convergence metrics
  - Computes efficiency vs optimal welfare
  - Verifies Nash equilibrium (exploitability check)

## Running Experiments

### Basic Usage

```bash
python experiments/run_sequential.py
```

This runs the medium experiment with:
- 5 entities, 10 licenses
- Complementarities enabled (synergy groups)
- Up to 500 iterations
- 1000 Monte Carlo samples per best response

### Customizing Experiments

Edit [config/config.py](config/config.py) or use the custom experiment function:

```python
from config.config import get_custom_experiment

config = get_custom_experiment(
    n_entities=7,
    n_licenses=15,
    budget_tightness=0.5,
    max_iterations=800,
    valuation_type='geographic'
)
```

## Key Metrics

- **Social Welfare**: Total value generated by allocation
- **Efficiency**: Actual welfare / Optimal welfare (%)
- **Revenue**: Total payments collected
- **Exploitability**: How much each entity could gain by deviating (measures Nash equilibrium quality)
- **Convergence**: Maximum bid change between iterations

## Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

## Example Output

The experiment produces:
1. **Console output**: Iteration progress, convergence metrics, final results
2. **Visualizations**:
   - Welfare convergence over time
   - Entity payoffs over time
   - Bid convergence (log scale)
   - Final license allocation matrix
3. **Nash equilibrium verification**: Exploitability per entity

## Future Work

- MPI-based parallel implementation for large-scale experiments
- Additional auction formats (second-price, VCG)
- Learning rate adjustments and exploration strategies
- Benchmarking against other equilibrium-finding algorithms

