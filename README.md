# Distributed Nash-Spectrum-Auctions in Game Theory

### Problem Describtion 
N entities bidding for spectrum licenses across M resources through simultaneous sealed-bid auctions, where each entity has a budget constraint. The goal is to find the nash equilibrium where no entity could improve payoff given any deicsion.

### Simulation Methods
#### Fictionous Play Monte Carlo Sampling
Fictitious Play is an iterative learning algorithm where each player repeatedly computes their best response against the historical average behavior of all other players. In our spectrum auction context, each entity updates their bidding strategy by optimizing against the empirical distribution of competitors' past bids.

