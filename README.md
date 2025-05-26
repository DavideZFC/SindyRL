# BanditAlgorithms

A collection of simple and modular implementations for solving the stochastic multi-armed bandit problem.

## Overview

This repository includes classic algorithms for K-armed bandits, implemented with clarity and educational value in mind. The goal is to offer a clean and accessible framework for experimenting with online decision-making strategies.

Included algorithms:
- **Epsilon-Greedy**
- **Upper Confidence Bound (UCB1, UCB2)**
- **MOSS (Minimax Optimal Strategy in the Stochastic case)**
- **ETC (Explore-Then-Commit)**
- **RLPE (Regularized Least Posterior Estimation)**
- **Softmax Exploration**

These algorithms are evaluated on simple environments with Gaussian or Bernoulli rewards, and their performance is compared in terms of cumulative regret.

## Project Structure

```
algos/                  # Bandit algorithms
│
├── UCB/                # UCB-based algorithms (UCB1, UCB2, MOSS, etc.)
│   ├── MOSS.py
│   ├── UCB1.py
│   ├── UCB2.py
│   └── ucb.py          # Base or helper functions
│
├── ETC.py              # Explore-Then-Commit
├── RLPE.py             # Regularized Least Posterior Estimation
└── epsilon_greedy.py   # Epsilon-Greedy algorithm

environments/           # Bandit environments
├── Bandit.py           # Base bandit class
├── BernoulliBandit.py  # Bernoulli reward model
└── GaussianBandit.py   # Gaussian reward model
```

Each algorithm can be plugged into different environments for comparative testing.

## Example Usage

To run a basic comparison between algorithms:

```bash
python prova.py
```

Example output includes plots like cumulative regret over time.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
