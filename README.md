# Multi-Agent Macroeconomic Simulation with Reinforcement Learning

A simulation where **monetary policy transmission emerges** from interactions between learned agents rather than being hardcoded. This is Agent-Based Computational Economics (ACE) meets Deep Reinforcement Learning.

## 🎯 Project Overview

This project simulates a complete economy with:
- **1 Central Bank** that learns optimal monetary policy (essentially learning a Taylor rule)
- **N Commercial Banks** that learn to set lending rates and manage risk
- **M Households** that learn consumption, saving, and borrowing behavior
- **K Firms** that learn pricing, hiring, and investment decisions

The agents interact through three markets:
- **Labor Market**: Wage determination and employment matching
- **Credit Market**: Loan origination and default dynamics
- **Goods Market**: Consumption and price level determination

**Key Innovation**: Monetary policy transmission (how interest rate changes affect inflation and output) emerges from agent learning rather than being imposed through equations.

## 📁 Project Structure

```
macro_sim/
├── config.py                 # All hyperparameters and economic parameters
├── environment.py            # Main simulation environment
├── main.py                   # Training script entry point
├── requirements.txt          # Dependencies
│
├── agents/                   # Economic agents
│   ├── base_agent.py         # Abstract base class
│   ├── central_bank.py       # Central Bank (monetary policy)
│   ├── commercial_bank.py    # Commercial Banks (credit intermediation)
│   ├── household.py          # Households (consumption, labor)
│   └── firm.py               # Firms (production, pricing)
│
├── markets/                  # Market clearing mechanisms
│   ├── labor_market.py       # Employment matching, wage dynamics
│   ├── credit_market.py      # Loan matching, default handling
│   └── goods_market.py       # Supply-demand clearing, price level
│
├── economics/                # Aggregate computations
│   └── aggregates.py         # GDP, inflation, unemployment calculation
│
├── networks/                 # Neural network architectures
│   └── policy_network.py     # Actor-Critic networks for PPO
│
├── training/                 # RL training infrastructure
│   ├── buffer.py             # Experience replay buffer with GAE
│   └── ppo.py                # PPO algorithm implementation
│
└── visualization/            # Plotting and analysis
    └── plots.py              # Training curves, macro variables, policy transmission
```

## 🚀 Quick Start

### Installation

```bash
# Clone or unzip the project
cd macro_sim

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training (500 epochs, 10-year simulation)
python main.py --mode train --epochs 500

# Faster test run
python main.py --mode train --epochs 100 --sim-length 60 --num-households 50

# Full scale
python main.py --mode train \
    --epochs 1000 \
    --sim-length 120 \
    --num-households 100 \
    --num-firms 20 \
    --num-banks 5 \
    --output-dir ./experiments
```

### Evaluation

```bash
python main.py --mode eval --model-path ./experiments/experiment_XXXX/models/best
```

## 🔧 Configuration

All parameters are in `config.py`. Key settings:

### Economic Parameters
```python
# Agent counts
num_commercial_banks: int = 5
num_households: int = 100
num_firms: int = 20

# Central Bank targets
inflation_target: float = 0.02  # 2% annual
initial_policy_rate: float = 0.03  # 3%

# Simulation
periods_per_year: int = 12  # Monthly
simulation_length: int = 120  # 10 years
```

### Training Parameters
```python
learning_rate: float = 3e-4
gamma: float = 0.99  # Discount factor
clip_epsilon: float = 0.2  # PPO clip
num_epochs: int = 1000
```

## 📊 What Emerges

After training, you should observe:

1. **Learned Taylor Rule**: The Central Bank learns to raise rates when inflation rises and cut when unemployment rises (without being told to do this!)

2. **Credit Channel**: Banks adjust lending standards based on policy rate changes, affecting credit availability

3. **Consumption Response**: Households reduce borrowing and consumption when rates rise

4. **Price Stickiness**: Firms learn gradual price adjustment (emergent Calvo pricing)

5. **Business Cycles**: Endogenous fluctuations in output and employment

## 📈 Outputs

Training produces:
- `models/`: Saved neural network weights
- `plots/`:
  - `macro_variables.png`: GDP, inflation, unemployment, rates over time
  - `training_curves.png`: Loss and entropy for each agent type
  - `policy_transmission.png`: How policy rate affects other variables
  - `agent_rewards.png`: Rewards by agent type
- `logs/`:
  - `simulation_history.csv`: Full time series data
  - `train_stats.json`: Training metrics

## 🧠 Technical Details

### Agent Reward Functions

**Central Bank**: Taylor rule-like loss
```
reward = -[(inflation - target)² + λ(output_gap)² + μ(rate_change)²]
```

**Commercial Banks**: Profit maximization
```
reward = net_interest_income - loan_losses - capital_adequacy_penalty
```

**Households**: Utility maximization
```
reward = log(consumption) - labor_disutility + savings_utility - debt_disutility
```

**Firms**: Profit maximization
```
reward = profit - inventory_holding_cost - bankruptcy_penalty
```

### Action Spaces (all discrete)

| Agent | Actions |
|-------|---------|
| Central Bank | Rate change: [-50, -25, 0, +25, +50] bps |
| Banks | Lending spread × Deposit spread × Risk tolerance (75 actions) |
| Households | Consumption rate × Borrow × Labor supply (30 actions) |
| Firms | Price change × Hiring × Investment (45 actions) |

### Training Approach

- **Independent PPO**: Each agent type has its own policy network
- **Parameter Sharing**: Agents of same type share weights (e.g., all households use same policy)
- **Heterogeneity**: Agents have different initial conditions (wealth, income, etc.)

## 🔬 Experiments to Try

1. **Shock Response**: After training, introduce an inflation shock and watch the central bank respond

2. **Policy Rule Comparison**: Compare learned policy to standard Taylor rule

3. **Credit Crisis**: Increase default rates and observe systemic effects

4. **Inequality Dynamics**: Track Gini coefficient evolution

5. **Zero Lower Bound**: What happens when rates can't go below zero?

## 📝 For Your Medium Article

Key angles to highlight:

1. **Emergence**: The most interesting finding is what EMERGES without being programmed
2. **Visualization**: The policy transmission plots tell a compelling story
3. **Honesty**: Document what doesn't work well (multi-agent RL is hard!)
4. **Comparison**: Compare learned behavior to textbook economics
5. **Code Quality**: This is production-grade code structure

## 🛠️ Extending the Project

Ideas for future work:
- Add a government sector (fiscal policy)
- Implement heterogeneous expectations
- Add international trade
- Include asset markets
- Implement more sophisticated matching (directed search)

## 📚 References

- Sutton & Barto (2018) - Reinforcement Learning
- Schulman et al. (2017) - PPO algorithm
- Tesfatsion (2006) - Agent-Based Computational Economics
- Smets & Wouters (2007) - DSGE model comparison

## License

MIT License - Use freely for research and learning.

---

*Built as part of a research portfolio project exploring the intersection of macroeconomics and deep reinforcement learning.*
