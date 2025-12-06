# System Architecture: Multi-Agent Economic Simulation

## Overview

This document describes the complete system architecture of the multi-agent macroeconomic simulation built with reinforcement learning.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │   config.py  │───▶│   train.py       │───▶│  checkpoints/*.pt        │   │
│  │  (Parameters)│    │  (Training Loop) │    │  (Saved Models)          │   │
│  └──────────────┘    └────────┬─────────┘    └──────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      MacroEconEnvironment                               │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │                         AGENTS                                    │  │ │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────────┐ │  │ │
│  │  │  │ Central   │ │Commercial │ │Households │ │      Firms        │ │  │ │
│  │  │  │ Bank (1)  │ │Banks (5)  │ │  (100)    │ │      (20)         │ │  │ │
│  │  │  └───────────┘ └───────────┘ └───────────┘ └───────────────────┘ │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │ │
│  │  │                         MARKETS                                   │  │ │
│  │  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐   │  │ │
│  │  │  │ Labor Market  │ │ Credit Market │ │    Goods Market       │   │  │ │
│  │  │  └───────────────┘ └───────────────┘ └───────────────────────┘   │  │ │
│  │  └──────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                               │                                              │
│                               ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        MultiAgentPPO                                    │ │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌─────────────┐ │ │
│  │  │ CB Network    │ │ Bank Network  │ │ HH Network    │ │Firm Network │ │ │
│  │  │ (128×128 MLP) │ │ (128×128 MLP) │ │ (128×128 MLP) │ │(128×128 MLP)│ │ │
│  │  └───────────────┘ └───────────────┘ └───────────────┘ └─────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │   app.py     │◀───│  checkpoints/*.pt│    │   Browser (Chart.js)     │   │
│  │   (Flask)    │───▶│  (Load Models)   │───▶│   Visualization          │   │
│  └──────────────┘    └──────────────────┘    └──────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          VALIDATION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │   test.py    │◀───│  checkpoints/*.pt│───▶│   validation_results.json│   │
│  │ (50 Tests)   │    │  (Run Episodes)  │    │   Reality Score          │   │
│  └──────────────┘    └──────────────────┘    └──────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
Economy_RL/
├── config.py                    # Central configuration (all parameters)
├── environment.py               # Main simulation orchestrator
├── train.py                     # Training script with PPO
├── test.py                      # 50-test validation suite
├── app.py                       # Flask web dashboard
│
├── agents/                      # Agent implementations
│   ├── __init__.py              # Exports all agent classes
│   ├── base_agent.py            # Abstract base class
│   ├── central_bank.py          # Monetary policy agent
│   ├── commercial_bank.py       # Credit intermediation
│   ├── household.py             # Consumption, labor supply
│   └── firm.py                  # Production, pricing, hiring
│
├── markets/                     # Market clearing mechanisms
│   ├── __init__.py              # Exports all market classes
│   ├── labor_market.py          # DMP search-and-matching
│   ├── credit_market.py         # Risk-based lending
│   └── goods_market.py          # Calvo pricing, demand
│
├── training/                    # RL training components
│   ├── __init__.py              # Exports PPO, Buffer
│   ├── ppo.py                   # Multi-agent PPO trainer
│   └── buffer.py                # Experience replay buffer
│
├── networks/                    # Neural network architectures
│   ├── __init__.py              # Exports ActorCritic
│   └── policy_network.py        # Actor-Critic MLP
│
├── economics/                   # Aggregate computations
│   ├── __init__.py              # Exports aggregates
│   └── aggregates.py            # GDP, inflation, Gini
│
├── templates/                   # Flask HTML templates
│   └── index.html               # Dashboard UI
│
├── checkpoints/                 # Saved model weights
│   └── run_YYYYMMDD_HHMMSS/
│       ├── best_model.pt
│       ├── final_model.pt
│       └── metrics.json
│
└── visualization/               # Plotting utilities
    └── plots.py
```

---

## Component Architecture

### 1. Configuration Layer (`config.py`)

```python
┌────────────────────────────────────────────────────────────┐
│                        Config                               │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  EconomicConfig  │  │   ActionConfig   │                │
│  ├──────────────────┤  ├──────────────────┤                │
│  │ num_households   │  │ cb_actions [21]  │                │
│  │ num_firms        │  │ bank_spreads [4] │                │
│  │ num_banks        │  │ hh_consumption[4]│                │
│  │ price_stickiness │  │ firm_prices [5]  │                │
│  │ separation_rate  │  │ firm_hiring [5]  │                │
│  │ taylor_coefs     │  │ ...              │                │
│  └──────────────────┘  └──────────────────┘                │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  NetworkConfig   │  │  TrainingConfig  │                │
│  ├──────────────────┤  ├──────────────────┤                │
│  │ hidden_sizes     │  │ learning_rate    │                │
│  │ input_sizes      │  │ gamma            │                │
│  │ activation       │  │ clip_epsilon     │                │
│  └──────────────────┘  └──────────────────┘                │
└────────────────────────────────────────────────────────────┘
```

**Purpose:** Single source of truth for all parameters.  
**Key Design:** Dataclasses with defaults allow easy modification.

---

### 2. Agent Architecture

```python
┌─────────────────────────────────────────────────────────────────┐
│                        BaseAgent (Abstract)                      │
├─────────────────────────────────────────────────────────────────┤
│  + id: int                                                       │
│  + config: Config                                                │
│  + state: AgentState                                             │
├─────────────────────────────────────────────────────────────────┤
│  + reset()                      # Initialize state               │
│  + get_observation() → ndarray  # State → neural network input   │
│  + decode_action(idx) → dict    # Action index → action values   │
│  + compute_reward() → float     # Compute agent's reward         │
│  + update_state()               # Apply action, update state     │
└─────────────────────────────────────────────────────────────────┘
                              ▲
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────┴─────────┐ ┌───────┴───────┐ ┌────────┴────────┐
│   CentralBank     │ │ CommercialBank│ │    Household    │
├───────────────────┤ ├───────────────┤ ├─────────────────┤
│ policy_rate       │ │ capital       │ │ savings         │
│ taylor_rule_rate  │ │ deposits      │ │ consumption     │
│ inflation_history │ │ loans, NPLs   │ │ is_employed     │
│                   │ │ lending_rate  │ │ permanent_income│
├───────────────────┤ ├───────────────┤ ├─────────────────┤
│ Actions:          │ │ Actions:      │ │ Actions:        │
│ - Rate level      │ │ - Spread      │ │ - Consumption % │
│   (0-10%, 21 acts)│ │ - Risk tol.   │ │ - Borrow Y/N    │
│                   │ │ (36 combos)   │ │ - Labor intens. │
│                   │ │               │ │   (24 combos)   │
└───────────────────┘ └───────────────┘ └─────────────────┘

                              ┌────────────────────┐
                              │       Firm         │
                              ├────────────────────┤
                              │ capital            │
                              │ inventory          │
                              │ num_workers        │
                              │ price              │
                              │ productivity       │
                              ├────────────────────┤
                              │ Actions:           │
                              │ - Price change     │
                              │ - Hiring (-2 to +2)│
                              │ - Investment rate  │
                              │   (100 combos)     │
                              └────────────────────┘
```

**Observation Flow:**
```
Global State → get_observation() → Normalization → [12-15 dim vector]
```

**Action Flow:**
```
Network output (softmax) → sample action_idx → decode_action() → {action dict}
```

---

### 3. Market Architecture

```python
┌─────────────────────────────────────────────────────────────────────┐
│                          MARKET CLEARING ORDER                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   1. Central Bank sets policy_rate                                   │
│                 │                                                    │
│                 ▼                                                    │
│   2. Banks set lending_rate = policy_rate + spread                   │
│                 │                                                    │
│                 ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    LABOR MARKET                              │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │  Inputs: Households, Firms                                   │   │
│   │  Process:                                                    │   │
│   │    1. Separations (employed → unemployed)                    │   │
│   │    2. Firms post vacancies                                   │   │
│   │    3. Matching: M = efficiency × √(V × U)                    │   │
│   │    4. Wage adjustment (Phillips Curve)                       │   │
│   │  Outputs: unemployment_rate, wage, matches                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                 │                                                    │
│                 ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   CREDIT MARKET                              │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │  Inputs: Banks, Households, Firms, policy_rate              │   │
│   │  Process:                                                    │   │
│   │    1. Households apply for loans (rate-sensitive demand)    │   │
│   │    2. Firms apply for loans (investment needs)              │   │
│   │    3. Banks evaluate risk, approve/reject                   │   │
│   │    4. DTI and collateral constraints enforced               │   │
│   │  Outputs: new_loans, credit_growth, avg_rates               │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                 │                                                    │
│                 ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    GOODS MARKET                              │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │  Inputs: Households (demand), Firms (supply)                │   │
│   │  Process:                                                    │   │
│   │    1. Aggregate demand = Σ household.consumption            │   │
│   │    2. Aggregate supply = Σ firm.inventory                   │   │
│   │    3. Allocate sales proportionally by price                │   │
│   │    4. Calvo pricing (35% can adjust)                        │   │
│   │    5. Compute inflation                                      │   │
│   │  Outputs: sales_by_firm, price_level, inflation             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4. Neural Network Architecture

```python
┌─────────────────────────────────────────────────────────────────┐
│                     ActorCritic Network                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: observation [12-15 dims]                                │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────────────────────┐                          │
│   │      Shared Feature Extractor     │                          │
│   │  Linear(input, 128) → ReLU        │                          │
│   └───────────────┬──────────────────┘                          │
│                   │                                              │
│          ┌────────┴────────┐                                    │
│          ▼                 ▼                                    │
│   ┌─────────────┐   ┌─────────────┐                             │
│   │ Actor Head  │   │ Critic Head │                             │
│   │ Linear(128) │   │ Linear(128) │                             │
│   │    ReLU     │   │    ReLU     │                             │
│   └──────┬──────┘   └──────┬──────┘                             │
│          ▼                 ▼                                    │
│   ┌─────────────┐   ┌─────────────┐                             │
│   │Policy Head  │   │ Value Head  │                             │
│   │Linear(acts) │   │ Linear(1)   │                             │
│   │  Softmax    │   │             │                             │
│   └──────┬──────┘   └──────┬──────┘                             │
│          ▼                 ▼                                    │
│   π(a|s): action      V(s): state                               │
│   probabilities        value                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Network Sizes:
┌────────────────┬───────────┬─────────────┬────────────────┐
│   Agent Type   │ Input Dim │ Output Dim  │  Parameters    │
├────────────────┼───────────┼─────────────┼────────────────┤
│ Central Bank   │    12     │     21      │    ~35K        │
│ Banks          │    12     │     36      │    ~36K        │
│ Households     │    15     │     24      │    ~36K        │
│ Firms          │    15     │    100      │    ~43K        │
├────────────────┼───────────┼─────────────┼────────────────┤
│ TOTAL          │    —      │      —      │   ~150K        │
└────────────────┴───────────┴─────────────┴────────────────┘
```

---

### 5. Training Architecture (PPO)

```python
┌─────────────────────────────────────────────────────────────────┐
│                     MultiAgentPPO Trainer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Training Loop                          │   │
│  │                                                           │   │
│  │   for epoch in range(800):                                │   │
│  │       │                                                   │   │
│  │       ▼  COLLECT ROLLOUT                                  │   │
│  │       ┌─────────────────────────────────────────────────┐ │   │
│  │       │  for step in range(300):                        │ │   │
│  │       │      obs = env.get_observations()               │ │   │
│  │       │      actions = networks.sample(obs)  # Batched  │ │   │
│  │       │      result = env.step(actions)                 │ │   │
│  │       │      buffer.store(obs, actions, rewards)        │ │   │
│  │       └─────────────────────────────────────────────────┘ │   │
│  │       │                                                   │   │
│  │       ▼  COMPUTE RETURNS (GAE)                            │   │
│  │       ┌─────────────────────────────────────────────────┐ │   │
│  │       │  for each buffer:                               │ │   │
│  │       │      advantages = GAE(rewards, values, γ, λ)    │ │   │
│  │       │      returns = advantages + values              │ │   │
│  │       └─────────────────────────────────────────────────┘ │   │
│  │       │                                                   │   │
│  │       ▼  PPO UPDATE (10 epochs)                           │   │
│  │       ┌─────────────────────────────────────────────────┐ │   │
│  │       │  for agent_type in [CB, Banks, HH, Firms]:      │ │   │
│  │       │      data = aggregate_all_agent_buffers()       │ │   │
│  │       │      for minibatch in shuffle(data):            │ │   │
│  │       │          ratio = π_new / π_old                  │ │   │
│  │       │          L_clip = min(ratio*A, clip(ratio)*A)   │ │   │
│  │       │          L_value = MSE(V, returns)              │ │   │
│  │       │          L_entropy = -H(π)                      │ │   │
│  │       │          loss = -L_clip + 0.5*L_value + 0.05*H  │ │   │
│  │       │          optimizer.step()                       │ │   │
│  │       └─────────────────────────────────────────────────┘ │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  KEY DESIGN: One network per agent TYPE, trained on pooled      │
│  experience from all agents of that type.                        │
│                                                                  │
│  Example: 100 households × 300 steps = 30,000 samples/epoch     │
│           All fed into ONE household network.                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**PPO Hyperparameters:**
```yaml
learning_rate: 3e-4
gamma: 0.97          # Monthly discount factor
gae_lambda: 0.95     # GAE parameter
clip_epsilon: 0.2    # PPO clipping
entropy_coef: 0.05   # Exploration bonus
value_coef: 0.5      # Value loss weight
update_epochs: 10    # PPO updates per rollout
minibatch_size: 32
```

---

### 6. Environment Step Sequence

```python
┌─────────────────────────────────────────────────────────────────┐
│               MacroEconEnvironment.step(actions)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Central Bank sets policy rate                          │
│  ────────────────────────────────────────                        │
│      action_idx = actions["central_bank"][0]                     │
│      cb_action = central_bank.decode_action(action_idx)          │
│      central_bank.update_state(cb_action)                        │
│      policy_rate = central_bank.get_policy_rate()                │
│                              │                                   │
│                              ▼                                   │
│  STEP 2: Banks set lending/deposit rates                        │
│  ────────────────────────────────────────                        │
│      for bank in banks:                                          │
│          action = bank.decode_action(actions["banks"][bank.id])  │
│          lending_rate = policy_rate + action["spread"]           │
│          bank.state.lending_rate = lending_rate                  │
│                              │                                   │
│                              ▼                                   │
│  STEP 3: Labor market clears                                     │
│  ────────────────────────────────────────                        │
│      labor_outcome = labor_market.clear(households, firms)       │
│      → Separations, matching, wage adjustment                    │
│                              │                                   │
│                              ▼                                   │
│  STEP 4: Credit market clears                                    │
│  ────────────────────────────────────────                        │
│      credit_outcome = credit_market.clear(banks, HH, firms)      │
│      → Loan applications, approvals, rate passthrough            │
│                              │                                   │
│                              ▼                                   │
│  STEP 5: Firms produce                                           │
│  ────────────────────────────────────────                        │
│      for firm in firms:                                          │
│          output = A × K^α × L^(1-α)   # Cobb-Douglas             │
│          inventory += output                                     │
│                              │                                   │
│                              ▼                                   │
│  STEP 6: Households consume                                      │
│  ────────────────────────────────────────                        │
│      for household in households:                                │
│          consumption = smooth(permanent_income) × rate           │
│                              │                                   │
│                              ▼                                   │
│  STEP 7: Goods market clears                                     │
│  ────────────────────────────────────────                        │
│      goods_outcome = goods_market.clear(households, firms)       │
│      → Sales allocation, Calvo pricing, inflation                │
│                              │                                   │
│                              ▼                                   │
│  STEP 8: All agents update state                                 │
│  ────────────────────────────────────────                        │
│      for agent in all_agents:                                    │
│          agent.update_state(action, market_outcomes)             │
│                              │                                   │
│                              ▼                                   │
│  STEP 9: Compute aggregates                                      │
│  ────────────────────────────────────────                        │
│      macro_state = aggregate_computer.compute(...)               │
│      GDP = C + I + ΔInventory                                    │
│      inflation = price_level_change                              │
│                              │                                   │
│                              ▼                                   │
│  STEP 10: Compute rewards                                        │
│  ────────────────────────────────────────                        │
│      for agent in all_agents:                                    │
│          reward = agent.compute_reward(action, macro_state)      │
│                              │                                   │
│                              ▼                                   │
│  RETURN: StepResult(macro_state, rewards, dones, info)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

                    CONFIG
                      │
                      ▼
              ┌───────────────┐
              │  Environment  │
              │   (126 agents)│
              └───────┬───────┘
                      │
         ┌────────────┴────────────┐
         │     OBSERVATIONS        │
         │  {type: {id: ndarray}}  │
         └────────────┬────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   Networks    │
              │  (4 MLPs)     │
              └───────┬───────┘
                      │
         ┌────────────┴────────────┐
         │      ACTIONS            │
         │  {type: {id: int}}      │
         └────────────┬────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Environment  │◄─────┐
              │    .step()    │      │
              └───────┬───────┘      │
                      │              │
         ┌────────────┴────────────┐ │
         │      STEP RESULT        │ │
         │  macro_state            │ │
         │  rewards {type: {id}}   │ │
         │  dones                  │ │
         │  info                   │ │
         └────────────┬────────────┘ │
                      │              │
                      ▼              │
              ┌───────────────┐      │
              │    Buffer     │      │
              │ (store trans.)│      │
              └───────┬───────┘      │
                      │              │
              ┌───────┴───────┐      │
              │  End of       │──NO──┘
              │  episode?     │
              └───────┬───────┘
                      │YES
                      ▼
              ┌───────────────┐
              │  Compute GAE  │
              │  PPO Update   │
              └───────────────┘
```

---

### 8. Reward Architecture

```python
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REWARD FUNCTIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CENTRAL BANK                                                                │
│  ─────────────                                                               │
│  reward = -10 × (chosen_rate - taylor_rate)²    # Taylor deviation           │
│         + bonus if deviation < 0.5%              # Precision bonus           │
│         - 0.5 × (inflation_gap² + 0.25 × output_gap²)  # Macro penalty       │
│         + 0.2 if rate > 1%                       # Avoid ZLB                 │
│                                                                              │
│  Range: [-20, +3]                                                            │
│                                                                              │
│  COMMERCIAL BANK                                                             │
│  ───────────────                                                             │
│  reward = profit / 3                             # Profit motive             │
│         - 2.5 × capital_gap / min_ratio          # Capital penalty           │
│         - 1.5 × (npl_ratio - 5%)                 # NPL penalty               │
│                                                                              │
│  Range: [-5, +5]                                                             │
│                                                                              │
│  HOUSEHOLD                                                                   │
│  ─────────                                                                   │
│  reward = log(consumption)                       # Utility from consumption  │
│         - 0.2 × (labor_intensity - 1)²           # Labor disutility          │
│         + 0.3 × buffer_ratio                     # Savings value             │
│         - 0.3 × debt_to_income                   # Debt burden               │
│         - 1.0 if unemployed                      # Unemployment penalty      │
│                                                                              │
│  Range: [-5, +5]                                                             │
│                                                                              │
│  FIRM                                                                        │
│  ────                                                                        │
│  reward = profit / 5                             # Profit motive             │
│         + 0.3 × cash / 30                        # Liquidity value           │
│         + 0.5 × capacity_utilization             # Efficiency bonus          │
│         - inventory_cost                         # Holding cost              │
│         - bankruptcy_penalty (if cash < -10)     # Avoid insolvency          │
│                                                                              │
│  Range: [-5, +5]                                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 9. Validation Architecture

```python
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION FRAMEWORK (test.py)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    RealWorldBenchmarks                               │    │
│  │  Source: FRED, BLS, BEA (1984-2019)                                  │    │
│  │  ├── GDP: mean=2.7%, std=2.1%, autocorr=0.85                        │    │
│  │  ├── Inflation: mean=2.8%, std=1.3%, persistence=0.85               │    │
│  │  ├── Unemployment: mean=5.8%, NAIRU=4.5%                            │    │
│  │  ├── Correlations: consumption-GDP=0.85, credit-GDP=0.70            │    │
│  │  └── Crisis facts: recession depth=-2.5%, frequency=15%             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       50 STATISTICAL TESTS                           │    │
│  │                                                                      │    │
│  │  Category 1: First Moments (mean values)                            │    │
│  │    - GDP growth mean, Inflation mean, Unemployment mean             │    │
│  │                                                                      │    │
│  │  Category 2: Second Moments (volatility)                            │    │
│  │    - GDP std, Inflation std, Unemployment std                       │    │
│  │                                                                      │    │
│  │  Category 3: Persistence (autocorrelation)                          │    │
│  │    - GDP AR(1), Inflation AR(1), Rate smoothing                     │    │
│  │                                                                      │    │
│  │  Category 4: Co-movements (correlations)                            │    │
│  │    - Consumption-GDP, Credit-GDP, Phillips curve                    │    │
│  │                                                                      │    │
│  │  Category 5: Policy Rules                                           │    │
│  │    - Taylor rule compliance, Rate response to shocks                │    │
│  │                                                                      │    │
│  │  Category 6: Business Cycle                                         │    │
│  │    - Recession depth, Duration, Recovery speed                      │    │
│  │                                                                      │    │
│  │  Category 7: Distribution                                           │    │
│  │    - Gini coefficient, Top 10% share, Bottom 50% share              │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        REALITY SCORE                                 │    │
│  │                                                                      │    │
│  │  Weighted average of all category scores:                           │    │
│  │    Score = Σ (category_weight × category_score)                     │    │
│  │                                                                      │    │
│  │  Current Result: 70%                                                 │    │
│  │    ├── Stability:      97%  ✓                                       │    │
│  │    ├── Phillips Curve: 96%  ✓                                       │    │
│  │    ├── Taylor Rule:    87%  ✓                                       │    │
│  │    ├── Persistence:    82%  ✓                                       │    │
│  │    ├── First Moments:  77%  ✓                                       │    │
│  │    ├── Transmission:   61%  ○                                       │    │
│  │    ├── Second Moments: 48%  ✗ (volatility too high)                 │    │
│  │    ├── Crisis:         44%  ✗ (recessions too deep)                 │    │
│  │    └── Co-movements:   20%  ✗ (wrong credit correlation)            │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. One Network Per Agent Type
- **Decision:** Share weights across agents of the same type
- **Rationale:** 100 households learning the same task should share knowledge
- **Benefit:** 4 networks instead of 126, faster training, better generalization

### 2. Discrete Action Spaces
- **Decision:** Discretize all continuous actions
- **Rationale:** Simpler PPO implementation, works well in practice
- **Trade-off:** Less granularity (e.g., 21 rate levels vs. continuous)

### 3. Monthly Timesteps
- **Decision:** 1 period = 1 month
- **Rationale:** Matches policy rate decision frequency
- **Trade-off:** Higher noise than quarterly data used in literature

### 4. Hardcoded Economic Mechanisms
- **Decision:** Embed Taylor Rule, Phillips Curve in environment
- **Rationale:** Guarantees economic plausibility
- **Trade-off:** Less "emergence," more "following built-in rules"

### 5. Consumption Smoothing
- **Decision:** 15% adjustment speed toward target consumption
- **Rationale:** Prevents demand death spirals
- **Trade-off:** Less realistic short-run dynamics

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Training time | ~3 hours (800 epochs on CPU) |
| Epoch time | ~13 seconds |
| Forward pass (batched) | <1ms per agent type |
| Memory usage | <500MB |
| Model size | ~600KB (4 networks) |
| Validation time | ~5 minutes (20 runs × 300 steps) |

---

## Extension Points

1. **New Agent Types:** Extend `BaseAgent`, add to environment
2. **New Markets:** Add market class, integrate into `step()` sequence
3. **New Actions:** Modify `ActionConfig`, update `decode_action()`
4. **New Observations:** Modify `get_observation()`, update network input size
5. **New Rewards:** Modify `compute_reward()` in agent class

---

## Files Quick Reference

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 254 | All parameters |
| `environment.py` | 480 | Simulation loop |
| `train.py` | 392 | Training script |
| `test.py` | 1761 | Validation |
| `agents/central_bank.py` | 310 | Monetary policy |
| `agents/commercial_bank.py` | 390 | Lending |
| `agents/household.py` | 516 | Consumption |
| `agents/firm.py` | 515 | Production |
| `markets/labor_market.py` | 251 | Employment |
| `markets/credit_market.py` | 245 | Loans |
| `markets/goods_market.py` | 212 | Sales |
| `training/ppo.py` | 354 | PPO algorithm |
| `training/buffer.py` | 175 | Experience buffer |
| `networks/policy_network.py` | 127 | Actor-Critic |
| `economics/aggregates.py` | 275 | GDP computation |
| **Total** | **~5,200** | |
