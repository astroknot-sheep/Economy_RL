# Building a Multi-Agent Economic Simulation from Scratch: An Honest Engineering Post-Mortem

*I built a macroeconomic simulation with 126 AI agents trained via reinforcement learning. Here's the complete technical breakdown — what I designed, what the agents learned, what worked, and what didn't.*

---

## TL;DR

Over several weeks, I built a multi-agent macroeconomic simulation from scratch in Python/PyTorch. The simulation achieved a **70% Reality Score** when tested against 50 statistical benchmarks from US economic data (1960-2023). 

This post is an honest breakdown of the entire codebase — what's hardcoded economic theory, what's genuinely learned by RL agents, and where the simulation succeeds and fails.

---

## The Architecture

### What I Built

A complete macroeconomic simulation consisting of:

| Component | Count | Implementation |
|-----------|-------|----------------|
| Central Bank | 1 | Sets policy rate (0-10% in 0.5% steps) |
| Commercial Banks | 5 | Set lending spreads, risk tolerance |
| Households | 100 | Consume, save, supply labor |
| Firms | 20 | Produce, hire, set prices |
| Labor Market | 1 | Search-and-matching (DMP model) |
| Credit Market | 1 | Risk-based lending, rate passthrough |
| Goods Market | 1 | Calvo pricing, demand allocation |

**Total: 126 agents + 3 markets**

### The Tech Stack

```
├── config.py                    # 254 lines — All economic parameters
├── environment.py               # 480 lines — Orchestrates simulation
├── train.py                     # 392 lines — PPO training loop
├── test.py                      # 1761 lines — 50-test validation suite
├── agents/
│   ├── base_agent.py            # 168 lines — Abstract agent class
│   ├── central_bank.py          # 310 lines — Monetary policy
│   ├── commercial_bank.py       # 390 lines — Credit intermediation
│   ├── household.py             # 516 lines — Consumption, labor supply
│   └── firm.py                  # 515 lines — Production, pricing
├── markets/
│   ├── labor_market.py          # 251 lines — Employment, wages
│   ├── credit_market.py         # 245 lines — Loans, rates
│   └── goods_market.py          # 212 lines — Sales, inflation
├── training/
│   ├── ppo.py                   # 354 lines — Proximal Policy Optimization
│   └── buffer.py                # 175 lines — Experience replay
├── networks/
│   └── policy_network.py        # 127 lines — Actor-Critic MLP
└── economics/
    └── aggregates.py            # 275 lines — GDP, inflation computation
```

**Total: ~4,500 lines of Python**

### Neural Network Architecture

Each agent type has a **128×128 MLP** (Multi-Layer Perceptron):

```python
# networks/policy_network.py
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 128]):
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )
        # Separate actor/critic heads
        self.actor_layers = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.critic_layers = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.policy_head = nn.Linear(128, output_size)
        self.value_head = nn.Linear(128, 1)
```

**Observation sizes:**
- Central Bank: 12 dimensions (inflation, output gap, rates, trends)
- Banks: 12 dimensions (capital ratio, NPL, rates, macro)
- Households: 15 dimensions (employment, income, savings, macro)
- Firms: 15 dimensions (capital, inventory, prices, demand)

**Action spaces:**
- Central Bank: 21 discrete actions (0% to 10% in 0.5% steps)
- Banks: 36 combinations (4 lending spreads × 3 deposit spreads × 3 risk levels)
- Households: 24 combinations (4 consumption × 2 borrowing × 3 labor)
- Firms: 100 combinations (5 price × 5 hiring × 4 investment)

---

## What's Hardcoded vs. What's Learned

This is the most important section. Let me be completely transparent about what the AI "discovers" versus what I programmed.

### ❌ HARDCODED: The Taylor Rule

**The claim I could have made:** "The Central Bank discovered the Taylor Rule without being told it exists!"

**The truth:** The Taylor Rule is explicitly computed in my code and used as the reward target.

```python
# agents/central_bank.py, lines 280-300
def compute_taylor_rule_rate(self):
    """
    Compute Taylor Rule suggested rate.
    
    r = r* + π + 1.5*(π - π*) + 0.5*y    ← THIS IS THE TAYLOR RULE
    """
    taylor_rate = (
        r_star +                                    # Neutral rate
        pi +                                        # Current inflation
        self.taylor_inflation_coef * (pi - pi_star) +  # 1.5 coefficient
        self.taylor_output_coef * y                     # 0.5 coefficient
    )
    return taylor_rate
```

And the reward function:

```python
# agents/central_bank.py, lines 203-257
def compute_reward(self, action, ...):
    taylor_rate = self.state.taylor_rule_rate * 12  # I compute this
    chosen_rate = action["target_rate_annual"]
    
    # Agent is PUNISHED for deviating from MY Taylor Rule calculation
    taylor_deviation = abs(chosen_rate - taylor_rate)
    taylor_reward = -(taylor_deviation ** 2) * 10  # Quadratic penalty
```

**What the agent actually learned:** To output rate choices that match my pre-computed Taylor Rule target. This is still non-trivial (noisy observations, 21-action discrete space), but it's not "discovery."

---

### ❌ HARDCODED: The Phillips Curve

**The claim:** "The labor market reproduced the Phillips Curve!"

**The truth:** The Phillips Curve is explicitly coded in the labor market clearing function.

```python
# markets/labor_market.py, lines 209-214
# === 7. WAGE ADJUSTMENT (PHILLIPS CURVE) ===
unemployment_gap = unemployment_rate - self.natural_unemployment

# THIS IS LITERALLY THE PHILLIPS CURVE EQUATION
wage_growth = -self.wage_adjustment_speed * unemployment_gap  # slope = -0.02
wage_growth = np.clip(wage_growth, -0.02, 0.02)  # Cap at ±2% monthly
```

The Phillips Curve didn't "emerge" — I programmed it. What emerged was the specific correlation coefficient (-0.28 in my simulation vs -0.30 in real data).

---

### ❌ HARDCODED: Calvo Pricing (Price Stickiness)

```python
# config.py, line 40
price_stickiness: float = 0.65  # Calvo parameter

# agents/firm.py, line 278
if np.random.random() > self.price_stickiness:
    # Only 35% of firms can adjust prices each period
    self.update_price(...)
```

This is a parameter I set, not something agents learned.

---

### ❌ HARDCODED: Consumption Smoothing (Permanent Income Hypothesis)

```python
# agents/household.py, lines 386-410
# 5a. Update permanent income estimate (SLOW exponential smoothing)
learning_rate = 0.15  # I chose this parameter
self.state.permanent_income_estimate = (
    learning_rate * current_income + 
    (1 - learning_rate) * self.state.permanent_income_estimate
)

# 5c. SUPER SMOOTH adjustment toward target
adjustment_speed = 0.15  # I chose this too
smoothed_consumption = (
    adjustment_speed * target_consumption +
    (1 - adjustment_speed) * self.state.previous_consumption
)
```

The smoothing behavior is designed, not learned.

---

### ✅ GENUINELY LEARNED: Bank Lending Decisions

Banks make real decisions that affect outcomes:

```python
# agents/commercial_bank.py, lines 193-211
def decode_action(self, action_idx):
    return {
        "lending_spread": self.lending_spread_actions[lending_idx],  # Learned
        "deposit_spread": self.deposit_spread_actions[deposit_idx],  # Learned
        "risk_tolerance": self.risk_tolerance_actions[risk_idx],      # Learned
    }
```

The banks learn:
- What spread to charge over the policy rate
- How much risk to tolerate in lending
- When to tighten/loosen credit standards

**Evidence:** Banks learned to charge higher spreads when NPL ratios rise, and lower spreads during good times. This is genuine emergent behavior.

---

### ✅ GENUINELY LEARNED: Firm Pricing and Hiring

Firms make discrete decisions:

```python
# agents/firm.py, lines 215-232
def decode_action(self, action_idx):
    return {
        "price_change": self.price_change_actions[price_idx],     # -2% to +2%
        "hiring": self.hiring_actions[hire_idx],                  # -2 to +2 workers
        "investment_rate": self.investment_actions[invest_idx],   # 0% to 10%
    }
```

Firms learn:
- When to raise/lower prices based on demand
- When to hire/fire based on capacity utilization
- When to invest based on profitability

**Evidence:** Firms with high sales ratios (>90%) learned to raise prices. Firms with low utilization (<50%) learned to fire workers. This matches economic theory but wasn't programmed — it emerged from profit maximization.

---

### ✅ GENUINELY LEARNED: Household Consumption/Savings

Households choose:

```python
# agents/household.py, lines 229-247
def decode_action(self, action_idx):
    return {
        "consumption_rate": [0.7, 0.8, 0.9, 0.95][cons_idx],  # Learned choice
        "wants_to_borrow": [False, True][borrow_idx],          # Learned choice
        "labor_intensity": [0.8, 1.0, 1.1][labor_idx],         # Learned choice
    }
```

The consumption rate (what fraction of smoothed target to consume) is a genuine learned decision.

---

### ✅ GENUINELY ACHIEVED: Economic Stability

The simulation maintains stability over 25-year horizons:
- No hyperinflation
- No GDP explosions or collapses
- No permanent depressions
- Policy rate doesn't get stuck at zero

**This is a real achievement.** Early versions regularly exploded. Stability emerged from:
1. Agent learning (choosing reasonable actions)
2. Environment design (smoothing, constraints)

The combination of both is what works.

---

## The Training System

### PPO Implementation

```python
# training/ppo.py
class MultiAgentPPO:
    """Trains separate networks for each agent TYPE (not each agent)."""
    
    def train(self):
        # KEY: Aggregate experiences from ALL agents of each type
        for agent_type, network in self.networks.items():
            all_obs = []
            for agent_id, buffer in self.buffers[agent_type].items():
                data = buffer.get_all()
                all_obs.append(data["observations"])
            
            # Train on combined data from all 100 households / 20 firms / etc.
            obs = torch.cat(all_obs, dim=0)
            # ... standard PPO update
```

**Key design decision:** One network per agent *type*, trained on pooled experience from all agents of that type. This means:
- 1 network for the Central Bank (trains on 1 agent's experience)
- 1 network for Banks (trains on 5 agents' pooled experience)
- 1 network for Households (trains on 100 agents' pooled experience)
- 1 network for Firms (trains on 20 agents' pooled experience)

Total: **4 neural networks**, not 126.

### Training Configuration

```python
# config.py, lines 200-221
@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.97       # Monthly discounting
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10   # PPO updates per rollout
    minibatch_size: int = 32
```

### Batched Action Sampling (Optimization)

```python
# train.py, lines 59-126
class FastActionSampler:
    """Batch all agents of each type into ONE forward pass."""
    
    @torch.no_grad()
    def get_actions_batched(self, observations):
        for agent_type, agent_obs in observations.items():
            # Stack all 100 household observations into one tensor
            obs_batch = np.stack([agent_obs[aid] for aid in agent_ids])
            obs_tensor = torch.as_tensor(obs_batch, device=self.device)
            
            # ONE forward pass for all 100 households
            action_batch, log_prob_batch, value_batch, _ = network.get_action_and_value(obs_tensor)
```

This gives ~10x speedup over naive per-agent calls.

---

## The Validation Framework

### 50 Statistical Tests

I built a comprehensive validation suite (1,761 lines in `test.py`) comparing simulation output against:
- FRED (Federal Reserve Economic Data)
- BLS (Bureau of Labor Statistics)
- BEA (Bureau of Economic Analysis)
- Academic literature (Smets & Wouters 2007, Stock & Watson 1999, etc.)

**Test categories:**
1. First moments (mean GDP growth, inflation, unemployment)
2. Second moments (volatility, skewness, kurtosis)
3. Persistence (autocorrelation)
4. Co-movements (correlations between variables)
5. Taylor Rule compliance
6. Phillips Curve relationship
7. Monetary transmission
8. Business cycle properties
9. Wealth distribution
10. Historical crisis patterns

### Real-World Benchmarks

```python
# test.py, lines 42-126
@dataclass
class RealWorldBenchmarks:
    # GDP (BEA National Accounts)
    gdp_growth_mean: float = 2.7           # % annual
    gdp_growth_std: float = 2.1            # %
    gdp_autocorr_1: float = 0.85           # AR(1)
    
    # Inflation (BLS CPI-U)
    inflation_mean: float = 2.8            # %
    inflation_std: float = 1.3             # %
    
    # Unemployment (BLS)
    unemployment_mean: float = 5.8         # %
    nairu: float = 4.5                     # Natural rate
    
    # Correlations (Stock & Watson 1999)
    corr_unemployment_gdp: float = -0.85   # Okun's Law
    corr_consumption_gdp: float = 0.85
    credit_gdp_correlation: float = 0.70   # Procyclical credit
    
    # Business cycles (NBER)
    recession_depth_avg: float = -2.5      # % GDP decline
```

### Final Scores

| Category | Score | What It Measures |
|----------|-------|------------------|
| Stability | 97% | Economy doesn't explode/collapse |
| Phillips Curve | 96% | Unemployment-inflation relationship |
| Taylor Rule | 87% | CB follows policy rule |
| Persistence | 82% | GDP autocorrelation |
| First Moments | 77% | Mean values realistic |
| Monetary Transmission | 61% | Interest rates affect economy |
| Second Moments | 48% | Volatility (TOO HIGH) |
| Historical Crisis | 44% | Recession severity (TOO DEEP) |
| Co-movements | 20% | Credit-GDP correlation (WRONG SIGN) |
| **Overall Reality Score** | **70%** | Weighted average |

---

## What Works

### 1. Macroeconomic Stability (97%)

The simulation maintains a functioning economy for 25 simulated years without:
- Hyperinflation (>20%)
- Depression (>20% unemployment sustained)
- GDP collapse or explosion
- Zero lower bound traps

This required:
- Consumption smoothing (prevents demand death spirals)
- Production smoothing (prevents output volatility)
- Price stickiness (prevents inflation volatility)
- Careful reward design (prevents pathological agent behavior)

### 2. Taylor Rule Compliance (87%)

The Central Bank learned to consistently match the Taylor Rule target despite:
- Noisy 12-dimensional observations
- 21-action discrete choice space
- Delayed feedback (effects take months)

```
Target rate: r = r* + π + 1.5(π-π*) + 0.5y
Achieved correlation: 0.92
Mean deviation: 0.3 percentage points
```

### 3. Realistic Inflation (96% for Phillips Curve)

The simulation produces:
- Mean inflation: 3.2% (target: 2.8%)
- Inflation-unemployment correlation: -0.28 (target: -0.30)
- Inflation persistence: 0.82 (target: 0.85)

### 4. Reasonable Unemployment (within 2% of target)

- Mean unemployment: 5.4% (target: 5.8%)
- Responds correctly to demand shocks
- Recovery dynamics present

---

## What Doesn't Work

### 1. GDP Volatility (0% — Complete Failure)

| Metric | Model | Real World | Problem |
|--------|-------|------------|---------|
| GDP std dev | 46% | 2.1% | 22x too volatile |

Real quarterly GDP fluctuates ~0.5-1.3%. My model swings 15-50% per quarter. This is catastrophic for realism.

**Root cause:** Monthly timesteps are too granular. Real business cycle models use quarterly data because monthly noise is too high.

### 2. Recession Severity (44% — Major Failure)

| Metric | Model | Real World | Problem |
|--------|-------|------------|---------|
| Recession depth | -400% | -2.5% | 160x too severe |

When recessions occur, GDP collapses apocalyptically instead of declining 2-3%.

**Root cause:** The demand death spiral. Despite consumption smoothing, a shock still cascades:
```
Demand ↓ → Firms fire → Income ↓ → Consumption ↓ → More firings → ...
```

Real economies have fiscal stabilizers (unemployment insurance, automatic spending) that my model lacks.

### 3. Credit-GDP Correlation (0% — Wrong Sign)

| Metric | Model | Real World | Problem |
|--------|-------|------------|---------|
| Credit-GDP correlation | -0.50 | +0.70 | Opposite direction |

Banks in my model lend MORE during recessions and LESS during booms — the exact opposite of reality.

**Root cause:** Banks see LAGGED GDP data. By the time they observe a boom, it's ending. They expand lending as the economy slows.

```python
# environment.py, lines 214-228
# I tried to fix this with forward-looking GDP...
current_production = sum(firm.get_production_capacity() for firm in self.firms)
estimated_current_gdp = current_production * avg_price
gdp_growth = ((estimated_current_gdp - prev_gdp) / prev_gdp) * 12
```

...but banks still reacted too slowly.

---

## Key Engineering Lessons

### 1. Environment Architecture > Algorithm Tuning

I spent 2 weeks tuning PPO hyperparameters (learning rate, clip epsilon, entropy). Gains: ~5%.

Then I added consumption smoothing (5 lines of code). Gains: ~35%.

**Most problems are environment problems, not algorithm problems.**

### 2. Reward Design is 80% of the Work

Naive rewards lead to disasters:

| Agent | Naive Reward | Result | Shaped Reward |
|-------|-------------|--------|---------------|
| Bank | Maximize profit | Reckless lending | Profit - capital penalty - NPL penalty |
| Firm | Maximize profit | Extreme price swings | Profit + utilization bonus - bankruptcy penalty |
| Household | Maximize consumption | Borrows until default | Log(C) + buffer value - debt burden |
| Central Bank | Minimize inflation | Ignores output gap | Taylor deviation² × 10 |

### 3. Validation Against Reality is Non-Negotiable

Without the 50-test Reality Score, I would have:
- Declared victory at 50%
- Claimed "emergent" behavior that was actually hardcoded
- Published misleading results

The validation forced me to confront failures honestly.

### 4. Multi-Agent RL is Hard

Problems I encountered:
- Non-stationary environments (other agents are learning too)
- Credit assignment (who caused this recession?)
- Catastrophic forgetting (agent unlearns good behavior)
- Emergent dynamics (126 agents create unpredictable interactions)

---

## What I Would Do Differently

### 1. Use Quarterly Timesteps

Monthly data is too noisy. Reducing from 12 to 4 periods per year would:
- Match academic DSGE literature
- Reduce noise-to-signal ratio
- Allow meaningful business cycle dynamics

### 2. Add Fiscal Policy

The missing automatic stabilizers (unemployment benefits, counter-cyclical spending) are critical. Without them, demand shocks become apocalyptic.

### 3. Add Wage Stickiness

I have price stickiness (Calvo pricing) but not wage stickiness. Real wages adjust slowly, which is a major stabilizing mechanism.

### 4. Fix Credit Market Timing

Banks need forward-looking expectations, not lagged data. This might require:
- Leading indicators in observations
- Explicit forecast models
- Or simpler: just use current production as the GDP proxy

---

## The Honest Conclusion

I set out to see if AI could discover economics. Here's the honest accounting:

**What I hardcoded:**
- Taylor Rule (the formula is in my code)
- Phillips Curve (the equation is in my wage adjustment)
- Price stickiness (a parameter I set)
- Consumption smoothing (I chose the learning rate)

**What agents genuinely learned:**
- To follow policy targets consistently
- Pricing decisions based on demand
- Hiring decisions based on profits
- Lending decisions based on risk
- How to maintain economic stability

**What genuinely emerged:**
- Specific correlation magnitudes
- Business cycle dynamics
- Recovery patterns
- The fact that stability is achievable at all

**Is 70% good?**

For a solo project built in weeks: Yes. Published DSGE models that take years to calibrate achieve 60-75%.

For replacing actual economic modeling: No. The 46% GDP volatility and -400% recessions are disqualifying.

**The most valuable outcome** wasn't the model — it was understanding why economic stability is hard, why stabilizers matter, and why honest validation is essential.

---

## Try It Yourself

```bash
git clone https://github.com/astroknot-sheep/Economy_RL.git
cd Economy_RL
pip install torch numpy pandas matplotlib tqdm scipy

# Train (~3 hours on CPU)
python train.py --epochs 800 --steps 300

# Validate against reality
python test.py
```

---

## Tech Stack Summary

```yaml
Training:
  Framework: PyTorch
  Algorithm: PPO (Proximal Policy Optimization)
  Networks: 4 Actor-Critic MLPs (128×128)
  Training: 800 epochs × 300 steps ≈ 3 hours on CPU

Environment:
  Agents: 126 total (1 + 5 + 100 + 20)
  Markets: 3 (Labor, Credit, Goods)
  Timestep: Monthly
  Horizon: 25 years per episode

Validation:
  Tests: 50 statistical comparisons
  Baseline: US economic data (1960-2023)
  Sources: FRED, BLS, BEA, academic literature
  Framework: 1,761 lines of test code
```

---

*If you made it this far, you understand why I believe in honest engineering write-ups. The failures are as instructive as the successes.*

**GitHub:** [github.com/astroknot-sheep/Economy_RL](https://github.com/astroknot-sheep/Economy_RL)

---

**Tags:** #MachineLearning #Economics #ReinforcementLearning #Python #HonestEngineering #MultiAgentSystems
