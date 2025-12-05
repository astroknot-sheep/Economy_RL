# 🔬 COMPREHENSIVE CODE AUDIT REPORT
## Economy_RL Codebase - Line-by-Line Review

**Generated:** December 5, 2025  
**Files Reviewed:** 15 core files  
**Total Lines:** ~3,500 lines

---

## 📋 EXECUTIVE SUMMARY

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 CRITICAL | 5 | Will cause incorrect economic behavior |
| 🟠 MAJOR | 11 | Significant issues affecting realism |
| 🟡 MINOR | 8 | Code quality / optimization issues |
| 🟢 INFO | 6 | Suggestions for improvement |

**Overall Health Score: 45/100** ⚠️

---

## 🔴 CRITICAL ISSUES (Must Fix)

### 1. **Broken Okun's Law - Production-Employment Decoupling**
**File:** `agents/firm.py`, Lines 346-391  
**Severity:** CRITICAL

```python
# PROBLEM: Production is based on EXPECTED DEMAND, not LABOR
# Line 388: Production = expected demand + inventory adjustment
desired_production = expected_demand + 0.3 * inventory_gap

# CONSEQUENCE: Firms can produce MORE with FEWER workers
# This breaks Okun's Law (correlation should be -0.85)
```

**Root Cause:** The Cobb-Douglas function is calculated at line 356, but then it's only used as a CEILING, not as the actual production:
```python
max_production = A * (K ** alpha) * (L ** (1 - alpha))  # Line 356
self.state.output = np.clip(desired_production, 0, max_production)  # Line 391
```

**FIX REQUIRED:**
```python
# Production MUST be tied to labor
labor_output = L * self.labor_productivity
capital_constrained = min(labor_output, A * (K ** alpha) * (L ** (1 - alpha)))
# THEN apply demand-side constraints
actual_production = min(capital_constrained, desired_production)
```

---

### 2. **Missing Production Smoothing - 46% Monthly GDP Swings**
**File:** `agents/firm.py`, Lines 388-392  
**Severity:** CRITICAL

```python
# PROBLEM: No inertia in production decisions
# Production jumps instantly to meet demand
desired_production = expected_demand + 0.3 * inventory_gap
self.state.output = np.clip(desired_production, 0, max_production)
```

**Evidence:** Test results show -46% to +49% monthly GDP changes (realistic: ±2%)

**FIX REQUIRED:**
```python
# Add production smoothing
smoothing_factor = 0.7  # 70% inertia
new_production = smoothing_factor * self.state.output + (1 - smoothing_factor) * desired_production
# Cap monthly change at ±10%
max_change = 0.10 * self.state.output
self.state.output = np.clip(new_production, 
                             self.state.output - max_change, 
                             self.state.output + max_change)
```

---

### 3. **Households Don't Update Consumption Before Goods Market**
**File:** `environment.py`, Lines 217-224  
**Severity:** CRITICAL

```python
# Lines 217-221: Household decisions are DECODED but NOT APPLIED
for h in self.households:
    if h.is_active and h.id in actions["households"]:
        action_idx = actions["households"][h.id]
        household_actions[h.id] = h.decode_action(action_idx)

# Line 224: Goods market uses OLD consumption values!
goods_outcome = self.goods_market.clear(self.households, self.firms)
```

**Problem:** Goods market reads `h.state.consumption` which hasn't been updated yet.

**FIX REQUIRED:** Move consumption update BEFORE goods market clearing:
```python
# === 7. HOUSEHOLD DECISIONS - UPDATE CONSUMPTION FIRST ===
for h in self.households:
    if h.is_active and h.id in household_actions:
        # Apply consumption decision before market clearing
        h._update_consumption(household_actions[h.id], market_outcomes)

# === 8. GOODS MARKET CLEARING ===
goods_outcome = self.goods_market.clear(self.households, self.firms)
```

---

### 4. **Labor Market Doesn't Match Workers to Firms Properly**
**File:** `markets/labor_market.py`, Lines 180-188  
**Severity:** CRITICAL

```python
# PROBLEM: Firm worker counts are updated in environment.py, not by actual matching
# Line 180-186 update firm_vacancies, but firm.state.num_workers is set elsewhere

# environment.py Lines 181-186:
for firm in self.firms:
    allocated_workers = len(labor_outcome.firm_workers[firm.id])
    firm.state.num_workers = max(allocated_workers, self.config.economic.min_firm_workers)
```

**Problem:** The number of matched workers `len(firm_workers[firm.id])` may not match what the firm actually needs. The `firm_workers` dict only contains households whose `employer_id` matches, but this doesn't reflect new hires within the same step.

**FIX REQUIRED:** Labor market should directly update firm worker counts and household employer assignments atomically.

---

### 5. **MPC Income Too Low for Monthly Model**
**File:** `config.py`, Line 66  
**Severity:** CRITICAL (Parameter Calibration)

```python
mpc_income: float = 0.07  # Phase 3: Monthly consumption from permanent income (~85% annual)
```

**Problem:** MPC of 0.07 means households consume only 7% of income per month. But with the Permanent Income Hypothesis implementation in `household.py` Lines 398-400:
```python
target_consumption = (
    self.mpc_income * self.state.permanent_income_estimate +
    self.mpc_wealth * max(self.state.savings, 0)
)
```

A monthly MPC of 0.07 means annual consumption is only ~84% of income (correct). But the **problem** is this is then SMOOTHED again at line 406, reducing it further.

**FIX REQUIRED:** Either:
1. Set `mpc_income = 0.85` and remove the averaging in PIH, OR
2. Keep 0.07 but ensure it actually represents monthly spending

---

## 🟠 MAJOR ISSUES (Should Fix)

### 6. **Phillips Curve Broken - Wages Don't Affect Prices**
**File:** `markets/labor_market.py`, Lines 207-216 vs `markets/goods_market.py`

```python
# Labor market: Wages adjust via Phillips curve (Lines 210-214)
wage_growth = -self.wage_adjustment * unemployment_gap
new_wage = self.current_wage * (1 + wage_growth)

# BUT: Goods market price dynamics (goods_market.py Line 159) only respond to:
price_adjustment = (1 - self.price_stickiness) * 0.05 * excess_demand_ratio
# WAGES ARE NOT INPUT TO PRICE DYNAMICS
```

**Result:** No cost-push inflation channel.

**FIX REQUIRED:** Add wage-cost passthrough in goods_market.py:
```python
# Add wage-cost inflation channel
wage_inflation = market_outcomes.get("wage_growth", 0.0)
cost_push = 0.5 * wage_inflation  # 50% passthrough
price_adjustment += cost_push
```

---

### 7. **Interest Rate Transmission Is Weak**
**File:** `agents/household.py`, Lines 484-499

```python
def get_borrowing_demand(self, lending_rate: float) -> float:
    annual_rate = lending_rate * 12
    demand_factor = np.exp(-self.rate_sensitivity * annual_rate)  # Line 494
```

**Problem:** With `rate_sensitivity = 15.0` (config.py Line 69):
- At 5% rates: demand_factor = exp(-15 × 0.05) = 0.47
- At 10% rates: demand_factor = exp(-15 × 0.10) = 0.22

This is ONLY for new borrowing. **Existing consumption isn't rate-sensitive**.

**FIX REQUIRED:** Add interest rate sensitivity to consumption decisions:
```python
# In household update_state, before consumption calculation:
real_rate = lending_rate * 12 - market_outcomes.get("inflation", 0.02)
consumption_adjustment = 1 - 0.5 * real_rate  # 0.5% less consumption per 1% real rate
desired_consumption *= np.clip(consumption_adjustment, 0.8, 1.2)
```

---

### 8. **Central Bank Rate Smoothing Too High**
**File:** `config.py`, Line 57 and `agents/central_bank.py`, Line 265

```python
interest_rate_smoothing: float = 0.75  # config.py

# central_bank.py Line 265:
new_rate = self.smoothing * old_rate + (1 - self.smoothing) * target_rate
```

**Problem:** 75% smoothing means the CB moves only 25% toward target each month. This creates excessive persistence and slow policy response.

**Calculation:** To move from 2% to 5% takes:
- Month 1: 2% + 0.25×(5-2) = 2.75%
- Month 6: ~4.0%
- Month 12: ~4.8%

It takes nearly a year to respond to inflation!

**FIX REQUIRED:** Reduce smoothing to 0.5-0.6, or make it state-dependent:
```python
# More aggressive when far from target
if abs(target_rate - old_rate) > 0.02 / 12:  # >2% deviation
    effective_smoothing = 0.5
else:
    effective_smoothing = 0.75
```

---

### 9. **Firm Investment Decision Ignores Interest Rates**
**File:** `agents/firm.py`, Lines 298-340

```python
# Line 300: Investment decision only uses action
investment_rate = action["investment_rate"]
desired_investment = max(self.state.capital * investment_rate, min_investment)

# PROBLEM: No sensitivity to lending_rate in market_outcomes
```

**Real Economics:** Investment should drop when interest rates rise (user cost of capital).

**FIX REQUIRED:**
```python
lending_rate = market_outcomes.get("lending_rate", 0.005)
# User cost of capital
user_cost = lending_rate + self.depreciation_rate
# Expected return (simplified)
expected_return = self.state.profit / max(self.state.capital, 1.0)
# Only invest if return > cost
if expected_return > user_cost:
    desired_investment = self.state.capital * investment_rate
else:
    desired_investment = min_investment  # Only replacement
```

---

### 10. **No Inventory-Sales Feedback to Production**
**File:** `agents/firm.py`, Lines 360-383

The firm uses `sales_history` to forecast demand, but there's no connection between actual sales and short-run production adjustment.

**FIX REQUIRED:** Add immediate inventory feedback:
```python
# If inventory-to-sales ratio is high, cut production
inv_to_sales = self.state.inventory / max(np.mean(self.state.sales_history[-3:]), 1.0)
if inv_to_sales > 3.0:  # 3 months of inventory
    desired_production *= 0.8  # Cut 20%
elif inv_to_sales < 1.0:  # Low inventory
    desired_production *= 1.1  # Boost 10%
```

---

### 11. **Bank Profits Don't Affect Lending Capacity**
**File:** `agents/commercial_bank.py`, Lines 301-314

```python
# Line 302-304: Net interest income calculated
interest_income = self.state.lending_rate * self.state.performing_loans
interest_expense = self.state.deposit_rate * self.state.deposits
self.state.net_interest_income = interest_income - interest_expense

# Line 313-314: Capital updated
self.state.capital += self.state.profit
```

**Problem:** When profits are negative, capital falls, but there's no mechanism to tighten lending beyond the capital ratio check. Banks should become risk-averse when capital falls.

**FIX REQUIRED:**
```python
# Add pro-cyclical risk adjustment
capital_buffer = self.state.capital_ratio - self.min_capital_ratio
if capital_buffer < 0.02:  # Near minimum
    self.state.risk_tolerance *= 0.5  # Become very cautious
```

---

### 12. **No Unemployment Benefits Exhaustion Tracking**
**File:** `agents/household.py`, Lines 337-341

```python
if self.state.benefit_months_remaining > 0:
    self.state.labor_income = skilled_wage * self.benefit_rate
else:
    self.state.labor_income = 0.0
```

**Problem:** When benefits run out, income goes to exactly 0, creating a consumption cliff. Real economies have social safety nets.

**FIX REQUIRED:**
```python
if self.state.benefit_months_remaining > 0:
    self.state.labor_income = skilled_wage * self.benefit_rate
else:
    # Welfare floor
    self.state.labor_income = self.config.economic.base_wage * 0.2
```

---

### 13. **Observation Sizes Don't Match Actual Observations**
**File:** `config.py` Lines 190-193 vs actual agent observations

```python
cb_input_size: int = 12      # config
bank_input_size: int = 12    
household_input_size: int = 15
firm_input_size: int = 15
```

**Verification:**
- Central Bank: `to_array()` returns 12 elements ✓
- Bank: `get_observation()` returns 6 + 6 = 12 elements ✓
- Household: `to_array()` returns 8 elements + 7 macro = 15 ✓
- Firm: `to_array()` returns 8 elements + 7 macro = 15 ✓

**Status:** OK ✓

---

### 14. **Matching Function Creates Too Much Unemployment Volatility**
**File:** `markets/labor_market.py`, Lines 147-150

```python
job_finding_rate = min(
    self.matching_efficiency * np.sqrt(tightness),
    0.4  # Cap at 40% per month
)
```

**Problem:** No floor on job finding rate. When labor market is slack (low tightness), matching can go to nearly zero, causing unemployment spikes.

**FIX REQUIRED:**
```python
job_finding_rate = np.clip(
    self.matching_efficiency * np.sqrt(tightness),
    0.05,  # Minimum 5% matching rate (frictional floor)
    0.4   # Maximum 40%
)
```

---

### 15. **TFP Growth Compounds Exponentially Without Bounds**
**File:** `agents/firm.py`, Line 344

```python
self.state.productivity *= (1 + self.tfp_growth_rate)
```

**Problem:** Over 30 years (360 months), productivity grows to:
- (1 + 0.02/12)^360 = 1.82×

This is fine, but if simulations run longer or parameter is changed, it could explode.

**FIX REQUIRED:**
```python
max_tfp = self.labor_productivity * 5.0  # Cap at 5x initial
self.state.productivity = min(
    self.state.productivity * (1 + self.tfp_growth_rate),
    max_tfp
)
```

---

### 16. **No Credit Crunch Mechanism**
**File:** `markets/credit_market.py`

When banks have losses, they should tighten lending. Currently, the only check is the capital ratio (Line 346-348 in `commercial_bank.py`):

```python
def can_lend(self, amount: float) -> bool:
    projected_ratio = self.state.capital / (self.state.loans + amount)
    return projected_ratio >= self.min_capital_ratio
```

**FIX REQUIRED:** Add credit crunch dynamics when losses are high:
```python
# In credit_market.py, before loan decisions:
for bank in active_banks:
    if bank.state.profit < 0:
        bank.state.risk_tolerance *= 0.9  # Become more cautious
```

---

## 🟡 MINOR ISSUES

### 17. **Hardcoded Magic Numbers**
**Files:** Multiple

Examples:
- `goods_market.py` Line 122: `elasticity = 2.0`
- `goods_market.py` Line 165: `alpha = 0.3`
- `household.py` Line 346: `capital_return_rate = 0.06 / 12`
- `central_bank.py` Line 224: `if taylor_deviation < 0.005`

**FIX:** Move all to `config.py`.

---

### 18. **Inefficient History Trimming**
**File:** `agents/firm.py`, Lines 360-361

```python
if len(self.state.sales_history) > 6:
    self.state.sales_history.pop(0)  # O(n) operation
```

**FIX:** Use `collections.deque(maxlen=6)` for O(1) operations.

---

### 19. **No Type Hints on Some Functions**
**Files:** `economics/aggregates.py`, `markets/*.py`

Some functions missing type hints, reducing IDE support.

---

### 20. **Unused Imports and Variables**
**Files:** Various

- `agents/firm.py`: `expected_demand` variable shadowing
- Some imported but unused typing generics

---

### 21. **No Docstrings for Many Methods**
**Files:** `RolloutBuffer` methods, some agent methods

---

### 22. **Buffer Overflow Not Handled Gracefully**
**File:** `training/buffer.py`, Line 62-64

```python
self.pos += 1
if self.pos >= self.buffer_size:
    self.full = True
    self.pos = 0  # Wraps around, overwriting old data
```

This is correct behavior but should be documented.

---

### 23. **Household Skill Level Not Used in Job Matching**
**File:** `markets/labor_market.py`

Households have `skill_level` but matching is random. Higher-skilled workers should match faster.

---

### 24. **No Load Balance/Checkpointing During Training**
**File:** `train.py`

Long training runs have no resume capability if interrupted.

---

## 🟢 INFO / SUGGESTIONS

1. **Consider using PyTorch's `torch.jit.script` for network inference speedup**

2. **Add unit tests for each agent's `update_state` method**

3. **Consider adding a "shock" mechanism for stress testing (oil price shock, financial crisis, etc.)**

4. **The Gini coefficient calculation could use scipy.stats.gini for better performance**

5. **Consider adding an investment-goods sector for more realistic dynamics**

6. **Add logging with proper log levels for debugging**

---

## 📊 PRIORITY MATRIX

| Issue | Impact | Effort | Priority |
|-------|--------|--------|----------|
| #1 Okun's Law | HIGH | MEDIUM | P0 |
| #2 Production Smoothing | HIGH | LOW | P0 |
| #3 Consumption Timing | HIGH | LOW | P0 |
| #6 Phillips Curve | HIGH | MEDIUM | P1 |
| #8 Rate Smoothing | MEDIUM | LOW | P1 |
| #9 Investment Rates | MEDIUM | MEDIUM | P1 |
| #14 Matching Floor | MEDIUM | LOW | P1 |
| #4 Labor Matching | MEDIUM | HIGH | P2 |
| #5 MPC Calibration | MEDIUM | LOW | P2 |
| #7 Rate Transmission | MEDIUM | MEDIUM | P2 |

---

## 🛠️ RECOMMENDED FIX ORDER

1. **Day 1:** Fix Production Smoothing (#2) - prevents 46% GDP swings
2. **Day 1:** Fix Consumption Timing (#3) - ensures demand affects supply
3. **Day 2:** Fix Okun's Law (#1) - tie production to labor
4. **Day 2:** Add Matching Floor (#14) - prevents unemployment spikes
5. **Day 3:** Fix Phillips Curve (#6) - add wage-price link
6. **Day 3:** Reduce Rate Smoothing (#8) - faster policy response
7. **Day 4:** Add Investment Rate Sensitivity (#9)
8. **Day 4:** Calibrate MPC (#5)

**Estimated time:** 4 developer-days for critical fixes

---

## ✅ WHAT'S WORKING WELL

1. **Clean code architecture** - Clear separation of agents, markets, economics
2. **Comprehensive NaN guards** - All observations have `np.nan_to_num`
3. **Good PPO implementation** - Proper advantage normalization, clipping
4. **Correct Taylor Rule formula** - CB has the right equation
5. **Calvo pricing structure** - Proper sticky price setup
6. **Wealth heterogeneity** - Good Pareto/log-normal initialization

---

*Report generated by comprehensive codebase audit.*
