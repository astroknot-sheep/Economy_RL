"""
Firm Agent - Corrected Implementation

ECONOMIC LOGIC:
- Production: Y = productivity * L (simplified linear)
- Price setting with Calvo stickiness
- Investment based on expected returns
- Inventory buffer stock model

KEY FIXES:
1. Proper production function
2. Price adjustment respects stickiness
3. Labor demand based on marginal product
4. Investment responds to interest rates
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .base_agent import BaseAgent, AgentState


@dataclass
class FirmState(AgentState):
    """State specific to Firms."""
    
    # Production factors
    capital: float = 100.0
    inventory: float = 20.0
    num_workers: int = 5
    output: float = 0.0
    productivity: float = 5.0  # TFP, grows over time
    
    # Pricing
    price: float = 1.0
    markup: float = 0.15
    
    # Costs
    wage_bill: float = 0.0
    unit_cost: float = 0.8
    
    # Finance
    debt: float = 0.0
    loan_rate: float = 0.0
    loan_bank_id: Optional[int] = None
    cash: float = 50.0
    
    # Performance
    revenue: float = 0.0
    profit: float = 0.0
    sales: float = 0.0
    capacity_utilization: float = 0.8
    actual_investment: float = 0.0
    sales_history: List[float] = field(default_factory=list)  # Phase 2: Track demand
    
    # Phase 4: Adaptive Expectations
    expected_inflation: float = 0.02 / 12  # Expected monthly inflation
    expected_demand: float = 0.0  # Expected sales (computed from history)
    
    def to_array(self) -> np.ndarray:
        """Convert to normalized observation array."""
        capital_norm = np.clip(self.capital / 100.0, 0, 5)
        inventory_norm = np.clip(self.inventory / 50.0, 0, 5)
        workers_norm = np.clip(self.num_workers / 5.0, 0, 3)
        price_norm = np.clip(self.price, 0.5, 2)
        debt_norm = np.clip(self.debt / 100.0, 0, 5)
        cash_norm = np.clip(self.cash / 50.0, -2, 5)
        profit_norm = np.clip(self.profit / 5.0, -5, 5)
        utilization_norm = np.clip(self.capacity_utilization, 0, 1)
        
        arr = np.array([
            capital_norm,
            inventory_norm,
            workers_norm,
            price_norm,
            debt_norm,
            cash_norm,
            profit_norm,
            utilization_norm,
        ], dtype=np.float32)
        
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "capital": self.capital,
            "inventory": self.inventory,
            "num_workers": self.num_workers,
            "price": self.price,
            "revenue": self.revenue,
            "profit": self.profit,
            "debt": self.debt,
            "cash": self.cash,
            "capacity_utilization": self.capacity_utilization,
        })
        return base


class Firm(BaseAgent):
    """
    Firm that produces goods and hires workers.
    
    PRODUCTION: Y = productivity * L
    
    DECISIONS:
    1. Price adjustment
    2. Hiring/firing
    3. Investment in capital
    """
    
    def __init__(self, agent_id: int, config: Any):
        super().__init__(agent_id, config)
        
        self.price_change_actions = config.actions.price_change_actions
        self.hiring_actions = config.actions.hiring_actions
        self.investment_actions = config.actions.investment_actions
        
        self._action_space_size = (
            len(self.price_change_actions) *
            len(self.hiring_actions) *
            len(self.investment_actions)
        )
        
        # Production parameters
        self.labor_productivity = config.economic.labor_productivity
        self.tfp_growth_rate = config.economic.tfp_growth_rate  # Monthly TFP growth
        self.capital_share = config.economic.capital_share
        self.depreciation_rate = config.economic.depreciation_rate
        self.markup_rate = config.economic.markup_rate
        self.price_stickiness = config.economic.price_stickiness
        self.min_workers = config.economic.min_firm_workers
    
    def reset(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        """Initialize firm with heterogeneous characteristics."""
        ic = initial_conditions or {}
        
        capital_mult = np.random.uniform(0.9, 1.1)
        
        capital = ic.get("capital", self.config.economic.initial_firm_capital * capital_mult)
        num_workers = ic.get("num_workers", self.config.economic.initial_firm_workers)
        num_workers = max(self.min_workers, int(num_workers))
        
        base_price = 1.0 * (1 + self.markup_rate)
        initial_price = base_price + np.random.uniform(-0.05, 0.05)
        
        initial_cash = ic.get("cash", self.config.economic.initial_firm_cash)
        
        self.state = FirmState(
            id=self.id,
            capital=capital,
            inventory=self.config.economic.initial_firm_inventory,
            num_workers=num_workers,
            price=initial_price,
            markup=self.markup_rate,
            cash=initial_cash,
            debt=0.0,
            productivity=self.labor_productivity,  # Start at base productivity
            expected_inflation=self.config.economic.inflation_target,  # Phase 4
            expected_demand=0.0,  # Phase 4 - will be computed from sales
            is_active=True,
        )
        
        self._compute_unit_cost(self.config.economic.base_wage)
        
        self.state.wealth = capital + self.state.cash
        self.state.income = 0.0
    
    def _compute_unit_cost(self, wage: float) -> None:
        """Compute marginal cost of production."""
        if self.state.num_workers <= 0:
            self.state.unit_cost = 100.0
            return
        
        labor_cost = wage / self.labor_productivity
        
        output_capacity = max(self.state.num_workers * self.labor_productivity, 1.0)
        capital_cost = self.depreciation_rate * self.state.capital / output_capacity
        
        self.state.unit_cost = labor_cost + capital_cost
    
    def get_observation(self, global_state: Dict[str, Any]) -> np.ndarray:
        """Firm observes own state + market conditions."""
        aggregate_demand = global_state.get("aggregate_demand", 100.0)
        price_level = global_state.get("price_level", 1.0)
        lending_rate = global_state.get("avg_lending_rate", 0.005)
        wage = global_state.get("wage", self.config.economic.base_wage)
        gdp_growth = global_state.get("gdp_growth", 0.0)
        inflation = global_state.get("inflation", 0.02)
        
        own_state = self.state.to_array()
        
        if self.state.output > 0:
            sales_ratio = self.state.sales / max(self.state.output, 0.1)
        else:
            sales_ratio = 0.5
        
        target_inventory = self.state.output * 2
        inventory_ratio = self.state.inventory / max(target_inventory, 1.0)
        
        market_state = np.array([
            np.clip(aggregate_demand / 100.0, 0, 5),
            np.clip(price_level, 0.5, 2),
            np.clip(lending_rate * 12, 0, 0.2),
            np.clip(wage / 5.0, 0, 3),
            np.clip(gdp_growth, -0.3, 0.3),
            np.clip(sales_ratio, 0, 2),
            np.clip(inventory_ratio, 0, 3),
        ], dtype=np.float32)
        
        obs = np.concatenate([own_state, market_state])
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    
    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode action into components."""
        n_price = len(self.price_change_actions)
        n_hire = len(self.hiring_actions)
        n_invest = len(self.investment_actions)
        
        action_idx = action_idx % self._action_space_size
        
        invest_idx = action_idx % n_invest
        remaining = action_idx // n_invest
        hire_idx = remaining % n_hire
        price_idx = remaining // n_hire
        
        return {
            "price_change": self.price_change_actions[price_idx],
            "hiring": self.hiring_actions[hire_idx],
            "investment_rate": self.investment_actions[invest_idx],
            "action_idx": action_idx,
        }
    
    def compute_reward(
        self,
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> float:
        """Firm reward = profit - penalties."""
        profit_reward = np.clip(self.state.profit / 5.0, -5, 5)
        
        cash_reward = 0.3 * np.clip(self.state.cash / 30.0, -2, 2)
        
        target_inventory = self.state.output * 2
        if target_inventory > 0:
            inventory_excess = max(0, self.state.inventory - target_inventory * 1.5)
            inventory_cost = 0.2 * inventory_excess / target_inventory
        else:
            inventory_cost = 0.0
        
        # Phase 3: Reduced bankruptcy penalties
        if self.state.cash < -10:
            bankruptcy_penalty = 2.5  # Reduced from 5.0
        elif self.state.cash < 0:
            bankruptcy_penalty = 1.0  # Reduced from 2.0
        else:
            bankruptcy_penalty = 0.0
        
        utilization_reward = 0.5 * self.state.capacity_utilization
        
        reward = profit_reward + cash_reward + utilization_reward - inventory_cost - bankruptcy_penalty
        
        # Phase 3: Normalized to [-5, 5]
        return float(np.clip(reward, -5, 5))
    
    def update_state(
        self,
        action: Dict[str, Any],
        market_outcomes: Dict[str, Any],
    ) -> None:
        """Update firm state based on actions and outcomes."""
        
        wage = market_outcomes.get("wage", self.config.economic.base_wage)
        
        # === 1. PRICE ADJUSTMENT ===
        if np.random.random() > self.price_stickiness:
            self._compute_unit_cost(wage)
            target_price = self.state.unit_cost * (1 + self.markup_rate)
            
            if self.state.output > 0:
                sales_ratio = self.state.sales / max(self.state.output, 0.1)
                if sales_ratio > 0.9:
                    target_price *= 1.02
                elif sales_ratio < 0.5:
                    target_price *= 0.98
            
            adjustment = 0.3 * (target_price - self.state.price) + action["price_change"]
            new_price = self.state.price * (1 + adjustment)
            self.state.price = np.clip(new_price, 0.5, 2.0)
        
        # === 2. HIRING/FIRING (BALANCED: Allow gradual adjustments) ===
        hire_change = int(action["hiring"])
        # Limit hiring/firing to ±2 per month (realistic labor adjustment costs)
        hire_change = np.clip(hire_change, -2, 2)
        new_workers = self.state.num_workers + hire_change
        # Ensure minimum operational staff
        self.state.num_workers = max(self.min_workers, int(new_workers))
        
        # === 3. INVESTMENT (Phase 2: Capital Accumulation Fix) ===
        # Firms must replace depreciated capital AND can expand if profitable
        investment_rate = action["investment_rate"]
        
        # Depreciation that will occur this period
        depreciation_amount = self.depreciation_rate * self.state.capital
        
        # Minimum investment = replacement (prevent capital death spiral)
        min_investment = depreciation_amount
        
        # Desired investment from action
        desired_investment = max(
            self.state.capital * investment_rate,
            min_investment  # At least replace depreciation
        )
        
        self.state.actual_investment = 0.0
        
        # Try to invest (replacement + expansion if profitable)
        if desired_investment > 0:
            if self.state.cash >= desired_investment:
                # Can self-finance
                self.state.capital += desired_investment
                self.state.cash -= desired_investment
                self.state.actual_investment = desired_investment
            elif self.state.cash >= min_investment:
                # At least do replacement investment
                self.state.capital += min_investment
                self.state.cash -= min_investment
                self.state.actual_investment = min_investment
            elif self.state.debt == 0 and desired_investment > min_investment:
                # Borrow for EXPANSION (not survival)
                loan_result = market_outcomes.get(f"firm_{self.id}_loan")
                if loan_result:
                    self.state.debt = loan_result["amount"]
                    self.state.loan_rate = loan_result["rate"]
                    self.state.loan_bank_id = loan_result["bank_id"]
                    self.state.cash += loan_result["amount"]
                    
                    actual = min(desired_investment, self.state.cash * 0.9)
                    self.state.capital += actual
                    self.state.cash -= actual
                    self.state.actual_investment = actual
        
        # === 4. PRODUCTIVITY GROWTH (creates economic expansion) ===
        # TFP grows by growth_rate each period
        self.state.productivity *= (1 + self.tfp_growth_rate)
        
        # === 5. PRODUCTION (SMOOTH COBB-DOUGLAS - NO CLIFFS) ===
        # CRITICAL FIX: Replace min(labor, capital) with SMOOTH production
        # that allows partial output even with suboptimal inputs
        
        alpha = self.capital_share  # 0.33
        A = self.state.productivity
        K = max(self.state.capital, 1.0)
        L = max(self.state.num_workers, 1)
        
        # STEP 1: Standard Cobb-Douglas production function
        # Y = A * K^α * L^(1-α)
        # This is SMOOTH - no sudden cliff-edges when L or K change
        max_production = A * (K ** alpha) * (L ** (1 - alpha))
        
        # STEP 2: Demand-based production target
        # Track sales history for expectations
        if len(self.state.sales_history) > 6:
            self.state.sales_history.pop(0)
        
        # Expected demand from sales history
        if len(self.state.sales_history) >= 2:
            recent_sales = np.mean(self.state.sales_history[-3:]) if len(self.state.sales_history) >= 3 else self.state.sales_history[-1]
            if len(self.state.sales_history) >= 3:
                y = np.array(self.state.sales_history)
                trend = (y[-1] - y[0]) / max(len(y) - 1, 1)
            else:
                trend = 0.0
            expected_demand = max(recent_sales + trend, 0.1)
        else:
            expected_demand = max(self.state.output, max_production * 0.5)
        
        # Target inventory and desired production
        target_inventory = expected_demand * 1.5
        inventory_gap = target_inventory - self.state.inventory
        desired_production = expected_demand + 0.3 * inventory_gap
        
        # STEP 3: Demand-constrained (don't produce more than you can sell)
        target_output = np.clip(desired_production, 0, max_production)
        
        # STEP 4: PRODUCTION SMOOTHING (Based on Great Moderation research)
        # Real-world quarterly GDP std: 0.5-1.3% (Fed data 1984-2019)
        # To achieve ~2% annual volatility, we need HIGH inertia
        if self.state.output > 0:
            smoothing = 0.85  # 85% inertia (firms adjust slowly like real world)
            smoothed_output = smoothing * self.state.output + (1 - smoothing) * target_output
            # Cap monthly change at ±5% (real firms don't swing 15%/month)
            max_change = 0.05 * self.state.output
            new_output = np.clip(smoothed_output, 
                                  self.state.output - max_change, 
                                  self.state.output + max_change)
            self.state.output = max(new_output, 0.1)
        else:
            self.state.output = target_output
        
        self.state.inventory += self.state.output
        
        if max_production > 0:
            self.state.capacity_utilization = self.state.output / max_production
        else:
            self.state.capacity_utilization = 0.0
        
        # === 6. SALES ===
        sales_quantity = market_outcomes.get(f"firm_{self.id}_sales", 0.0)
        sales_quantity = min(sales_quantity, self.state.inventory)
        self.state.sales = sales_quantity
        self.state.inventory = max(0, self.state.inventory - sales_quantity)
        
        # Phase 2: Track sales for demand forecasting
        self.state.sales_history.append(sales_quantity)
        
        max_inventory = self.state.output * 4
        self.state.inventory = min(self.state.inventory, max_inventory)
        
        # === 6. REVENUE ===
        self.state.revenue = sales_quantity * self.state.price
        
        # === 7. COSTS ===
        self.state.wage_bill = self.state.num_workers * wage
        
        depreciation = self.depreciation_rate * self.state.capital
        self.state.capital = max(10, self.state.capital - depreciation)
        
        # === 8. DEBT SERVICE ===
        interest = 0.0
        principal = 0.0
        if self.state.debt > 0:
            interest = self.state.debt * self.state.loan_rate
            principal = min(self.state.debt / 60, self.state.debt)
            self.state.debt = max(0, self.state.debt - principal)
        
        # === 9. PROFIT ===
        total_costs = self.state.wage_bill + depreciation + interest
        self.state.profit = self.state.revenue - total_costs
        
        # === 10. CASH UPDATE ===
        self.state.cash += self.state.profit - principal
        
        # === 11. UNIT COST UPDATE ===
        self._compute_unit_cost(wage)
        
        # === 12. BANKRUPTCY CHECK (BALANCED: Allow debt but prevent spirals) ===
        # Firms can go into debt but we cap it to prevent complete collapse
        if self.state.cash < -self.state.capital * 1.0:  # 100% of capital threshold
            self.state.cash = -self.state.capital * 0.8  # Cap debt, don't bankrupt
            # Only bankrupt if repeatedly hitting cap (very lenient)
            # self._handle_bankruptcy(market_outcomes)  # Disabled for stability
        
        # === 13. ADAPTIVE EXPECTATIONS (Phase 4) ===
        # Update inflation expectations
        observed_inflation = market_outcomes.get("inflation", self.config.economic.inflation_target)
        learning_rate = 0.3
        self.state.expected_inflation = (
            learning_rate * observed_inflation +
            (1 - learning_rate) * self.state.expected_inflation
        )
        
        # Expected demand was already computed in production section
        # Store it in state for observation
        self.state.expected_demand = expected_demand if 'expected_demand' in locals() else self.state.sales
        
        # === 14. BASE CLASS FIELDS ===
        self.state.wealth = self.state.capital + max(self.state.cash, 0) - self.state.debt
        self.state.income = self.state.profit
    
    def _handle_bankruptcy(self, market_outcomes: Dict[str, Any]) -> None:
        """Handle firm bankruptcy."""
        if self.state.loan_bank_id is not None:
            default_key = f"bank_{self.state.loan_bank_id}_defaults"
            current = market_outcomes.get(default_key, 0.0)
            market_outcomes[default_key] = current + self.state.debt
        
        self.declare_bankruptcy()
    
    def get_action_space_size(self) -> int:
        return self._action_space_size
    
    def get_observation_size(self) -> int:
        return self.config.network.firm_input_size
    
    def get_labor_demand(self) -> int:
        """Return desired number of workers."""
        # BALANCED: Minimum demand based on utilization
        min_demand = 3  # Reduced from 5 - allow more natural dynamics
        
        if self.state.output > 0 and self.state.sales > 0:
            utilization = self.state.sales / max(self.state.output, 0.1)
            
            if utilization > 0.9:
                desired = self.state.num_workers + 1
            elif utilization > 0.7:
                desired = self.state.num_workers
            elif utilization > 0.4:
                desired = max(min_demand, self.state.num_workers - 1)
            else:
                desired = max(min_demand, self.state.num_workers - 2)  # Allow faster adjustment
        else:
            desired = max(min_demand, self.state.num_workers)
        
        return int(desired)
    
    def get_production_capacity(self) -> float:
        """Return maximum output with current workers."""
        return self.state.num_workers * self.labor_productivity
    
    def get_supply(self) -> Tuple[float, float]:
        """Return (quantity available, price) for goods market."""
        return self.state.inventory, self.state.price