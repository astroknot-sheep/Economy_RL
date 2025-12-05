"""
Aggregate Economic Computations
Computes GDP, inflation, unemployment, and other macro variables.

KEY CORRECTIONS:
1. GDP computed correctly as C + I + ΔInventory
2. Output gap computed from production vs potential
3. Proper handling of real vs nominal variables
"""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MacroState:
    """Complete macroeconomic state."""
    
    # Real variables
    gdp: float = 100.0
    gdp_growth: float = 0.0
    output_gap: float = 0.0
    potential_gdp: float = 100.0
    
    # Prices
    inflation: float = 0.02  # Annualized
    price_level: float = 1.0
    
    # Labor market
    unemployment: float = 0.045
    employment: int = 0
    wage: float = 5.0
    
    # Credit
    credit_growth: float = 0.0
    total_credit: float = 0.0
    default_rate: float = 0.025
    
    # Interest rates (monthly)
    policy_rate: float = 0.002
    avg_lending_rate: float = 0.005
    avg_deposit_rate: float = 0.002
    
    # Aggregates
    total_consumption: float = 0.0
    total_investment: float = 0.0
    aggregate_demand: float = 0.0
    aggregate_supply: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "gdp": self.gdp,
            "gdp_growth": self.gdp_growth,
            "output_gap": self.output_gap,
            "potential_gdp": self.potential_gdp,
            "inflation": self.inflation,
            "price_level": self.price_level,
            "unemployment": self.unemployment,
            "employment": self.employment,
            "wage": self.wage,
            "credit_growth": self.credit_growth,
            "total_credit": self.total_credit,
            "default_rate": self.default_rate,
            "policy_rate": self.policy_rate,
            "avg_lending_rate": self.avg_lending_rate,
            "avg_deposit_rate": self.avg_deposit_rate,
            "total_consumption": self.total_consumption,
            "total_investment": self.total_investment,
            "aggregate_demand": self.aggregate_demand,
            "aggregate_supply": self.aggregate_supply,
        }


class AggregateComputer:
    """
    Computes aggregate macroeconomic variables.
    
    GDP ACCOUNTING:
    GDP = C + I + ΔInventory
    """
    
    def __init__(self, config: Any):
        self.config = config
        
        self.gdp_history: List[float] = []
        self.potential_gdp_history: List[float] = []
        
        self.potential_gdp = 100.0
        self.previous_inventory = 0.0
        self.smoothed_inventory_change = 0.0  # Exponential smoothing for stability
    
    def compute(
        self,
        households: List[Any],
        firms: List[Any],
        banks: List[Any],
        labor_outcome: Any,
        credit_outcome: Any,
        goods_outcome: Any,
        policy_rate: float,
    ) -> MacroState:
        """Compute all aggregate variables."""
        
        state = MacroState()
        
        # === MONETARY POLICY ===
        state.policy_rate = policy_rate
        
        # === LABOR MARKET ===
        state.unemployment = labor_outcome.unemployment_rate
        state.employment = labor_outcome.total_employed
        state.wage = labor_outcome.wage
        
        # === CREDIT MARKET ===
        state.avg_lending_rate = credit_outcome.avg_lending_rate
        state.avg_deposit_rate = credit_outcome.avg_deposit_rate
        state.credit_growth = credit_outcome.credit_growth
        state.total_credit = sum(b.state.loans for b in banks if b.is_active)
        
        # === GOODS MARKET ===
        state.price_level = goods_outcome.price_level
        state.inflation = goods_outcome.inflation
        state.aggregate_demand = goods_outcome.aggregate_demand
        state.aggregate_supply = goods_outcome.aggregate_supply
        
        # === GDP COMPUTATION ===
        # GDP = C + I + ΔInventory (but inventory needs smoothing!)
        
        # 1. Final sales (main GDP components)
        state.total_consumption = sum(
            h.state.consumption for h in households if h.is_active
        )
        
        state.total_investment = sum(
            f.state.actual_investment for f in firms if f.is_active
        )
        
        final_sales = state.total_consumption + state.total_investment
        
        # 2. Inventory investment with EXPONENTIAL SMOOTHING
        # This prevents inventory swings from causing GDP explosions
        current_inventory = sum(f.state.inventory for f in firms if f.is_active)
        raw_inventory_change = current_inventory - self.previous_inventory
        
        # Apply exponential smoothing (α=0.2 for stability)
        alpha = 0.2
        self.smoothed_inventory_change = (
            alpha * raw_inventory_change + 
            (1 - alpha) * self.smoothed_inventory_change
        )
        
        # 3. Cap inventory contribution to 5% of final sales
        # This prevents inventory from dominating GDP
        max_inv_contribution = 0.05 * max(final_sales, 1.0)
        inventory_contribution = np.clip(
            self.smoothed_inventory_change,
            -max_inv_contribution,
            max_inv_contribution
        )
        
        self.previous_inventory = current_inventory
        
        # 4. Nominal GDP
        nominal_gdp = final_sales + inventory_contribution
        
        # 5. Real GDP (deflate by price level)
        state.gdp = nominal_gdp / max(state.price_level, 0.5)
        state.gdp = max(state.gdp, 1.0)  # Floor to prevent negatives
        
        # === GDP GROWTH ===
        self.gdp_history.append(state.gdp)
        
        if len(self.gdp_history) >= 2:
            prev_gdp = self.gdp_history[-2]
            if prev_gdp > 0:
                state.gdp_growth = (state.gdp - prev_gdp) / prev_gdp
            else:
                state.gdp_growth = 0.0
        else:
            state.gdp_growth = 0.0
        
        # === POTENTIAL GDP ===
        self._update_potential_gdp(state.gdp, state.unemployment)
        state.potential_gdp = self.potential_gdp
        
        # === OUTPUT GAP ===
        if self.potential_gdp > 0:
            state.output_gap = (state.gdp - self.potential_gdp) / self.potential_gdp
        else:
            state.output_gap = 0.0
        
        state.output_gap = np.clip(state.output_gap, -0.2, 0.2)
        
        # === DEFAULT RATE ===
        total_loans = sum(b.state.loans for b in banks if b.is_active)
        total_npls = sum(b.state.non_performing_loans for b in banks if b.is_active)
        if total_loans > 0:
            state.default_rate = total_npls / total_loans
        else:
            state.default_rate = 0.0
        
        return state
    
    def _update_potential_gdp(self, current_gdp: float, unemployment: float) -> None:
        """Update potential GDP estimate using Okun's Law."""
        natural_u = self.config.economic.natural_unemployment_rate
        
        if len(self.gdp_history) < 4:
            self.potential_gdp = current_gdp
            return
        
        okun = 2.0
        u_gap = unemployment - natural_u
        
        if u_gap != 0:
            implied_potential = current_gdp / (1 - okun * u_gap)
        else:
            implied_potential = current_gdp
        
        alpha = 0.1
        self.potential_gdp = alpha * implied_potential + (1 - alpha) * self.potential_gdp
        
        monthly_trend = 0.02 / 12
        self.potential_gdp *= (1 + monthly_trend)
        
        self.potential_gdp = max(self.potential_gdp, 50)
    
    def reset(self) -> None:
        """Reset state."""
        self.gdp_history = []
        self.potential_gdp_history = []
        self.potential_gdp = 100.0
        self.previous_inventory = 0.0


def compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient."""
    if not values or len(values) < 2:
        return 0.0
    
    values = sorted(values)
    n = len(values)
    
    min_val = min(values)
    if min_val < 0:
        values = [v - min_val for v in values]
    
    total = sum(values)
    if total == 0:
        return 0.0
    
    cumsum = np.cumsum(values)
    gini = (2 * np.sum((np.arange(1, n + 1) * values))) / (n * total) - (n + 1) / n
    
    return max(0.0, min(1.0, gini))


def compute_wealth_distribution(households: List[Any]) -> Dict[str, float]:
    """Compute wealth distribution statistics."""
    active = [h for h in households if h.is_active]
    if not active:
        return {}
    
    wealth = [h.state.wealth for h in active]
    
    return {
        "mean_wealth": np.mean(wealth),
        "median_wealth": np.median(wealth),
        "std_wealth": np.std(wealth),
        "gini_wealth": compute_gini(wealth),
        "top_10_share": sum(sorted(wealth)[-int(len(wealth)*0.1):]) / max(sum(wealth), 1),
        "bottom_50_share": sum(sorted(wealth)[:int(len(wealth)*0.5)]) / max(sum(wealth), 1),
    }