"""
Goods Market - Corrected Implementation

ECONOMIC LOGIC:
- Calvo-style price stickiness
- Demand allocated based on price competitiveness
- Inflation computed from price level changes

PRICE ADJUSTMENT:
- Each period, fraction (1-stickiness) of firms adjust prices
- Adjusters move toward markup over marginal cost
- Also respond to excess demand/supply
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class GoodsMarketOutcome:
    """Result of goods market clearing."""
    
    total_sales: float
    aggregate_demand: float
    aggregate_supply: float
    excess_demand: float
    price_level: float
    inflation: float  # Annualized
    capacity_utilization: float
    
    # Sales by firm
    sales_by_firm: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sales": self.total_sales,
            "aggregate_demand": self.aggregate_demand,
            "aggregate_supply": self.aggregate_supply,
            "excess_demand": self.excess_demand,
            "price_level": self.price_level,
            "inflation": self.inflation,
            "capacity_utilization": self.capacity_utilization,
        }


class GoodsMarket:
    """
    Goods market with sticky prices and demand rationing.
    
    KEY MECHANISMS:
    1. Calvo-style price stickiness
    2. Demand allocation based on relative prices
    3. Proper inflation computation
    """
    
    def __init__(self, config: Any):
        self.config = config
        
        self.price_stickiness = config.economic.price_stickiness
        
        # State
        self.price_level: float = 1.0
        self.price_history: List[float] = [1.0]
    
    def reset(self) -> None:
        """Reset goods market state."""
        self.price_level = 1.0
        self.price_history = [1.0]
    
    def clear(
        self,
        households: List[Any],
        firms: List[Any],
    ) -> GoodsMarketOutcome:
        """
        Clear the goods market.
        
        SEQUENCE:
        1. Compute aggregate demand (household consumption)
        2. Compute aggregate supply (firm inventories)
        3. Allocate demand to firms based on prices
        4. Update price level and compute inflation
        """
        active_households = [h for h in households if h.is_active]
        active_firms = [f for f in firms if f.is_active]
        
        if not active_households or not active_firms:
            return self._empty_outcome()
        
        # === 1. AGGREGATE DEMAND ===
        # Total desired consumption from households
        aggregate_demand = sum(h.state.consumption for h in active_households)
        
        # === 2. AGGREGATE SUPPLY ===
        # Total goods available (inventories)
        firm_supply = {}
        for f in active_firms:
            quantity, price = f.get_supply()
            firm_supply[f.id] = {
                "quantity": quantity,
                "price": price,
            }
        
        aggregate_supply = sum(fs["quantity"] for fs in firm_supply.values())
        
        # === 3. COMPUTE AVERAGE PRICE LEVEL ===
        if aggregate_supply > 0:
            weighted_price = sum(
                fs["quantity"] * fs["price"]
                for fs in firm_supply.values()
            ) / aggregate_supply
        else:
            weighted_price = self.price_level
        
        # === 4. ALLOCATE DEMAND TO FIRMS ===
        sales_by_firm = {}
        total_sales = 0.0
        
        if aggregate_supply > 0:
            # Price elasticity of demand
            elasticity = 2.0
            
            # Compute demand shares based on relative prices
            prices = [firm_supply[f.id]["price"] for f in active_firms]
            avg_price = np.mean(prices)
            
            # Higher price → lower demand share
            raw_shares = []
            for f in active_firms:
                relative_price = firm_supply[f.id]["price"] / max(avg_price, 0.1)
                share = (1 / relative_price) ** elasticity
                raw_shares.append(share)
            
            total_share = sum(raw_shares)
            demand_shares = [s / total_share for s in raw_shares] if total_share > 0 else [1/len(active_firms)] * len(active_firms)
            
            # Allocate demand
            for i, f in enumerate(active_firms):
                firm_demand = aggregate_demand * demand_shares[i]
                available = firm_supply[f.id]["quantity"]
                
                actual_sales = min(firm_demand, available)
                sales_by_firm[f.id] = actual_sales
                total_sales += actual_sales
        else:
            for f in active_firms:
                sales_by_firm[f.id] = 0.0
        
        # === 5. EXCESS DEMAND ===
        excess_demand = aggregate_demand - aggregate_supply
        excess_demand_ratio = excess_demand / max(aggregate_supply, 1.0)
        
        # === 6. PRICE LEVEL UPDATE ===
        old_price_level = self.price_level
        
        # Price pressure from excess demand
        # Calvo: only (1-stickiness) fraction adjusts
        price_adjustment = (1 - self.price_stickiness) * 0.05 * excess_demand_ratio
        price_adjustment = np.clip(price_adjustment, -0.008, 0.008)  # Cap monthly change
        
        new_price_level = old_price_level * (1 + price_adjustment)
        
        # Also incorporate actual firm prices
        alpha = 0.3
        new_price_level = alpha * weighted_price + (1 - alpha) * new_price_level
        
        # Bounds
        new_price_level = np.clip(new_price_level, 0.5, 3.0)
        
        self.price_level = new_price_level
        self.price_history.append(new_price_level)
        
        # === 7. INFLATION ===
        if old_price_level > 0:
            monthly_inflation = (new_price_level - old_price_level) / old_price_level
        else:
            monthly_inflation = 0.0
        
        # Annualize
        annual_inflation = ((1 + monthly_inflation) ** 12) - 1
        annual_inflation = np.clip(annual_inflation, -0.10, 0.30)
        
        # === 8. CAPACITY UTILIZATION ===
        if aggregate_supply > 0:
            capacity_utilization = total_sales / aggregate_supply
        else:
            capacity_utilization = 0.0
        
        return GoodsMarketOutcome(
            total_sales=total_sales,
            aggregate_demand=aggregate_demand,
            aggregate_supply=aggregate_supply,
            excess_demand=excess_demand,
            price_level=self.price_level,
            inflation=annual_inflation,
            capacity_utilization=capacity_utilization,
            sales_by_firm=sales_by_firm,
        )
    
    def _empty_outcome(self) -> GoodsMarketOutcome:
        """Return empty outcome when no agents."""
        return GoodsMarketOutcome(
            total_sales=0.0,
            aggregate_demand=0.0,
            aggregate_supply=0.0,
            excess_demand=0.0,
            price_level=1.0,
            inflation=0.0,
            capacity_utilization=0.0,
            sales_by_firm={},
        )