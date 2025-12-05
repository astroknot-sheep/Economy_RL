"""
Credit Market - Corrected Implementation

ECONOMIC LOGIC:
- Banks set rates as spread over policy rate (passthrough)
- Credit rationing based on borrower risk
- Debt-to-income constraints enforced

INTEREST RATE PASSTHROUGH:
lending_rate = policy_rate + spread + risk_premium

This is the key monetary transmission mechanism!
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class CreditMarketOutcome:
    """Result of credit market clearing."""
    
    total_new_loans: float
    total_loan_demand: float
    credit_growth: float
    avg_lending_rate: float
    avg_deposit_rate: float
    credit_spread: float
    
    # Individual loan results
    loan_results: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_new_loans": self.total_new_loans,
            "total_loan_demand": self.total_loan_demand,
            "credit_growth": self.credit_growth,
            "avg_lending_rate": self.avg_lending_rate,
            "avg_deposit_rate": self.avg_deposit_rate,
            "credit_spread": self.credit_spread,
        }


class CreditMarket:
    """
    Credit market with risk-based pricing and rationing.
    
    KEY MECHANISMS:
    1. Interest rate passthrough from policy rate
    2. Risk-based credit rationing
    3. Debt-to-income constraints
    """
    
    def __init__(self, config: Any):
        self.config = config
        
        # Credit history
        self.previous_credit: float = 0.0
        self.credit_history: List[float] = []
    
    def reset(self) -> None:
        """Reset credit market state."""
        self.previous_credit = 0.0
        self.credit_history = []
    
    def clear(
        self,
        banks: List[Any],
        households: List[Any],
        firms: List[Any],
        policy_rate: float,
        gdp_growth: float = 0.0,  # Added for procyclical lending
    ) -> CreditMarketOutcome:
        """
        Clear the credit market.
        
        SEQUENCE:
        1. Collect loan demand from households and firms
        2. Banks evaluate and approve loans
        3. Compute credit growth and spreads
        """
        active_banks = [b for b in banks if b.is_active]
        active_households = [h for h in households if h.is_active]
        active_firms = [f for f in firms if f.is_active]
        
        if not active_banks:
            return self._empty_outcome(policy_rate)
        
        loan_results = {}
        total_new_loans = 0.0
        total_demand = 0.0
        
        # === 1. HOUSEHOLD LOAN APPLICATIONS ===
        for h in active_households:
            if h.state.loan_amount > 0:
                continue
            
            # Rate-sensitive demand
            avg_rate = np.mean([b.state.lending_rate for b in active_banks])
            demand = h.get_borrowing_demand(avg_rate)
            
            if demand <= 0:
                continue
            
            total_demand += demand
            
            # Find best offer
            borrower_risk = h.get_risk_score()
            best_offer = None
            
            for bank in active_banks:
                offer = bank.get_loan_offer(borrower_risk, gdp_growth)  # Pass GDP growth
                if offer is not None:
                    if best_offer is None or offer["rate"] < best_offer["rate"]:
                        best_offer = offer
            
            if best_offer:
                # Check DTI constraint
                max_dti = self.config.economic.max_dti
                if h.state.labor_income > 0:
                    monthly_payment = demand * best_offer["rate"] * 1.02  # Rough estimate
                    projected_dti = monthly_payment / h.state.labor_income
                    
                    if projected_dti > max_dti:
                        demand = h.state.labor_income * max_dti / (best_offer["rate"] * 1.02)
                
                # Phase 4: COLLATERAL REQUIREMENTS
                # Low-wealth households face tighter constraints
                collateral_value = max(h.state.savings, 0)
                if collateral_value < h.state.labor_income * 3:
                    # Require collateral: max loan = 3x savings for low-wealth
                    max_collateral_loan = collateral_value * 3
                    demand = min(demand, max_collateral_loan)
                
                # Phase 4: WEALTH-BASED RATE ADJUSTMENT
                # Poor households pay higher rates
                wealth_ratio = h.state.savings / max(h.state.labor_income * 6, 1.0) if h.state.labor_income > 0 else 0
                if wealth_ratio < 0:
                    # Negative wealth - very risky
                    best_offer["rate"] += 0.03 / 12  # +3% annual
                elif wealth_ratio < 0.5:
                    # Low wealth
                    best_offer["rate"] += 0.02 / 12  # +2% annual
                
                # Cap at bank's max
                actual_amount = min(demand, best_offer["max_amount"])
                
                if actual_amount > 0:
                    loan_results[f"household_{h.id}_loan"] = {
                        "amount": actual_amount,
                        "rate": best_offer["rate"],
                        "bank_id": best_offer["bank_id"],
                    }
                    
                    # Track bank's new loans
                    bank_key = f"bank_{best_offer['bank_id']}_new_loans"
                    loan_results[bank_key] = loan_results.get(bank_key, 0) + actual_amount
                    
                    total_new_loans += actual_amount
        
        # === 2. FIRM LOAN APPLICATIONS ===
        for f in active_firms:
            if f.state.debt > 0:
                continue
            
            # Firms with negative cash or investment needs
            if f.state.cash < 0:
                demand = abs(f.state.cash) * 1.5
            else:
                demand = f.state.capital * 0.1
            
            if demand <= 0:
                continue
            
            total_demand += demand
            
            # Firm risk based on profitability
            if f.state.revenue > 0:
                profit_margin = f.state.profit / f.state.revenue
                risk = 0.3 - profit_margin * 0.5
            else:
                risk = 0.5
            risk = np.clip(risk, 0, 1)
            
            # Find best offer
            best_offer = None
            for bank in active_banks:
                offer = bank.get_loan_offer(risk, gdp_growth)  # Pass GDP growth
                if offer is not None:
                    if best_offer is None or offer["rate"] < best_offer["rate"]:
                        best_offer = offer
            
            if best_offer:
                actual_amount = min(demand, best_offer["max_amount"])
                
                if actual_amount > 0:
                    loan_results[f"firm_{f.id}_loan"] = {
                        "amount": actual_amount,
                        "rate": best_offer["rate"],
                        "bank_id": best_offer["bank_id"],
                    }
                    
                    bank_key = f"bank_{best_offer['bank_id']}_new_loans"
                    loan_results[bank_key] = loan_results.get(bank_key, 0) + actual_amount
                    
                    total_new_loans += actual_amount
        
        # === 3. COMPUTE AGGREGATES ===
        current_credit = sum(b.state.loans for b in active_banks)
        
        if self.previous_credit > 0:
            credit_growth = (current_credit - self.previous_credit) / self.previous_credit
        else:
            credit_growth = 0.0
        
        self.previous_credit = current_credit
        self.credit_history.append(current_credit)
        
        # Average rates
        avg_lending = np.mean([b.state.lending_rate for b in active_banks])
        avg_deposit = np.mean([b.state.deposit_rate for b in active_banks])
        credit_spread = avg_lending - policy_rate
        
        return CreditMarketOutcome(
            total_new_loans=total_new_loans,
            total_loan_demand=total_demand,
            credit_growth=credit_growth,
            avg_lending_rate=avg_lending,
            avg_deposit_rate=avg_deposit,
            credit_spread=credit_spread,
            loan_results=loan_results,
        )
    
    def _empty_outcome(self, policy_rate: float) -> CreditMarketOutcome:
        """Return empty outcome when no banks."""
        return CreditMarketOutcome(
            total_new_loans=0.0,
            total_loan_demand=0.0,
            credit_growth=0.0,
            avg_lending_rate=policy_rate + 0.003,
            avg_deposit_rate=policy_rate,
            credit_spread=0.003,
            loan_results={},
        )