"""
Commercial Bank Agent - Corrected Implementation

ECONOMIC LOGIC:
- Banks intermediate between depositors and borrowers
- Set lending rate as spread over policy rate
- Manage capital adequacy (Basel-style)
- Profit from net interest margin

INTEREST RATE PASSTHROUGH:
lending_rate = policy_rate + spread + risk_premium
deposit_rate = policy_rate - deposit_spread

This is the key monetary transmission mechanism!
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .base_agent import BaseAgent, AgentState


@dataclass
class CommercialBankState(AgentState):
    """State specific to Commercial Banks."""
    
    # Balance sheet
    capital: float = 100.0
    deposits: float = 500.0
    loans: float = 400.0
    reserves: float = 50.0
    loan_loss_reserves: float = 10.0
    
    # Loan portfolio
    performing_loans: float = 390.0
    non_performing_loans: float = 10.0
    
    # Interest rates (monthly)
    lending_rate: float = 0.005   # ~6% annual
    deposit_rate: float = 0.002   # ~2.4% annual
    
    # Risk appetite
    risk_tolerance: float = 0.5
    
    # Performance (monthly)
    net_interest_income: float = 0.0
    loan_losses: float = 0.0
    profit: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to normalized observation array."""
        capital_ratio = self.capital / max(self.loans, 1.0)
        npl_ratio = self.non_performing_loans / max(self.loans, 1.0)
        loan_to_deposit = self.loans / max(self.deposits, 1.0)
        reserve_ratio = self.reserves / max(self.deposits, 1.0)
        
        arr = np.array([
            np.clip(capital_ratio, 0, 0.5),
            np.clip(npl_ratio, 0, 0.3),
            np.clip(loan_to_deposit, 0, 1.5),
            np.clip(self.lending_rate * 12, 0, 0.2),  # Annualized
            np.clip(self.deposit_rate * 12, 0, 0.1),
            np.clip(self.risk_tolerance, 0, 1),
            np.clip(reserve_ratio, 0, 0.3),
            np.clip(self.profit / 5.0, -5, 5),
        ], dtype=np.float32)
        
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "capital": self.capital,
            "deposits": self.deposits,
            "loans": self.loans,
            "npl_ratio": self.npl_ratio,
            "lending_rate_annual": self.lending_rate * 12,
            "deposit_rate_annual": self.deposit_rate * 12,
            "capital_ratio": self.capital_ratio,
            "profit": self.profit,
        })
        return base
    
    @property
    def capital_ratio(self) -> float:
        return self.capital / max(self.loans, 1.0)
    
    @property
    def npl_ratio(self) -> float:
        return self.non_performing_loans / max(self.loans, 1.0)
    
    @property
    def net_interest_margin(self) -> float:
        """Net interest margin (spread between lending and deposit rates)."""
        return self.lending_rate - self.deposit_rate


class CommercialBank(BaseAgent):
    """
    Commercial Bank that intermediates credit.
    
    DECISIONS:
    1. Lending spread over policy rate
    2. Deposit spread below policy rate
    3. Risk tolerance (lending standards)
    
    MONETARY TRANSMISSION:
    When policy rate rises:
    - Bank raises lending rate
    - Credit becomes more expensive
    - Less borrowing → less spending → lower output/inflation
    """
    
    def __init__(self, agent_id: int, config: Any):
        super().__init__(agent_id, config)
        
        self.lending_spread_actions = config.actions.lending_spread_actions
        self.deposit_spread_actions = config.actions.deposit_spread_actions
        self.risk_tolerance_actions = config.actions.risk_tolerance_actions
        
        self._action_space_size = (
            len(self.lending_spread_actions) *
            len(self.deposit_spread_actions) *
            len(self.risk_tolerance_actions)
        )
        
        self.min_capital_ratio = config.economic.min_capital_ratio
        self.recovery_rate = config.economic.recovery_rate
    
    def reset(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        """Initialize bank with some heterogeneity."""
        ic = initial_conditions or {}
        
        # Random variation
        capital_mult = np.random.uniform(0.9, 1.1)
        deposit_mult = np.random.uniform(0.9, 1.1)
        
        base_capital = self.config.economic.initial_bank_capital
        base_deposits = self.config.economic.initial_bank_deposits
        
        capital = ic.get("capital", base_capital * capital_mult)
        deposits = ic.get("deposits", base_deposits * deposit_mult)
        
        loans = deposits * 0.8  # 80% loan-to-deposit
        reserves = deposits * 0.1
        
        # Initial rates (monthly)
        initial_lending = 0.005  # ~6% annual
        initial_deposit = 0.002  # ~2.4% annual
        
        self.state = CommercialBankState(
            id=self.id,
            capital=capital,
            deposits=deposits,
            loans=loans,
            reserves=reserves,
            loan_loss_reserves=loans * self.config.economic.loan_loss_reserve_rate,
            performing_loans=loans * 0.975,
            non_performing_loans=loans * 0.025,
            lending_rate=initial_lending,
            deposit_rate=initial_deposit,
            risk_tolerance=0.5,
            is_active=True,
        )
        
        self.state.wealth = capital
        self.state.income = 0.0
    
    def get_observation(self, global_state: Dict[str, Any]) -> np.ndarray:
        """Bank observes own state + macro conditions."""
        policy_rate = global_state.get("policy_rate", 0.002)
        credit_growth = global_state.get("credit_growth", 0.0)
        default_rate = global_state.get("default_rate", 0.02)
        gdp_growth = global_state.get("gdp_growth", 0.0)
        inflation = global_state.get("inflation", 0.02)
        unemployment = global_state.get("unemployment", 0.045)
        
        own_state = self.state.to_array()[:6]
        
        macro_state = np.array([
            np.clip(policy_rate * 12, 0, 0.15),  # Annualized
            np.clip(credit_growth, -0.3, 0.3),
            np.clip(default_rate, 0, 0.2),
            np.clip(gdp_growth, -0.2, 0.2),
            np.clip(inflation, -0.05, 0.15),
            np.clip(unemployment, 0, 0.2),
        ], dtype=np.float32)
        
        obs = np.concatenate([own_state, macro_state])
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    
    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode action into rate spreads and risk tolerance."""
        n_lending = len(self.lending_spread_actions)
        n_deposit = len(self.deposit_spread_actions)
        n_risk = len(self.risk_tolerance_actions)
        
        action_idx = action_idx % self._action_space_size
        
        risk_idx = action_idx % n_risk
        remaining = action_idx // n_risk
        deposit_idx = remaining % n_deposit
        lending_idx = remaining // n_deposit
        
        return {
            "lending_spread": self.lending_spread_actions[lending_idx],
            "deposit_spread": self.deposit_spread_actions[deposit_idx],
            "risk_tolerance": self.risk_tolerance_actions[risk_idx],
            "action_idx": action_idx,
        }
    
    def compute_rates(
        self,
        action: Dict[str, Any],
        policy_rate: float
    ) -> Tuple[float, float]:
        """
        Compute actual lending and deposit rates from spreads.
        
        This is the key interest rate passthrough mechanism!
        """
        # Lending rate = policy rate + spread
        lending_rate = policy_rate + action["lending_spread"]
        
        # Deposit rate = policy rate + spread (spread is negative)
        deposit_rate = policy_rate + action["deposit_spread"]
        
        # Ensure positive spread (lending > deposit)
        deposit_rate = min(deposit_rate, lending_rate - 0.001)
        
        # Floor rates at zero
        lending_rate = max(lending_rate, 0.001)
        deposit_rate = max(deposit_rate, 0.0)
        
        return lending_rate, deposit_rate
    
    def compute_reward(
        self,
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> float:
        """Bank reward = profit - capital penalty - NPL penalty."""
        # Profit component
        profit_reward = self.state.profit / 3.0
        profit_reward = np.clip(profit_reward, -5, 5)
        
        # Capital adequacy penalty (Phase 3: reduced coefficient)
        if self.state.capital_ratio < self.min_capital_ratio:
            gap = self.min_capital_ratio - self.state.capital_ratio
            capital_penalty = 2.5 * gap / self.min_capital_ratio  # Reduced from 5.0
        else:
            capital_penalty = 0.0
        
        # NPL penalty (Phase 3: reduced coefficient)
        if self.state.npl_ratio > 0.05:
            npl_penalty = 1.5 * (self.state.npl_ratio - 0.05)  # Reduced from 3.0
        else:
            npl_penalty = 0.0
        
        reward = profit_reward - capital_penalty - npl_penalty
        
        # Phase 3: Normalized to [-5, 5]
        return float(np.clip(reward, -5, 5))
    
    def update_state(
        self,
        action: Dict[str, Any],
        market_outcomes: Dict[str, Any],
    ) -> None:
        """Update bank state."""
        policy_rate = market_outcomes.get("policy_rate", 0.002)
        
        # === 1. UPDATE RATES ===
        lending_rate, deposit_rate = self.compute_rates(action, policy_rate)
        self.state.lending_rate = lending_rate
        self.state.deposit_rate = deposit_rate
        self.state.risk_tolerance = action["risk_tolerance"]
        
        # === 2. NEW LOANS ===
        new_loans = market_outcomes.get(f"bank_{self.id}_new_loans", 0.0)
        
        # === 3. DEFAULTS ===
        defaults = market_outcomes.get(f"bank_{self.id}_defaults", 0.0)
        recovery = defaults * self.recovery_rate
        
        # === 4. UPDATE LOAN PORTFOLIO ===
        self.state.loans = max(10, self.state.loans + new_loans - defaults + recovery)
        
        # NPL dynamics: some cure, some new
        npl_cure_rate = 0.03
        new_npl_rate = 0.01
        
        cured = self.state.non_performing_loans * npl_cure_rate
        new_npls = self.state.performing_loans * new_npl_rate + defaults * 0.5
        
        self.state.non_performing_loans = max(0, self.state.non_performing_loans - cured + new_npls)
        self.state.performing_loans = max(0, self.state.loans - self.state.non_performing_loans)
        
        # === 5. NET INTEREST INCOME ===
        interest_income = self.state.lending_rate * self.state.performing_loans
        interest_expense = self.state.deposit_rate * self.state.deposits
        self.state.net_interest_income = interest_income - interest_expense
        
        # === 6. LOAN LOSSES ===
        self.state.loan_losses = defaults * (1 - self.recovery_rate)
        
        # === 7. PROFIT ===
        self.state.profit = self.state.net_interest_income - self.state.loan_losses
        
        # === 8. CAPITAL UPDATE ===
        self.state.capital += self.state.profit
        self.state.capital = max(1, self.state.capital)
        
        # === 9. DEPOSITS ===
        rate_effect = 0.01 * (self.state.deposit_rate * 12 - 0.02)
        deposit_growth = np.random.normal(0.002, 0.003) + rate_effect
        deposit_change = market_outcomes.get(f"bank_{self.id}_deposit_change", 0.0)
        self.state.deposits = max(100, self.state.deposits * (1 + deposit_growth) + deposit_change)
        
        # === 10. RESERVES ===
        self.state.reserves = self.state.deposits * 0.1
        
        # === 11. BANKRUPTCY CHECK ===
        if self.state.capital_ratio < 0.02:
            self.declare_bankruptcy()
        
        # === 12. BASE CLASS FIELDS ===
        self.state.wealth = self.state.capital
        self.state.income = self.state.profit
        self.state.debt = 0
    
    def get_action_space_size(self) -> int:
        return self._action_space_size
    
    def get_observation_size(self) -> int:
        return self.config.network.bank_input_size
    
    def can_lend(self, amount: float) -> bool:
        """Check if bank can extend a loan."""
        if not self.is_active:
            return False
        
        new_loans = self.state.loans + amount
        projected_ratio = self.state.capital / new_loans
        
        return projected_ratio >= self.min_capital_ratio
    
    def get_loan_offer(self, borrower_risk: float, gdp_growth: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Generate a loan offer for a borrower.
        
        PROCYCLICAL LENDING (Research finding: Credit-GDP correlation +0.70):
        - Banks lend MORE during economic expansions (high GDP growth)
        - Banks lend LESS during contractions (low GDP growth)
        This is realistic bank behavior observed in ECB and Fed research.
        """
        if not self.is_active:
            return None
        
        # PROCYCLICAL: Adjust risk tolerance based on GDP growth
        # During good times (high GDP growth), banks take more risk
        # During bad times (low GDP growth), banks tighten standards
        cyclical_risk_adjustment = 0.2 * gdp_growth  # +/- 20% of growth rate
        adjusted_risk_tolerance = self.state.risk_tolerance + cyclical_risk_adjustment
        adjusted_risk_tolerance = np.clip(adjusted_risk_tolerance, 0.1, 0.9)
        
        if borrower_risk > adjusted_risk_tolerance:
            return None
        
        # PROCYCLICAL: Adjust lending capacity based on GDP growth
        # Banks expand lending during booms, contract during busts
        cycle_factor = 1.0 + 0.5 * gdp_growth  # ±50% of growth rate
        cycle_factor = np.clip(cycle_factor, 0.5, 1.5)
        
        if not self.can_lend(self.state.capital * 0.1 * cycle_factor):
            return None
        
        risk_premium = borrower_risk * 0.02 / 12
        # Lower rates during expansions (procyclical)
        rate_adjustment = -0.001 * gdp_growth  # Lower rates when GDP high
        offered_rate = self.state.lending_rate + risk_premium + rate_adjustment
        offered_rate = max(0.001, offered_rate)
        
        return {
            "bank_id": self.id,
            "rate": offered_rate,
            "max_amount": self.state.capital * 0.15 * cycle_factor,
        }