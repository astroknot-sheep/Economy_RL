"""
Household Agent - Corrected Implementation

ECONOMIC LOGIC:
- Consumption follows Permanent Income Hypothesis with buffer stock
- Labor supply decision (search intensity)
- Borrowing decision (rate-sensitive)

CONSUMPTION FUNCTION:
C = MPC_income * Y + MPC_wealth * W

Where:
- MPC_income ≈ 0.85 (high for current income)
- MPC_wealth ≈ 0.003 monthly (~4% annual)
- Buffer stock: target savings = 6 months expenses
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

from .base_agent import BaseAgent, AgentState


@dataclass
class HouseholdState(AgentState):
    """State specific to Households."""
    
    # Employment
    is_employed: bool = True
    employer_id: Optional[int] = None
    months_unemployed: int = 0
    labor_income: float = 5.0
    skill_level: float = 1.0  # Phase 2: Pareto-distributed (0.3-5.0x multiplier)
    
    # Financial position
    savings: float = 30.0
    consumption: float = 4.0
    
    # Borrowing
    loan_amount: float = 0.0
    loan_rate: float = 0.0
    loan_bank_id: Optional[int] = None
    months_remaining: int = 0
    
    # Unemployment insurance
    benefit_months_remaining: int = 0
    
    # Permanent Income Hypothesis
    permanent_income_estimate: float = 5.0  # Smoothed income expectation
    previous_consumption: float = 4.0  # For consumption smoothing
    
    # Phase 4: Adaptive Expectations
    expected_inflation: float = 0.02 / 12  # Expected monthly inflation
    expected_income: float = 5.0  # Expected future income
    
    # Derived
    disposable_income: float = 5.0
    debt_service: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to normalized observation array."""
        employed_float = 1.0 if self.is_employed else 0.0
        income_norm = np.clip(self.labor_income / 5.0, 0, 3)
        savings_norm = np.clip(self.savings / 30.0, -1, 5)
        debt_norm = np.clip(self.loan_amount / 50.0, 0, 3)
        consumption_norm = np.clip(self.consumption / 5.0, 0, 3)
        
        # Debt-to-income ratio
        if self.labor_income > 0:
            dti = self.debt_service / self.labor_income
        else:
            dti = 0.0
        
        # Buffer ratio (savings / target)
        target_savings = self.labor_income * 6
        if target_savings > 0:
            buffer_ratio = self.savings / target_savings
        else:
            buffer_ratio = 1.0
        
        arr = np.array([
            employed_float,
            income_norm,
            savings_norm,
            debt_norm,
            consumption_norm,
            np.clip(dti, 0, 1),
            np.clip(buffer_ratio, 0, 3),
            np.clip(self.months_unemployed / 12, 0, 2),
        ], dtype=np.float32)
        
        return np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "is_employed": self.is_employed,
            "labor_income": self.labor_income,
            "savings": self.savings,
            "consumption": self.consumption,
            "loan_amount": self.loan_amount,
            "disposable_income": self.disposable_income,
        })
        return base


class Household(BaseAgent):
    """
    Household that consumes, saves, works, and borrows.
    
    DECISIONS:
    1. Consumption rate
    2. Whether to borrow
    3. Labor supply intensity
    
    UTILITY:
    U = log(C) - disutility(L) + buffer_value - debt_cost
    """
    
    def __init__(self, agent_id: int, config: Any):
        super().__init__(agent_id, config)
        
        self.consumption_rate_actions = config.actions.consumption_rate_actions
        self.borrowing_actions = config.actions.borrowing_actions
        self.labor_supply_actions = config.actions.labor_supply_actions
        
        self._action_space_size = (
            len(self.consumption_rate_actions) *
            len(self.borrowing_actions) *
            len(self.labor_supply_actions)
        )
        
        # Consumption parameters
        self.mpc_income = config.economic.mpc_income
        self.mpc_wealth = config.economic.mpc_wealth
        self.precautionary_months = config.economic.precautionary_months
        
        # Borrowing parameters
        self.rate_sensitivity = config.economic.borrowing_rate_sensitivity
        self.max_dti = config.economic.max_dti
        self.loan_term = config.economic.loan_term_months
        
        # Unemployment
        self.benefit_rate = config.economic.unemployment_benefit_rate
        self.benefit_duration = config.economic.benefit_duration
    
    def reset(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        """Initialize household with heterogeneity."""
        ic = initial_conditions or {}
        
        # Phase 2: SKILL HETEROGENEITY (Pareto distribution)
        # Creates wage inequality - most have skill ~1.0, few have very high skills
        # Pareto with shape=1.5 gives realistic distribution
        pareto_sample = np.random.pareto(1.5) + 1  # Shift to start at 1
        skill_level = np.clip(pareto_sample, 0.3, 5.0)  # 0.3x to 5.0x multiplier
        
        # Phase 2: WEALTH HETEROGENEITY (log-normal distribution)
        # Creates initial wealth inequality
        base_savings = self.config.economic.initial_household_savings
        log_normal_mult = np.random.lognormal(mean=0.0, sigma=1.2)
        # Clip to create realistic distribution (some poor, some rich)
        savings_mult = np.clip(log_normal_mult, 0.1, 10.0)
        
        initial_savings = ic.get(
            "savings",
            base_savings * savings_mult
        )
        
        # Most start employed
        is_employed = ic.get("is_employed", np.random.random() > 0.05)
        
        # Phase 2: SKILL-DEPENDENT WAGES
        base_wage = self.config.economic.base_wage
        labor_income = (base_wage * skill_level) if is_employed else 0.0
        
        initial_consumption = labor_income * 0.8 if labor_income > 0 else base_wage * 0.3
        
        self.state = HouseholdState(
            id=self.id,
            is_employed=is_employed,
            employer_id=None,
            months_unemployed=0 if is_employed else 1,
            labor_income=labor_income,
            savings=initial_savings,
            consumption=initial_consumption,
            loan_amount=0.0,
            loan_rate=0.0,
            loan_bank_id=None,
            months_remaining=0,
            benefit_months_remaining=0 if is_employed else self.benefit_duration,
            permanent_income_estimate=labor_income if labor_income > 0 else base_wage,
            previous_consumption=initial_consumption,
            skill_level=skill_level,  # Store skill
            expected_inflation=self.config.economic.inflation_target,  # Phase 4
            expected_income=labor_income if labor_income > 0 else base_wage,  # Phase 4
            is_active=True,
        )
        
        self.state.wealth = initial_savings
        self.state.income = labor_income
        self.state.debt = 0.0
    
    def get_observation(self, global_state: Dict[str, Any]) -> np.ndarray:
        """Household observes own state + macro conditions."""
        inflation = global_state.get("inflation", 0.02)
        unemployment = global_state.get("unemployment", 0.045)
        lending_rate = global_state.get("avg_lending_rate", 0.005)
        deposit_rate = global_state.get("avg_deposit_rate", 0.002)
        gdp_growth = global_state.get("gdp_growth", 0.0)
        wage = global_state.get("wage", 5.0)
        price_level = global_state.get("price_level", 1.0)
        
        own_state = self.state.to_array()
        
        macro_state = np.array([
            np.clip(inflation, -0.05, 0.15),
            np.clip(unemployment, 0, 0.2),
            np.clip(lending_rate * 12, 0, 0.2),
            np.clip(deposit_rate * 12, 0, 0.1),
            np.clip(gdp_growth, -0.2, 0.2),
            np.clip(wage / 5.0, 0.5, 2),
            np.clip(price_level, 0.5, 2),
        ], dtype=np.float32)
        
        obs = np.concatenate([own_state, macro_state])
        return np.nan_to_num(obs, nan=0.0, posinf=2.0, neginf=-1.0)
    
    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Decode action into components."""
        n_cons = len(self.consumption_rate_actions)
        n_borrow = len(self.borrowing_actions)
        n_labor = len(self.labor_supply_actions)
        
        action_idx = action_idx % self._action_space_size
        
        labor_idx = action_idx % n_labor
        remaining = action_idx // n_labor
        borrow_idx = remaining % n_borrow
        cons_idx = remaining // n_borrow
        
        return {
            "consumption_rate": self.consumption_rate_actions[cons_idx],
            "wants_to_borrow": self.borrowing_actions[borrow_idx] == 1,
            "labor_intensity": self.labor_supply_actions[labor_idx],
            "action_idx": action_idx,
        }
    
    def compute_reward(
        self,
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> float:
        """
        Household utility function.
        
        U = log(C) - labor_disutility + buffer_value - debt_burden - unemployment_penalty
        """
        # Log consumption utility
        consumption = max(self.state.consumption, 0.1)
        consumption_utility = np.log(consumption)
        
        # Labor disutility
        labor_intensity = action.get("labor_intensity", 1.0)
        labor_disutility = 0.2 * (labor_intensity - 1.0) ** 2
        
        # Buffer stock value
        target_savings = self.state.labor_income * self.precautionary_months
        if target_savings > 0:
            buffer_ratio = self.state.savings / target_savings
            buffer_value = 0.3 * min(buffer_ratio, 1.5)
        else:
            buffer_value = 0.0
        
        # Debt burden
        if self.state.labor_income > 0:
            dti = self.state.debt_service / self.state.labor_income
        else:
            dti = 0.5 if self.state.loan_amount > 0 else 0.0
        debt_burden = 0.3 * dti
        
        # Phase 3: Reduced unemployment penalty
        unemployment_penalty = 1.0 if not self.state.is_employed else 0.0  # Reduced from 2.0
        
        reward = (
            consumption_utility 
            - labor_disutility 
            + buffer_value 
            - debt_burden 
            - unemployment_penalty
        )
        
        # Phase 3: Normalized to [-5, 5]
        return float(np.clip(reward, -5, 5))
    
    def update_state(
        self,
        action: Dict[str, Any],
        market_outcomes: Dict[str, Any],
    ) -> None:
        """Update household state."""
        wage = market_outcomes.get("wage", self.config.economic.base_wage)
        deposit_rate = market_outcomes.get("deposit_rate", 0.002)
        
        # === 1. EMPLOYMENT STATUS ===
        employed_key = f"household_{self.id}_employed"
        if employed_key in market_outcomes:
            new_employed = market_outcomes[employed_key]
            
            if new_employed and not self.state.is_employed:
                # Got a job
                self.state.months_unemployed = 0
                self.state.benefit_months_remaining = 0
            elif not new_employed and self.state.is_employed:
                # Lost job
                self.state.months_unemployed = 1
                self.state.benefit_months_remaining = self.benefit_duration
            elif not new_employed:
                self.state.months_unemployed += 1
                self.state.benefit_months_remaining = max(
                    0, self.state.benefit_months_remaining - 1
                )
            
            self.state.is_employed = new_employed
        
        # === 2. INCOME ===
        labor_intensity = action.get("labor_intensity", 1.0)
        
        # Phase 2: SKILL-DEPENDENT WAGES
        base_wage = wage  # Market wage
        skilled_wage = base_wage * self.state.skill_level
        
        if self.state.is_employed:
            self.state.labor_income = skilled_wage * labor_intensity
        else:
            if self.state.benefit_months_remaining > 0:
                # Benefits based on skilled wage
                self.state.labor_income = skilled_wage * self.benefit_rate
            else:
                self.state.labor_income = 0.0
        
        # Phase 2: CAPITAL INCOME (returns on wealth)
        # Combination of deposit rate + capital returns (6% annual = 0.5% monthly)
        if self.state.savings > 0:
            capital_return_rate = 0.06 / 12  # 6% annual
            interest_income = self.state.savings * (deposit_rate + capital_return_rate)
        else:
            interest_income = 0.0
        
        # === 3. DEBT SERVICE ===
        if self.state.loan_amount > 0 and self.state.months_remaining > 0:
            # Monthly payment (simple amortization)
            monthly_rate = self.state.loan_rate
            if monthly_rate > 0:
                payment = self.state.loan_amount * (
                    monthly_rate * (1 + monthly_rate) ** self.state.months_remaining
                ) / ((1 + monthly_rate) ** self.state.months_remaining - 1)
            else:
                payment = self.state.loan_amount / self.state.months_remaining
            
            payment = min(payment, self.state.loan_amount)
            
            interest_payment = self.state.loan_amount * monthly_rate
            principal_payment = payment - interest_payment
            
            self.state.debt_service = payment
            self.state.loan_amount = max(0, self.state.loan_amount - principal_payment)
            self.state.months_remaining -= 1
            
            if self.state.months_remaining <= 0:
                self.state.loan_amount = 0
                self.state.loan_rate = 0
                self.state.loan_bank_id = None
        else:
            self.state.debt_service = 0.0
        
        # === 4. DISPOSABLE INCOME ===
        self.state.disposable_income = (
            self.state.labor_income 
            + interest_income 
            - self.state.debt_service
        )
        
        # === 5. CONSUMPTION (SUPER-SMOOTHED to prevent demand collapse) ===
    
        #  5a. Update permanent income estimate (SLOW exponential smoothing)
        # CRITICAL FIX: Households adjust expectations VERY slowly during recessions
        # This prevents panic consumption cuts that cause GDP to collapse -400%
        learning_rate = 0.15  # REDUCED from 0.3 - slower adjustment
        current_income = self.state.labor_income
        self.state.permanent_income_estimate = (
            learning_rate * current_income + 
            (1 - learning_rate) * self.state.permanent_income_estimate
        )
        
        # 5b. Calculate target consumption based on PIH
        # C* = MPC_income × Y_permanent + MPC_wealth × W
        target_consumption = (
            self.mpc_income * self.state.permanent_income_estimate +
            self.mpc_wealth * max(self.state.savings, 0)
        )
        
        # 5c. SUPER SMOOTH adjustment toward target (15% adjustment per period)
        # CRITICAL FIX: Prevents consumption from cliff-diving
        # Real households maintain consumption during recessions by drawing savings
        adjustment_speed = 0.15  # REDUCED from 0.3 - much slower
        smoothed_consumption = (
            adjustment_speed * target_consumption +
            (1 - adjustment_speed) * self.state.previous_consumption
        )
        
        # 5d. Apply action multiplier (household's discretionary choice)
        consumption_rate = action.get("consumption_rate", 0.85)
        desired_consumption = consumption_rate * smoothed_consumption
        
        # 5e. Subsistence floor (30% of base wage - prevents starvation)
        subsistence_floor = 0.3 * self.config.economic.base_wage
        desired_consumption = max(desired_consumption, subsistence_floor)
        
        # 5f. Budget constraint (can't consume more than available)
        available_resources = self.state.disposable_income + max(self.state.savings, 0)
        self.state.consumption = min(desired_consumption, available_resources)
        
        # 5g. Track for next period's smoothing
        self.state.previous_consumption = self.state.consumption
        
        # === 6. SAVINGS UPDATE ===
        self.state.savings += self.state.disposable_income - self.state.consumption
        
        # === 7. NEW BORROWING ===
        if action.get("wants_to_borrow", False) and self.state.loan_amount == 0:
            loan_result = market_outcomes.get(f"household_{self.id}_loan")
            if loan_result:
                self.state.loan_amount = loan_result["amount"]
                self.state.loan_rate = loan_result["rate"]
                self.state.loan_bank_id = loan_result["bank_id"]
                self.state.months_remaining = self.loan_term
                self.state.savings += loan_result["amount"]
        
        # === 8. DEFAULT CHECK ===
        if self.state.loan_amount > 0:
            if self.state.disposable_income > 0:
                dti = self.state.debt_service / self.state.disposable_income
            else:
                dti = 1.0
            
            if dti > self.max_dti and self.state.savings < self.state.debt_service:
                self._handle_default(market_outcomes)
        
        # === 9. ADAPTIVE EXPECTATIONS (Phase 4) ===
        # Update inflation expectations based on observed inflation
        observed_inflation = market_outcomes.get("inflation", self.config.economic.inflation_target)
        learning_rate = 0.3
        self.state.expected_inflation = (
            learning_rate * observed_inflation +
            (1 - learning_rate) * self.state.expected_inflation
        )
        
        # Update income expectations (already using permanent income estimate)
        self.state.expected_income = self.state.permanent_income_estimate
        
        # === 10. BASE CLASS FIELDS ===
        self.state.wealth = self.state.savings - self.state.loan_amount
        self.state.income = self.state.labor_income
        self.state.debt = self.state.loan_amount
    
    def _handle_default(self, market_outcomes: Dict[str, Any]) -> None:
        """Handle loan default."""
        if self.state.loan_bank_id is not None:
            default_key = f"bank_{self.state.loan_bank_id}_defaults"
            current = market_outcomes.get(default_key, 0.0)
            market_outcomes[default_key] = current + self.state.loan_amount
        
        self.state.loan_amount = 0
        self.state.loan_rate = 0
        self.state.loan_bank_id = None
        self.state.months_remaining = 0
    
    def get_action_space_size(self) -> int:
        return self._action_space_size
    
    def get_observation_size(self) -> int:
        return self.config.network.household_input_size
    
    def get_borrowing_demand(self, lending_rate: float) -> float:
        """Compute rate-sensitive borrowing demand."""
        if self.state.loan_amount > 0:
            return 0.0
        
        if not self.state.is_employed:
            return 0.0
        
        # Rate sensitivity
        annual_rate = lending_rate * 12
        demand_factor = np.exp(-self.rate_sensitivity * annual_rate)
        
        # Base demand (proportion of income)
        base_demand = self.state.labor_income * 6
        
        return base_demand * demand_factor
    
    def get_risk_score(self) -> float:
        """Return borrower risk score [0, 1]."""
        risk = 0.0
        
        if not self.state.is_employed:
            risk += 0.4
        
        if self.state.savings < self.state.labor_income * 2:
            risk += 0.2
        
        if self.state.months_unemployed > 0:
            risk += 0.1 * min(self.state.months_unemployed, 4)
        
        return min(risk, 1.0)