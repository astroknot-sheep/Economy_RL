"""
Central Bank Agent - FIXED Implementation

CRITICAL FIXES:
1. Reward function now rewards FOLLOWING Taylor Rule, not just low volatility
2. Rate change penalty REMOVED - we WANT the CB to move rates
3. Added Taylor Rule deviation penalty - CB is punished for NOT following Taylor
4. Better observation space with momentum/trend indicators
5. Action space is LEVELS not changes - CB picks a rate level

TAYLOR RULE (Taylor 1993):
r = r* + π + 1.5*(π - π*) + 0.5*y

The CB should learn to approximate this rule through RL.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

from .base_agent import BaseAgent, AgentState


@dataclass
class CentralBankState(AgentState):
    """State specific to Central Bank."""
    
    # Policy instruments (monthly rates)
    policy_rate: float = 0.004  # Start at ~5% annual, not 0%
    
    # Observed macro variables
    inflation: float = 0.02  # Annualized
    inflation_gap: float = 0.0
    output_gap: float = 0.0
    unemployment: float = 0.045
    unemployment_gap: float = 0.0
    credit_growth: float = 0.0
    
    # Computed reference
    taylor_rule_rate: float = 0.004
    
    # History for smoothing and trends
    inflation_history: List[float] = field(default_factory=list)
    output_gap_history: List[float] = field(default_factory=list)
    rate_history: List[float] = field(default_factory=list)
    
    def to_array(self) -> np.ndarray:
        """Convert to normalized observation array."""
        # Current values
        inflation_norm = np.clip(self.inflation, -0.05, 0.15)
        inflation_gap_norm = np.clip(self.inflation_gap, -0.05, 0.10)
        output_gap_norm = np.clip(self.output_gap, -0.15, 0.15)
        unemployment_norm = np.clip(self.unemployment, 0, 0.20)
        unemployment_gap_norm = np.clip(self.unemployment_gap, -0.05, 0.10)
        
        # Current rate vs Taylor
        current_rate_annual = self.policy_rate * 12
        taylor_rate_annual = self.taylor_rule_rate * 12
        rate_gap = current_rate_annual - taylor_rate_annual
        
        # Trend indicators (are things getting worse or better?)
        if len(self.inflation_history) >= 3:
            inflation_trend = self.inflation - np.mean(self.inflation_history[-3:])
        else:
            inflation_trend = 0.0
            
        if len(self.output_gap_history) >= 3:
            output_trend = self.output_gap - np.mean(self.output_gap_history[-3:])
        else:
            output_trend = 0.0
        
        # Moving averages
        if len(self.inflation_history) >= 6:
            avg_inflation_6m = np.mean(self.inflation_history[-6:])
        else:
            avg_inflation_6m = self.inflation
            
        if len(self.output_gap_history) >= 6:
            avg_output_gap_6m = np.mean(self.output_gap_history[-6:])
        else:
            avg_output_gap_6m = self.output_gap
        
        arr = np.array([
            inflation_norm,
            inflation_gap_norm,
            output_gap_norm,
            unemployment_norm,
            unemployment_gap_norm,
            np.clip(current_rate_annual, 0, 0.15),
            np.clip(taylor_rate_annual, 0, 0.15),
            np.clip(rate_gap, -0.10, 0.10),
            np.clip(inflation_trend, -0.05, 0.05),
            np.clip(output_trend, -0.10, 0.10),
            np.clip(avg_inflation_6m, -0.05, 0.15),
            np.clip(avg_output_gap_6m, -0.15, 0.15),
        ], dtype=np.float32)
        
        return np.nan_to_num(arr, nan=0.0, posinf=0.15, neginf=-0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "policy_rate_annual": self.policy_rate * 12,
            "taylor_rate_annual": self.taylor_rule_rate * 12,
            "inflation": self.inflation,
            "output_gap": self.output_gap,
            "unemployment": self.unemployment,
        })
        return base


class CentralBank(BaseAgent):
    """
    Central Bank that sets monetary policy.
    
    KEY DESIGN CHANGES:
    1. Actions are RATE LEVELS, not changes (0% to 10% in 0.5% steps)
    2. Reward ENCOURAGES following Taylor Rule
    3. NO penalty for rate changes - we want active policy
    4. Trend/momentum in observations helps forward-looking policy
    """
    
    def __init__(self, agent_id: int, config: Any):
        super().__init__(agent_id, config)
        
        # Action space: rate LEVELS from config (annual rates)
        # Convert to monthly for internal use
        self.rate_levels_annual = config.actions.central_bank_actions
        self.rate_levels = [r / 12 for r in self.rate_levels_annual]  # Convert to monthly
        self._action_space_size = len(self.rate_levels)
        
        # Policy parameters (keep both naming conventions for compatibility)
        self.r_star = config.economic.neutral_real_rate / 12  # Monthly
        self.neutral_rate = config.economic.neutral_real_rate  # Annual (for Taylor Rule)
        self.inflation_target = config.economic.inflation_target / 12  # Monthly
        self.taylor_inflation_coef = config.economic.taylor_inflation_coef
        self.taylor_output_coef = config.economic.taylor_output_coef
        self.taylor_inflation_weight = 1.5
        self.taylor_output_weight = 0.5
        
        # STRUCTURAL FIX: Reduce smoothing to allow aggressive policy
        # Real central banks smooth rates, but not 80% - more like 50-60%
        self.smoothing = 0.50  # Reduced from 0.80 - allow bigger moves
        
        self.nairu = config.economic.natural_unemployment_rate
    
    def reset(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        """Initialize central bank state."""
        ic = initial_conditions or {}
        
        # Start at neutral rate, not zero!
        initial_rate = ic.get("policy_rate", 0.004)  # ~5% annual
        
        self.state = CentralBankState(
            id=self.id,
            policy_rate=initial_rate,
            taylor_rule_rate=initial_rate,
            inflation=self.inflation_target,
            inflation_gap=0.0,
            output_gap=0.0,
            unemployment=self.nairu,
            unemployment_gap=0.0,
            credit_growth=0.0,
            is_active=True,
        )
        
        self.state.rate_history = [initial_rate] * 6
        self.state.inflation_history = [self.inflation_target] * 6
        self.state.output_gap_history = [0.0] * 6
        
        self.state.wealth = 0
        self.state.income = 0
        self.state.debt = 0
    
    def get_observation(self, global_state: Dict[str, Any]) -> np.ndarray:
        """Construct observation from macro state."""
        # Update internal state from global
        self.state.inflation = global_state.get("inflation", 0.02)
        self.state.output_gap = global_state.get("output_gap", 0.0)
        self.state.unemployment = global_state.get("unemployment", 0.045)
        self.state.credit_growth = global_state.get("credit_growth", 0.0)
        
        # Compute gaps
        self.state.inflation_gap = self.state.inflation - self.inflation_target
        self.state.unemployment_gap = self.state.unemployment - self.nairu
        
        # Compute Taylor Rule rate
        self.state.taylor_rule_rate = self.compute_taylor_rule_rate()
        
        return self.state.to_array()
    
    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Convert action index to rate level."""
        action_idx = action_idx % self._action_space_size
        target_rate = self.rate_levels[action_idx]
        
        return {
            "target_rate": target_rate,
            "target_rate_annual": target_rate * 12,
            "action_idx": action_idx,
        }
    
    def compute_reward(
        self,
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        global_state: Dict[str, Any],
    ) -> float:
        """
        Central Bank reward - AGGRESSIVE TAYLOR RULE FOLLOWING.
        
        STRUCTURAL FIX: Make Taylor Rule THE PRIMARY objective.
        Remove soft caps, use quadratic penalty, reduce competing signals.
        """
        # Get rates
        taylor_rate = self.state.taylor_rule_rate * 12  # Annual
        chosen_rate = action["target_rate_annual"]
        
        # === REWARD COMPONENTS ===
        
        # 1. Taylor Rule following reward (AGGRESSIVE PENALTY - NO CAP)
        # CRITICAL FIX: Quadratic penalty with 10x multiplier, NO soft cap
        taylor_deviation = abs(chosen_rate - taylor_rate)
        taylor_reward = -(taylor_deviation ** 2) * 10  # Severe punishment
        
        # Bonus if very close to Taylor
        if taylor_deviation < 0.005:
            taylor_reward += 2.0  # Large bonus for precision
        elif taylor_deviation < 0.01:
            taylor_reward += 1.0
        
        # 2. Macro outcomes (REDUCED - Don't compete with Taylor)
        # Taylor Rule ALREADY incorporates inflation and output gaps
        # So we shouldn't double-penalize
        inflation_gap = self.state.inflation_gap
        output_gap = self.state.output_gap
        macro_penalty = -0.5 * (inflation_gap ** 2 + 0.25 * output_gap ** 2)  # Reduced from -2
        
        # 3. Stability bonus (REDUCED - Taylor is king)
        if abs(inflation_gap) < 0.01 and abs(output_gap) < 0.05:
            stability_bonus = 0.3  # Reduced from 0.5
        elif abs(inflation_gap) < 0.02:
            stability_bonus = 0.1  # Reduced from 0.2
        else:
            stability_bonus = 0.0
        
        # 4. Activity bonus - reward for NOT being at zero bound
        if chosen_rate > 0.01:
            activity_bonus = 0.2
        else:
            activity_bonus = 0.0
        
        # Total reward - Taylor Rule is 90% of the signal
        reward = taylor_reward + macro_penalty + stability_bonus + activity_bonus
        
        # WIDER CLIPPING: Allow larger penalties to drive learning
        return float(np.clip(reward, -20, 3))  # Increased from [-5, 2]

    
    def update_state(
        self,
        action: Dict[str, Any],
        market_outcomes: Dict[str, Any],
    ) -> None:
        """Update policy rate with smoothing."""
        target_rate = action["target_rate"]
        old_rate = self.state.policy_rate
        
        # Apply smoothing toward target
        new_rate = self.smoothing * old_rate + (1 - self.smoothing) * target_rate
        
        # Bounds: 0% to 12% annual (0% to 1% monthly)
        self.state.policy_rate = np.clip(new_rate, 0.0, 0.01)
        
        # Update history
        self.state.rate_history.append(self.state.policy_rate)
        if len(self.state.rate_history) > 24:
            self.state.rate_history = self.state.rate_history[-24:]
    
    def compute_taylor_rule_rate(self) -> float:
        """
        Compute Taylor Rule suggested rate.
        
        r = r* + π + 1.5*(π - π*) + 0.5*y
        """
        r_star = self.neutral_rate
        pi = self.state.inflation
        pi_star = self.inflation_target
        y = self.state.output_gap
        
        taylor_rate = (
            r_star + 
            pi + 
            self.taylor_inflation_coef * (pi - pi_star) + 
            self.taylor_output_coef * y
        )
        
        taylor_rate_monthly = taylor_rate / 12
        
        return np.clip(taylor_rate_monthly, 0.0, 0.01)
    
    def get_action_space_size(self) -> int:
        return self._action_space_size
    
    def get_observation_size(self) -> int:
        return 12
    
    def get_policy_rate(self) -> float:
        """Return current policy rate (monthly)."""
        return self.state.policy_rate if self.state else 0.004