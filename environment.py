"""
Macroeconomic Simulation Environment - Corrected Implementation

This environment orchestrates all agents and markets for a
multi-agent macroeconomic simulation with proper monetary
policy transmission.

KEY FIXES:
1. Proper ordering of market clearing
2. Correct information flow between markets
3. Consistent units (monthly rates, real vs nominal)
4. Clean separation between agent decisions and market outcomes
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from config import Config, DEFAULT_CONFIG
from agents import CentralBank, CommercialBank, Household, Firm
from markets import LaborMarket, CreditMarket, GoodsMarket
from economics import AggregateComputer, MacroState, compute_wealth_distribution


@dataclass
class StepResult:
    """Result from a single simulation step."""
    macro_state: MacroState
    rewards: Dict[str, Dict[int, float]]
    dones: Dict[str, bool]
    info: Dict[str, Any]


class MacroEconEnvironment:
    """
    Multi-agent macroeconomic simulation.
    
    TIMING WITHIN EACH PERIOD:
    1. Central Bank sets policy rate
    2. Banks set lending/deposit rates (passthrough)
    3. Labor market clears (employment, wages)
    4. Credit market clears (new loans)
    5. Firms produce
    6. Households consume
    7. Goods market clears (sales, prices, inflation)
    8. All agents update state
    9. Aggregates computed
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
        
        np.random.seed(self.config.seed)
        
        self._init_agents()
        self._init_markets()
        
        self.aggregate_computer = AggregateComputer(self.config)
        
        self.current_step = 0
        self.macro_state: Optional[MacroState] = None
        self.history: List[Dict[str, Any]] = []
    
    def _init_agents(self) -> None:
        """Initialize all agents."""
        self.central_bank = CentralBank(agent_id=0, config=self.config)
        
        self.banks: List[CommercialBank] = [
            CommercialBank(agent_id=i, config=self.config)
            for i in range(self.config.economic.num_commercial_banks)
        ]
        
        self.households: List[Household] = [
            Household(agent_id=i, config=self.config)
            for i in range(self.config.economic.num_households)
        ]
        
        self.firms: List[Firm] = [
            Firm(agent_id=i, config=self.config)
            for i in range(self.config.economic.num_firms)
        ]
    
    def _init_markets(self) -> None:
        """Initialize markets."""
        self.labor_market = LaborMarket(self.config)
        self.credit_market = CreditMarket(self.config)
        self.goods_market = GoodsMarket(self.config)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        self.current_step = 0
        
        # Reset all agents
        self.central_bank.reset()
        for bank in self.banks:
            bank.reset()
        for household in self.households:
            household.reset()
        for firm in self.firms:
            firm.reset()
        
        # CRITICAL FIX: Initial labor market matching
        # Assign employed households to firms at start
        employed_households = [h for h in self.households if h.state.is_employed]
        firm_slots = []
        for f in self.firms:
            for _ in range(f.state.num_workers):
                firm_slots.append(f.id)
        
        # Randomly assign workers to firms
        import random
        random.shuffle(employed_households)
        for i, h in enumerate(employed_households):
            if i < len(firm_slots):
                h.state.employer_id = firm_slots[i]
            else:
                # More workers than slots - mark as unemployed
                h.state.is_employed = False
                h.state.employer_id = None
        
        # Reset markets
        self.labor_market.reset()
        self.credit_market.reset()
        self.goods_market.reset()
        
        # Reset aggregates
        self.aggregate_computer.reset()
        
        # Initial macro state
        self.macro_state = MacroState(
            policy_rate=self.central_bank.get_policy_rate(),
            inflation=self.config.economic.inflation_target * 12,
            unemployment=self.config.economic.natural_unemployment_rate,
            gdp=100.0,
            potential_gdp=100.0,
        )
        
        self.history = []
        
        return self._get_observations()
    
    def _get_observations(self) -> Dict[str, Dict[int, np.ndarray]]:
        """Get observations for all agents."""
        global_state = self.macro_state.to_dict() if self.macro_state else {}
        
        return {
            "central_bank": {
                0: self.central_bank.get_observation(global_state)
            },
            "banks": {
                bank.id: bank.get_observation(global_state)
                for bank in self.banks if bank.is_active
            },
            "households": {
                h.id: h.get_observation(global_state)
                for h in self.households if h.is_active
            },
            "firms": {
                f.id: f.get_observation(global_state)
                for f in self.firms if f.is_active
            },
        }
    
    def step(self, actions: Dict[str, Dict[int, int]]) -> StepResult:
        """
        Execute one step of the simulation.
        
        ORDERING IS CRITICAL FOR CORRECT DYNAMICS.
        """
        self.current_step += 1
        
        # === 1. CENTRAL BANK SETS POLICY RATE ===
        cb_action_idx = actions["central_bank"][0]
        cb_action = self.central_bank.decode_action(cb_action_idx)
        self.central_bank.update_state(cb_action, {})
        policy_rate = self.central_bank.get_policy_rate()
        
        # === 2. BANKS SET RATES ===
        bank_actions = {}
        for bank in self.banks:
            if bank.is_active and bank.id in actions["banks"]:
                action_idx = actions["banks"][bank.id]
                action = bank.decode_action(action_idx)
                bank_actions[bank.id] = action
                
                lending_rate, deposit_rate = bank.compute_rates(action, policy_rate)
                bank.state.lending_rate = lending_rate
                bank.state.deposit_rate = deposit_rate
                bank.state.risk_tolerance = action["risk_tolerance"]
        
        # === 3. LABOR MARKET CLEARING ===
        prev_unemployment = self.macro_state.unemployment if self.macro_state else 0.05
        labor_outcome = self.labor_market.clear(
            self.households,
            self.firms,
            previous_unemployment=prev_unemployment,
        )
        
        # BALANCED: Minimum workers per firm (reduced from 4 to 2)
        min_workers_per_firm = max(2, self.config.economic.min_firm_workers)
        
        for firm in self.firms:
            if firm.is_active and firm.id in labor_outcome.firm_workers:
                allocated_workers = len(labor_outcome.firm_workers[firm.id])
                firm.state.num_workers = max(
                    allocated_workers,
                    min_workers_per_firm
                )
        
        # === 4. CREDIT MARKET CLEARING ===
        # CRITICAL FIX: Compute GDP growth from CURRENT production (forward-looking)
        # NOT from stale historical data. This fixes Credit-GDP correlation.
        
        # Calculate expected current GDP from production capacity
        current_production = sum(
            firm.get_production_capacity() for firm in self.firms if firm.is_active
        )
        # Estimate current GDP (production × average price)
        avg_price = np.mean([firm.state.price for firm in self.firms if firm.is_active])
        estimated_current_gdp = current_production * avg_price
        
        # Growth rate compared to LAST actual GDP
        if len(self.history) >= 1:
            prev_gdp = self.history[-1].get("gdp", 1.0)
            # ANNUALIZE the growth (multiply by 12) so banks react meaningfully
            gdp_growth = ((estimated_current_gdp - prev_gdp) / max(prev_gdp, 1.0)) * 12
        else:
            gdp_growth = 0.0
        
        credit_outcome = self.credit_market.clear(
            self.banks,
            self.households,
            self.firms,
            policy_rate,
            gdp_growth,  # Pass for procyclical lending
        )
        
        # === 5. BUILD MARKET OUTCOMES ===
        market_outcomes = {
            "policy_rate": policy_rate,
            "wage": labor_outcome.wage,
            "deposit_rate": credit_outcome.avg_deposit_rate,
            "lending_rate": credit_outcome.avg_lending_rate,
        }
        
        market_outcomes.update(credit_outcome.loan_results)
        
        for h in self.households:
            if h.is_active:
                market_outcomes[f"household_{h.id}_employed"] = h.state.is_employed
        
        # === 6. FIRM DECISIONS AND PRODUCTION ===
        for f in self.firms:
            if f.is_active and f.id in actions["firms"]:
                action_idx = actions["firms"][f.id]
                action = f.decode_action(action_idx)
        
        # === 7. HOUSEHOLD DECISIONS (FIXED: Compute consumption BEFORE market) ===
        household_actions = {}
        for h in self.households:
            if h.is_active and h.id in actions["households"]:
                action_idx = actions["households"][h.id]
                household_actions[h.id] = h.decode_action(action_idx)
                
                # CRITICAL FIX: Pre-compute consumption for goods market
                # This ensures demand is current, not stale
                consumption_rate = household_actions[h.id].get("consumption_rate", 0.85)
                
                # Simplified consumption calculation for market clearing
                # (Full update happens after market)
                available = h.state.disposable_income + max(h.state.savings, 0)
                target = h.state.permanent_income_estimate * 0.85  # Approx target
                h.state.consumption = min(consumption_rate * target, available)
                h.state.consumption = max(h.state.consumption, 0.3 * self.config.economic.base_wage)
        
        # === 8. GOODS MARKET CLEARING (now uses updated consumption) ===
        goods_outcome = self.goods_market.clear(self.households, self.firms)
        
        for firm_id, sales in goods_outcome.sales_by_firm.items():
            market_outcomes[f"firm_{firm_id}_sales"] = sales
        
        # === 9. UPDATE ALL AGENTS ===
        for h in self.households:
            if h.is_active and h.id in household_actions:
                h.update_state(household_actions[h.id], market_outcomes)
        
        for f in self.firms:
            if f.is_active and f.id in actions["firms"]:
                action = f.decode_action(actions["firms"][f.id])
                f.update_state(action, market_outcomes)
        
        for bank in self.banks:
            if bank.is_active and bank.id in bank_actions:
                bank.update_state(bank_actions[bank.id], market_outcomes)
        
        # === 10. COMPUTE AGGREGATES ===
        self.macro_state = self.aggregate_computer.compute(
            self.households,
            self.firms,
            self.banks,
            labor_outcome,
            credit_outcome,
            goods_outcome,
            policy_rate,
        )
        
        macro_dict = self.macro_state.to_dict()
        self.central_bank.state.inflation_history.append(macro_dict["inflation"])
        self.central_bank.state.output_gap_history.append(macro_dict["output_gap"])
        
        # === 11. COMPUTE REWARDS ===
        rewards = self._compute_rewards(actions, market_outcomes)
        
        # === 12. CHECK TERMINATION ===
        done = self.current_step >= self.config.economic.simulation_length
        dones = {
            "central_bank": done,
            "banks": done,
            "households": done,
            "firms": done,
        }
        
        # === 13. RECORD HISTORY ===
        self._record_history(actions, rewards)
        
        # === 14. BUILD INFO ===
        info = {
            "step": self.current_step,
            "macro_state": self.macro_state.to_dict(),
            "labor": {
                "unemployment": labor_outcome.unemployment_rate,
                "wage": labor_outcome.wage,
                "vacancies": labor_outcome.vacancies,
            },
            "credit": {
                "growth": credit_outcome.credit_growth,
                "lending_rate": credit_outcome.avg_lending_rate * 12,
                "spread": credit_outcome.credit_spread * 12,
            },
            "goods": {
                "inflation": goods_outcome.inflation,
                "price_level": goods_outcome.price_level,
                "capacity_utilization": goods_outcome.capacity_utilization,
            },
        }
        
        return StepResult(
            macro_state=self.macro_state,
            rewards=rewards,
            dones=dones,
            info=info,
        )
    
    def _compute_rewards(
        self,
        actions: Dict[str, Dict[int, int]],
        market_outcomes: Dict[str, Any],
    ) -> Dict[str, Dict[int, float]]:
        """Compute rewards for all agents."""
        global_state = self.macro_state.to_dict()
        
        rewards = {
            "central_bank": {},
            "banks": {},
            "households": {},
            "firms": {},
        }
        
        cb_action = self.central_bank.decode_action(actions["central_bank"][0])
        rewards["central_bank"][0] = self.central_bank.compute_reward(
            cb_action, {}, global_state
        )
        
        for bank in self.banks:
            if bank.is_active and bank.id in actions["banks"]:
                action = bank.decode_action(actions["banks"][bank.id])
                rewards["banks"][bank.id] = bank.compute_reward(
                    action, market_outcomes, global_state
                )
        
        for h in self.households:
            if h.is_active and h.id in actions["households"]:
                action = h.decode_action(actions["households"][h.id])
                rewards["households"][h.id] = h.compute_reward(
                    action, {}, global_state
                )
        
        for f in self.firms:
            if f.is_active and f.id in actions["firms"]:
                action = f.decode_action(actions["firms"][f.id])
                rewards["firms"][f.id] = f.compute_reward(
                    action, {}, global_state
                )
        
        return rewards
    
    def _record_history(
        self,
        actions: Dict[str, Dict[int, int]],
        rewards: Dict[str, Dict[int, float]],
    ) -> None:
        """Record step for analysis."""
        record = {
            "step": self.current_step,
            "macro_state": self.macro_state.to_dict() if self.macro_state else {},
            "wealth_distribution": compute_wealth_distribution(self.households),
            "cb_reward": rewards["central_bank"].get(0, 0),
            "mean_bank_reward": np.mean(list(rewards["banks"].values())) if rewards["banks"] else 0,
            "mean_household_reward": np.mean(list(rewards["households"].values())) if rewards["households"] else 0,
            "mean_firm_reward": np.mean(list(rewards["firms"].values())) if rewards["firms"] else 0,
        }
        self.history.append(record)
    
    def get_agent_configs(self) -> Dict[str, Dict]:
        """Get configuration for each agent type."""
        return {
            "central_bank": {
                "obs_size": self.config.network.cb_input_size,
                "action_size": len(self.config.actions.central_bank_actions),
            },
            "banks": {
                "obs_size": self.config.network.bank_input_size,
                "action_size": (
                    len(self.config.actions.lending_spread_actions) *
                    len(self.config.actions.deposit_spread_actions) *
                    len(self.config.actions.risk_tolerance_actions)
                ),
            },
            "households": {
                "obs_size": self.config.network.household_input_size,
                "action_size": (
                    len(self.config.actions.consumption_rate_actions) *
                    len(self.config.actions.borrowing_actions) *
                    len(self.config.actions.labor_supply_actions)
                ),
            },
            "firms": {
                "obs_size": self.config.network.firm_input_size,
                "action_size": (
                    len(self.config.actions.price_change_actions) *
                    len(self.config.actions.hiring_actions) *
                    len(self.config.actions.investment_actions)
                ),
            },
        }
    
    def get_history_dataframe(self):
        """Convert history to pandas DataFrame."""
        import pandas as pd
        
        records = []
        for h in self.history:
            record = {"step": h["step"]}
            record.update(h["macro_state"])
            record.update({f"wealth_{k}": v for k, v in h["wealth_distribution"].items()})
            record["cb_reward"] = h["cb_reward"]
            record["mean_bank_reward"] = h["mean_bank_reward"]
            record["mean_household_reward"] = h["mean_household_reward"]
            record["mean_firm_reward"] = h["mean_firm_reward"]
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_current_summary(self) -> Dict[str, float]:
        """Get summary of current state."""
        if not self.macro_state:
            return {}
        
        # Get CB rates in annual terms
        policy_rate = self.central_bank.state.policy_rate * 12 if self.central_bank.state else 0.05
        taylor_rate = self.central_bank.state.taylor_rule_rate * 12 if self.central_bank.state else 0.05
        
        return {
            "GDP": self.macro_state.gdp,
            "Inflation (%)": self.macro_state.inflation * 100,
            "Unemployment (%)": self.macro_state.unemployment * 100,
            "Policy Rate (%)": policy_rate * 100,
            "Taylor Rate (%)": taylor_rate * 100,
            "Output Gap (%)": self.macro_state.output_gap * 100,
        }