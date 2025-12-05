"""
Configuration for Multi-Agent Macroeconomic Simulation

CALIBRATION SOURCES:
- Smets & Wouters (2007): Standard DSGE parameters
- Taylor (1993, 1999): Monetary policy rule coefficients  
- Ball & Mankiw (2002): NAIRU and Phillips curve
- BLS/BEA: Depreciation rates, capital share

TIME CONVENTION:
- 1 period = 1 month
- 12 periods = 1 year
- Annual rates converted to monthly where applicable
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class EconomicConfig:
    """Economic parameters calibrated to US economy."""
    
    # === AGENT COUNTS ===
    num_households: int = 100
    num_firms: int = 20
    num_commercial_banks: int = 5
    
    # === TIME ===
    periods_per_year: int = 12
    simulation_length: int = 120  # 10 years
    
    # === PRODUCTION ===
    capital_share: float = 0.33
    labor_productivity: float = 5.0
    tfp_growth_rate: float = 0.02 / 12  # 2% annual TFP growth → monthly
    depreciation_rate: float = 0.10 / 12  # 10% annual → monthly
    markup_rate: float = 0.15
    price_stickiness: float = 0.65  # Calvo parameter (Phase 2: reduced for stronger Phillips)
    
    # === LABOR MARKET ===
    natural_unemployment_rate: float = 0.045  # 4.5% NAIRU
    separation_rate: float = 0.02  # Monthly
    job_finding_rate: float = 0.28  # Monthly
    matching_efficiency: float = 0.6
    wage_adjustment_speed: float = 0.02  # Phillips curve slope
    base_wage: float = 5.0
    min_firm_workers: int = 2
    unemployment_benefit_rate: float = 0.4
    benefit_duration: int = 6  # Months
    
    # === MONETARY POLICY ===
    inflation_target: float = 0.02 / 12  # 2% annual → monthly
    taylor_inflation_coef: float = 1.5
    taylor_output_coef: float = 0.5
    interest_rate_smoothing: float = 0.75
    neutral_real_rate: float = 0.02 / 12  # 2% annual → monthly
    
    # === BANKING ===
    min_capital_ratio: float = 0.08  # Basel III
    recovery_rate: float = 0.4
    loan_loss_reserve_rate: float = 0.025
    
    # === HOUSEHOLDS ===
    mpc_income: float = 0.07  # Phase 3: Monthly consumption from permanent income (~85% annual)
    mpc_wealth: float = 0.003  # Monthly (~4% annual)
    precautionary_months: float = 6.0
    borrowing_rate_sensitivity: float = 15.0  # Phase 2: increased for stronger transmission
    deposit_rate_sensitivity: float = 8.0  # Phase 2: added for savings response
    max_dti: float = 0.5
    loan_term_months: int = 60
    
    # === INITIAL CONDITIONS ===
    initial_firm_capital: float = 100.0
    initial_firm_workers: int = 5
    initial_firm_inventory: float = 20.0
    initial_firm_cash: float = 50.0
    initial_household_savings: float = 30.0
    initial_bank_capital: float = 100.0
    initial_bank_deposits: float = 500.0


@dataclass
class ActionConfig:
    """Discrete action spaces for all agents."""
    
    # Central Bank: ABSOLUTE policy rate levels (annual)
    # Range: 0% to 10% in 0.5% steps = 21 actions
    # CRITICAL: Using absolute levels prevents getting stuck at 0%
    central_bank_actions: List[float] = field(default_factory=lambda: [
        0.000,  # 0.0% - zero lower bound
        0.005,  # 0.5%
        0.010,  # 1.0%
        0.015,  # 1.5%
        0.020,  # 2.0% - near neutral
        0.025,  # 2.5%
        0.030,  # 3.0%
        0.035,  # 3.5%
        0.040,  # 4.0%
        0.045,  # 4.5%
        0.050,  # 5.0% - neutral
        0.055,  # 5.5%
        0.060,  # 6.0%
        0.065,  # 6.5%
        0.070,  # 7.0%
        0.075,  # 7.5%
        0.080,  # 8.0%
        0.085,  # 8.5%
        0.090,  # 9.0%
        0.095,  # 9.5%
        0.100,  # 10.0%
    ])
    
    # Commercial Banks: lending spread over policy rate (monthly)
    lending_spread_actions: List[float] = field(default_factory=lambda: [
        0.002,   # 2.4% annual spread
        0.003,   # 3.6% annual
        0.004,   # 4.8% annual
        0.005,   # 6.0% annual
    ])
    
    # Commercial Banks: deposit spread below policy rate
    deposit_spread_actions: List[float] = field(default_factory=lambda: [
        -0.002,  # Pay 2.4% less than policy
        -0.001,  # Pay 1.2% less
        0.0,     # Pay policy rate
    ])
    
    # Commercial Banks: risk tolerance
    risk_tolerance_actions: List[float] = field(default_factory=lambda: [
        0.3,  # Conservative
        0.5,  # Moderate
        0.7,  # Aggressive
    ])
    
    # Households: consumption rate out of available funds
    consumption_rate_actions: List[float] = field(default_factory=lambda: [
        0.7,   # Low consumption
        0.8,   # Moderate
        0.9,   # High
        0.95,  # Very high
    ])
    
    # Households: borrowing decision
    borrowing_actions: List[int] = field(default_factory=lambda: [
        0,  # Don't borrow
        1,  # Borrow if needed
    ])
    
    # Households: labor supply intensity
    labor_supply_actions: List[float] = field(default_factory=lambda: [
        0.8,   # Part-time
        1.0,   # Full-time
        1.1,   # Overtime
    ])
    
    # Firms: price adjustment
    price_change_actions: List[float] = field(default_factory=lambda: [
        -0.02,  # Cut 2%
        -0.01,  # Cut 1%
        0.0,    # Hold
        0.01,   # Raise 1%
        0.02,   # Raise 2%
    ])
    
    # Firms: hiring/firing
    hiring_actions: List[int] = field(default_factory=lambda: [
        -2,  # Fire 2
        -1,  # Fire 1
        0,   # Hold
        1,   # Hire 1
        2,   # Hire 2
    ])
    
    # Firms: investment rate
    investment_actions: List[float] = field(default_factory=lambda: [
        0.0,    # No investment
        0.02,   # 2% of capital
        0.05,   # 5% of capital
        0.10,   # 10% of capital
    ])


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    
    # Input sizes (observation dimensions)
    cb_input_size: int = 12
    bank_input_size: int = 12
    household_input_size: int = 15
    firm_input_size: int = 15
    
    # Hidden layers
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"


@dataclass
class TrainingConfig:
    """PPO training configuration."""
    
    device: str = "cpu"
    learning_rate: float = 3e-4
    gamma: float = 0.97  # Phase 3: Monthly discounting (0.995 too close to 1 for monthly)
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.05  # Phase 3: Increased for better exploration
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training loop
    num_epochs: int = 300
    steps_per_epoch: int = 120
    update_epochs: int = 10  # PPO updates per rollout
    minibatch_size: int = 32
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50


@dataclass
class Config:
    """Master configuration."""
    
    seed: int = 42
    economic: EconomicConfig = field(default_factory=EconomicConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Default configuration instance
DEFAULT_CONFIG = Config()


def create_config(
    num_households: int = 100,
    num_firms: int = 20,
    num_banks: int = 5,
    simulation_length: int = 120,
    num_epochs: int = 300,
    seed: int = 42,
) -> Config:
    """Create a configuration with custom parameters."""
    config = Config(seed=seed)
    config.economic.num_households = num_households
    config.economic.num_firms = num_firms
    config.economic.num_commercial_banks = num_banks
    config.economic.simulation_length = simulation_length
    config.training.num_epochs = num_epochs
    return config