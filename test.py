"""
Ultimate Single Model Validation System
=========================================

This script tests ONE model to its MAXIMUM capability against real-world
economic data and produces a comprehensive "Reality Score" - how close
the simulation matches actual economic behavior.

Tests based on:
- Federal Reserve Economic Data (FRED) 1984-2019
- Bureau of Labor Statistics (BLS)
- Bureau of Economic Analysis (BEA)
- Academic literature: Smets & Wouters (2007), Christiano et al. (2005),
  Stock & Watson (1999), Romer & Romer (2004), Taylor (1993),
  Blanchard & Kahn (1980), Kydland & Prescott (1982), Gali (2015)

Usage:
    python test_single_model_comprehensive.py --model /path/to/model.pt --runs 20
"""

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import os
from datetime import datetime
from scipy import stats
from scipy.signal import correlate
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# REAL-WORLD BENCHMARK DATA (POST-1984 GREAT MODERATION ERA)
# =============================================================================

@dataclass
class RealWorldBenchmarks:
    """
    Comprehensive real-world economic benchmarks from US data.
    Primary source: FRED, BLS, BEA (1984-2019 "Great Moderation" era)
    """
    
    # === GDP (BEA National Accounts) ===
    gdp_growth_mean: float = 2.7           # % annual
    gdp_growth_std: float = 2.1            # %
    gdp_growth_skew: float = -0.5          # Negative skew (recessions)
    gdp_growth_kurtosis: float = 4.2       # Fat tails
    gdp_autocorr_1: float = 0.85           # AR(1)
    gdp_autocorr_4: float = 0.60           # AR(4)
    
    # === INFLATION (BLS CPI-U) ===
    inflation_mean: float = 2.8            # %
    inflation_std: float = 1.3             # %
    inflation_target: float = 2.0          # Fed target
    inflation_persistence: float = 0.85    # AR(1)
    inflation_half_life: float = 4.5       # Quarters to mean-revert
    
    # === UNEMPLOYMENT (BLS) ===
    unemployment_mean: float = 5.8         # %
    unemployment_std: float = 1.6          # %
    unemployment_min: float = 3.5          # %
    unemployment_max: float = 10.0         # %
    nairu: float = 4.5                     # Natural rate
    unemployment_persistence: float = 0.95 # Very sticky
    
    # === INTEREST RATES (Federal Reserve) ===
    fed_funds_mean: float = 4.2            # %
    fed_funds_std: float = 2.8             # %
    fed_funds_min: float = 0.0             # ZLB
    fed_funds_max: float = 11.5            # %
    rate_smoothing: float = 0.75           # Inertia coefficient
    
    # === TAYLOR RULE COEFFICIENTS ===
    taylor_inflation: float = 1.5          # Response to π gap
    taylor_output: float = 0.5             # Response to y gap
    taylor_r_star: float = 2.0             # Neutral real rate
    
    # === PHILLIPS CURVE ===
    phillips_slope: float = -0.3           # Δπ/Δu slope (flattened)
    okun_coefficient: float = 2.0          # GDP loss per 1% unemployment
    sacrifice_ratio: float = 2.0           # Unemployment cost of 1% disinflation
    
    # === BUSINESS CYCLE CORRELATIONS (Stock & Watson 1999) ===
    corr_consumption_gdp: float = 0.85
    corr_investment_gdp: float = 0.90
    corr_unemployment_gdp: float = -0.85
    corr_inflation_gdp: float = 0.30
    corr_hours_gdp: float = 0.88
    corr_wages_gdp: float = 0.12
    corr_productivity_gdp: float = 0.40
    
    # === VOLATILITY RATIOS (rel. to GDP) ===
    vol_consumption_rel: float = 0.75      # Smoother
    vol_investment_rel: float = 3.0        # 3x more volatile
    vol_hours_rel: float = 0.65
    vol_wages_rel: float = 0.40
    
    # === MONETARY TRANSMISSION (Romer & Romer 2004) ===
    rate_to_gdp_lag: int = 18              # Months to peak effect
    rate_to_inflation_lag: int = 24        # Months
    rate_to_gdp_magnitude: float = -0.4    # % GDP per 1% rate
    rate_to_inflation_magnitude: float = -0.3
    
    # === WEALTH DISTRIBUTION (Fed SCF) ===
    gini_wealth: float = 0.85
    top_10_share: float = 0.70
    top_1_share: float = 0.35
    bottom_50_share: float = 0.02
    
    # === BUSINESS CYCLE FACTS (NBER) ===
    avg_expansion_months: int = 58
    avg_recession_months: int = 11
    recession_frequency: float = 0.15      # % of time
    recession_depth_avg: float = -2.5      # % GDP decline
    
    # === CREDIT (Fed Z.1) ===
    credit_gdp_ratio: float = 160          # % of GDP
    credit_growth_mean: float = 6.5        # %
    credit_growth_std: float = 4.2         # %
    credit_gdp_correlation: float = 0.70


# =============================================================================
# HISTORICAL CRISIS DATABASE - Real-world economic events for validation
# =============================================================================

HISTORICAL_CRISES = [
    # USA Events
    {"country": "USA", "name": "2008 Financial Crisis", "year": 2008, "type": "Financial_Crisis",
     "gdp_impact": -4.0, "peak_unemployment": 10.0, "peak_inflation": 3.8, "policy_rate_change": -525,
     "recovery_quarters": 18, "severity": 9.5},
    {"country": "USA", "name": "Dot-com Bubble", "year": 2001, "type": "Asset_Bubble",
     "gdp_impact": -0.3, "peak_unemployment": 6.3, "peak_inflation": 3.4, "policy_rate_change": -475,
     "recovery_quarters": 8, "severity": 5.0},
    {"country": "USA", "name": "COVID-19 Recession", "year": 2020, "type": "Pandemic_Shock",
     "gdp_impact": -3.4, "peak_unemployment": 14.7, "peak_inflation": 1.2, "policy_rate_change": -150,
     "recovery_quarters": 4, "severity": 7.5},
    {"country": "USA", "name": "1990-91 Recession", "year": 1991, "type": "Cyclical_Recession",
     "gdp_impact": -1.4, "peak_unemployment": 7.8, "peak_inflation": 5.4, "policy_rate_change": -300,
     "recovery_quarters": 3, "severity": 4.5},
    {"country": "USA", "name": "Volcker Disinflation", "year": 1982, "type": "Policy_Induced",
     "gdp_impact": -2.7, "peak_unemployment": 10.8, "peak_inflation": 13.5, "policy_rate_change": 800,
     "recovery_quarters": 6, "severity": 6.0},
    
    # Japan Events
    {"country": "Japan", "name": "Lost Decade", "year": 1990, "type": "Asset_Bubble_Burst",
     "gdp_impact": -0.5, "peak_unemployment": 5.5, "peak_inflation": 1.7, "policy_rate_change": -600,
     "recovery_quarters": 52, "severity": 9.0},
    {"country": "Japan", "name": "2008 Crisis", "year": 2009, "type": "External_Shock",
     "gdp_impact": -5.5, "peak_unemployment": 5.4, "peak_inflation": 1.4, "policy_rate_change": -40,
     "recovery_quarters": 8, "severity": 7.0},
    
    # Germany Events
    {"country": "Germany", "name": "Reunification Shock", "year": 1991, "type": "Structural_Shock",
     "gdp_impact": -1.5, "peak_unemployment": 11.0, "peak_inflation": 5.1, "policy_rate_change": 200,
     "recovery_quarters": 20, "severity": 7.5},
    {"country": "Germany", "name": "2008 Crisis", "year": 2009, "type": "Financial_Crisis",
     "gdp_impact": -5.6, "peak_unemployment": 8.0, "peak_inflation": 2.8, "policy_rate_change": 0,
     "recovery_quarters": 6, "severity": 7.5},
    {"country": "Germany", "name": "2022 Energy Crisis", "year": 2022, "type": "Energy_Crisis",
     "gdp_impact": -0.3, "peak_unemployment": 5.7, "peak_inflation": 8.7, "policy_rate_change": 400,
     "recovery_quarters": 8, "severity": 6.0},
    
    # UK Events
    {"country": "UK", "name": "Black Wednesday", "year": 1992, "type": "Currency_Crisis",
     "gdp_impact": 0.5, "peak_unemployment": 10.7, "peak_inflation": 3.7, "policy_rate_change": 0,
     "recovery_quarters": 4, "severity": 6.5},
    {"country": "UK", "name": "2008 Crisis", "year": 2009, "type": "Financial_Crisis",
     "gdp_impact": -4.3, "peak_unemployment": 8.5, "peak_inflation": 4.5, "policy_rate_change": -450,
     "recovery_quarters": 20, "severity": 8.5},
    {"country": "UK", "name": "COVID-19", "year": 2020, "type": "Pandemic_Shock",
     "gdp_impact": -9.3, "peak_unemployment": 5.2, "peak_inflation": 0.9, "policy_rate_change": -65,
     "recovery_quarters": 6, "severity": 7.5},
    
    # India Events
    {"country": "India", "name": "1991 BOP Crisis", "year": 1991, "type": "BOP_Crisis",
     "gdp_impact": 1.1, "peak_unemployment": 6.0, "peak_inflation": 13.9, "policy_rate_change": 0,
     "recovery_quarters": 8, "severity": 8.5},
    {"country": "India", "name": "Demonetization", "year": 2016, "type": "Policy_Shock",
     "gdp_impact": -0.5, "peak_unemployment": 5.0, "peak_inflation": 4.5, "policy_rate_change": 0,
     "recovery_quarters": 4, "severity": 5.5},
    {"country": "India", "name": "COVID-19", "year": 2020, "type": "Pandemic_Shock",
     "gdp_impact": -6.6, "peak_unemployment": 23.5, "peak_inflation": 6.2, "policy_rate_change": -115,
     "recovery_quarters": 6, "severity": 8.0},
    
    # Brazil Events
    {"country": "Brazil", "name": "Real Plan", "year": 1994, "type": "Hyperinflation_End",
     "gdp_impact": 4.0, "peak_unemployment": 7.6, "peak_inflation": 2477, "policy_rate_change": 0,
     "recovery_quarters": 20, "severity": 7.0},
    {"country": "Brazil", "name": "2014-16 Recession", "year": 2015, "type": "Commodity_Recession",
     "gdp_impact": -7.0, "peak_unemployment": 13.7, "peak_inflation": 10.7, "policy_rate_change": 700,
     "recovery_quarters": 11, "severity": 9.0},
    
    # Italy Events
    {"country": "Italy", "name": "Sovereign Debt Crisis", "year": 2012, "type": "Debt_Crisis",
     "gdp_impact": -2.8, "peak_unemployment": 12.7, "peak_inflation": 3.3, "policy_rate_change": 0,
     "recovery_quarters": 12, "severity": 8.5},
    {"country": "Italy", "name": "COVID-19", "year": 2020, "type": "Pandemic_Shock",
     "gdp_impact": -9.0, "peak_unemployment": 9.3, "peak_inflation": 0.6, "policy_rate_change": 0,
     "recovery_quarters": 6, "severity": 8.0},
    
    # Canada Events
    {"country": "Canada", "name": "2008 Crisis", "year": 2009, "type": "External_Shock",
     "gdp_impact": -3.7, "peak_unemployment": 8.7, "peak_inflation": 2.3, "policy_rate_change": -400,
     "recovery_quarters": 4, "severity": 6.0},
    {"country": "Canada", "name": "Oil Crash", "year": 2015, "type": "Commodity_Shock",
     "gdp_impact": -0.5, "peak_unemployment": 7.2, "peak_inflation": 1.1, "policy_rate_change": -100,
     "recovery_quarters": 8, "severity": 5.0},
    
    # China Events
    {"country": "China", "name": "2015 Stock Crash", "year": 2015, "type": "Asset_Bubble",
     "gdp_impact": -0.5, "peak_unemployment": 4.1, "peak_inflation": 1.4, "policy_rate_change": 0,
     "recovery_quarters": 4, "severity": 6.0},
    {"country": "China", "name": "Property Crisis", "year": 2022, "type": "Property_Crisis",
     "gdp_impact": -1.2, "peak_unemployment": 5.3, "peak_inflation": 0.2, "policy_rate_change": -50,
     "recovery_quarters": 12, "severity": 7.0},
    
    # France Events
    {"country": "France", "name": "2008 Crisis", "year": 2009, "type": "Financial_Crisis",
     "gdp_impact": -2.9, "peak_unemployment": 10.0, "peak_inflation": 3.2, "policy_rate_change": 0,
     "recovery_quarters": 8, "severity": 6.5},
    {"country": "France", "name": "COVID-19", "year": 2020, "type": "Pandemic_Shock",
     "gdp_impact": -7.9, "peak_unemployment": 8.9, "peak_inflation": 0.5, "policy_rate_change": 0,
     "recovery_quarters": 6, "severity": 7.0},
]

# Compute aggregate crisis statistics for benchmarking
def get_crisis_benchmarks():
    """Compute aggregate statistics from historical crises."""
    crises = HISTORICAL_CRISES
    
    return {
        "avg_gdp_impact": np.mean([c["gdp_impact"] for c in crises]),
        "avg_peak_unemployment": np.mean([c["peak_unemployment"] for c in crises]),
        "avg_recovery_quarters": np.mean([c["recovery_quarters"] for c in crises]),
        "avg_severity": np.mean([c["severity"] for c in crises]),
        "financial_crisis_gdp": np.mean([c["gdp_impact"] for c in crises if "Financial" in c["type"]]),
        "pandemic_gdp": np.mean([c["gdp_impact"] for c in crises if "Pandemic" in c["type"]]),
        "mild_recession_gdp": np.mean([c["gdp_impact"] for c in crises if abs(c["gdp_impact"]) < 2]),
        "severe_recession_gdp": np.mean([c["gdp_impact"] for c in crises if c["gdp_impact"] < -3]),
        "n_crises": len(crises),
        "crisis_types": list(set(c["type"] for c in crises)),
    }

CRISIS_BENCHMARKS = get_crisis_benchmarks()


# =============================================================================
# COMPREHENSIVE TEST SUITE
# =============================================================================

class UltimateModelValidator:
    """
    Ultimate validator that tests a single model to maximum capability.
    """
    
    def __init__(self):
        self.benchmarks = RealWorldBenchmarks()
        self.imports_ok = False
        self.test_results = []
        self.category_scores = {}
        self._setup_imports()
    
    def _setup_imports(self):
        """Import simulation components."""
        try:
            from config import DEFAULT_CONFIG
            from environment import MacroEconEnvironment
            from training import MultiAgentPPO, PPOConfig
            self.config = DEFAULT_CONFIG
            self.MacroEconEnvironment = MacroEconEnvironment
            self.MultiAgentPPO = MultiAgentPPO
            self.PPOConfig = PPOConfig
            self.imports_ok = True
        except ImportError as e:
            print(f"⚠️  Import error: {e}")
            self.imports_ok = False
    
    def load_model(self, model_path: str) -> Tuple[Any, bool]:
        """Load model from path."""
        if not self.imports_ok:
            return None, False
        
        env = self.MacroEconEnvironment(self.config)
        agent_configs = env.get_agent_configs()
        
        ppo = self.MultiAgentPPO(agent_configs=agent_configs, device="cpu")
        
        try:
            ppo.load(model_path)
            return ppo, True
        except Exception as e:
            print(f"  ⚠️  Failed to load: {e}")
            return ppo, False
    
    def run_simulations(self, ppo, n_runs: int, duration: int) -> List[pd.DataFrame]:
        """Run multiple simulations for robust statistics."""
        results = []
        
        for i in range(n_runs):
            print(f"\r  Simulation {i+1}/{n_runs}...", end="", flush=True)
            try:
                self.config.economic.simulation_length = duration
                env = self.MacroEconEnvironment(self.config)
                obs = env.reset()
                
                for _ in range(duration):
                    actions, _, _ = ppo.get_actions(obs, deterministic=True)
                    result = env.step(actions)
                    obs = env._get_observations()
                    if result.dones.get("central_bank", False):
                        break
                
                df = env.get_history_dataframe()
                results.append(df)
            except Exception as e:
                print(f"\n    ⚠️  Run {i+1} failed: {e}")
        
        print()
        return results
    
    # =========================================================================
    # DATA EXTRACTION HELPERS
    # =========================================================================
    
    def _to_array(self, series) -> np.ndarray:
        """Convert to numpy array."""
        if isinstance(series, pd.Series):
            return series.values
        return np.array(series)
    
    def _to_percent(self, arr: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """Convert to percent if needed."""
        if np.abs(arr).max() < threshold:
            return arr * 100
        return arr
    
    def _gdp_growth(self, df: pd.DataFrame) -> np.ndarray:
        """Compute annualized GDP growth."""
        gdp = np.maximum(df['gdp'].values, 1.0)
        return np.diff(np.log(gdp)) * 12 * 100
    
    def _autocorr(self, x: np.ndarray, lag: int = 1) -> float:
        """Compute autocorrelation."""
        if len(x) <= lag:
            return np.nan
        corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def _cross_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-correlation."""
        n = min(len(x), len(y))
        if n < 10:
            return np.nan
        corr = np.corrcoef(x[:n], y[:n])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    def _lagged_corr(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        """Compute lagged correlation."""
        if len(x) <= lag or len(y) <= lag:
            return np.nan
        n = min(len(x) - lag, len(y) - lag)
        if n < 10:
            return np.nan
        corr = np.corrcoef(x[:n], y[lag:lag+n])[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    # =========================================================================
    # SCORING SYSTEM
    # =========================================================================
    
    def _score_value(self, model_val: float, benchmark: float, 
                     tolerance: float, higher_better: bool = False) -> float:
        """
        Score how close model value is to benchmark.
        Returns 0-100 score.
        """
        if np.isnan(model_val) or np.isnan(benchmark):
            return 0.0
        
        error = abs(model_val - benchmark)
        relative_error = error / max(abs(benchmark), tolerance, 0.01)
        
        # Score decreases with error
        score = max(0, 100 * (1 - relative_error / 2))
        return min(100, score)
    
    def _score_range(self, model_val: float, low: float, high: float) -> float:
        """Score if value is within acceptable range."""
        if np.isnan(model_val):
            return 0.0
        
        if low <= model_val <= high:
            # Score based on how centered in range
            mid = (low + high) / 2
            range_size = high - low
            distance_from_mid = abs(model_val - mid) / (range_size / 2)
            return 100 * (1 - 0.3 * distance_from_mid)  # Penalize edges slightly
        else:
            # Outside range - penalize based on distance
            if model_val < low:
                overshoot = (low - model_val) / max(abs(low), 1)
            else:
                overshoot = (model_val - high) / max(abs(high), 1)
            return max(0, 50 * (1 - overshoot))
    
    def _score_correlation(self, model_corr: float, target_corr: float, 
                           sign_matters: bool = True) -> float:
        """Score correlation match."""
        if np.isnan(model_corr):
            return 0.0
        
        if sign_matters:
            # Check if signs match
            if np.sign(model_corr) != np.sign(target_corr) and abs(target_corr) > 0.1:
                return max(0, 30 * (1 - abs(model_corr - target_corr)))
        
        error = abs(model_corr - target_corr)
        return max(0, 100 * (1 - error))
    
    def _score_boolean(self, condition: bool) -> float:
        """Score boolean condition."""
        return 100.0 if condition else 0.0
    
    def _add_test(self, name: str, category: str, model_val: float, 
                  benchmark: float, score: float, weight: float = 1.0,
                  details: str = ""):
        """Add a test result."""
        self.test_results.append({
            "name": name,
            "category": category,
            "model_value": model_val,
            "benchmark": benchmark,
            "score": score,
            "weight": weight,
            "details": details
        })
    
    # =========================================================================
    # TEST CATEGORY 1: FIRST MOMENT STATISTICS (Means)
    # =========================================================================
    
    def test_first_moments(self, dfs: List[pd.DataFrame]) -> float:
        """Test mean values of key variables."""
        print("\n  📊 Testing First Moments (Means)...")
        
        all_gdp_growth = []
        all_inflation = []
        all_unemployment = []
        all_policy_rate = []
        
        for df in dfs:
            all_gdp_growth.extend(self._gdp_growth(df).tolist())
            
            infl = self._to_percent(self._to_array(df['inflation']))
            all_inflation.extend(infl.tolist())
            
            unemp = self._to_percent(self._to_array(df['unemployment']))
            all_unemployment.extend(unemp.tolist())
            
            rate = df['policy_rate'].values
            if rate.max() < 0.1:
                rate = rate * 12 * 100
            else:
                rate = rate * 100
            all_policy_rate.extend(rate.tolist())
        
        scores = []
        
        # GDP Growth Mean
        gdp_mean = np.mean(all_gdp_growth)
        s = self._score_value(gdp_mean, self.benchmarks.gdp_growth_mean, 2.0)
        self._add_test("GDP Growth Mean", "First Moments", gdp_mean, 
                      self.benchmarks.gdp_growth_mean, s, 2.0,
                      f"{gdp_mean:.2f}% vs {self.benchmarks.gdp_growth_mean}%")
        scores.append(s * 2.0)
        
        # Inflation Mean
        infl_mean = np.mean(all_inflation)
        s = self._score_value(infl_mean, self.benchmarks.inflation_mean, 1.5)
        self._add_test("Inflation Mean", "First Moments", infl_mean,
                      self.benchmarks.inflation_mean, s, 2.0)
        scores.append(s * 2.0)
        
        # Unemployment Mean
        unemp_mean = np.mean(all_unemployment)
        s = self._score_value(unemp_mean, self.benchmarks.unemployment_mean, 2.0)
        self._add_test("Unemployment Mean", "First Moments", unemp_mean,
                      self.benchmarks.unemployment_mean, s, 2.0)
        scores.append(s * 2.0)
        
        # Policy Rate Mean
        rate_mean = np.mean(all_policy_rate)
        s = self._score_value(rate_mean, self.benchmarks.fed_funds_mean, 2.0)
        self._add_test("Policy Rate Mean", "First Moments", rate_mean,
                      self.benchmarks.fed_funds_mean, s, 1.5)
        scores.append(s * 1.5)
        
        # NAIRU proximity
        s = self._score_value(unemp_mean, self.benchmarks.nairu, 1.5)
        self._add_test("Unemployment Near NAIRU", "First Moments", unemp_mean,
                      self.benchmarks.nairu, s, 1.0)
        scores.append(s * 1.0)
        
        # Inflation near target
        s = self._score_value(infl_mean, self.benchmarks.inflation_target, 1.0)
        self._add_test("Inflation Near Target", "First Moments", infl_mean,
                      self.benchmarks.inflation_target, s, 1.5)
        scores.append(s * 1.5)
        
        category_score = sum(scores) / sum([2.0, 2.0, 2.0, 1.5, 1.0, 1.5])
        self.category_scores["First Moments"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 2: SECOND MOMENT STATISTICS (Volatilities)
    # =========================================================================
    
    def test_second_moments(self, dfs: List[pd.DataFrame]) -> float:
        """Test volatilities and standard deviations."""
        print("  📈 Testing Second Moments (Volatilities)...")
        
        all_gdp_growth = []
        all_inflation = []
        all_unemployment = []
        all_policy_rate = []
        
        for df in dfs:
            all_gdp_growth.extend(self._gdp_growth(df).tolist())
            all_inflation.extend(self._to_percent(self._to_array(df['inflation'])).tolist())
            all_unemployment.extend(self._to_percent(self._to_array(df['unemployment'])).tolist())
            rate = df['policy_rate'].values
            if rate.max() < 0.1:
                rate = rate * 12 * 100
            all_policy_rate.extend(rate.tolist())
        
        scores = []
        
        # GDP Volatility
        gdp_std = np.std(all_gdp_growth)
        s = self._score_range(gdp_std, 0.5, 15.0)  # ABMs typically higher
        self._add_test("GDP Volatility", "Second Moments", gdp_std,
                      self.benchmarks.gdp_growth_std, s, 1.5)
        scores.append(s * 1.5)
        
        # Inflation Volatility
        infl_std = np.std(all_inflation)
        s = self._score_range(infl_std, 0.3, 5.0)
        self._add_test("Inflation Volatility", "Second Moments", infl_std,
                      self.benchmarks.inflation_std, s, 1.5)
        scores.append(s * 1.5)
        
        # Unemployment Volatility
        unemp_std = np.std(all_unemployment)
        s = self._score_range(unemp_std, 0.5, 5.0)
        self._add_test("Unemployment Volatility", "Second Moments", unemp_std,
                      self.benchmarks.unemployment_std, s, 1.5)
        scores.append(s * 1.5)
        
        # Policy Rate Volatility
        rate_std = np.std(all_policy_rate)
        s = self._score_range(rate_std, 0.3, 8.0)
        self._add_test("Policy Rate Volatility", "Second Moments", rate_std,
                      self.benchmarks.fed_funds_std, s, 1.0)
        scores.append(s * 1.0)
        
        # GDP Skewness (should be negative - recessions)
        gdp_skew = stats.skew(all_gdp_growth)
        s = self._score_range(gdp_skew, -2.0, 0.5)
        self._add_test("GDP Skewness (Negative)", "Second Moments", gdp_skew,
                      self.benchmarks.gdp_growth_skew, s, 1.0)
        scores.append(s * 1.0)
        
        # GDP Kurtosis (fat tails)
        gdp_kurt = stats.kurtosis(all_gdp_growth)
        s = self._score_range(gdp_kurt, 0.0, 10.0)
        self._add_test("GDP Kurtosis (Fat Tails)", "Second Moments", gdp_kurt,
                      self.benchmarks.gdp_growth_kurtosis, s, 0.5)
        scores.append(s * 0.5)
        
        category_score = sum(scores) / sum([1.5, 1.5, 1.5, 1.0, 1.0, 0.5])
        self.category_scores["Second Moments"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 3: PERSISTENCE (Autocorrelations)
    # =========================================================================
    
    def test_persistence(self, dfs: List[pd.DataFrame]) -> float:
        """Test time series persistence."""
        print("  🔄 Testing Persistence (Autocorrelations)...")
        
        gdp_ac1 = []
        gdp_ac4 = []
        infl_ac1 = []
        unemp_ac1 = []
        rate_ac1 = []
        
        for df in dfs:
            if len(df) > 12:
                gdp_g = self._gdp_growth(df)
                gdp_ac1.append(self._autocorr(gdp_g, 1))
                if len(gdp_g) > 4:
                    gdp_ac4.append(self._autocorr(gdp_g, 4))
                
                infl = self._to_percent(self._to_array(df['inflation']))
                infl_ac1.append(self._autocorr(infl, 1))
                
                unemp = self._to_percent(self._to_array(df['unemployment']))
                unemp_ac1.append(self._autocorr(unemp, 1))
                
                rate = df['policy_rate'].values
                rate_ac1.append(self._autocorr(rate, 1))
        
        scores = []
        
        # GDP AR(1)
        if gdp_ac1:
            avg = np.nanmean(gdp_ac1)
            s = self._score_correlation(avg, self.benchmarks.gdp_autocorr_1)
            self._add_test("GDP Persistence AR(1)", "Persistence", avg,
                          self.benchmarks.gdp_autocorr_1, s, 1.5)
            scores.append(s * 1.5)
        
        # GDP AR(4)
        if gdp_ac4:
            avg = np.nanmean(gdp_ac4)
            s = self._score_correlation(avg, self.benchmarks.gdp_autocorr_4)
            self._add_test("GDP Persistence AR(4)", "Persistence", avg,
                          self.benchmarks.gdp_autocorr_4, s, 1.0)
            scores.append(s * 1.0)
        
        # Inflation Persistence
        if infl_ac1:
            avg = np.nanmean(infl_ac1)
            s = self._score_correlation(avg, self.benchmarks.inflation_persistence)
            self._add_test("Inflation Persistence", "Persistence", avg,
                          self.benchmarks.inflation_persistence, s, 1.5)
            scores.append(s * 1.5)
        
        # Unemployment Persistence
        if unemp_ac1:
            avg = np.nanmean(unemp_ac1)
            s = self._score_correlation(avg, self.benchmarks.unemployment_persistence)
            self._add_test("Unemployment Persistence", "Persistence", avg,
                          self.benchmarks.unemployment_persistence, s, 1.5)
            scores.append(s * 1.5)
        
        # Interest Rate Smoothing
        if rate_ac1:
            avg = np.nanmean(rate_ac1)
            s = self._score_correlation(avg, self.benchmarks.rate_smoothing)
            self._add_test("Interest Rate Smoothing", "Persistence", avg,
                          self.benchmarks.rate_smoothing, s, 1.0)
            scores.append(s * 1.0)
        
        if not scores:
            return 0.0
        
        category_score = sum(scores) / sum([1.5, 1.0, 1.5, 1.5, 1.0][:len(scores)])
        self.category_scores["Persistence"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 4: BUSINESS CYCLE CO-MOVEMENTS
    # =========================================================================
    
    def test_comovements(self, dfs: List[pd.DataFrame]) -> float:
        """Test business cycle co-movements."""
        print("  🔗 Testing Business Cycle Co-movements...")
        
        unemp_gdp_corrs = []
        infl_gdp_corrs = []
        cons_gdp_corrs = []
        inv_gdp_corrs = []
        credit_gdp_corrs = []
        
        for df in dfs:
            gdp_g = self._gdp_growth(df)
            
            # Unemployment-GDP
            unemp = self._to_percent(self._to_array(df['unemployment']))
            unemp_change = np.diff(unemp)
            if len(unemp_change) == len(gdp_g):
                c = self._cross_corr(unemp_change, gdp_g)
                if not np.isnan(c):
                    unemp_gdp_corrs.append(c)
            
            # Inflation-GDP
            infl = self._to_percent(self._to_array(df['inflation']))
            infl_change = np.diff(infl)
            if len(infl_change) == len(gdp_g):
                c = self._cross_corr(infl_change, gdp_g)
                if not np.isnan(c):
                    infl_gdp_corrs.append(c)
            
            # Consumption-GDP
            if 'consumption' in df.columns:
                cons = df['consumption'].values
                cons_g = np.diff(np.log(np.maximum(cons, 1))) * 12 * 100
                if len(cons_g) == len(gdp_g):
                    c = self._cross_corr(cons_g, gdp_g)
                    if not np.isnan(c):
                        cons_gdp_corrs.append(c)
            
            # Investment-GDP
            if 'investment' in df.columns:
                inv = df['investment'].values
                inv_g = np.diff(np.log(np.maximum(inv, 1))) * 12 * 100
                if len(inv_g) == len(gdp_g):
                    c = self._cross_corr(inv_g, gdp_g)
                    if not np.isnan(c):
                        inv_gdp_corrs.append(c)
            
            # Credit-GDP
            if 'total_credit' in df.columns:
                credit = df['total_credit'].values
                credit_g = np.diff(np.log(np.maximum(credit, 1))) * 12 * 100
                if len(credit_g) == len(gdp_g):
                    c = self._cross_corr(credit_g, gdp_g)
                    if not np.isnan(c):
                        credit_gdp_corrs.append(c)
        
        scores = []
        weights = []
        
        # Unemployment-GDP (should be negative)
        if unemp_gdp_corrs:
            avg = np.nanmean(unemp_gdp_corrs)
            s = self._score_correlation(avg, self.benchmarks.corr_unemployment_gdp)
            self._add_test("Unemployment-GDP Correlation", "Co-movements", avg,
                          self.benchmarks.corr_unemployment_gdp, s, 2.0)
            scores.append(s * 2.0)
            weights.append(2.0)
        
        # Inflation-GDP
        if infl_gdp_corrs:
            avg = np.nanmean(infl_gdp_corrs)
            s = self._score_correlation(avg, self.benchmarks.corr_inflation_gdp, False)
            self._add_test("Inflation-GDP Correlation", "Co-movements", avg,
                          self.benchmarks.corr_inflation_gdp, s, 1.0)
            scores.append(s * 1.0)
            weights.append(1.0)
        
        # Consumption-GDP
        if cons_gdp_corrs:
            avg = np.nanmean(cons_gdp_corrs)
            s = self._score_correlation(avg, self.benchmarks.corr_consumption_gdp)
            self._add_test("Consumption-GDP Correlation", "Co-movements", avg,
                          self.benchmarks.corr_consumption_gdp, s, 1.5)
            scores.append(s * 1.5)
            weights.append(1.5)
        
        # Investment-GDP
        if inv_gdp_corrs:
            avg = np.nanmean(inv_gdp_corrs)
            s = self._score_correlation(avg, self.benchmarks.corr_investment_gdp)
            self._add_test("Investment-GDP Correlation", "Co-movements", avg,
                          self.benchmarks.corr_investment_gdp, s, 1.5)
            scores.append(s * 1.5)
            weights.append(1.5)
        
        # Credit-GDP
        if credit_gdp_corrs:
            avg = np.nanmean(credit_gdp_corrs)
            s = self._score_correlation(avg, self.benchmarks.credit_gdp_correlation)
            self._add_test("Credit-GDP Correlation", "Co-movements", avg,
                          self.benchmarks.credit_gdp_correlation, s, 1.0)
            scores.append(s * 1.0)
            weights.append(1.0)
        
        if not scores:
            return 0.0
        
        category_score = sum(scores) / sum(weights)
        self.category_scores["Co-movements"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 5: TAYLOR RULE COMPLIANCE
    # =========================================================================
    
    def test_taylor_rule(self, dfs: List[pd.DataFrame]) -> float:
        """Test Taylor Rule compliance."""
        print("  🏛️ Testing Taylor Rule Compliance...")
        
        all_deviations = []
        high_infl_correct = 0
        high_infl_total = 0
        low_infl_correct = 0
        low_infl_total = 0
        rate_changes_correct = 0
        rate_changes_total = 0
        
        for df in dfs:
            infl = self._to_percent(self._to_array(df['inflation']))
            
            if 'output_gap' in df.columns:
                output_gap = self._to_percent(self._to_array(df['output_gap']))
            else:
                output_gap = np.zeros(len(infl))
            
            rate = df['policy_rate'].values
            if rate.max() < 0.1:
                policy_rate = rate * 12 * 100
            else:
                policy_rate = rate * 100
            
            # Compute Taylor Rule rate
            taylor_rate = (self.benchmarks.taylor_r_star + infl +
                          self.benchmarks.taylor_inflation * (infl - self.benchmarks.inflation_target) +
                          self.benchmarks.taylor_output * output_gap)
            taylor_rate = np.clip(taylor_rate, 0, 20)
            
            all_deviations.extend(np.abs(policy_rate - taylor_rate).tolist())
            
            # Directional correctness
            n = min(len(infl), len(policy_rate), len(output_gap))
            for i in range(1, n):
                infl_gap = infl[i] - self.benchmarks.inflation_target
                rate_change = policy_rate[i] - policy_rate[i-1]
                
                if abs(infl_gap) > 0.5:
                    rate_changes_total += 1
                    if (infl_gap > 0 and rate_change > 0) or (infl_gap < 0 and rate_change < 0):
                        rate_changes_correct += 1
                
                if infl_gap > 1.5:
                    high_infl_total += 1
                    if rate_change > 0:
                        high_infl_correct += 1
                elif infl_gap < -0.5:
                    low_infl_total += 1
                    if rate_change < 0:
                        low_infl_correct += 1
        
        scores = []
        
        # Taylor Rule Deviation
        if all_deviations:
            avg_dev = np.mean(all_deviations)
            s = self._score_range(avg_dev, 0, 5.0)
            self._add_test("Taylor Rule Deviation", "Taylor Rule", avg_dev,
                          0.0, s, 2.0)
            scores.append(s * 2.0)
        
        # Rate increase when inflation high
        if high_infl_total > 10:
            pct = high_infl_correct / high_infl_total * 100
            s = self._score_range(pct, 30, 100)
            self._add_test("Rate ↑ when π High", "Taylor Rule", pct,
                          70.0, s, 2.0)
            scores.append(s * 2.0)
        
        # Rate decrease when inflation low
        if low_infl_total > 10:
            pct = low_infl_correct / low_infl_total * 100
            s = self._score_range(pct, 30, 100)
            self._add_test("Rate ↓ when π Low", "Taylor Rule", pct,
                          70.0, s, 2.0)
            scores.append(s * 2.0)
        
        # Overall directional correctness
        if rate_changes_total > 20:
            pct = rate_changes_correct / rate_changes_total * 100
            s = self._score_range(pct, 40, 100)
            self._add_test("Overall Directional Correctness", "Taylor Rule", pct,
                          70.0, s, 1.5)
            scores.append(s * 1.5)
        
        if not scores:
            return 0.0
        
        category_score = sum(scores) / sum([2.0, 2.0, 2.0, 1.5][:len(scores)])
        self.category_scores["Taylor Rule"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 6: PHILLIPS CURVE
    # =========================================================================
    
    def test_phillips_curve(self, dfs: List[pd.DataFrame]) -> float:
        """Test Phillips Curve relationships."""
        print("  📉 Testing Phillips Curve...")
        
        all_unemp = []
        all_infl_change = []
        all_unemp_gap = []
        all_output_gap = []
        sacrifice_ratios = []
        
        for df in dfs:
            unemp = self._to_percent(self._to_array(df['unemployment']))
            infl = self._to_percent(self._to_array(df['inflation']))
            
            infl_change = np.diff(infl)
            all_unemp.extend(unemp[1:].tolist())
            all_infl_change.extend(infl_change.tolist())
            
            if 'output_gap' in df.columns:
                output_gap = self._to_percent(self._to_array(df['output_gap']))
                unemp_gap = unemp - self.benchmarks.nairu
                all_unemp_gap.extend(unemp_gap.tolist())
                all_output_gap.extend(output_gap.tolist())
            
            # Sacrifice ratio
            if len(infl) > 20:
                infl_reduction = infl[0] - infl[-1]
                unemp_increase = np.max(unemp) - unemp[0]
                if infl_reduction > 0.5 and unemp_increase > 0:
                    sr = unemp_increase / infl_reduction
                    if 0.1 < sr < 10:
                        sacrifice_ratios.append(sr)
        
        scores = []
        
        # Phillips Curve slope
        if len(all_unemp) > 50:
            corr = np.corrcoef(all_unemp, all_infl_change)[0, 1]
            if not np.isnan(corr):
                # Should be negative or near zero
                s = self._score_range(corr, -0.8, 0.3)
                self._add_test("Phillips Curve Correlation", "Phillips Curve", corr,
                              self.benchmarks.phillips_slope, s, 2.0)
                scores.append(s * 2.0)
        
        # Okun's Law
        if len(all_unemp_gap) > 50:
            corr = np.corrcoef(all_unemp_gap, all_output_gap)[0, 1]
            if not np.isnan(corr):
                # Should be negative
                s = self._score_range(corr, -1.0, 0.2)
                self._add_test("Okun's Law Correlation", "Phillips Curve", corr,
                              -0.7, s, 2.0)
                scores.append(s * 2.0)
        
        # Sacrifice Ratio
        if sacrifice_ratios:
            avg_sr = np.mean(sacrifice_ratios)
            s = self._score_value(avg_sr, self.benchmarks.sacrifice_ratio, 1.5)
            self._add_test("Sacrifice Ratio", "Phillips Curve", avg_sr,
                          self.benchmarks.sacrifice_ratio, s, 1.5)
            scores.append(s * 1.5)
        
        if not scores:
            return 0.0
        
        category_score = sum(scores) / sum([2.0, 2.0, 1.5][:len(scores)])
        self.category_scores["Phillips Curve"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 7: MONETARY TRANSMISSION
    # =========================================================================
    
    def test_monetary_transmission(self, dfs: List[pd.DataFrame]) -> float:
        """Test monetary policy transmission mechanism."""
        print("  💰 Testing Monetary Transmission...")
        
        rate_to_gdp = []
        rate_to_infl = []
        transmission_exists = []
        
        for df in dfs:
            if len(df) < 30:
                continue
            
            rate = df['policy_rate'].values
            gdp = df['gdp'].values
            infl = df['inflation'].values
            
            rate_change = np.diff(rate)
            gdp_growth = np.diff(np.log(np.maximum(gdp, 1)))
            infl_change = np.diff(infl)
            
            # Lagged effects (12-18 months for GDP, 18-24 for inflation)
            for lag in [12, 15, 18]:
                if len(rate_change) > lag + 10:
                    n = min(len(rate_change) - lag, len(gdp_growth) - lag)
                    if n > 10:
                        c = np.corrcoef(rate_change[:n], gdp_growth[lag:lag+n])[0, 1]
                        if not np.isnan(c):
                            rate_to_gdp.append(c)
            
            for lag in [18, 21, 24]:
                if len(rate_change) > lag + 10:
                    n = min(len(rate_change) - lag, len(infl_change) - lag)
                    if n > 10:
                        c = np.corrcoef(rate_change[:n], infl_change[lag:lag+n])[0, 1]
                        if not np.isnan(c):
                            rate_to_infl.append(c)
            
            # Check if transmission exists at all
            if rate_to_gdp:
                transmission_exists.append(abs(np.mean(rate_to_gdp)) > 0.05)
        
        scores = []
        
        # Rate -> GDP effect
        if rate_to_gdp:
            avg = np.mean(rate_to_gdp)
            # Should be negative (higher rates = lower GDP)
            s = self._score_range(avg, -0.7, 0.2)
            self._add_test("Rate → GDP Lagged Effect", "Monetary Transmission", avg,
                          self.benchmarks.rate_to_gdp_magnitude, s, 2.0)
            scores.append(s * 2.0)
        
        # Rate -> Inflation effect
        if rate_to_infl:
            avg = np.mean(rate_to_infl)
            # Should be negative
            s = self._score_range(avg, -0.6, 0.2)
            self._add_test("Rate → Inflation Lagged Effect", "Monetary Transmission", avg,
                          self.benchmarks.rate_to_inflation_magnitude, s, 2.0)
            scores.append(s * 2.0)
        
        # Transmission mechanism exists
        if transmission_exists:
            pct = sum(transmission_exists) / len(transmission_exists) * 100
            s = self._score_range(pct, 30, 100)
            self._add_test("Transmission Mechanism Active", "Monetary Transmission", pct,
                          70.0, s, 1.5)
            scores.append(s * 1.5)
        
        if not scores:
            return 0.0
        
        category_score = sum(scores) / sum([2.0, 2.0, 1.5][:len(scores)])
        self.category_scores["Monetary Transmission"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 8: STABILITY
    # =========================================================================
    
    def test_stability(self, dfs: List[pd.DataFrame]) -> float:
        """Test economic stability."""
        print("  🛡️ Testing Economic Stability...")
        
        n = len(dfs)
        explosions = 0
        collapses = 0
        hyperinflations = 0
        deflations = 0
        depressions = 0
        zlb_stuck = 0
        
        for df in dfs:
            gdp = df['gdp'].values
            infl = self._to_percent(self._to_array(df['inflation']))
            unemp = self._to_percent(self._to_array(df['unemployment']))
            rate = df['policy_rate'].values
            
            if gdp.max() > gdp[0] * 10:
                explosions += 1
            if gdp.min() < gdp[0] * 0.1:
                collapses += 1
            if infl.max() > 50:
                hyperinflations += 1
            if infl.min() < -10:
                deflations += 1
            if unemp.max() > 25:
                depressions += 1
            
            # Stuck at ZLB
            if rate.max() < 0.1:
                rate_pct = rate * 12 * 100
            else:
                rate_pct = rate * 100
            if np.sum(rate_pct < 0.5) > len(rate_pct) * 0.8:
                zlb_stuck += 1
        
        scores = []
        
        # No explosions
        s = self._score_boolean(explosions == 0)
        self._add_test("No GDP Explosions", "Stability", explosions, 0, s, 2.5)
        scores.append(s * 2.5)
        
        # No collapses
        s = self._score_boolean(collapses == 0)
        self._add_test("No GDP Collapses", "Stability", collapses, 0, s, 2.5)
        scores.append(s * 2.5)
        
        # No hyperinflation
        s = self._score_boolean(hyperinflations == 0)
        self._add_test("No Hyperinflation", "Stability", hyperinflations, 0, s, 2.0)
        scores.append(s * 2.0)
        
        # No severe deflation
        s = self._score_boolean(deflations == 0)
        self._add_test("No Severe Deflation", "Stability", deflations, 0, s, 1.5)
        scores.append(s * 1.5)
        
        # No depression
        s = self._score_boolean(depressions == 0)
        self._add_test("No Depression", "Stability", depressions, 0, s, 2.0)
        scores.append(s * 2.0)
        
        # Not stuck at ZLB
        zlb_pct = zlb_stuck / n * 100 if n > 0 else 0
        s = self._score_range(100 - zlb_pct, 50, 100)
        self._add_test("Not Stuck at ZLB", "Stability", 100 - zlb_pct, 100, s, 1.0)
        scores.append(s * 1.0)
        
        category_score = sum(scores) / sum([2.5, 2.5, 2.0, 1.5, 2.0, 1.0])
        self.category_scores["Stability"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 9: BUSINESS CYCLE FEATURES
    # =========================================================================
    
    def test_business_cycle_features(self, dfs: List[pd.DataFrame]) -> float:
        """Test business cycle features."""
        print("  🔁 Testing Business Cycle Features...")
        
        recession_counts = []
        recession_frequencies = []
        recession_depths = []
        recovery_speeds = []
        
        for df in dfs:
            gdp_g = self._gdp_growth(df)
            
            # Count recessions
            in_recession = gdp_g < -1.0
            episodes = 0
            was_in = False
            for r in in_recession:
                if r and not was_in:
                    episodes += 1
                was_in = r
            
            recession_counts.append(episodes)
            recession_frequencies.append(np.sum(in_recession) / len(in_recession))
            
            # Recession depth
            if np.any(in_recession):
                recession_depths.append(np.min(gdp_g))
            
            # Recovery speed
            for i in range(len(in_recession) - 6):
                if in_recession[i] and not np.any(in_recession[i+1:i+4]):
                    recovery_speeds.append(np.mean(gdp_g[i+1:i+7]))
        
        scores = []
        
        # Economy has recessions
        has_recessions = sum(r > 0 for r in recession_counts) / len(recession_counts) > 0.3
        s = self._score_boolean(has_recessions)
        self._add_test("Economy Has Recessions", "Business Cycle Features", 
                      1 if has_recessions else 0, 1, s, 2.0)
        scores.append(s * 2.0)
        
        # Recession frequency
        avg_freq = np.mean(recession_frequencies) * 100
        s = self._score_range(avg_freq, 5, 35)
        self._add_test("Recession Frequency", "Business Cycle Features", avg_freq,
                      self.benchmarks.recession_frequency * 100, s, 1.5)
        scores.append(s * 1.5)
        
        # Recession depth
        if recession_depths:
            avg_depth = np.mean(recession_depths)
            s = self._score_range(avg_depth, -15, -0.5)
            self._add_test("Recession Depth", "Business Cycle Features", avg_depth,
                          self.benchmarks.recession_depth_avg, s, 1.0)
            scores.append(s * 1.0)
        
        # Recovery exists
        if recovery_speeds:
            avg_recovery = np.mean(recovery_speeds)
            s = self._score_range(avg_recovery, 0.5, 15)
            self._add_test("Recovery Speed", "Business Cycle Features", avg_recovery,
                          3.0, s, 1.0)
            scores.append(s * 1.0)
        
        category_score = sum(scores) / sum([2.0, 1.5, 1.0, 1.0][:len(scores)])
        self.category_scores["Business Cycle Features"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 10: WEALTH DISTRIBUTION
    # =========================================================================
    
    def test_wealth_distribution(self, dfs: List[pd.DataFrame]) -> float:
        """Test wealth distribution."""
        print("  💎 Testing Wealth Distribution...")
        
        all_gini = []
        all_top10 = []
        all_top1 = []
        all_bottom50 = []
        
        for df in dfs:
            if 'wealth_gini_wealth' in df.columns:
                vals = df['wealth_gini_wealth'].dropna().values
                all_gini.extend(vals.tolist())
            if 'wealth_top_10_share' in df.columns:
                vals = df['wealth_top_10_share'].dropna().values * 100
                all_top10.extend(vals.tolist())
            if 'wealth_top_1_share' in df.columns:
                vals = df['wealth_top_1_share'].dropna().values * 100
                all_top1.extend(vals.tolist())
            if 'wealth_bottom_50_share' in df.columns:
                vals = df['wealth_bottom_50_share'].dropna().values * 100
                all_bottom50.extend(vals.tolist())
        
        scores = []
        
        if all_gini:
            avg = np.mean(all_gini)
            s = self._score_value(avg, self.benchmarks.gini_wealth, 0.2)
            self._add_test("Gini Coefficient", "Wealth Distribution", avg,
                          self.benchmarks.gini_wealth, s, 1.5)
            scores.append(s * 1.5)
        
        if all_top10:
            avg = np.mean(all_top10)
            s = self._score_value(avg, self.benchmarks.top_10_share * 100, 15)
            self._add_test("Top 10% Share", "Wealth Distribution", avg,
                          self.benchmarks.top_10_share * 100, s, 1.0)
            scores.append(s * 1.0)
        
        if all_top1:
            avg = np.mean(all_top1)
            s = self._score_value(avg, self.benchmarks.top_1_share * 100, 10)
            self._add_test("Top 1% Share", "Wealth Distribution", avg,
                          self.benchmarks.top_1_share * 100, s, 1.0)
            scores.append(s * 1.0)
        
        if all_bottom50:
            avg = np.mean(all_bottom50)
            s = self._score_range(avg, 0, 20)
            self._add_test("Bottom 50% Share", "Wealth Distribution", avg,
                          self.benchmarks.bottom_50_share * 100, s, 0.5)
            scores.append(s * 0.5)
        
        if not scores:
            return 50.0  # Neutral if no data
        
        category_score = sum(scores) / sum([1.5, 1.0, 1.0, 0.5][:len(scores)])
        self.category_scores["Wealth Distribution"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 11: POLICY EFFECTIVENESS
    # =========================================================================
    
    def test_policy_effectiveness(self, dfs: List[pd.DataFrame]) -> float:
        """Test if policy is actually effective."""
        print("  🎯 Testing Policy Effectiveness...")
        
        rate_variability = []
        policy_impacts = []
        stabilization_scores = []
        
        for df in dfs:
            rate = df['policy_rate'].values
            gdp = df['gdp'].values
            infl = self._to_percent(self._to_array(df['inflation']))
            
            if rate.max() < 0.1:
                rate_pct = rate * 12 * 100
            else:
                rate_pct = rate * 100
            
            rate_variability.append(np.std(rate_pct))
            
            # Policy impact
            if len(df) > 24:
                rate_changes = np.diff(rate_pct)
                tightening = rate_changes > 0.1
                
                if np.sum(tightening) > 5:
                    impacts = []
                    for i in range(len(tightening)):
                        if tightening[i] and i + 12 < len(infl):
                            impacts.append(infl[i + 12] - infl[i])
                    if impacts:
                        policy_impacts.append(-np.mean(impacts))
            
            # Stabilization
            gdp_vol = np.std(np.diff(np.log(np.maximum(gdp, 1))))
            infl_vol = np.std(infl)
            stabilization_scores.append(1 / (1 + gdp_vol + infl_vol/10))
        
        scores = []
        
        # Rate variability
        if rate_variability:
            avg = np.mean(rate_variability)
            s = self._score_range(avg, 0.3, 8)
            self._add_test("Policy Rate Variability", "Policy Effectiveness", avg,
                          self.benchmarks.fed_funds_std, s, 1.5)
            scores.append(s * 1.5)
        
        # Policy impact
        if policy_impacts:
            avg = np.mean(policy_impacts)
            s = self._score_range(avg, -0.5, 2.0)
            self._add_test("Policy Impact Score", "Policy Effectiveness", avg,
                          0.5, s, 2.0)
            scores.append(s * 2.0)
        
        # Stabilization
        if stabilization_scores:
            avg = np.mean(stabilization_scores)
            s = self._score_range(avg, 0.1, 1.0)
            self._add_test("Stabilization Score", "Policy Effectiveness", avg,
                          0.5, s, 1.5)
            scores.append(s * 1.5)
        
        if not scores:
            return 50.0
        
        category_score = sum(scores) / sum([1.5, 2.0, 1.5][:len(scores)])
        self.category_scores["Policy Effectiveness"] = category_score
        return category_score
    
    # =========================================================================
    # TEST CATEGORY 12: HISTORICAL CRISIS COMPARISON
    # =========================================================================
    
    def test_historical_crisis_comparison(self, dfs: List[pd.DataFrame]) -> float:
        """Compare simulation recessions against historical real-world crises."""
        print("  🌍 Testing Historical Crisis Comparison...")
        
        # Extract recession characteristics from simulation
        sim_recessions = []
        for df in dfs:
            gdp = df['gdp'].values
            gdp_g = np.diff(np.log(np.maximum(gdp, 1))) * 12 * 100  # Annualized
            
            infl = self._to_percent(self._to_array(df['inflation']))
            unemp = self._to_percent(self._to_array(df['unemployment']))
            rate = df['policy_rate'].values
            if rate.max() < 0.1:
                rate_pct = rate * 12 * 100
            else:
                rate_pct = rate * 100
            
            # Find recession periods
            in_recession = gdp_g < -1.0
            
            # Extract recession characteristics
            if np.any(in_recession):
                # GDP impact during recession
                recession_gdp = [gdp_g[i] for i in range(len(in_recession)) if in_recession[i]]
                if recession_gdp:
                    sim_recessions.append({
                        "gdp_impact": np.min(recession_gdp),
                        "peak_unemployment": np.max(unemp),
                        "peak_inflation": np.max(infl),
                        "policy_rate_range": np.max(rate_pct) - np.min(rate_pct),
                    })
        
        scores = []
        
        # Compare to real crisis benchmarks
        if sim_recessions:
            # Average simulation recession depth
            sim_avg_depth = np.mean([r["gdp_impact"] for r in sim_recessions])
            real_avg_depth = CRISIS_BENCHMARKS["avg_gdp_impact"]
            
            # Score: Is recession depth realistic? (Real-world avg is about -2.5%)
            depth_diff = abs(sim_avg_depth - real_avg_depth)
            s = max(0, 100 - depth_diff * 5)  # 20% tolerance
            self._add_test("Recession Depth vs Real World", "Historical Crisis",
                          sim_avg_depth, real_avg_depth, s, 2.0)
            scores.append(s * 2.0)
            
            # Compare peak unemployment
            sim_peak_unemp = np.mean([r["peak_unemployment"] for r in sim_recessions])
            real_peak_unemp = CRISIS_BENCHMARKS["avg_peak_unemployment"]
            
            unemp_diff = abs(sim_peak_unemp - real_peak_unemp) / real_peak_unemp * 100
            s = max(0, 100 - unemp_diff)
            self._add_test("Peak Unemployment vs Real World", "Historical Crisis",
                          sim_peak_unemp, real_peak_unemp, s, 1.5)
            scores.append(s * 1.5)
            
            # Severity comparison (mild vs severe recessions)
            mild_count = sum(1 for r in sim_recessions if r["gdp_impact"] > -3)
            severe_count = sum(1 for r in sim_recessions if r["gdp_impact"] <= -3)
            total = mild_count + severe_count
            
            if total > 0:
                # Real-world: ~60% mild, ~40% severe based on our database
                mild_ratio = mild_count / total
                target_mild = sum(1 for c in HISTORICAL_CRISES if c["gdp_impact"] > -3) / len(HISTORICAL_CRISES)
                
                s = 100 - abs(mild_ratio - target_mild) * 100
                self._add_test("Recession Severity Distribution", "Historical Crisis",
                              mild_ratio * 100, target_mild * 100, s, 1.0)
                scores.append(s * 1.0)
            
            # Policy response magnitude
            if sim_recessions:
                sim_policy_range = np.mean([r["policy_rate_range"] for r in sim_recessions])
                real_policy_response = np.mean([abs(c["policy_rate_change"]) / 100 for c in HISTORICAL_CRISES 
                                                if c["policy_rate_change"] != 0])
                
                s = self._score_value(sim_policy_range * 100, real_policy_response * 100, tolerance=50)
                self._add_test("Policy Response Magnitude", "Historical Crisis",
                              sim_policy_range, real_policy_response * 100, s, 1.5)
                scores.append(s * 1.5)
        else:
            # No recessions detected - partial score
            self._add_test("Recession Detection", "Historical Crisis",
                          0, 1, 30.0, 1.0)
            scores.append(30.0 * 1.0)
        
        # Compare against specific crisis types
        # Financial crisis similarity
        financial_crises = [c for c in HISTORICAL_CRISES if "Financial" in c["type"]]
        if financial_crises and sim_recessions:
            avg_financial_gdp = np.mean([c["gdp_impact"] for c in financial_crises])
            worst_sim = min(r["gdp_impact"] for r in sim_recessions)
            
            # Can simulation produce financial-crisis-like severity?
            s = 100 if worst_sim < avg_financial_gdp * 0.8 else 50 if worst_sim < avg_financial_gdp * 0.5 else 20
            self._add_test("Can Produce Severe Crises", "Historical Crisis",
                          worst_sim, avg_financial_gdp, s, 1.0)
            scores.append(s * 1.0)
        
        if not scores:
            return 50.0
        
        category_score = sum(scores) / sum([2.0, 1.5, 1.0, 1.5, 1.0][:len(scores)])
        self.category_scores["Historical Crisis"] = category_score
        return category_score
    
    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    
    def run_all_tests(self, dfs: List[pd.DataFrame]) -> float:
        """Run all test categories and compute final score."""
        
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE VALIDATION")
        print("="*60)
        
        # Run all categories
        self.test_first_moments(dfs)
        self.test_second_moments(dfs)
        self.test_persistence(dfs)
        self.test_comovements(dfs)
        self.test_taylor_rule(dfs)
        self.test_phillips_curve(dfs)
        self.test_monetary_transmission(dfs)
        self.test_stability(dfs)
        self.test_business_cycle_features(dfs)
        self.test_wealth_distribution(dfs)
        self.test_policy_effectiveness(dfs)
        self.test_historical_crisis_comparison(dfs)  # NEW: Compare to real crises
        
        # Compute final weighted score
        weights = {
            "First Moments": 2.0,
            "Second Moments": 1.5,
            "Persistence": 1.5,
            "Co-movements": 2.0,
            "Taylor Rule": 2.0,
            "Phillips Curve": 1.5,
            "Monetary Transmission": 2.0,
            "Stability": 2.5,
            "Business Cycle Features": 1.5,
            "Wealth Distribution": 1.0,
            "Policy Effectiveness": 1.5,
            "Historical Crisis": 1.5,  # NEW: Real-world crisis comparison
        }
        
        total_weighted = 0
        total_weight = 0
        
        for cat, score in self.category_scores.items():
            w = weights.get(cat, 1.0)
            total_weighted += score * w
            total_weight += w
        
        final_score = total_weighted / total_weight if total_weight > 0 else 0
        return final_score
    
    # =========================================================================
    # PRINT DETAILED RESULTS
    # =========================================================================
    
    def print_results(self, final_score: float, model_path: str):
        """Print detailed results."""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE VALIDATION RESULTS")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Total Tests: {len(self.test_results)}")
        
        # Category breakdown
        print("\n" + "-"*60)
        print("CATEGORY SCORES")
        print("-"*60)
        
        for cat in sorted(self.category_scores.keys()):
            score = self.category_scores[cat]
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            status = "✓" if score >= 60 else "⚠" if score >= 40 else "✗"
            print(f"{status} {cat:<25} [{bar}] {score:>5.1f}%")
        
        # Detailed test results by category
        print("\n" + "-"*60)
        print("DETAILED TEST RESULTS")
        print("-"*60)
        
        categories = sorted(set(t["category"] for t in self.test_results))
        for cat in categories:
            print(f"\n{cat}:")
            cat_tests = [t for t in self.test_results if t["category"] == cat]
            for t in cat_tests:
                status = "✓" if t["score"] >= 60 else "⚠" if t["score"] >= 40 else "✗"
                print(f"  {status} {t['name']:<35} Score: {t['score']:>5.1f}  "
                      f"(Model: {t['model_value']:>8.2f}, Target: {t['benchmark']:>8.2f})")
        
        # Final score
        print("\n" + "="*80)
        print("🎯 FINAL REALITY SCORE")
        print("="*80)
        
        bar = "█" * int(final_score / 2.5) + "░" * (40 - int(final_score / 2.5))
        print(f"\n  [{bar}] {final_score:.1f}%\n")
        
        # Grade
        if final_score >= 90:
            grade = "A+"
            desc = "EXCEPTIONAL - Nearly indistinguishable from real economy"
        elif final_score >= 85:
            grade = "A"
            desc = "EXCELLENT - Captures most real-world dynamics"
        elif final_score >= 80:
            grade = "A-"
            desc = "VERY GOOD - Strong match to empirical facts"
        elif final_score >= 75:
            grade = "B+"
            desc = "GOOD - Reasonable approximation of reality"
        elif final_score >= 70:
            grade = "B"
            desc = "ABOVE AVERAGE - Shows realistic tendencies"
        elif final_score >= 65:
            grade = "B-"
            desc = "DECENT - Some realistic features present"
        elif final_score >= 60:
            grade = "C+"
            desc = "FAIR - Basic economic relationships present"
        elif final_score >= 55:
            grade = "C"
            desc = "AVERAGE - Needs improvement in key areas"
        elif final_score >= 50:
            grade = "C-"
            desc = "BELOW AVERAGE - Missing important dynamics"
        elif final_score >= 45:
            grade = "D"
            desc = "POOR - Significant deviations from reality"
        else:
            grade = "F"
            desc = "FAILING - Does not capture real economic behavior"
        
        print(f"  Grade: {grade}")
        print(f"  {desc}")
        
        # Key insights
        print("\n" + "-"*60)
        print("KEY INSIGHTS")
        print("-"*60)
        
        # Best categories
        best_cats = sorted(self.category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\n  ✅ Strongest Areas:")
        for cat, score in best_cats:
            print(f"     • {cat}: {score:.1f}%")
        
        # Worst categories
        worst_cats = sorted(self.category_scores.items(), key=lambda x: x[1])[:3]
        print("\n  ⚠️ Areas for Improvement:")
        for cat, score in worst_cats:
            print(f"     • {cat}: {score:.1f}%")
        
        # Specific failed tests
        failed = [t for t in self.test_results if t["score"] < 40]
        if failed:
            print("\n  ❌ Critical Issues:")
            for t in failed[:5]:
                print(f"     • {t['name']}: {t['score']:.1f}%")
        
        print("\n" + "="*80)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    def save_results(self, final_score: float, model_path: str, output_path: str):
        """Save results to JSON."""
        
        def to_native(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "final_score": float(final_score),
            "grade": self._compute_grade(final_score),
            "total_tests": len(self.test_results),
            "category_scores": {k: float(v) for k, v in self.category_scores.items()},
            "test_details": [
                {
                    "name": t["name"],
                    "category": t["category"],
                    "model_value": to_native(t["model_value"]),
                    "benchmark": to_native(t["benchmark"]),
                    "score": float(t["score"]),
                    "weight": float(t["weight"])
                }
                for t in self.test_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
    
    def _compute_grade(self, score: float) -> str:
        if score >= 90: return "A+"
        if score >= 85: return "A"
        if score >= 80: return "A-"
        if score >= 75: return "B+"
        if score >= 70: return "B"
        if score >= 65: return "B-"
        if score >= 60: return "C+"
        if score >= 55: return "C"
        if score >= 50: return "C-"
        if score >= 45: return "D"
        return "F"
    
    # =========================================================================
    # MAIN VALIDATION METHOD
    # =========================================================================
    
    def validate(self, model_path: str, n_runs: int = 20, 
                 duration: int = 240) -> float:
        """Main validation entry point."""
        
        print("="*80)
        print("🔬 ULTIMATE MODEL VALIDATION SYSTEM")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Simulations: {n_runs}")
        print(f"Duration: {duration} months ({duration//12} years)")
        
        if not self.imports_ok:
            print("\n❌ Failed to import required modules")
            return 0.0
        
        # Load model
        print("\n📥 Loading model...")
        ppo, loaded = self.load_model(model_path)
        
        if not loaded:
            print("❌ Failed to load model")
            return 0.0
        
        print("✓ Model loaded successfully")
        
        # Run simulations
        print(f"\n🔄 Running {n_runs} simulations...")
        dfs = self.run_simulations(ppo, n_runs, duration)
        
        if not dfs:
            print("❌ No successful simulations")
            return 0.0
        
        print(f"✓ {len(dfs)} simulations completed")
        
        # Run all tests
        final_score = self.run_all_tests(dfs)
        
        # Print results
        self.print_results(final_score, model_path)
        
        return final_score


# =============================================================================
# MAIN
# =============================================================================

def main():
    # HARDCODED MODEL PATH - Using OLD model with BALANCED settings
    MODEL_PATH = "/Users/dhriman/Desktop/Personal Projects/Economy_RL/checkpoints/run_20251205_153712/checkpoint_epoch_250.pt"
    
    parser = argparse.ArgumentParser(
        description="Ultimate single model validation system"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to model file (.pt)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of simulation runs (default: 20)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=240,
        help="Simulation duration in months (default: 240 = 20 years)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Use hardcoded path if no argument provided
    model_path = args.model if args.model != MODEL_PATH else MODEL_PATH
    
    print(f"\n🎯 Using model: {model_path}\n")
    
    validator = UltimateModelValidator()
    final_score = validator.validate(
        model_path=model_path,
        n_runs=args.runs,
        duration=args.duration
    )
    
    if final_score > 0:
        validator.save_results(final_score, model_path, args.output)


if __name__ == "__main__":
    main()