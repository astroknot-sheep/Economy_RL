import numpy as np
import pandas as pd
import os
import json
from config import DEFAULT_CONFIG, Config
from environment import MacroEconEnvironment
from training import MultiAgentPPO, PPOConfig

# Configuration
CHECKPOINT_PATH = "checkpoints/run_20251204_174226/checkpoint_epoch_250.pt"
SIM_STEPS = 240  # 20 years

def load_model(config):
    env = MacroEconEnvironment(config)
    agent_configs = env.get_agent_configs()
    ppo_config = PPOConfig(
        learning_rate=config.training.learning_rate,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_epsilon=config.training.clip_epsilon,
        entropy_coef=config.training.entropy_coef,
        n_epochs=config.training.update_epochs,
        batch_size=config.training.minibatch_size,
    )
    ppo = MultiAgentPPO(agent_configs=agent_configs, ppo_config=ppo_config, device="cpu")
    
    if os.path.exists(CHECKPOINT_PATH):
        ppo.load(CHECKPOINT_PATH)
        print(f"Loaded model from {CHECKPOINT_PATH}")
        return ppo
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return None

def run_analysis():
    config = DEFAULT_CONFIG
    config.economic.simulation_length = SIM_STEPS
    
    ppo = load_model(config)
    if not ppo:
        return

    env = MacroEconEnvironment(config)
    obs = env.reset()
    
    print("Running simulation...")
    for _ in range(SIM_STEPS):
        actions, _, _ = ppo.get_actions(obs, deterministic=True)
        env.step(actions)
        obs = env._get_observations()
    
    df = env.get_history_dataframe()
    
    # --- Analysis ---
    
    # 1. Data Preparation
    # Convert to percentages where appropriate
    df['inflation_pct'] = df['inflation'] * 100
    df['unemployment_pct'] = df['unemployment'] * 100
    df['gdp_growth'] = df['gdp'].pct_change() * 100
    df['policy_rate_pct'] = df['policy_rate'] * 100
    
    # Drop initial burn-in (first 12 steps)
    df_clean = df.iloc[12:].copy()
    
    # 2. Volatility (Standard Deviation)
    volatility = {
        "GDP Growth": df_clean['gdp_growth'].std(),
        "Inflation": df_clean['inflation_pct'].std(),
        "Unemployment": df_clean['unemployment_pct'].std(),
        "Policy Rate": df_clean['policy_rate_pct'].std()
    }
    
    # 3. Persistence (Autocorrelation at lag 1)
    persistence = {
        "Inflation": df_clean['inflation_pct'].autocorr(lag=1),
        "GDP Growth": df_clean['gdp_growth'].autocorr(lag=1),
        "Unemployment": df_clean['unemployment_pct'].autocorr(lag=1)
    }
    
    # 4. Correlations
    # Phillips Curve: Unemployment vs Inflation
    phillips_corr = df_clean['unemployment_pct'].corr(df_clean['inflation_pct'])
    
    # Okun's Law: GDP Growth vs Change in Unemployment
    df_clean['unemp_change'] = df_clean['unemployment_pct'].diff()
    okun_corr = df_clean['gdp_growth'].corr(df_clean['unemp_change'])
    
    # Taylor Rule: Policy Rate vs Inflation
    taylor_corr = df_clean['policy_rate_pct'].corr(df_clean['inflation_pct'])
    
    # 5. Means
    means = {
        "GDP Growth": df_clean['gdp_growth'].mean(),
        "Inflation": df_clean['inflation_pct'].mean(),
        "Unemployment": df_clean['unemployment_pct'].mean()
    }

    # --- US Benchmarks (Great Moderation approx) ---
    benchmarks = {
        "volatility": {
            "GDP Growth": 0.5,  # Quarterly approx
            "Inflation": 0.5,
            "Unemployment": 0.8
        },
        "persistence": {
            "Inflation": 0.8,   # High persistence
            "Unemployment": 0.95 # Very high persistence
        },
        "correlations": {
            "Phillips": -0.3,   # Weak negative
            "Okun": -0.5,       # Strong negative
            "Taylor": 0.7       # Strong positive
        }
    }
    
    print("\n" + "="*50)
    print("BRUTALLY HONEST MODEL ANALYSIS (Epoch 250)")
    print("="*50)
    
    print("\n1. STABILITY & VOLATILITY (Standard Deviation)")
    print(f"{'Metric':<15} | {'Model':<10} | {'Reality (Approx)':<15} | {'Verdict'}")
    print("-" * 60)
    print(f"{'GDP Growth':<15} | {volatility['GDP Growth']:<10.2f} | {benchmarks['volatility']['GDP Growth']:<15} | {'⚠️ Too Volatile' if volatility['GDP Growth'] > 1.0 else '✅ Realistic'}")
    print(f"{'Inflation':<15} | {volatility['Inflation']:<10.2f} | {benchmarks['volatility']['Inflation']:<15} | {'⚠️ Unstable' if volatility['Inflation'] > 2.0 else '✅ Stable'}")
    print(f"{'Unemployment':<15} | {volatility['Unemployment']:<10.2f} | {benchmarks['volatility']['Unemployment']:<15} | {'⚠️ Too Rigid' if volatility['Unemployment'] < 0.2 else '✅ Realistic'}")

    print("\n2. DYNAMICS & PERSISTENCE (Autocorrelation)")
    print(f"{'Metric':<15} | {'Model':<10} | {'Reality':<15} | {'Verdict'}")
    print("-" * 60)
    print(f"{'Inflation':<15} | {persistence['Inflation']:<10.2f} | {benchmarks['persistence']['Inflation']:<15} | {'⚠️ No Memory' if persistence['Inflation'] < 0.5 else '✅ Persistent'}")
    print(f"{'Unemployment':<15} | {persistence['Unemployment']:<10.2f} | {benchmarks['persistence']['Unemployment']:<15} | {'⚠️ Jittery' if persistence['Unemployment'] < 0.8 else '✅ Smooth'}")

    print("\n3. ECONOMIC RELATIONSHIPS (Correlations)")
    print(f"{'Relation':<15} | {'Model':<10} | {'Theory':<15} | {'Verdict'}")
    print("-" * 60)
    print(f"{'Phillips Curve':<15} | {phillips_corr:<10.2f} | Negative       | {'✅ Valid' if phillips_corr < -0.1 else '❌ Broken'}")
    print(f"{'Okun\'s Law':<15} | {okun_corr:<10.2f} | Negative       | {'✅ Valid' if okun_corr < -0.2 else '❌ Broken'}")
    print(f"{'Taylor Rule':<15} | {taylor_corr:<10.2f} | Positive       | {'✅ Valid' if taylor_corr > 0.4 else '❌ Weak'}")

    print("\n4. KEY LEVELS (Means)")
    print(f"Inflation Mean: {means['Inflation']:.2f}% (Target: 2-3%) -> {'✅ On Target' if 1.5 <= means['Inflation'] <= 3.5 else '❌ Missed Target'}")
    print(f"Unemployment Mean: {means['Unemployment']:.2f}% (Target: 4-6%) -> {'✅ Realistic' if 3.5 <= means['Unemployment'] <= 6.5 else '❌ Unrealistic'}")

if __name__ == "__main__":
    run_analysis()
