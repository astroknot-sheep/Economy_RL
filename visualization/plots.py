"""
Visualization Utilities
Plotting and animation for macroeconomic simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Any, Optional
import pandas as pd


def plot_macro_variables(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot key macroeconomic variables over time.
    
    Args:
        history_df: DataFrame with simulation history
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Macroeconomic Simulation Results", fontsize=14)
    
    # GDP and Growth
    ax = axes[0, 0]
    ax.plot(history_df["step"], history_df["gdp"], label="GDP", color="blue")
    if "potential_gdp" in history_df.columns:
        ax.plot(history_df["step"], history_df["potential_gdp"], 
                label="Potential GDP", color="blue", linestyle="--", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("GDP")
    ax.set_title("GDP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Inflation
    ax = axes[0, 1]
    ax.plot(history_df["step"], history_df["inflation"] * 100, color="red")
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.5, label="Target (2%)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Inflation (%)")
    ax.set_title("Inflation Rate (Annualized)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Unemployment
    ax = axes[1, 0]
    ax.plot(history_df["step"], history_df["unemployment"] * 100, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("Unemployment (%)")
    ax.set_title("Unemployment Rate")
    ax.grid(True, alpha=0.3)
    
    # Interest Rates
    ax = axes[1, 1]
    ax.plot(history_df["step"], history_df["policy_rate"] * 100, 
            label="Policy Rate", color="green")
    if "avg_lending_rate" in history_df.columns:
        ax.plot(history_df["step"], history_df["avg_lending_rate"] * 100,
                label="Lending Rate", color="purple")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Interest Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Credit Growth
    ax = axes[2, 0]
    if "credit_growth" in history_df.columns:
        ax.plot(history_df["step"], history_df["credit_growth"] * 100, color="brown")
    ax.set_xlabel("Step")
    ax.set_ylabel("Credit Growth (%)")
    ax.set_title("Credit Growth")
    ax.grid(True, alpha=0.3)
    
    # Wealth Inequality (Gini)
    ax = axes[2, 1]
    if "wealth_gini_wealth" in history_df.columns:
        ax.plot(history_df["step"], history_df["wealth_gini_wealth"], color="purple")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Wealth Inequality")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_training_curves(
    train_stats: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training statistics for all agent types.
    
    Args:
        train_stats: Dict mapping agent_type -> {metric: [values]}
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure
    """
    agent_types = list(train_stats.keys())
    n_agents = len(agent_types)
    
    fig, axes = plt.subplots(n_agents, 3, figsize=(14, 4 * n_agents))
    if n_agents == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Training Curves by Agent Type", fontsize=14)
    
    for i, agent_type in enumerate(agent_types):
        stats = train_stats[agent_type]
        
        # Policy Loss
        ax = axes[i, 0]
        if "policy_loss" in stats:
            ax.plot(stats["policy_loss"])
        ax.set_xlabel("Update")
        ax.set_ylabel("Policy Loss")
        ax.set_title(f"{agent_type}: Policy Loss")
        ax.grid(True, alpha=0.3)
        
        # Value Loss
        ax = axes[i, 1]
        if "value_loss" in stats:
            ax.plot(stats["value_loss"])
        ax.set_xlabel("Update")
        ax.set_ylabel("Value Loss")
        ax.set_title(f"{agent_type}: Value Loss")
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes[i, 2]
        if "entropy" in stats:
            ax.plot(stats["entropy"])
        ax.set_xlabel("Update")
        ax.set_ylabel("Entropy")
        ax.set_title(f"{agent_type}: Policy Entropy")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_policy_transmission(
    history_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot monetary policy transmission mechanism.
    
    Shows how policy rate changes affect other variables.
    
    Args:
        history_df: DataFrame with simulation history
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Monetary Policy Transmission", fontsize=14)
    
    # Policy rate vs Lending rate
    ax = axes[0, 0]
    ax.scatter(history_df["policy_rate"] * 100, 
               history_df["avg_lending_rate"] * 100, 
               alpha=0.5, s=10)
    ax.set_xlabel("Policy Rate (%)")
    ax.set_ylabel("Lending Rate (%)")
    ax.set_title("Pass-through to Lending Rates")
    
    # Fit line
    z = np.polyfit(history_df["policy_rate"], history_df["avg_lending_rate"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(history_df["policy_rate"].min(), history_df["policy_rate"].max(), 100)
    ax.plot(x_line * 100, p(x_line) * 100, "r--", label=f"Slope: {z[0]:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Policy rate vs Credit growth
    ax = axes[0, 1]
    ax.scatter(history_df["policy_rate"] * 100,
               history_df["credit_growth"] * 100,
               alpha=0.5, s=10)
    ax.set_xlabel("Policy Rate (%)")
    ax.set_ylabel("Credit Growth (%)")
    ax.set_title("Policy Rate vs Credit Growth")
    ax.grid(True, alpha=0.3)
    
    # Policy rate vs Inflation (with lag)
    ax = axes[1, 0]
    lag = 6  # 6 months lag
    if len(history_df) > lag:
        ax.scatter(history_df["policy_rate"].iloc[:-lag] * 100,
                   history_df["inflation"].iloc[lag:] * 100,
                   alpha=0.5, s=10)
    ax.set_xlabel("Policy Rate (%) - t")
    ax.set_ylabel("Inflation (%) - t+6")
    ax.set_title("Policy Rate vs Lagged Inflation")
    ax.grid(True, alpha=0.3)
    
    # Policy rate vs Output gap
    ax = axes[1, 1]
    if "output_gap" in history_df.columns:
        lag = 3
        if len(history_df) > lag:
            ax.scatter(history_df["policy_rate"].iloc[:-lag] * 100,
                       history_df["output_gap"].iloc[lag:] * 100,
                       alpha=0.5, s=10)
    ax.set_xlabel("Policy Rate (%) - t")
    ax.set_ylabel("Output Gap (%) - t+3")
    ax.set_title("Policy Rate vs Lagged Output Gap")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_agent_rewards(
    history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot rewards by agent type over time.
    
    Args:
        history: List of history records from environment
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Agent Rewards Over Time", fontsize=14)
    
    steps = [h["step"] for h in history]
    
    # Central Bank
    ax = axes[0, 0]
    cb_rewards = [h["cb_reward"] for h in history]
    ax.plot(steps, cb_rewards, color="green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Central Bank")
    ax.grid(True, alpha=0.3)
    
    # Banks
    ax = axes[0, 1]
    bank_rewards = [h["mean_bank_reward"] for h in history]
    ax.plot(steps, bank_rewards, color="blue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Commercial Banks")
    ax.grid(True, alpha=0.3)
    
    # Households
    ax = axes[1, 0]
    hh_rewards = [h["mean_household_reward"] for h in history]
    ax.plot(steps, hh_rewards, color="orange")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Households")
    ax.grid(True, alpha=0.3)
    
    # Firms
    ax = axes[1, 1]
    firm_rewards = [h["mean_firm_reward"] for h in history]
    ax.plot(steps, firm_rewards, color="purple")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Firms")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def create_animation(
    history_df: pd.DataFrame,
    save_path: str,
    fps: int = 10,
) -> None:
    """
    Create animated visualization of the simulation.
    
    Args:
        history_df: DataFrame with simulation history
        save_path: Path to save animation (mp4 or gif)
        fps: Frames per second
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Initialize empty plots
    lines = []
    
    # GDP
    ax = axes[0, 0]
    line_gdp, = ax.plot([], [], color="blue")
    ax.set_xlim(0, len(history_df))
    ax.set_ylim(history_df["gdp"].min() * 0.9, history_df["gdp"].max() * 1.1)
    ax.set_xlabel("Step")
    ax.set_ylabel("GDP")
    ax.set_title("GDP")
    ax.grid(True, alpha=0.3)
    lines.append(line_gdp)
    
    # Inflation
    ax = axes[0, 1]
    line_inf, = ax.plot([], [], color="red")
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.5)
    ax.set_xlim(0, len(history_df))
    ax.set_ylim(-5, 10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Inflation (%)")
    ax.set_title("Inflation Rate")
    ax.grid(True, alpha=0.3)
    lines.append(line_inf)
    
    # Unemployment
    ax = axes[1, 0]
    line_unemp, = ax.plot([], [], color="orange")
    ax.set_xlim(0, len(history_df))
    ax.set_ylim(0, 20)
    ax.set_xlabel("Step")
    ax.set_ylabel("Unemployment (%)")
    ax.set_title("Unemployment Rate")
    ax.grid(True, alpha=0.3)
    lines.append(line_unemp)
    
    # Policy Rate
    ax = axes[1, 1]
    line_rate, = ax.plot([], [], color="green")
    ax.set_xlim(0, len(history_df))
    ax.set_ylim(0, 15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Policy Rate")
    ax.grid(True, alpha=0.3)
    lines.append(line_rate)
    
    fig.suptitle("Macroeconomic Simulation", fontsize=14)
    plt.tight_layout()
    
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    
    def animate(frame):
        x = history_df["step"].iloc[:frame+1]
        lines[0].set_data(x, history_df["gdp"].iloc[:frame+1])
        lines[1].set_data(x, history_df["inflation"].iloc[:frame+1] * 100)
        lines[2].set_data(x, history_df["unemployment"].iloc[:frame+1] * 100)
        lines[3].set_data(x, history_df["policy_rate"].iloc[:frame+1] * 100)
        return lines
    
    anim = animation.FuncAnimation(
        fig, 
        animate, 
        init_func=init,
        frames=len(history_df), 
        interval=1000//fps, 
        blit=True
    )
    
    # Save
    if save_path.endswith(".gif"):
        anim.save(save_path, writer="pillow", fps=fps)
    else:
        anim.save(save_path, writer="ffmpeg", fps=fps)
    
    plt.close()


def plot_comparison(
    histories: Dict[str, pd.DataFrame],
    variable: str,
    title: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare a variable across multiple simulation runs.
    
    Args:
        histories: Dict mapping run name to DataFrame
        variable: Column name to plot
        title: Plot title
        save_path: If provided, save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, df in histories.items():
        ax.plot(df["step"], df[variable], label=name, alpha=0.8)
    
    ax.set_xlabel("Step")
    ax.set_ylabel(variable)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig
