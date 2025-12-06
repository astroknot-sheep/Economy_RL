"""
PPO Trainer - Corrected Implementation

KEY FIXES:
1. Train on ALL agents, not just the first one
2. Proper advantage normalization
3. Correct clipping and loss computation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from networks import ActorCritic
from .buffer import RolloutBuffer


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    target_kl: Optional[float] = 0.02


class MultiAgentPPO:
    """
    PPO trainer for multiple agent types.
    
    Maintains separate networks and buffers for each agent type,
    but trains them all properly.
    """
    
    def __init__(
        self,
        agent_configs: Dict[str, Dict],
        ppo_config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ):
        """
        Args:
            agent_configs: Dict mapping agent_type -> {obs_size, action_size}
            ppo_config: PPO hyperparameters
            device: torch device
        """
        self.agent_configs = agent_configs
        self.config = ppo_config or PPOConfig()
        self.device = device
        
        self.networks: Dict[str, ActorCritic] = {}
        self.optimizers: Dict[str, optim.Adam] = {}
        self.buffers: Dict[str, Dict[int, RolloutBuffer]] = {}
        
        self._init_networks()
    
    def _init_networks(self) -> None:
        """Initialize networks for each agent type."""
        for agent_type, config in self.agent_configs.items():
            network = ActorCritic(
                input_size=config["obs_size"],
                output_size=config["action_size"],
                hidden_sizes=[128, 128],
            ).to(self.device)
            
            self.networks[agent_type] = network
            self.optimizers[agent_type] = optim.Adam(
                network.parameters(),
                lr=self.config.learning_rate,
            )
            self.buffers[agent_type] = {}
    
    def init_buffers(
        self,
        agent_counts: Dict[str, int],
        buffer_size: int,
    ) -> None:
        """Initialize rollout buffers for all agents."""
        for agent_type, count in agent_counts.items():
            obs_size = self.agent_configs[agent_type]["obs_size"]
            
            self.buffers[agent_type] = {}
            for agent_id in range(count):
                self.buffers[agent_type][agent_id] = RolloutBuffer(
                    buffer_size=buffer_size,
                    observation_size=obs_size,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                    device=self.device,
                )
    
    def get_actions(
        self,
        observations: Dict[str, Dict[int, np.ndarray]],
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
        """
        Get actions for all agents.
        
        Returns:
            actions: {agent_type: {agent_id: action}}
            log_probs: {agent_type: {agent_id: log_prob}}
            values: {agent_type: {agent_id: value}}
        """
        actions = {}
        log_probs = {}
        values = {}
        
        for agent_type, agent_obs in observations.items():
            network = self.networks[agent_type]
            network.eval()
            
            actions[agent_type] = {}
            log_probs[agent_type] = {}
            values[agent_type] = {}
            
            with torch.no_grad():
                for agent_id, obs in agent_obs.items():
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    
                    action, log_prob, value, _ = network.get_action_and_value(
                        obs_tensor, deterministic=deterministic
                    )
                    
                    actions[agent_type][agent_id] = action.item()
                    log_probs[agent_type][agent_id] = log_prob.item()
                    values[agent_type][agent_id] = value.item()
        
        return actions, log_probs, values
    
    def store_transitions(
        self,
        observations: Dict[str, Dict[int, np.ndarray]],
        actions: Dict[str, Dict[int, int]],
        rewards: Dict[str, Dict[int, float]],
        dones: Dict[str, bool],
        log_probs: Dict[str, Dict[int, float]],
        values: Dict[str, Dict[int, float]],
    ) -> None:
        """Store transitions for all agents."""
        for agent_type in observations:
            done = dones.get(agent_type, False)
            
            for agent_id in observations[agent_type]:
                if agent_id not in self.buffers[agent_type]:
                    continue
                
                buffer = self.buffers[agent_type][agent_id]
                
                obs = observations[agent_type].get(agent_id)
                action = actions[agent_type].get(agent_id)
                reward = rewards[agent_type].get(agent_id, 0.0)
                log_prob = log_probs[agent_type].get(agent_id, 0.0)
                value = values[agent_type].get(agent_id, 0.0)
                
                if obs is not None and action is not None:
                    buffer.add(obs, action, reward, done, log_prob, value)
    
    def compute_returns(
        self,
        last_observations: Dict[str, Dict[int, np.ndarray]],
        last_dones: Dict[str, bool],
    ) -> None:
        """Compute returns and advantages for all buffers."""
        for agent_type in self.buffers:
            network = self.networks[agent_type]
            network.eval()
            
            done = last_dones.get(agent_type, False)
            
            for agent_id, buffer in self.buffers[agent_type].items():
                if buffer.pos == 0 and not buffer.full:
                    continue
                
                # Get last value
                obs = last_observations.get(agent_type, {}).get(agent_id)
                if obs is not None:
                    with torch.no_grad():
                        obs_tensor = torch.tensor(
                            obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        _, last_value = network.forward(obs_tensor)
                        last_value = last_value.item()
                else:
                    last_value = 0.0
                
                buffer.compute_returns_and_advantages(last_value, done)
    
    def train(self) -> Dict[str, Dict[str, float]]:
        """
        Train all networks using collected experiences.
        
        KEY FIX: Aggregate experiences from ALL agents of each type,
        not just the first one.
        """
        stats = {}
        
        for agent_type, network in self.networks.items():
            # Aggregate all data from all agents of this type
            all_obs = []
            all_actions = []
            all_log_probs = []
            all_advantages = []
            all_returns = []
            
            for agent_id, buffer in self.buffers[agent_type].items():
                if buffer.pos == 0 and not buffer.full:
                    continue
                
                data = buffer.get_all()
                all_obs.append(data["observations"])
                all_actions.append(data["actions"])
                all_log_probs.append(data["old_log_probs"])
                all_advantages.append(data["advantages"])
                all_returns.append(data["returns"])
            
            if not all_obs:
                stats[agent_type] = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
                continue
            
            # Concatenate all data
            obs = torch.cat(all_obs, dim=0)
            actions = torch.cat(all_actions, dim=0)
            old_log_probs = torch.cat(all_log_probs, dim=0)
            advantages = torch.cat(all_advantages, dim=0)
            returns = torch.cat(all_returns, dim=0)
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Train
            network.train()
            optimizer = self.optimizers[agent_type]
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            n_updates = 0
            
            for epoch in range(self.config.n_epochs):
                # Shuffle indices
                indices = torch.randperm(len(obs))
                
                for start in range(0, len(obs), self.config.batch_size):
                    batch_idx = indices[start:start + self.config.batch_size]
                    
                    batch_obs = obs[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_advantages = advantages[batch_idx]
                    batch_returns = returns[batch_idx]
                    
                    # Forward pass
                    new_log_probs, values, entropy = network.evaluate_actions(
                        batch_obs, batch_actions
                    )
                    
                    # Policy loss (clipped PPO)
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon
                    ) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = nn.functional.mse_loss(values, batch_returns)
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    loss = (
                        policy_loss
                        + self.config.value_loss_coef * value_loss
                        + self.config.entropy_coef * entropy_loss
                    )
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        network.parameters(),
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    n_updates += 1
                
                # Early stopping on KL divergence
                if self.config.target_kl is not None:
                    with torch.no_grad():
                        new_log_probs, _, _ = network.evaluate_actions(obs, actions)
                        kl = (old_log_probs - new_log_probs).mean().item()
                        if kl > self.config.target_kl * 1.5:
                            break
            
            stats[agent_type] = {
                "policy_loss": total_policy_loss / max(n_updates, 1),
                "value_loss": total_value_loss / max(n_updates, 1),
                "entropy": total_entropy / max(n_updates, 1),
                "n_samples": len(obs),
            }
        
        return stats
    
    def reset_buffers(self) -> None:
        """Reset all buffers."""
        for agent_type in self.buffers:
            for buffer in self.buffers[agent_type].values():
                buffer.reset()
    
    def save(self, path: str) -> None:
        """Save all networks."""
        checkpoint = {
            "networks": {
                agent_type: network.state_dict()
                for agent_type, network in self.networks.items()
            },
            "optimizers": {
                agent_type: opt.state_dict()
                for agent_type, opt in self.optimizers.items()
            },
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str) -> None:
        """Load all networks."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for agent_type, state_dict in checkpoint["networks"].items():
            if agent_type in self.networks:
                self.networks[agent_type].load_state_dict(state_dict)
        
        for agent_type, state_dict in checkpoint["optimizers"].items():
            if agent_type in self.optimizers:
                self.optimizers[agent_type].load_state_dict(state_dict)