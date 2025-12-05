"""
Experience Buffer for PPO
"""

import torch
import numpy as np
from typing import Dict, Generator


class RolloutBuffer:
    """
    Buffer for storing rollout experiences and computing GAE.
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.observation_size = observation_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Storage
        self.observations = np.zeros((buffer_size, observation_size), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Add a transition."""
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
    ) -> None:
        """Compute returns and GAE advantages."""
        size = self.buffer_size if self.full else self.pos
        
        last_gae = 0.0
        
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_done = float(last_done)
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
            
            # TD error
            delta = (
                self.rewards[t]
                + self.gamma * next_value * (1 - next_done)
                - self.values[t]
            )
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            self.advantages[t] = last_gae
        
        # Returns = advantages + values
        self.returns[:size] = self.advantages[:size] + self.values[:size]
    
    def get_samples(
        self,
        batch_size: int,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generate random minibatches."""
        size = self.buffer_size if self.full else self.pos
        indices = np.random.permutation(size)
        
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield {
                "observations": torch.tensor(
                    self.observations[batch_indices],
                    device=self.device,
                    dtype=torch.float32
                ),
                "actions": torch.tensor(
                    self.actions[batch_indices],
                    device=self.device,
                    dtype=torch.long
                ),
                "old_log_probs": torch.tensor(
                    self.log_probs[batch_indices],
                    device=self.device,
                    dtype=torch.float32
                ),
                "advantages": torch.tensor(
                    self.advantages[batch_indices],
                    device=self.device,
                    dtype=torch.float32
                ),
                "returns": torch.tensor(
                    self.returns[batch_indices],
                    device=self.device,
                    dtype=torch.float32
                ),
            }
            
            start_idx += batch_size
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        size = self.buffer_size if self.full else self.pos
        
        return {
            "observations": torch.tensor(
                self.observations[:size],
                device=self.device,
                dtype=torch.float32
            ),
            "actions": torch.tensor(
                self.actions[:size],
                device=self.device,
                dtype=torch.long
            ),
            "old_log_probs": torch.tensor(
                self.log_probs[:size],
                device=self.device,
                dtype=torch.float32
            ),
            "advantages": torch.tensor(
                self.advantages[:size],
                device=self.device,
                dtype=torch.float32
            ),
            "returns": torch.tensor(
                self.returns[:size],
                device=self.device,
                dtype=torch.float32
            ),
        }
    
    def reset(self) -> None:
        """Reset buffer."""
        self.pos = 0
        self.full = False