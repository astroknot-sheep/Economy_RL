"""
Neural Networks for Policy and Value Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    
    Uses separate heads for policy and value to allow
    different learning dynamics.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 128],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            activation_fn = nn.ReLU
        
        # Shared feature extractor (first layers)
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes[:-1]):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Separate actor and critic heads
        last_hidden = hidden_sizes[-1] if hidden_sizes else input_size
        
        self.actor_layers = nn.Sequential(
            nn.Linear(prev_size if layers else input_size, last_hidden),
            activation_fn(),
        )
        self.critic_layers = nn.Sequential(
            nn.Linear(prev_size if layers else input_size, last_hidden),
            activation_fn(),
        )
        
        self.policy_head = nn.Linear(last_hidden, output_size)
        self.value_head = nn.Linear(last_hidden, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning log-probs and values."""
        shared_features = self.shared(x)
        
        actor_features = self.actor_layers(shared_features)
        critic_features = self.critic_layers(shared_features)
        
        logits = self.policy_head(actor_features)
        log_probs = F.log_softmax(logits, dim=-1)
        
        values = self.value_head(critic_features).squeeze(-1)
        
        return log_probs, values
    
    def get_action_and_value(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute value."""
        log_probs, values = self.forward(x)
        probs = torch.exp(log_probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        action_log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action, action_log_prob, values, entropy
    
    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-prob, value, entropy for given actions."""
        log_probs, values = self.forward(x)
        probs = torch.exp(log_probs)
        
        action_log_prob = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_prob, values, entropy