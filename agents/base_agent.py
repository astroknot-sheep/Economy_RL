"""
Base Agent Class
Abstract base class for all economic agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class AgentState:
    """Base state container for all agents."""
    id: int
    wealth: float = 0.0
    debt: float = 0.0
    income: float = 0.0
    is_active: bool = True
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for network input."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging."""
        return {
            "id": self.id,
            "wealth": self.wealth,
            "debt": self.debt,
            "income": self.income,
            "is_active": self.is_active,
        }


class BaseAgent(ABC):
    """Abstract base class for all economic agents."""
    
    def __init__(self, agent_id: int, config: Any):
        self.id = agent_id
        self.config = config
        self.state: Optional[AgentState] = None
        self.history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def reset(self, initial_conditions: Optional[Dict[str, Any]] = None) -> None:
        """Reset agent to initial state."""
        pass
    
    @abstractmethod
    def get_observation(self, global_state: Dict[str, Any]) -> np.ndarray:
        """Construct observation vector from global state."""
        pass
    
    @abstractmethod
    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """Convert discrete action index to action values."""
        pass
    
    @abstractmethod
    def compute_reward(
        self, 
        action: Dict[str, Any], 
        next_state: Dict[str, Any],
        global_state: Dict[str, Any]
    ) -> float:
        """Compute reward for this timestep."""
        pass
    
    @abstractmethod
    def update_state(
        self, 
        action: Dict[str, Any], 
        market_outcomes: Dict[str, Any]
    ) -> None:
        """Update internal state based on action and market outcomes."""
        pass
    
    def record_history(self, step: int, action: Dict[str, Any], reward: float) -> None:
        """Record state/action/reward for analysis."""
        self.history.append({
            "step": step,
            "state": self.state.to_dict() if self.state else None,
            "action": action,
            "reward": reward,
        })
    
    def get_action_space_size(self) -> int:
        """Return size of discrete action space."""
        raise NotImplementedError
    
    def get_observation_size(self) -> int:
        """Return size of observation vector."""
        raise NotImplementedError
    
    @property
    def is_active(self) -> bool:
        """Check if agent is still active."""
        return self.state.is_active if self.state else False
    
    def declare_bankruptcy(self) -> None:
        """Handle bankruptcy."""
        if self.state:
            self.state.is_active = False


class AgentPopulation:
    """Manages a population of homogeneous agents."""
    
    def __init__(
        self, 
        agent_class: type,
        num_agents: int,
        config: Any,
    ):
        self.agent_class = agent_class
        self.num_agents = num_agents
        self.config = config
        
        self.agents: List[BaseAgent] = [
            agent_class(agent_id=i, config=config)
            for i in range(num_agents)
        ]
    
    def reset_all(self, initial_conditions: Optional[List[Dict[str, Any]]] = None) -> None:
        """Reset all agents."""
        for i, agent in enumerate(self.agents):
            ic = initial_conditions[i] if initial_conditions else None
            agent.reset(ic)
    
    def get_observations_batch(self, global_state: Dict[str, Any]) -> np.ndarray:
        """Get observations for all agents as a batch."""
        obs_list = [
            agent.get_observation(global_state) 
            for agent in self.agents 
            if agent.is_active
        ]
        if obs_list:
            return np.stack(obs_list, axis=0)
        return np.array([])
    
    def get_active_agents(self) -> List[BaseAgent]:
        """Return list of non-bankrupt agents."""
        return [a for a in self.agents if a.is_active]
    
    def get_active_count(self) -> int:
        """Return count of active agents."""
        return sum(1 for a in self.agents if a.is_active)
    
    def aggregate_state(self) -> Dict[str, float]:
        """Compute aggregate statistics across population."""
        active = self.get_active_agents()
        if not active:
            return {}
        
        total_wealth = sum(a.state.wealth for a in active)
        total_debt = sum(a.state.debt for a in active)
        total_income = sum(a.state.income for a in active)
        
        return {
            "total_wealth": total_wealth,
            "total_debt": total_debt,
            "total_income": total_income,
            "mean_wealth": total_wealth / len(active),
            "mean_debt": total_debt / len(active),
            "mean_income": total_income / len(active),
            "num_active": len(active),
        }