"""Debug: Check active firms"""

import numpy as np
from config import DEFAULT_CONFIG
from environment import MacroEconEnvironment
from training import MultiAgentPPO

MODEL = "/Users/dhriman/Desktop/Personal Projects/Economy_RL/checkpoints/run_20251204_174226/checkpoint_epoch_250.pt"

config = DEFAULT_CONFIG
config.economic.simulation_length = 20

env = MacroEconEnvironment(config)
ppo = MultiAgentPPO(agent_configs=env.get_agent_configs(), device="cpu")
ppo.load(MODEL)

obs = env.reset()

for step in range(12):
    actions, _, _ = ppo.get_actions(obs, deterministic=True)
    result = env.step(actions)
    obs = env._get_observations()
    
    active_firms = sum(1 for f in env.firms if f.is_active)
    total_workers = sum(f.state.num_workers for f in env.firms if f.is_active)
    
    print(f"Step {step+1}: Active Firms={active_firms}/20, Workers={total_workers}, "
          f"Unemp={result.macro_state.unemployment*100:.1f}%")
    
    # Show firm details if any inactive
    inactive = [f.id for f in env.firms if not f.is_active]
    if inactive:
        print(f"  Inactive firms: {inactive}")
