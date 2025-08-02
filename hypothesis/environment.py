import gym
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from gym import spaces

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HypothesisEnvironment(gym.Env):
    """Custom RL environment for scientific hypothesis generation."""
    
    def __init__(self, sim_results_dir='simulation/results'):
        super(HypothesisEnvironment, self).__init__()
        self.sim_results_dir = Path(sim_results_dir)
        self.results_files = list(self.sim_results_dir.glob('*.parquet'))
        if not self.results_files:
            logger.error("No simulation results found in {}".format(self.sim_results_dir))
            raise ValueError("No simulation results available")
        
        # Define state and action spaces
        self.state_dim = 10  # Example: reduced dimension of simulation output
        self.action_dim = 3  # Example: number of possible hypotheses
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        # Initialize state
        self.current_file_idx = 0
        self.current_state = None
        self.hypothesis_history = set()  # Track generated hypotheses for novelty
        
        # Load first simulation result
        self.load_simulation_result()
        
    def load_simulation_result(self):
        """Load simulation result as state."""
        try:
            df = pd.read_parquet(self.results_files[self.current_file_idx])
            # Example: aggregate simulation output to state vector
            state = df.values.mean(axis=0)[:self.state_dim]
            if len(state) < self.state_dim:
                state = np.pad(state, (0, self.state_dim - len(state)), mode='constant')
            self.current_state = state.astype(np.float32)
        except Exception as e:
            logger.error("Error loading simulation result {}: {}".format(self.results_files[self.current_file_idx], str(e)))
            self.current_state = np.zeros(self.state_dim, dtype=np.float32)

    def reset(self):
        """Reset environment to initial state."""
        try:
            self.current_file_idx = np.random.randint(0, len(self.results_files))
            self.load_simulation_result()
            logger.info("Environment reset to simulation result {}".format(self.results_files[self.current_file_idx]))
            return self.current_state
        except Exception as e:
            logger.error("Error resetting environment: {}".format(str(e)))
            return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment."""
        try:
            # Map action to hypothesis (consistent with agent.py)
            hypothesis_map = {
                0: "Increase temperature affects protein stability",
                1: "Alter gravitational field changes orbital dynamics",
                2: "Modify residue sequence impacts folding energy"
            }
            hypothesis = hypothesis_map.get(action, "Unknown hypothesis")
            
            # Calculate reward
            novelty = 1.0 if hypothesis not in self.hypothesis_history else 0.1
            feasibility = self.calculate_feasibility(hypothesis)
            reward = 0.7 * novelty + 0.3 * feasibility  # Weighted reward
            
            # Update hypothesis history
            self.hypothesis_history.add(hypothesis)
            
            # Move to next simulation result
            self.current_file_idx = (self.current_file_idx + 1) % len(self.results_files)
            self.load_simulation_result()
            
            # Check if done (e.g., cycled through all results)
            done = len(self.hypothesis_history) >= len(self.results_files) * 2
            
            info = {"hypothesis": hypothesis, "novelty": novelty, "feasibility": feasibility}
            logger.info("Step: Action {}, Hypothesis: {}, Reward: {:.4f}".format(action, hypothesis, reward))
            
            return self.current_state, reward, done, info
        
        except Exception as e:
            logger.error("Error in step: {}".format(str(e)))
            return self.current_state, 0.0, False, {}

    def calculate_feasibility(self, hypothesis):
        """Estimate hypothesis feasibility based on simulation data."""
        try:
            # Placeholder: Feasibility based on state magnitude (customize per domain)
            state_norm = np.linalg.norm(self.current_state)
            max_norm = 10.0  # Example threshold
            feasibility = min(1.0, state_norm / max_norm)  # Normalize to [0,1]
            return feasibility
        except Exception as e:
            logger.error("Error calculating feasibility: {}".format(str(e)))
            return 0.5  # Default feasibility

    def render(self, mode='human'):
        """Render the current state (optional)."""
        logger.info("Current state: {}".format(self.current_state))

if __name__ == "__main__":
    # Example usage for testing
    env = HypothesisEnvironment()
    obs = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
