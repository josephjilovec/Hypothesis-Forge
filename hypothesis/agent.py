"""
Reinforcement Learning agent for hypothesis generation.
Uses Proximal Policy Optimization (PPO) to propose novel hypotheses.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

from config.config import RL_LEARNING_RATE, RL_BATCH_SIZE, RL_GAMMA, RL_EPISODES, CHECKPOINT_DIR
from utils.logging_config import logger


class HypothesisGenerationEnv(gym.Env):
    """Gymnasium environment for hypothesis generation."""

    def __init__(self, simulation_results: List[Dict[str, Any]]):
        """
        Initialize hypothesis generation environment.

        Args:
            simulation_results: List of simulation result dictionaries
        """
        super().__init__()

        self.simulation_results = simulation_results
        self.current_index = 0
        self.current_result = None

        # Action space: [hypothesis_type, parameter_1, parameter_2, novelty_score]
        # hypothesis_type: 0=structural, 1=functional, 2=interaction, 3=evolutionary
        # parameters: normalized values between -1 and 1
        self.action_space = spaces.Box(
            low=np.array([0, -1.0, -1.0, 0.0]),
            high=np.array([3, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Observation space: features from simulation results
        # [sim_type, num_steps, runtime, success, normalized_features...]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # Fixed size observation
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment."""
        super().reset(seed=seed)

        if self.simulation_results:
            self.current_index = np.random.randint(0, len(self.simulation_results))
            self.current_result = self.simulation_results[self.current_index]
        else:
            self.current_result = {
                "simulation_type": "unknown",
                "num_steps": 0,
                "runtime_seconds": 0.0,
                "success": False,
            }

        observation = self._get_observation()
        info = {"current_result": self.current_result}

        return observation, info

    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.

        Args:
            action: Action array [hypothesis_type, param1, param2, novelty]

        Returns:
            observation, reward, terminated, truncated, info
        """
        hypothesis_type = int(action[0]) % 4
        param1 = float(action[1])
        param2 = float(action[2])
        novelty = float(np.clip(action[3], 0.0, 1.0))

        # Generate hypothesis based on action
        hypothesis = self._generate_hypothesis(hypothesis_type, param1, param2, novelty)

        # Calculate reward based on novelty and coherence
        reward = self._calculate_reward(hypothesis, novelty)

        # Move to next simulation result
        self.current_index = (self.current_index + 1) % len(self.simulation_results) if self.simulation_results else 0
        if self.simulation_results:
            self.current_result = self.simulation_results[self.current_index]

        observation = self._get_observation()
        terminated = False
        truncated = False
        info = {"hypothesis": hypothesis, "reward": reward}

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Extract observation from current simulation result."""
        obs = np.zeros(20, dtype=np.float32)

        # Encode simulation type
        sim_type = self.current_result.get("simulation_type", "unknown")
        if "protein" in sim_type.lower():
            obs[0] = 1.0
        elif "gravitational" in sim_type.lower() or "gravity" in sim_type.lower():
            obs[0] = 2.0
        else:
            obs[0] = 0.0

        # Encode numerical features
        obs[1] = float(self.current_result.get("num_steps", 0)) / 100.0  # Normalize
        obs[2] = float(self.current_result.get("runtime_seconds", 0.0)) / 300.0  # Normalize to 5 min
        obs[3] = 1.0 if self.current_result.get("success", False) else -1.0

        # Add random noise for exploration
        obs[4:] = np.random.randn(16).astype(np.float32) * 0.1

        return obs

    def _generate_hypothesis(
        self, hypothesis_type: int, param1: float, param2: float, novelty: float
    ) -> Dict[str, Any]:
        """Generate a hypothesis based on action."""
        hypothesis_templates = {
            0: "Structural hypothesis: The protein structure exhibits {param1} characteristics that may influence {param2}.",
            1: "Functional hypothesis: The observed {param1} suggests a functional role in {param2} processes.",
            2: "Interaction hypothesis: There may be interactions between {param1} and {param2} components.",
            3: "Evolutionary hypothesis: The {param1} pattern indicates evolutionary pressure toward {param2}.",
        }

        template = hypothesis_templates.get(hypothesis_type, hypothesis_templates[0])
        hypothesis_text = template.format(
            param1=f"parameter_{param1:.2f}",
            param2=f"parameter_{param2:.2f}",
        )

        return {
            "type": hypothesis_type,
            "text": hypothesis_text,
            "parameters": {"param1": param1, "param2": param2},
            "novelty_score": novelty,
            "simulation_context": self.current_result,
        }

    def _calculate_reward(self, hypothesis: Dict[str, Any], novelty: float) -> float:
        """Calculate reward for generated hypothesis."""
        # Base reward from novelty
        reward = novelty * 10.0

        # Bonus for coherent hypotheses (parameters in reasonable range)
        params = hypothesis["parameters"]
        coherence = 1.0 - abs(params["param1"]) - abs(params["param2"])
        reward += max(0, coherence) * 5.0

        # Bonus for successful simulations
        if self.current_result.get("success", False):
            reward += 2.0

        return float(reward)


class HypothesisAgent:
    """PPO agent for hypothesis generation."""

    def __init__(self, simulation_results: List[Dict[str, Any]]):
        """
        Initialize hypothesis generation agent.

        Args:
            simulation_results: List of simulation result dictionaries
        """
        self.simulation_results = simulation_results
        self.env = None
        self.model = None

    def train(self, total_timesteps: int = 10000):
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total training timesteps
        """
        logger.info("Initializing hypothesis generation environment...")

        if not self.simulation_results:
            logger.warning("No simulation results available, creating dummy data...")
            self.simulation_results = [{
                "simulation_type": "protein_folding",
                "num_steps": 50,
                "runtime_seconds": 120.0,
                "success": True,
            }]

        # Create environment
        self.env = HypothesisGenerationEnv(self.simulation_results)

        logger.info("Training PPO agent...")
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=RL_LEARNING_RATE,
            n_steps=RL_BATCH_SIZE,
            gamma=RL_GAMMA,
            verbose=1,
        )

        self.model.learn(total_timesteps=total_timesteps)

        # Save model
        checkpoint_path = CHECKPOINT_DIR / "hypothesis_agent"
        self.model.save(str(checkpoint_path))
        logger.info(f"Saved trained model to {checkpoint_path}")

    def generate_hypotheses(self, num_hypotheses: int = 10) -> List[Dict[str, Any]]:
        """
        Generate hypotheses using trained agent.

        Args:
            num_hypotheses: Number of hypotheses to generate

        Returns:
            List of generated hypotheses
        """
        if self.model is None:
            logger.warning("Model not trained, using random policy...")
            return self._generate_random_hypotheses(num_hypotheses)

        hypotheses = []
        obs, info = self.env.reset()

        for _ in range(num_hypotheses):
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = self.env.step(action)

            if "hypothesis" in info:
                hypotheses.append(info["hypothesis"])

            if terminated or truncated:
                obs, info = self.env.reset()

        return hypotheses

    def _generate_random_hypotheses(self, num_hypotheses: int) -> List[Dict[str, Any]]:
        """Generate random hypotheses as fallback."""
        hypotheses = []
        for i in range(num_hypotheses):
            hypothesis = {
                "type": i % 4,
                "text": f"Random hypothesis {i+1}: Potential relationship between observed patterns.",
                "parameters": {
                    "param1": np.random.uniform(-1, 1),
                    "param2": np.random.uniform(-1, 1),
                },
                "novelty_score": np.random.uniform(0, 1),
                "simulation_context": self.simulation_results[0] if self.simulation_results else {},
            }
            hypotheses.append(hypothesis)

        return hypotheses


def main():
    """Main entry point for hypothesis agent."""
    # Load simulation results
    from pathlib import Path
    from config.config import RESULTS_DIR
    import pandas as pd

    results_file = RESULTS_DIR / "simulation_results.parquet"
    if results_file.exists():
        results_df = pd.read_parquet(results_file)
        simulation_results = results_df.to_dict("records")
    else:
        logger.warning("No simulation results found, using dummy data")
        simulation_results = []

    # Train agent
    agent = HypothesisAgent(simulation_results)
    agent.train(total_timesteps=5000)  # Reduced for faster execution

    # Generate hypotheses
    hypotheses = agent.generate_hypotheses(num_hypotheses=20)
    logger.info(f"Generated {len(hypotheses)} hypotheses")

    return hypotheses


if __name__ == "__main__":
    main()

