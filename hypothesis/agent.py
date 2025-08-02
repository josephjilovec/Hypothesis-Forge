import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from collections import deque
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """Neural network for PPO policy approximation."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

class PPOAgent:
    """Proximal Policy Optimization (PPO) agent for hypothesis generation."""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, device='cpu'):
        self.device = torch.device(device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = deque(maxlen=10000)
        logger.info(f"Initialized PPOAgent with state_dim={state_dim}, action_dim={action_dim} on {self.device}")

    def select_action(self, state):
        """Select an action (hypothesis) based on current policy."""
        try:
            state = torch.FloatTensor(state).to(self.device)
            action_probs, _ = self.policy(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob
        except Exception as e:
            logger.error(f"Error selecting action: {str(e)}")
            return None, None

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store transition in memory for training."""
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0
        for r, v, nv, d in zip(reversed(rewards), reversed(values), reversed(next_values), reversed(dones)):
            delta = r + self.gamma * nv * (1 - d) - v
            gae = delta + self.gamma * 0.95 * (1 - d) * gae
            advantages.insert(0, gae)
        return torch.FloatTensor(advantages).to(self.device)

    def train(self, env, num_episodes=1000, max_steps=100, batch_size=64):
        """Train the PPO agent using simulation environment."""
        try:
            for episode in range(num_episodes):
                state = env.reset()
                episode_reward = 0
                states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []

                for step in range(max_steps):
                    action, log_prob = self.select_action(state)
                    if action is None:
                        logger.warning("Skipping step due to action selection error")
                        continue

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    dones.append(done)
                    log_probs.append(log_prob)

                    state = next_state
                    if done:
                        break

                # Compute values and advantages
                states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
                _, values = self.policy(states_tensor)
                _, next_values = self.policy(next_states_tensor)
                advantages = self.compute_gae(rewards, values, next_values, dones)

                # Update policy
                self.update_policy(states, actions, log_probs, advantages, batch_size)

                # Log progress
                if episode % 10 == 0:
                    logger.info(f"Episode {episode}, Reward: {episode_reward:.4f}")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")

    def update_policy(self, states, actions, old_log_probs, advantages, batch_size):
        """Update policy using PPO clipped objective."""
        try:
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_states = torch.FloatTensor(np.array(states)[batch_indices]).to(self.device)
                batch_actions = torch.LongTensor(np.array(actions)[batch_indices]).to(self.device)
                batch_old_log_probs = torch.stack([old_log_probs[i] for i in batch_indices]).to(self.device)
                batch_advantages = advantages[batch_indices]

                # Forward pass
                action_probs, values = self.policy(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)

                # PPO objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), batch_advantages)
                loss = actor_loss + 0.5 * critic_loss

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        except Exception as e:
            logger.error(f"Policy update failed: {str(e)}")

    def generate_hypothesis(self, state):
        """Generate a 'what-if' hypothesis based on simulation state."""
        try:
            action, _ = self.select_action(state)
            if action is None:
                return None
            # Example mapping: action index to hypothesis (customize based on domain)
            hypothesis_map = {
                0: "Increase temperature affects protein stability",
                1: "Alter gravitational field changes orbital dynamics",
                2: "Modify residue sequence impacts folding energy"
                # Add more domain-specific hypotheses
            }
            return hypothesis_map.get(action, "Unknown hypothesis")
        except Exception as e:
            logger.error(f"Error generating hypothesis: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage for testing
    state_dim = 10  # Example: dimension of simulation output
    action_dim = 3  # Example: number of possible hypotheses
    agent = PPOAgent(state_dim, action_dim, device='cuda' if torch.cuda.is_available() else 'cpu')
    # Dummy environment for testing
    from unittest.mock import MagicMock
    env = MagicMock()
    env.reset.return_value = np.random.randn(state_dim)
    env.step.return_value = (np.random.randn(state_dim), 1.0, False, {})
    agent.train(env, num_episodes=10, max_steps=5)
