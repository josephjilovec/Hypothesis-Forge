import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import patch, Mock
import json
import logging
from hypothesis.agent import PPOAgent, PolicyNetwork
from hypothesis.environment import HypothesisEnvironment
from hypothesis.hypothesis_ranking import HypothesisRanker

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def agent():
    """Fixture to create a PPOAgent instance."""
    state_dim = 10
    action_dim = 3
    return PPOAgent(state_dim=state_dim, action_dim=action_dim, device='cpu')

@pytest.fixture
def environment(tmp_path):
    """Fixture to create a HypothesisEnvironment instance with mock simulation data."""
    results_dir = tmp_path / "simulation" / "results"
    results_dir.mkdir(parents=True)
    df = pd.DataFrame(np.random.randn(5, 10))
    df.to_parquet(results_dir / "test_results.parquet")
    return HypothesisEnvironment(sim_results_dir=str(results_dir))

@pytest.fixture
def ranker(tmp_path):
    """Fixture to create a HypothesisRanker instance with temporary directories."""
    history_file = tmp_path / "hypothesis" / "history.json"
    output_file = tmp_path / "hypothesis" / "ranked_hypotheses.json"
    return HypothesisRanker(history_file=str(history_file), output_file=str(output_file))

@pytest.fixture
def mock_hypotheses():
    """Fixture to create mock hypotheses for testing."""
    return [
        {'hypothesis': 'Increase temperature affects protein stability', 'state': [1.0] * 10},
        {'hypothesis': 'Alter gravitational field changes orbital dynamics', 'state': [2.0] * 10}
    ]

def test_agent_init(agent):
    """Test PPOAgent initialization."""
    assert isinstance(agent.policy, PolicyNetwork)
    assert agent.policy.actor[-1].out_features == 3  # action_dim
    assert agent.policy.critic[-1].out_features == 1  # value output
    assert agent.device == torch.device('cpu')
    assert len(agent.memory) == 0
    assert agent.gamma == 0.99
    assert agent.clip_epsilon == 0.2

def test_select_action(agent):
    """Test PPOAgent action selection."""
    state = np.random.randn(10)
    action, log_prob = agent.select_action(state)
    assert action in range(3)  # action_dim = 3
    assert isinstance(log_prob, torch.Tensor)
    assert log_prob.shape == ()

def test_store_transition(agent):
    """Test storing transitions in agent memory."""
    state = np.random.randn(10)
    action = 1
    reward = 0.5
    next_state = np.random.randn(10)
    done = False
    log_prob = torch.tensor(0.1)
    agent.store_transition(state, action, reward, next_state, done, log_prob)
    assert len(agent.memory) == 1
    transition = agent.memory[0]
    assert np.array_equal(transition[0], state)
    assert transition[1] == action
    assert transition[2] == reward
    assert np.array_equal(transition[3], next_state)
    assert transition[4] == done
    assert transition[5] == log_prob

def test_compute_gae(agent):
    """Test Generalized Advantage Estimation calculation."""
    rewards = [1.0, 0.5, 0.2]
    values = torch.tensor([0.8, 0.6, 0.4])
    next_values = torch.tensor([0.6, 0.4, 0.2])
    dones = [0, 0, 1]
    advantages = agent.compute_gae(rewards, values, next_values, dones)
    assert isinstance(advantages, torch.Tensor)
    assert advantages.shape == (3,)
    assert torch.all(advantages >= -10) and torch.all(advantages <= 10)  # Reasonable range

def test_generate_hypothesis(agent):
    """Test hypothesis generation."""
    state = np.random.randn(10)
    hypothesis = agent.generate_hypothesis(state)
    assert isinstance(hypothesis, str)
    assert hypothesis in [
        "Increase temperature affects protein stability",
        "Alter gravitational field changes orbital dynamics",
        "Modify residue sequence impacts folding energy",
        "Unknown hypothesis"
    ]

def test_environment_init(environment):
    """Test HypothesisEnvironment initialization."""
    assert environment.sim_results_dir == Path('simulation/results')
    assert len(environment.results_files) >= 1
    assert environment.state_dim == 10
    assert environment.action_dim == 3
    assert isinstance(environment.action_space, gym.spaces.Discrete)
    assert isinstance(environment.observation_space, gym.spaces.Box)
    assert environment.observation_space.shape == (10,)

def test_environment_reset(environment):
    """Test environment reset."""
    state = environment.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (10,)
    assert np.all(np.isfinite(state))

def test_environment_step(environment):
    """Test environment step."""
    environment.reset()
    action = 0
    next_state, reward, done, info = environment.step(action)
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (10,)
    assert isinstance(reward, float)
    assert 0 <= reward <= 1
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'hypothesis' in info
    assert 'novelty' in info
    assert 'feasibility' in info
    assert info['hypothesis'] in [
        "Increase temperature affects protein stability",
        "Alter gravitational field changes orbital dynamics",
        "Modify residue sequence impacts folding energy"
    ]

def test_environment_feasibility(environment):
    """Test feasibility calculation."""
    environment.current_state = np.array([1.0] * 10)
    feasibility = environment.calculate_feasibility("Increase temperature affects protein stability")
    assert isinstance(feasibility, float)
    assert 0 <= feasibility <= 1
    # Test with high norm
    environment.current_state = np.array([10.0] * 10)
    feasibility_high = environment.calculate_feasibility("Alter gravitational field changes orbital dynamics")
    assert feasibility_high > feasibility  # Gravitational hypothesis prefers higher norms

def test_ranker_init(ranker, tmp_path):
    """Test HypothesisRanker initialization."""
    assert ranker.history_file == tmp_path / "hypothesis" / "history.json"
    assert ranker.output_file == tmp_path / "hypothesis" / "ranked_hypotheses.json"
    assert ranker.output_file.parent.exists()
    assert isinstance(ranker.hypothesis_history, list)
    assert isinstance(ranker.embedding_cache, dict)

def test_ranker_load_history(ranker, tmp_path):
    """Test loading hypothesis history."""
    history_file = tmp_path / "hypothesis" / "history.json"
    history_data = [{'hypothesis': 'Test hypothesis', 'state': [1.0] * 10}]
    history_file.parent.mkdir(parents=True)
    with open(history_file, 'w') as f:
        json.dump(history_data, f)
    
    history = ranker.load_history()
    assert len(history) == 1
    assert history[0]['hypothesis'] == 'Test hypothesis'

def test_ranker_save_history(ranker, tmp_path, mock_hypotheses):
    """Test saving hypothesis history."""
    history_file = tmp_path / "hypothesis" / "history.json"
    ranker.save_history(mock_hypotheses)
    assert history_file.exists()
    with open(history_file, 'r') as f:
        saved_history = json.load(f)
    assert len(saved_history) == 2
    assert saved_history[0]['hypothesis'] == mock_hypotheses[0]['hypothesis']

def test_ranker_compute_novelty(ranker, mock_hypotheses):
    """Test novelty computation."""
    ranker.hypothesis_history = [mock_hypotheses[0]]
    novelty = ranker.compute_novelty(mock_hypotheses[1]['hypothesis'], np.array(mock_hypotheses[1]['state']))
    assert isinstance(novelty, float)
    assert 0 <= novelty <= 1
    # First hypothesis should have max novelty
    ranker.hypothesis_history = []
    novelty_first = ranker.compute_novelty(mock_hypotheses[0]['hypothesis'], np.array(mock_hypotheses[0]['state']))
    assert novelty_first == 1.0

def test_ranker_compute_feasibility(ranker):
    """Test feasibility computation."""
    state = np.array([5.0] * 10)
    feasibility = ranker.compute_feasibility("Increase temperature affects protein stability", state)
    assert isinstance(feasibility, float)
    assert 0 <= feasibility <= 1
    # Test gravitational hypothesis with high norm
    feasibility_grav = ranker.compute_feasibility("Alter gravitational field changes orbital dynamics", state * 2)
    assert feasibility_grav > feasibility

def test_ranker_rank_hypotheses(ranker, mock_hypotheses, tmp_path):
    """Test ranking hypotheses."""
    ranked = ranker.rank_hypotheses(mock_hypotheses)
    assert len(ranked) == 2
    assert all(key in ranked[0] for key in ['hypothesis', 'state', 'novelty', 'feasibility', 'score'])
    assert ranked[0]['score'] >= ranked[1]['score']  # Sorted by score
    assert (tmp_path / "hypothesis" / "ranked_hypotheses.json").exists()

def test_integration_agent_environment(agent, environment):
    """Test integration of agent and environment."""
    state = environment.reset()
    for _ in range(5):
        action, _ = agent.select_action(state)
        next_state, reward, done, info = environment.step(action)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        state = next_state
        if done:
            break

def test_integration_ranker(ranker, mock_hypotheses):
    """Test integration of ranker with hypotheses."""
    with patch('hypothesis.hypothesis_ranking.HypothesisRanker.save_ranked_hypotheses') as mock_save:
        result = ranker.run(mock_hypotheses)
        assert len(result) == 2
        assert mock_save.called
        assert all('score' in hyp for hyp in result)

if __name__ == "__main__":
    pytest.main(["-v"])
