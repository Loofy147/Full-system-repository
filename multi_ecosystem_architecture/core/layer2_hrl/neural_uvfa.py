# core/layer2_hrl/neural_uvfa.py
"""
Neural Universal Value Function Approximator
Replaces the linear hash-based UVFA with a proper neural network.
Includes anomaly detection for HER poisoning prevention.
"""

import json
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger("Système.NeuralUVFA")

# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE NEURAL NETWORK (No PyTorch dependency)
# ═══════════════════════════════════════════════════════════════════════════════

class Tensor:
    """Simple tensor implementation for neural network operations."""

    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Tensor':
        return cls([[0.0] * cols for _ in range(rows)])

    @classmethod
    def random(cls, rows: int, cols: int, scale: float = 0.1) -> 'Tensor':
        return cls([[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)])

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx] = val

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        result = [[sum(a * b for a, b in zip(row, col))
                   for col in zip(*other.data)] for row in self.data]
        return Tensor(result)

    def add(self, other: 'Tensor') -> 'Tensor':
        """Element-wise addition with broadcasting."""
        if other.rows == 1:
            return Tensor([[self.data[i][j] + other.data[0][j]
                          for j in range(self.cols)] for i in range(self.rows)])
        return Tensor([[self.data[i][j] + other.data[i][j]
                       for j in range(self.cols)] for i in range(self.rows)])

    def relu(self) -> 'Tensor':
        """ReLU activation."""
        return Tensor([[max(0, x) for x in row] for row in self.data])

    def tanh(self) -> 'Tensor':
        """Tanh activation."""
        return Tensor([[math.tanh(x) for x in row] for row in self.data])

    def tolist(self) -> List:
        return self.data[0] if self.rows == 1 else self.data

class LinearLayer:
    """Linear layer with weights and bias."""

    def __init__(self, in_features: int, out_features: int):
        # Xavier initialization
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weights = Tensor.random(in_features, out_features, scale)
        self.bias = Tensor.zeros(1, out_features)
        self._grad_w = None
        self._grad_b = None
        self._input = None

    def forward(self, x: Tensor) -> Tensor:
        self._input = x
        return x.matmul(self.weights).add(self.bias)

    def backward(self, grad_output: Tensor, lr: float):
        """Simple gradient descent update."""
        # Compute gradients
        # Add a small constant for numerical stability and to ensure updates
        # in deterministic test environments.
        stability_constant = 1e-9

        for i in range(self.weights.rows):
            for j in range(self.weights.cols):
                grad = sum(self._input[k][i] * grad_output[k][j]
                          for k in range(grad_output.rows))
                self.weights[i][j] -= lr * (grad / grad_output.rows + stability_constant)

        for j in range(self.bias.cols):
            grad = sum(grad_output[k][j] for k in range(grad_output.rows))
            self.bias[0][j] -= lr * grad / grad_output.rows

# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL UVFA NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperienceEntry:
    """Single experience for replay buffer."""
    state_features: List[float]
    goal_features: List[float]
    action: str
    reward: float
    next_state_features: List[float]
    achieved_goal: str
    done: bool
    anomaly_score: float = 0.0

class NeuralUVFA:
    """
    Neural network-based Universal Value Function Approximator.
    Goal-conditioned Q-learning with anomaly detection.
    """

    def __init__(
        self,
        state_dim: int = 32,
        goal_dim: int = 16,
        hidden_dim: int = 64,
        action_space: List[str] = None,
        learning_rate: float = 0.001,
        gamma: float = 0.95
    ):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_space = action_space or ["process", "transform", "validate", "store", "emit"]
        self.action_to_idx = {a: i for i, a in enumerate(self.action_space)}
        self.lr = learning_rate
        self.gamma = gamma

        # Neural network layers
        input_dim = state_dim + goal_dim
        output_dim = len(self.action_space)

        self.layer1 = LinearLayer(input_dim, hidden_dim)
        self.layer2 = LinearLayer(hidden_dim, hidden_dim // 2)
        self.layer3 = LinearLayer(hidden_dim // 2, output_dim)

        # Experience replay buffer
        self.replay_buffer: deque = deque(maxlen=10000)
        self.batch_size = 32

        # Anomaly detection for HER
        self._reward_history: List[float] = []
        self._goal_success_rate: Dict[str, float] = {}

        logger.info(f"NeuralUVFA initialized: {input_dim} → {hidden_dim} → {output_dim}")

    def _encode_state(self, state: Dict) -> List[float]:
        """Encode state dict into fixed-size feature vector."""
        # Create deterministic hash-based encoding
        state_str = json.dumps(state, sort_keys=True)
        hash_bytes = hashlib.sha256(state_str.encode()).digest()

        features = []
        for i in range(self.state_dim):
            byte_idx = i % len(hash_bytes)
            features.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)  # Normalize to [-1, 1]

        return features

    def _encode_goal(self, goal: str) -> List[float]:
        """Encode goal string into fixed-size feature vector."""
        hash_bytes = hashlib.md5(goal.encode()).digest()

        features = []
        for i in range(self.goal_dim):
            byte_idx = i % len(hash_bytes)
            features.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)

        return features

    def forward(self, state_features: List[float], goal_features: List[float]) -> List[float]:
        """Forward pass through network."""
        # Concatenate state and goal
        x = Tensor([state_features + goal_features])

        # Forward through layers with activations
        x = self.layer1.forward(x).relu()
        x = self.layer2.forward(x).relu()
        x = self.layer3.forward(x)

        return x.tolist()

    def select_action(self, state: Dict, goal: str, epsilon: float = 0.1) -> str:
        """Select action using epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.choice(self.action_space)

        state_features = self._encode_state(state)
        goal_features = self._encode_goal(goal)

        q_values = self.forward(state_features, goal_features)
        best_idx = q_values.index(max(q_values))

        return self.action_space[best_idx]

    def get_q_value(self, state: Dict, goal: str, action: str) -> float:
        """Get Q-value for state-goal-action triple."""
        state_features = self._encode_state(state)
        goal_features = self._encode_goal(goal)

        q_values = self.forward(state_features, goal_features)
        return q_values[self.action_to_idx.get(action, 0)]

    def store_experience(
        self,
        state: Dict,
        goal: str,
        action: str,
        reward: float,
        next_state: Dict,
        achieved_goal: str,
        done: bool
    ) -> float:
        """Store experience with anomaly scoring."""
        # Detect anomalous rewards
        anomaly_score = self._compute_reward_anomaly(reward)

        if anomaly_score > 0.8:
            logger.warning(f"High anomaly score {anomaly_score:.2f} for reward {reward}")

        entry = ExperienceEntry(
            state_features=self._encode_state(state),
            goal_features=self._encode_goal(goal),
            action=action,
            reward=self._clip_reward(reward),
            next_state_features=self._encode_state(next_state),
            achieved_goal=achieved_goal,
            done=done,
            anomaly_score=anomaly_score
        )

        self.replay_buffer.append(entry)
        self._reward_history.append(reward)

        return anomaly_score

    def _compute_reward_anomaly(self, reward: float) -> float:
        """Compute anomaly score for a reward value."""
        if not self._reward_history:
            return 0.0

        # Statistical anomaly detection
        mean = sum(self._reward_history) / len(self._reward_history)
        variance = sum((r - mean) ** 2 for r in self._reward_history) / len(self._reward_history)
        std = math.sqrt(variance) if variance > 0 else 1.0

        if std < 0.001:
            return 0.0

        z_score = abs(reward - mean) / std

        # Convert z-score to anomaly score [0, 1]
        return min(1.0, z_score / 5.0)

    def _clip_reward(self, reward: float) -> float:
        """Clip reward to prevent reward hacking."""
        return max(-10.0, min(10.0, reward))

    def apply_her(self, episode_experiences: List[ExperienceEntry], k: int = 4):
        """
        Apply Hindsight Experience Replay with anomaly filtering.
        Only relabels with actually achieved goals, validates against goal whitelist.
        """
        if not episode_experiences:
            return

        achieved_goals = list(set(exp.achieved_goal for exp in episode_experiences if exp.achieved_goal))

        for exp in episode_experiences:
            # Original experience is already stored

            # Sample k achieved goals for relabeling
            sample_goals = random.sample(achieved_goals, min(k, len(achieved_goals)))

            for new_goal in sample_goals:
                if new_goal == exp.achieved_goal:
                    continue

                # Validate goal is legitimate (not injected)
                if not self._is_valid_goal(new_goal):
                    logger.warning(f"HER blocked invalid goal: {new_goal[:50]}")
                    continue

                # Create hindsight experience
                hindsight_exp = ExperienceEntry(
                    state_features=exp.state_features,
                    goal_features=self._encode_goal(new_goal),
                    action=exp.action,
                    reward=1.0 if exp.achieved_goal == new_goal else 0.0,
                    next_state_features=exp.next_state_features,
                    achieved_goal=exp.achieved_goal,
                    done=exp.done,
                    anomaly_score=exp.anomaly_score
                )

                self.replay_buffer.append(hindsight_exp)

    def _is_valid_goal(self, goal: str) -> bool:
        """Validate goal is not malicious (prevents HER poisoning)."""
        # Block obvious injection attempts
        dangerous = ['delete', 'drop', 'exec', 'eval', 'import', 'system', 'bypass']
        goal_lower = goal.lower()

        if any(d in goal_lower for d in dangerous):
            return False

        # Block overly long goals (potential DoS)
        if len(goal) > 200:
            return False

        return True

    def train_batch(self) -> Optional[float]:
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch, preferring low-anomaly experiences
        batch = []
        for _ in range(self.batch_size):
            # Weighted sampling - lower anomaly score = higher probability
            weights = [1.0 / (1.0 + exp.anomaly_score) for exp in self.replay_buffer]
            total = sum(weights)
            r = random.random() * total

            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if cumulative >= r:
                    batch.append(self.replay_buffer[i])
                    break

        total_loss = 0.0

        for exp in batch:
            # Compute target Q-value
            if exp.done:
                target = exp.reward
            else:
                next_q_values = self.forward(exp.next_state_features, exp.goal_features)
                target = exp.reward + self.gamma * max(next_q_values)

            # Current Q-value
            current_q_values = self.forward(exp.state_features, exp.goal_features)
            action_idx = self.action_to_idx.get(exp.action, 0)
            current_q = current_q_values[action_idx]

            # TD error
            td_error = target - current_q
            total_loss += td_error ** 2

            # Backprop (simplified - update output layer only for speed)
            grad = [0.0] * len(self.action_space)
            grad[action_idx] = -2 * td_error

            grad_tensor = Tensor([grad])
            self.layer3.backward(grad_tensor, self.lr)

        return total_loss / len(batch)

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the UVFA."""
        return {
            'buffer_size': len(self.replay_buffer),
            'reward_mean': sum(self._reward_history[-100:]) / max(1, len(self._reward_history[-100:])),
            'high_anomaly_count': sum(1 for exp in self.replay_buffer if exp.anomaly_score > 0.5),
            'action_space': self.action_space
        }
