"""
Control Plane: Uncertainty-Guided Adaptive Splitter (paper Sec. 4.2, Table 8).

A PPO agent observes the 3-dim MDP state s_t = [U_t, R_cpu, B_net] (Table 8)
-- embedding uncertainty (Eq. 11), CPU utilization, and EMA-estimated uplink
bandwidth -- and selects a split layer k in {0,...,L}, where L is the
encoder's number of splittable blocks (8 for models.AudioResNet18). The
reward is Eq. 12:

    r_t = alpha * A_task - beta * (Lat_t / T_max) - eta * (E_t / E_budget)

`SimulatedEdgeCloudEnv` is an offline training environment standing in for
the paper's "historical traces collected from diverse hardware platforms"
(Sec. 4.2.3) -- it reproduces the MDP's structure (state/action/reward
shapes and qualitative trends: later splits cost more latency, more
uncertain samples benefit more from offloading) but does NOT reproduce the
paper's measured latency/energy numbers (Tables 2-7), since no
hardware-in-the-loop trace collection is available in this repository.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SimulatedEdgeCloudEnv:
    """Offline RL training environment implementing Table 8's MDP."""

    def __init__(self, config: Dict, num_blocks: int = 8):
        cp_config = config['control_plane']
        self.num_blocks = num_blocks  # L
        self.num_actions = num_blocks + 1  # k in {0, ..., L}
        self.state_dim = 3  # [U_t, R_cpu, B_net]

        reward_config = cp_config['reward']
        self.alpha = reward_config['alpha']
        self.beta = reward_config['beta']
        self.eta = reward_config['eta']
        self.t_max_ms = reward_config['t_max_ms']
        self.e_budget_mj = reward_config['e_budget_mj']

        self.max_episode_steps = cp_config.get('max_episode_steps', 100)
        self.episode_step = 0
        self.state = self._sample_state()

    def _sample_state(self) -> np.ndarray:
        """Draw a fresh (uncertainty, cpu, bandwidth) trace sample."""
        uncertainty = np.random.uniform(0.0, 1.0)
        cpu = np.random.uniform(0.1, 1.0)
        bandwidth = np.random.uniform(0.05, 1.0)
        return np.array([uncertainty, cpu, bandwidth], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset to a new episode."""
        self.episode_step = 0
        self.state = self._sample_state()
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute split decision `action` under the current state and return
        (next_state, reward, done, info).
        """
        self.episode_step += 1
        uncertainty, cpu, bandwidth = self.state
        edge_ratio = action / self.num_blocks  # 0 = full offload, 1 = full on-device

        # Task accuracy: a server-routing bonus for uncertain samples,
        # reflecting Sec. 4.2.3's "offloads to the server even at minimal
        # split depth" behavior for high-uncertainty inputs.
        offload_bonus = (1 - edge_ratio) * uncertainty
        accuracy = np.clip(0.7 + 0.2 * offload_bonus, 0.0, 1.0)

        # Latency: edge compute grows with edge_ratio and CPU pressure;
        # transmission grows with what's offloaded and shrinks with
        # bandwidth (Sec. 6.2.2's qualitative latency breakdown).
        edge_compute_ms = 5.0 + 40.0 * edge_ratio * cpu
        transmit_ms = 5.0 + 80.0 * (1 - edge_ratio) / (bandwidth + 0.05)
        server_compute_ms = 10.0 * (1 - edge_ratio)
        latency_ms = edge_compute_ms + transmit_ms + server_compute_ms

        # Energy: edge compute energy grows with edge_ratio; radio energy
        # dominates for offloaded data (Sec. 6.2.3).
        energy_mj = 20.0 * edge_ratio + 60.0 * (1 - edge_ratio)

        reward = (
            self.alpha * accuracy
            - self.beta * (latency_ms / self.t_max_ms)
            - self.eta * (energy_mj / self.e_budget_mj)
        )

        self.state = self._sample_state()
        done = self.episode_step >= self.max_episode_steps

        info = {
            'split_layer': action,
            'accuracy': float(accuracy),
            'latency_ms': float(latency_ms),
            'energy_mj': float(energy_mj),
            'edge_ratio': float(edge_ratio),
        }
        return self.state, reward, done, info


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic network for PPO (Sec. 4.2.3)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        return self.policy(features), self.value(features)

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        action_probs, state_value = self.forward(state.unsqueeze(0))
        dist = Categorical(action_probs.squeeze(0))
        action = dist.sample()
        return action.item(), dist.log_prob(action), state_value.squeeze(0)

    def evaluate(self, states: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        return dist.log_prob(actions), state_values.squeeze(-1), dist.entropy()


class PPOSplitAgent:
    """PPO agent learning the Control Plane's split-point policy."""

    def __init__(self, config: Dict, env: SimulatedEdgeCloudEnv):
        ppo_config = config['control_plane']['ppo']
        self.env = env
        self.device = torch.device(config['experiment']['device'])

        self.policy = ActorCritic(
            state_dim=env.state_dim,
            action_dim=env.num_actions,
            hidden_dim=ppo_config.get('hidden_dim', 128)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=ppo_config.get('learning_rate', 3e-4)
        )

        self.gamma = ppo_config.get('gamma', 0.99)
        self.lambda_gae = ppo_config.get('lambda_gae', 0.95)
        self.epsilon_clip = ppo_config.get('epsilon_clip', 0.2)
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)
        self.value_coef = ppo_config.get('value_coef', 0.5)
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        self.update_epochs = ppo_config.get('update_epochs', 10)
        self.batch_size = ppo_config.get('batch_size', 64)

        self.states: List = []
        self.actions: List = []
        self.rewards: List = []
        self.log_probs: List = []
        self.values: List = []
        self.dones: List = []

    def select_action(self, state: np.ndarray) -> int:
        """Select an action and record the transition's pre-reward fields."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        return action

    def store_transition(self, reward: float, done: bool):
        """Record the reward/done for the most recent select_action() call."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, rewards: List[float], values: List[float],
                     dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation."""
        advantages, returns = [], []
        advantage, next_value = 0.0, 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            advantage = delta + self.gamma * self.lambda_gae * advantage * mask
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
            next_value = values[t]

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self) -> Dict:
        """Update the policy using clipped-surrogate PPO."""
        if len(self.states) == 0:
            return {}

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages, returns = self.compute_gae(self.rewards, self.values, self.dones)

        total_policy_loss = total_value_loss = total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                log_probs, values, entropy = self.policy.evaluate(
                    states[batch_idx], actions[batch_idx]
                )
                ratios = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratios * advantages[batch_idx]
                surr2 = torch.clamp(
                    ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip
                ) * advantages[batch_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[batch_idx])
                entropy_loss = -entropy.mean()

                loss = (policy_loss + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

        n = max(num_updates, 1)
        return {
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n,
            'entropy': total_entropy / n,
        }

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class SplitController:
    """
    Runtime split-point controller using a trained policy.

    Deployment-time callers supply the real measured state components --
    U_t from DistributionalMemory.entropy(), R_cpu and B_net from
    ResourceMonitor -- rather than the simulated environment used only for
    offline PPO training.
    """

    def __init__(self, config: Dict, num_blocks: int = 8,
                 policy_path: Optional[str] = None):
        self.num_blocks = num_blocks
        self.device = torch.device(config['experiment']['device'])

        self.policy = ActorCritic(
            state_dim=3,
            action_dim=num_blocks + 1,
            hidden_dim=config['control_plane']['ppo'].get('hidden_dim', 128)
        ).to(self.device)

        if policy_path:
            self.load_policy(policy_path)
        self.policy.eval()

        self.history = deque(maxlen=100)

    def get_split_layer(self, uncertainty: float, cpu_util: float,
                         bandwidth_mbps: float,
                         bandwidth_norm_mbps: float = 50.0) -> int:
        """
        Args:
            uncertainty: U_t in [0, log C] from DistributionalMemory.entropy()
            cpu_util: R_cpu in [0, 1] from ResourceMonitor
            bandwidth_mbps: Raw B_net estimate in Mbps from ResourceMonitor
            bandwidth_norm_mbps: Bandwidth value treated as "1.0" for
                normalization (deployment-specific link capacity)

        Returns:
            Split layer index k in {0, ..., L}
        """
        bandwidth_norm = min(bandwidth_mbps / bandwidth_norm_mbps, 1.0)
        state = np.array([uncertainty, cpu_util, bandwidth_norm], dtype=np.float32)
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor.unsqueeze(0))
            split_layer = torch.argmax(action_probs, dim=-1).item()

        self.history.append({'state': state, 'split_layer': split_layer})
        return split_layer

    def load_policy(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])

    def get_statistics(self) -> Dict:
        """Summary statistics over recent split decisions."""
        if not self.history:
            return {}

        split_layers = [h['split_layer'] for h in self.history]
        return {
            'avg_split_layer': float(np.mean(split_layers)),
            'std_split_layer': float(np.std(split_layers)),
            'split_distribution': np.bincount(
                split_layers, minlength=self.num_blocks + 1
            ).tolist(),
            'num_decisions': len(self.history),
        }
