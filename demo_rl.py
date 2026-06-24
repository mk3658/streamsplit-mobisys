#!/usr/bin/env python3
"""
Demo script for the Control Plane's RL components (Sec. 4.2, Table 8).

Tests the RL module components without full training.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from edge.rl_splitting import (
    ActorCritic,
    PPOSplitAgent,
    SimulatedEdgeCloudEnv,
    SplitController,
)


def load_config():
    with open('configs/streamsplit.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_environment():
    """Test SimulatedEdgeCloudEnv."""
    print("=" * 60)
    print("Testing SimulatedEdgeCloudEnv")
    print("=" * 60)

    config = load_config()
    env = SimulatedEdgeCloudEnv(config, num_blocks=config['encoder']['num_blocks'])

    print("\nEnvironment created:")
    print(f"  State dimension: {env.state_dim} (expected 3: [U_t, R_cpu, B_net])")
    print(f"  Number of actions: {env.num_actions} (expected L+1)")
    print(f"  Max episode steps: {env.max_episode_steps}")

    state = env.reset()
    print("\nInitial state:")
    print(f"  Shape: {state.shape}")
    print(f"  Values: {state}")

    print("\nRunning test episode:")
    episode_reward = 0
    for step in range(10):
        action = np.random.randint(0, env.num_actions)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        print(f"  Step {step}: k={action}, reward={reward:.3f}, "
              f"accuracy={info['accuracy']:.3f}, "
              f"latency={info['latency_ms']:.1f}ms, "
              f"energy={info['energy_mj']:.1f}mJ")

        if done:
            break

    print(f"\nTotal reward: {episode_reward:.3f}")
    print("\nEnvironment test passed!")
    return True


def test_agent():
    """Test PPOSplitAgent."""
    print("\n" + "=" * 60)
    print("Testing PPOSplitAgent")
    print("=" * 60)

    config = load_config()
    env = SimulatedEdgeCloudEnv(config, num_blocks=config['encoder']['num_blocks'])
    agent = PPOSplitAgent(config, env)

    print("\nAgent created:")
    print(f"  Policy parameters: "
          f"{sum(p.numel() for p in agent.policy.parameters())}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Epsilon clip: {agent.epsilon_clip}")

    print("\nTesting action selection:")
    state = env.reset()
    for i in range(5):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(reward, done)

        print(f"  Selection {i}: k={action}, reward={reward:.3f}")
        state = next_state

    print(f"\nBuffer size: {len(agent.states)} transitions")
    metrics = agent.update()
    if metrics:
        print(f"  Policy loss: {metrics['policy_loss']:.4f}")
        print(f"  Value loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy']:.4f}")

    print("\nAgent test passed!")
    return True


def test_controller():
    """Test SplitController."""
    print("\n" + "=" * 60)
    print("Testing SplitController")
    print("=" * 60)

    config = load_config()
    controller = SplitController(
        config, num_blocks=config['encoder']['num_blocks'], policy_path=None
    )

    print("\nController created:")
    print(f"  Policy parameters: "
          f"{sum(p.numel() for p in controller.policy.parameters())}")

    print("\nTesting split decisions:")
    for i in range(10):
        uncertainty = np.random.uniform(0.0, 1.0)
        cpu_util = np.random.uniform(0.1, 1.0)
        bandwidth_mbps = np.random.uniform(1.0, 50.0)

        split_layer = controller.get_split_layer(
            uncertainty, cpu_util, bandwidth_mbps
        )

        if i < 5:
            print(f"  Decision {i}: k={split_layer}, U_t={uncertainty:.3f}, "
                  f"R_cpu={cpu_util:.3f}, B_net={bandwidth_mbps:.1f}Mbps")

    stats = controller.get_statistics()
    print("\nController statistics:")
    print(f"  Total decisions: {stats['num_decisions']}")
    print(f"  Avg split layer: {stats['avg_split_layer']:.2f}")
    print(f"  Split distribution: {stats['split_distribution']}")

    print("\nController test passed!")
    return True


def test_actor_critic_network():
    """Test the ActorCritic network architecture."""
    print("=" * 60)
    print("Testing ActorCritic Network")
    print("=" * 60)

    state_dim, action_dim, hidden_dim = 3, 9, 128
    network = ActorCritic(state_dim, action_dim, hidden_dim)

    print("\nNetwork architecture:")
    print(f"  State dim: {state_dim} (paper Table 8: [U_t, R_cpu, B_net])")
    print(f"  Action dim: {action_dim} (paper: L+1, here L=8)")
    print(f"  Total parameters: {sum(p.numel() for p in network.parameters())}")

    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    action_probs, state_values = network(states)

    print("\nForward pass:")
    print(f"  Input shape: {states.shape}")
    print(f"  Action probs shape: {action_probs.shape}")
    print(f"  State values shape: {state_values.shape}")
    print(f"  Action probs sum: {action_probs.sum(dim=1).mean():.3f} "
          "(should be ~1.0)")

    single_state = torch.randn(state_dim)
    action, log_prob, value = network.act(single_state)
    print("\nAction sampling:")
    print(f"  Sampled action: {action}")
    print(f"  Log probability: {log_prob.item():.3f}")
    print(f"  State value: {value.item():.3f}")

    actions = torch.randint(0, action_dim, (batch_size,))
    log_probs, values, entropy = network.evaluate(states, actions)
    print("\nEvaluation:")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Mean entropy: {entropy.mean():.3f}")

    print("\nNetwork test passed!")
    return True


def main():
    """Run all RL module tests."""
    print("\n" + "=" * 60)
    print("StreamSplit Control Plane Tests")
    print("=" * 60)

    try:
        network_ok = test_actor_critic_network()
        env_ok = test_environment()
        agent_ok = test_agent()
        controller_ok = test_controller()

        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Actor-Critic Network: {'PASS' if network_ok else 'FAIL'}")
        print(f"Environment:          {'PASS' if env_ok else 'FAIL'}")
        print(f"PPO Agent:            {'PASS' if agent_ok else 'FAIL'}")
        print(f"Split Controller:     {'PASS' if controller_ok else 'FAIL'}")

        if all([network_ok, env_ok, agent_ok, controller_ok]):
            print("\nAll Control Plane tests passed!")
            return 0
        print("\nSome tests failed!")
        return 1

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
