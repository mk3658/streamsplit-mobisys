#!/usr/bin/env python3
"""
Training script for the Control Plane's PPO split-point agent (Sec. 4.2).

Trains against SimulatedEdgeCloudEnv, an offline stand-in for the paper's
hardware traces (see edge/rl_splitting.py module docstring).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from edge.rl_splitting import PPOSplitAgent, SimulatedEdgeCloudEnv, SplitController
from utils.device import get_device, optimize_for_device, print_device_info
from utils.logger import Logger

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def plot_training_curves(rewards, policy_losses, value_losses, save_path):
    """Plot and save training curves."""
    if plt is None:
        print("matplotlib not available, skipping training curve plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(policy_losses)
    axes[1].set_title('Policy Loss')
    axes[1].set_xlabel('Update')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(value_losses)
    axes[2].set_title('Value Loss')
    axes[2].set_xlabel('Update')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def evaluate_policy(agent, env, num_episodes=10):
    """Evaluate trained policy with greedy action selection."""
    episode_rewards = []
    split_decisions = []
    accuracies = []
    latencies = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_splits = []

        for _ in range(env.max_episode_steps):
            state_tensor = torch.FloatTensor(state).to(agent.device)
            with torch.no_grad():
                action_probs, _ = agent.policy(state_tensor.unsqueeze(0))
                action = torch.argmax(action_probs, dim=-1).item()

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_splits.append(info['split_layer'])
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        split_decisions.extend(episode_splits)
        accuracies.append(info['accuracy'])
        latencies.append(info['latency_ms'])

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_accuracy': np.mean(accuracies),
        'mean_latency_ms': np.mean(latencies),
        'split_distribution': np.bincount(
            split_decisions, minlength=env.num_actions
        ).tolist(),
    }


def train_rl_agent(config, args):
    """Train the Control Plane's PPO agent."""
    device = get_device(force_cpu=args.force_cpu)
    print_device_info(device)
    optimize_for_device(device)
    config['experiment']['device'] = str(device)

    log_dir = Path(config['experiment']['log_dir']) / 'rl_training'
    checkpoint_dir = Path(config['experiment']['checkpoint_dir']) / 'rl'
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(log_dir, 'rl_training')

    env = SimulatedEdgeCloudEnv(
        config, num_blocks=config['encoder']['num_blocks']
    )
    agent = PPOSplitAgent(config, env)

    print("=" * 60)
    print("Control Plane RL Training (Table 8 MDP)")
    print("=" * 60)
    print(f"State dimension: {env.state_dim}")
    print(f"Number of actions (L+1): {env.num_actions}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max episode steps: {env.max_episode_steps}")
    print(f"Update frequency: every {args.update_frequency} steps")
    print("=" * 60)

    episode_rewards, policy_losses, value_losses, entropies = [], [], [], []
    total_steps = 0
    best_reward = -float('inf')

    pbar = tqdm(range(args.num_episodes), desc="Training")
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for _ in range(env.max_episode_steps):
            action = agent.select_action(state)
            next_state, reward, done, _info = env.step(action)
            agent.store_transition(reward, done)

            episode_reward += reward
            episode_length += 1
            total_steps += 1
            state = next_state

            if total_steps % args.update_frequency == 0:
                metrics = agent.update()
                if metrics:
                    policy_losses.append(metrics['policy_loss'])
                    value_losses.append(metrics['value_loss'])
                    entropies.append(metrics['entropy'])

            if done:
                break

        episode_rewards.append(episode_reward)
        pbar.set_postfix({
            'reward': f'{episode_reward:.3f}',
            'length': episode_length,
            'best': f'{best_reward:.3f}',
        })

        if (episode + 1) % args.log_frequency == 0:
            avg_reward = np.mean(episode_rewards[-args.log_frequency:])
            logger.log({
                'episode': episode + 1,
                'avg_reward': avg_reward,
                'episode_reward': episode_reward,
                'total_steps': total_steps,
                'policy_loss': policy_losses[-1] if policy_losses else 0,
                'value_loss': value_losses[-1] if value_losses else 0,
                'entropy': entropies[-1] if entropies else 0,
            })

        if (episode + 1) % args.eval_frequency == 0:
            eval_metrics = evaluate_policy(agent, env, num_episodes=10)
            print(f"\nEpisode {episode + 1}/{args.num_episodes} eval: "
                  f"reward={eval_metrics['mean_reward']:.3f} "
                  f"accuracy={eval_metrics['mean_accuracy']:.3f} "
                  f"latency={eval_metrics['mean_latency_ms']:.1f}ms")

            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                agent.save(str(checkpoint_dir / 'best_policy.pt'))

        if (episode + 1) % args.checkpoint_frequency == 0:
            agent.save(str(checkpoint_dir / f'policy_ep{episode + 1}.pt'))

    final_eval = evaluate_policy(agent, env, num_episodes=50)
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    print(f"Mean Reward: {final_eval['mean_reward']:.3f} "
          f"+/- {final_eval['std_reward']:.3f}")
    print(f"Mean Accuracy: {final_eval['mean_accuracy']:.3f}")
    print(f"Mean Latency: {final_eval['mean_latency_ms']:.1f}ms")
    print("Split Layer Distribution:")
    total = sum(final_eval['split_distribution'])
    for k, count in enumerate(final_eval['split_distribution']):
        print(f"  k={k}: {count} ({100 * count / total:.1f}%)")

    agent.save(str(checkpoint_dir / 'final_policy.pt'))
    plot_training_curves(
        episode_rewards, policy_losses, value_losses,
        str(log_dir / 'training_curves.png')
    )

    logger.save_metrics()
    print("\nTraining completed successfully!")


def test_split_controller(config, policy_path):
    """Exercise a trained SplitController with synthetic state samples."""
    print("=" * 60)
    print("Testing Split Controller")
    print("=" * 60)

    controller = SplitController(
        config, num_blocks=config['encoder']['num_blocks'],
        policy_path=policy_path
    )

    for i in range(20):
        uncertainty = np.random.uniform(0.0, 1.0)
        cpu_util = np.random.uniform(0.1, 1.0)
        bandwidth_mbps = np.random.uniform(1.0, 50.0)

        split_layer = controller.get_split_layer(
            uncertainty, cpu_util, bandwidth_mbps
        )

        if (i + 1) % 5 == 0:
            print(f"\nDecision {i + 1}: split_layer={split_layer}, "
                  f"U_t={uncertainty:.3f}, R_cpu={cpu_util:.3f}, "
                  f"B_net={bandwidth_mbps:.1f}Mbps")

    stats = controller.get_statistics()
    print("\n" + "=" * 60)
    print("Split Controller Statistics")
    print("=" * 60)
    print(f"Total Decisions: {stats['num_decisions']}")
    print(f"Avg Split Layer: {stats['avg_split_layer']:.2f}")
    print(f"Split Distribution: {stats['split_distribution']}")
    print("\nController test completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Train the Control Plane PPO agent'
    )
    parser.add_argument('--config', type=str,
                         default='configs/streamsplit.yaml')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--update_frequency', type=int, default=256)
    parser.add_argument('--log_frequency', type=int, default=10)
    parser.add_argument('--eval_frequency', type=int, default=50)
    parser.add_argument('--checkpoint_frequency', type=int, default=100)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--test', action='store_true',
                         help='Test trained controller')
    parser.add_argument('--policy_path', type=str,
                         default='checkpoints/rl/best_policy.pt')

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.test:
        test_split_controller(config, args.policy_path)
    else:
        train_rl_agent(config, args)


if __name__ == '__main__':
    main()
