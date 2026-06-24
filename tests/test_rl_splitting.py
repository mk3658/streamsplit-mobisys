#!/usr/bin/env python3
"""
Unit tests for edge.rl_splitting (Table 8 MDP, Eq. 12 reward).
"""

import numpy as np

from edge.rl_splitting import SimulatedEdgeCloudEnv, SplitController


def make_config(num_blocks=8):
    return {
        'experiment': {'device': 'cpu'},
        'encoder': {'num_blocks': num_blocks},
        'control_plane': {
            'max_episode_steps': 100,
            'reward': {
                'alpha': 10,
                'beta': 5,
                'eta': 3,
                't_max_ms': 200.0,
                'e_budget_mj': 100.0,
            },
            'ppo': {
                'hidden_dim': 32,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'lambda_gae': 0.95,
                'epsilon_clip': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'update_epochs': 2,
                'batch_size': 8,
            },
        },
    }


def test_state_dim_is_three():
    config = make_config()
    env = SimulatedEdgeCloudEnv(config, num_blocks=8)
    state = env.reset()
    assert state.shape == (3,)
    assert env.state_dim == 3


def test_action_space_is_l_plus_one():
    config = make_config(num_blocks=8)
    env = SimulatedEdgeCloudEnv(config, num_blocks=8)
    assert env.num_actions == 9  # L=8 -> k in {0, ..., 8}


def test_reward_matches_eq12():
    config = make_config()
    env = SimulatedEdgeCloudEnv(config, num_blocks=8)
    reward_cfg = config['control_plane']['reward']

    env.reset()
    for action in range(env.num_actions):
        _next_state, reward, _done, info = env.step(action)

        expected = (
            reward_cfg['alpha'] * info['accuracy']
            - reward_cfg['beta'] * (info['latency_ms'] / reward_cfg['t_max_ms'])
            - reward_cfg['eta'] * (info['energy_mj'] / reward_cfg['e_budget_mj'])
        )
        assert abs(reward - expected) < 1e-5


def test_on_device_cheaper_energy_than_full_offload():
    # Sec. 6.2.3: radio transmission dominates battery energy, so full
    # on-device execution (k=L) consumes less energy than full offload
    # (k=0), despite costing more edge compute.
    config = make_config()
    env = SimulatedEdgeCloudEnv(config, num_blocks=8)
    env.reset()

    _, _, _, info_offload = env.step(0)  # k=0: full offload
    _, _, _, info_on_device = env.step(env.num_blocks)  # k=L: full on-device

    assert info_on_device['energy_mj'] < info_offload['energy_mj']


def test_split_controller_action_in_range():
    config = make_config(num_blocks=8)
    controller = SplitController(config, num_blocks=8, policy_path=None)

    for _ in range(20):
        k = controller.get_split_layer(
            uncertainty=np.random.uniform(0, 1),
            cpu_util=np.random.uniform(0, 1),
            bandwidth_mbps=np.random.uniform(1, 50),
        )
        assert 0 <= k <= 8


if __name__ == '__main__':
    test_state_dim_is_three()
    test_action_space_is_l_plus_one()
    test_reward_matches_eq12()
    test_on_device_cheaper_energy_than_full_offload()
    test_split_controller_action_in_range()
    print("All RL splitting tests passed!")
