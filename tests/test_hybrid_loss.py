#!/usr/bin/env python3
"""
Unit tests for server.hybrid_loss (Eq. 1, 3, 4, 6, 13, 14) and
server.temporal_buffer (Sec. 4.3.1).
"""

import torch
import torch.nn.functional as F

from server.hybrid_loss import HybridLoss, LaplacianRegularization, SlicedWassersteinDistance
from server.temporal_buffer import TemporalBuffer


def make_config():
    return {
        'hybrid_loss': {
            'num_projections': 50,
            'lambda_sw': 0.1,
            'lambda_lap': 0.01,
            'window': 100,
            'k_neighbors': 5,
            'sync_every': 100,
        }
    }


def test_swd_compares_against_uniform_prior_not_a_second_batch():
    sw = SlicedWassersteinDistance(num_projections=50)
    collapsed = F.normalize(torch.ones(64, 32) + 0.01 * torch.randn(64, 32), dim=1)
    diverse = F.normalize(torch.randn(64, 32), dim=1)

    loss_collapsed = sw(collapsed)
    loss_diverse = sw(diverse)

    # A near-collapsed batch should be farther from the uniform prior than
    # a genuinely diverse one (Sec. 3.1: minimizing L_SW <=> maximizing
    # entropy / diversity).
    assert loss_collapsed > loss_diverse


def test_laplacian_zero_with_no_edges():
    lap = LaplacianRegularization()
    embeddings = torch.randn(5, 16)
    loss = lap(embeddings, edges=[])
    assert torch.isclose(loss, torch.tensor(0.0))


def test_laplacian_matches_manual_computation():
    lap = LaplacianRegularization()
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    edges = [(0, 1), (1, 2)]

    loss = lap(embeddings, edges)
    expected = ((1.0) ** 2 + (1.0 ** 2 + 2.0 ** 2)) / 2  # mean over |E|=2
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-5)


def test_temporal_buffer_connects_temporally_not_by_feature_distance():
    buffer = TemporalBuffer(window_size=10, k_neighbors=1)
    # Frame 0 and frame 1 are temporally adjacent but feature-far apart;
    # frame 0 and frame 5 are feature-close but temporally distant.
    buffer.insert(torch.tensor([0.0, 0.0]), frame_index=0)
    buffer.insert(torch.tensor([10.0, 10.0]), frame_index=1)
    buffer.insert(torch.tensor([0.01, 0.01]), frame_index=5)

    embeddings, edges = buffer.build_graph()
    assert embeddings.shape[0] == 3
    # Frame 0 (position 0) should connect to frame 1 (position 1), its
    # temporal nearest neighbor, not frame 5 despite being feature-closer.
    frame0_neighbors = [j for (i, j) in edges if i == 0]
    assert frame0_neighbors == [1]


def test_temporal_buffer_evicts_outside_window():
    buffer = TemporalBuffer(window_size=3, k_neighbors=1)
    for i in range(10):
        buffer.insert(torch.randn(4), frame_index=i)
    assert len(buffer) <= 3


def test_hybrid_loss_combines_components_with_configured_weights():
    config = make_config()
    hybrid = HybridLoss(config)

    embeddings = F.normalize(torch.randn(20, 32), dim=1)
    edges = [(i, i + 1) for i in range(19)]
    task_loss = torch.tensor(1.234)

    result = hybrid(embeddings, edges, task_loss=task_loss, return_components=True)
    expected_total = (
        result['loss_task']
        + config['hybrid_loss']['lambda_sw'] * result['loss_sw']
        + config['hybrid_loss']['lambda_lap'] * result['loss_lap']
    )
    assert torch.isclose(result['loss_total'], expected_total, atol=1e-5)
    assert torch.isclose(result['loss_task'], task_loss)


def test_hybrid_loss_defaults_task_loss_to_zero():
    config = make_config()
    hybrid = HybridLoss(config)
    embeddings = F.normalize(torch.randn(10, 32), dim=1)
    edges = [(0, 1)]

    result = hybrid(embeddings, edges, return_components=True)
    assert torch.isclose(result['loss_task'], torch.tensor(0.0))


if __name__ == '__main__':
    test_swd_compares_against_uniform_prior_not_a_second_batch()
    test_laplacian_zero_with_no_edges()
    test_laplacian_matches_manual_computation()
    test_temporal_buffer_connects_temporally_not_by_feature_distance()
    test_temporal_buffer_evicts_outside_window()
    test_hybrid_loss_combines_components_with_configured_weights()
    test_hybrid_loss_defaults_task_loss_to_zero()
    print("All hybrid loss / temporal buffer tests passed!")
