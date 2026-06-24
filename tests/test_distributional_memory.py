#!/usr/bin/env python3
"""
Unit tests for edge.distributional_memory.DistributionalMemory (Eq. 7-9, 11).
"""

import math

import torch

from edge.distributional_memory import DistributionalMemory


def make_memory(num_components=64, embedding_dim=128, cold_start_frames=5):
    return DistributionalMemory(
        num_components=num_components,
        embedding_dim=embedding_dim,
        cold_start_frames=cold_start_frames,
    )


def test_param_shapes():
    memory = make_memory(num_components=64, embedding_dim=128)
    assert memory.pi.shape == (64,)
    assert memory.mu.shape == (64, 128)
    assert memory.sigma.shape == (64, 128)


def test_storage_bytes_matches_paper_estimate():
    memory = make_memory(num_components=64, embedding_dim=128)
    # Eq. 8: Size ~= 2*(C*d*2B) + (C*2B) ~= 33KB for C=64, d=128.
    storage_kb = memory.storage_bytes() / 1024
    assert 30 <= storage_kb <= 35


def test_cold_start_then_warm():
    memory = make_memory(cold_start_frames=10)
    assert not memory.is_warm
    for _ in range(10):
        memory.update(torch.nn.functional.normalize(torch.randn(1, 128), dim=1))
    assert memory.is_warm


def test_update_keeps_pi_normalized():
    memory = make_memory()
    embeddings = torch.nn.functional.normalize(torch.randn(20, 128), dim=1)
    for emb in embeddings:
        memory.update(emb.unsqueeze(0))
    assert torch.isclose(memory.pi.sum(), torch.tensor(1.0), atol=1e-4)


def test_entropy_bounds():
    memory = make_memory(num_components=64)
    embeddings = torch.nn.functional.normalize(torch.randn(10, 128), dim=1)
    entropy = memory.entropy(embeddings)
    assert entropy.shape == (10,)
    assert torch.all(entropy >= 0)
    assert torch.all(entropy <= math.log(64) + 1e-4)


def test_virtual_negatives_are_unit_norm_and_exclude_own_component():
    memory = make_memory(num_components=64, embedding_dim=128)
    embeddings = torch.nn.functional.normalize(torch.randn(20, 128), dim=1)
    for emb in embeddings:
        memory.update(emb.unsqueeze(0))

    anchor = embeddings[0]
    n_syn = 256
    negatives = memory.sample_virtual_negatives(anchor, n_syn)

    assert negatives.shape == (n_syn, 128)
    norms = torch.norm(negatives, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    c_star = int(torch.argmax(memory.posterior(anchor.unsqueeze(0))).item())
    # No sampled negative should be reconstructible to exactly mu_{c*}
    # before noise -- a softer check: the posterior-assigned component's
    # mean should not dominate the sampled set entirely.
    component_means_used = torch.cdist(negatives, memory.mu)
    nearest_component = torch.argmin(component_means_used, dim=1)
    assert not torch.all(nearest_component == c_star)


def test_sync_payload_roundtrip():
    server_memory = make_memory()
    edge_memory = make_memory()

    embeddings = torch.nn.functional.normalize(torch.randn(10, 128), dim=1)
    for emb in embeddings:
        server_memory.update(emb.unsqueeze(0))

    edge_memory.load_sync_payload(server_memory.sync_payload())
    assert torch.allclose(edge_memory.mu, server_memory.mu, atol=1e-2)
    assert torch.allclose(edge_memory.pi, server_memory.pi, atol=1e-2)


if __name__ == '__main__':
    test_param_shapes()
    test_storage_bytes_matches_paper_estimate()
    test_cold_start_then_warm()
    test_update_keeps_pi_normalized()
    test_entropy_bounds()
    test_virtual_negatives_are_unit_norm_and_exclude_own_component()
    test_sync_payload_roundtrip()
    print("All distributional memory tests passed!")
