"""
Hybrid loss for the Cloud Refiner (paper Sec. 3, Sec. 4.3.2, Eq. 13).

Combines a pluggable task loss with two distribution-based regularizers:
- Sliced-Wasserstein Distance (Eq. 1, Eq. 3): pulls the embedding
  distribution toward the uniform prior on the hypersphere, preventing
  dimensional collapse (Diversity, Sec. 3.1).
- Laplacian regularization (Eq. 4, Eq. 6/14): penalizes large jumps across
  a temporal adjacency graph, enforcing manifold smoothness (Affinity,
  Sec. 3.2).
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlicedWassersteinDistance(nn.Module):
    """
    Sliced-Wasserstein distance between an embedding batch and the uniform
    prior U(S^{d-1}) on the hypersphere (Eq. 1, Eq. 3) -- NOT a distance
    between two embedding batches.
    """

    def __init__(self, num_projections: int = 50):
        """
        Args:
            num_projections: M, number of random projection directions
                (paper: 50)
        """
        super().__init__()
        self.num_projections = num_projections

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings on the hypersphere [N, D]

        Returns:
            Scalar Sliced-Wasserstein distance to U(S^{d-1})
        """
        n, dim = embeddings.shape
        device = embeddings.device

        prior_samples = F.normalize(torch.randn(n, dim, device=device), dim=1)
        projections = F.normalize(
            torch.randn(dim, self.num_projections, device=device), dim=0
        )

        proj_embeddings = embeddings @ projections  # [N, M]
        proj_prior = prior_samples @ projections     # [N, M]

        sorted_embeddings, _ = torch.sort(proj_embeddings, dim=0)
        sorted_prior, _ = torch.sort(proj_prior, dim=0)

        sw_distance = torch.mean((sorted_embeddings - sorted_prior) ** 2)
        return torch.sqrt(sw_distance + 1e-8)


class LaplacianRegularization(nn.Module):
    """Dirichlet energy over a temporal adjacency graph (Eq. 4, Eq. 6/14)."""

    def forward(self, embeddings: torch.Tensor,
                edges: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Args:
            embeddings: Embeddings stacked in temporal order [N, D]
            edges: Temporal adjacency edges from TemporalBuffer.build_graph(),
                with implicit uniform weight W_ij=1 (see module docs of
                server.temporal_buffer)

        Returns:
            (1/|E|) * sum_{(i,j) in E} ||z_i - z_j||^2
        """
        if len(edges) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        i_idx = torch.tensor([e[0] for e in edges], device=embeddings.device)
        j_idx = torch.tensor([e[1] for e in edges], device=embeddings.device)
        diff = embeddings[i_idx] - embeddings[j_idx]
        return torch.sum(diff ** 2, dim=1).mean()


class HybridLoss(nn.Module):
    """Server objective L_task + lambda1 * L_SW + lambda2 * L_Lap (Eq. 13)."""

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with a `hybrid_loss` section
        """
        super().__init__()
        hl_config = config['hybrid_loss']

        self.sw_loss = SlicedWassersteinDistance(
            num_projections=hl_config['num_projections']
        )
        self.lap_loss = LaplacianRegularization()
        self.lambda_sw = hl_config['lambda_sw']
        self.lambda_lap = hl_config['lambda_lap']

    def forward(self,
                embeddings: torch.Tensor,
                edges: List[Tuple[int, int]],
                task_loss: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Args:
            embeddings: Buffer embeddings [N, D], L2-normalized, stacked in
                temporal order (matching `edges`' indices)
            edges: Temporal k-NN graph from TemporalBuffer.build_graph()
            task_loss: Precomputed L_task (Eq. 13). Pluggable per Sec. 4.3.2:
                a global InfoNCE loss in pure self-supervised settings, or a
                cross-entropy loss when labels are available. Defaults to 0
                if not supplied.
            return_components: Return a dict of loss components instead of
                just the total

        Returns:
            Total loss, or a dict of {loss_total, loss_task, loss_sw, loss_lap}
        """
        loss_sw = self.sw_loss(embeddings)
        loss_lap = self.lap_loss(embeddings, edges)
        loss_task = (
            task_loss if task_loss is not None
            else torch.tensor(0.0, device=embeddings.device)
        )

        loss_total = loss_task + self.lambda_sw * loss_sw + self.lambda_lap * loss_lap

        if return_components:
            return {
                'loss_total': loss_total,
                'loss_task': loss_task,
                'loss_sw': loss_sw,
                'loss_lap': loss_lap,
            }
        return loss_total
