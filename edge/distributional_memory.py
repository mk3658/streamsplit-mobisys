"""
Distributional Memory: GMM-based replacement for explicit negative-sample
storage on the edge (paper Sec. 4.1.2-4.1.3).

Unlike a memory bank, no raw embeddings are ever stored here -- only the
Gaussian Mixture Model's sufficient statistics (pi, mu, sigma), which fit
in ~33KB for the paper's default C=64 components, d=128 (Eq. 8). The GMM is
updated online via streaming EM (Eq. 7) and used for two purposes:

1. Boundary-aware virtual negative sampling (Eq. 9) for the edge contrastive
   loss (Eq. 10), synthesized fresh per anchor and discarded immediately.
2. Embedding uncertainty U_t = H(p(c|z_t)) (Eq. 11), the zero-cost signal
   consumed by the Control Plane (edge/rl_splitting.py).
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F


class DistributionalMemory:
    """Streaming GMM over the embedding space (Eq. 7)."""

    def __init__(self, num_components: int = 64, embedding_dim: int = 128,
                 tau: float = 0.1, cold_start_frames: int = 50,
                 ema_lr: float = 0.01, eps: float = 1e-6):
        """
        Args:
            num_components: Number of Gaussian components C (paper: 64)
            embedding_dim: Embedding dimension d (paper: 128)
            tau: Temperature controlling boundary-aware sampling hardness (Eq. 9)
            cold_start_frames: Frames to observe before sampling negatives (Sec. 4.1.2)
            ema_lr: Online EM update rate
            eps: Numerical stability constant
        """
        self.num_components = num_components
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.cold_start_frames = cold_start_frames
        self.ema_lr = ema_lr
        self.eps = eps

        self.pi = torch.ones(num_components) / num_components
        self.mu = torch.randn(num_components, embedding_dim) * 0.1
        self.sigma = torch.ones(num_components, embedding_dim) * 0.5

        self.n_seen = 0

    @property
    def is_warm(self) -> bool:
        """Whether the cold-start period (Sec. 4.1.2) has elapsed."""
        return self.n_seen >= self.cold_start_frames

    def _log_gaussian(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Log density of each embedding under each diagonal-covariance component."""
        diff = embeddings.unsqueeze(1) - self.mu.unsqueeze(0)  # [N, C, D]
        var = self.sigma.unsqueeze(0)  # [1, C, D]
        return -0.5 * torch.sum(
            diff ** 2 / var + torch.log(2 * math.pi * var), dim=-1
        )  # [N, C]

    def posterior(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Posterior responsibilities p(c|z) via Bayes' rule. [N, D] -> [N, C]."""
        log_prob = self._log_gaussian(embeddings) + torch.log(self.pi + self.eps)
        return torch.softmax(log_prob, dim=-1)

    def entropy(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Embedding uncertainty U_t = H(p(c|z_t)) (Eq. 11).

        Args:
            embeddings: [N, D]

        Returns:
            Entropy values [N], in [0, log(C)].
        """
        p = self.posterior(embeddings)
        return -torch.sum(p * torch.log(p + self.eps), dim=-1)

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor):
        """Streaming EM update, O(C*d) per call (Sec. 4.1.2)."""
        embeddings = embeddings.detach()
        responsibilities = self.posterior(embeddings)  # [N, C]
        n_k = responsibilities.sum(dim=0)  # [C]
        active = n_k > self.eps

        if active.any():
            new_pi = n_k / embeddings.shape[0]
            n_k_safe = n_k.clamp_min(self.eps).unsqueeze(1)
            new_mu = (responsibilities.t() @ embeddings) / n_k_safe
            diff = embeddings.unsqueeze(1) - new_mu.unsqueeze(0)
            new_sigma = torch.einsum(
                'nc,ncd->cd', responsibilities, diff ** 2
            ) / n_k_safe

            lr = self.ema_lr
            self.pi = torch.where(active, (1 - lr) * self.pi + lr * new_pi, self.pi)
            mask = active.unsqueeze(1)
            self.mu = torch.where(mask, (1 - lr) * self.mu + lr * new_mu, self.mu)
            self.sigma = torch.where(
                mask, (1 - lr) * self.sigma + lr * new_sigma.clamp_min(self.eps), self.sigma
            )

        self.pi = self.pi / self.pi.sum()
        self.n_seen += embeddings.shape[0]

    @torch.no_grad()
    def sample_virtual_negatives(self, anchor: torch.Tensor, n_syn: int) -> torch.Tensor:
        """
        Boundary-aware virtual negative sampling (Eq. 9).

        Samples components c != c* weighted by proximity of mu_c to the
        anchor's assigned component mu_c*, draws one Gaussian sample per
        chosen component, and L2-normalizes. Nothing is stored -- the
        caller uses and discards these tensors immediately (Sec. 4.1.3).

        Args:
            anchor: Single anchor embedding [D]
            n_syn: Number of virtual negatives to synthesize

        Returns:
            Virtual negative embeddings [n_syn, D], L2-normalized.
        """
        posterior = self.posterior(anchor.unsqueeze(0)).squeeze(0)  # [C]
        c_star = int(torch.argmax(posterior).item())

        mu_star = self.mu[c_star]
        sq_dist = torch.sum((mu_star.unsqueeze(0) - self.mu) ** 2, dim=1)  # [C]
        logits = torch.log(self.pi + self.eps) - sq_dist / (2 * self.tau ** 2)
        logits[c_star] = float('-inf')

        probs = torch.softmax(logits, dim=0)
        component_idx = torch.multinomial(probs, n_syn, replacement=True)

        std = torch.sqrt(self.sigma[component_idx])
        samples = self.mu[component_idx] + torch.randn(n_syn, self.embedding_dim) * std
        return F.normalize(samples, dim=1)

    def storage_bytes(self) -> int:
        """FP16 storage footprint of (pi, mu, sigma) -- Eq. 8, ~33KB for C=64, d=128."""
        c, d = self.num_components, self.embedding_dim
        return 2 * (c * d * 2) + (c * 2)

    def sync_payload(self) -> Dict[str, torch.Tensor]:
        """FP16 GMM parameters for Lazy Synchronization (Sec. 4.3.3)."""
        return {
            'pi': self.pi.half(),
            'mu': self.mu.half(),
            'sigma': self.sigma.half(),
        }

    def load_sync_payload(self, payload: Dict[str, torch.Tensor]):
        """Apply GMM parameters received via Lazy Synchronization (Sec. 4.3.3)."""
        self.pi = payload['pi'].float()
        self.mu = payload['mu'].float()
        self.sigma = payload['sigma'].float()
