"""
Streaming contrastive learning on the edge (paper Sec. 4.1, Eq. 10).

A single encoder processes an anchor frame and its augmented positive pair;
negatives are virtual, synthesized on-the-fly from the local
DistributionalMemory (Eq. 9) and never physically stored. The paper
explicitly does not use temporal neighbors as positives, to avoid
buffering latency.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributional_memory import DistributionalMemory


class StreamingContrastiveLearning(nn.Module):
    """Streaming InfoNCE with GMM-sampled virtual negatives (Eq. 10)."""

    def __init__(self, config: Dict, encoder: nn.Module,
                 memory: DistributionalMemory):
        """
        Args:
            config: Configuration dictionary
            encoder: Audio encoder (e.g. models.AudioResNet18)
            memory: DistributionalMemory instance providing virtual negatives
        """
        super().__init__()

        contrastive_config = config['edge']['contrastive']
        self.encoder = encoder
        self.memory = memory
        self.temperature = contrastive_config['temperature']
        self.n_syn = contrastive_config['n_syn']

    def forward(self, x: torch.Tensor,
                x_aug: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Anchor samples [batch_size, ...]
            x_aug: Augmented positive samples [batch_size, ...]

        Returns:
            Tuple of (loss, info_dict)
        """
        z = F.normalize(self.encoder(x), dim=1)
        z_pos = F.normalize(self.encoder(x_aug), dim=1)

        losses = []
        entropies = []
        for i in range(z.shape[0]):
            z_i, z_pos_i = z[i], z_pos[i]

            # Update the GMM with the latest embedding (Sec. 4.1.2, online EM).
            z_i_detached = z_i.detach().unsqueeze(0)
            self.memory.update(z_i_detached)
            entropies.append(self.memory.entropy(z_i_detached).item())

            pos_logit = torch.dot(z_i, z_pos_i) / self.temperature

            if self.memory.is_warm:
                negatives = self.memory.sample_virtual_negatives(
                    z_i.detach(), self.n_syn
                )
                neg_logits = (negatives @ z_i) / self.temperature
                logits = torch.cat([pos_logit.unsqueeze(0), neg_logits])
            else:
                # Cold start (Sec. 4.1.2): not enough statistics yet to
                # sample meaningful virtual negatives.
                logits = pos_logit.unsqueeze(0)

            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), labels))

        loss = torch.stack(losses).mean()

        info = {
            'loss_edge': loss.item(),
            'gmm_entropy': sum(entropies) / len(entropies),
            'gmm_warm': self.memory.is_warm,
        }
        return loss, info

    def compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a normalized embedding without affecting GMM state."""
        with torch.no_grad():
            return F.normalize(self.encoder(x), dim=1)
