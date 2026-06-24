"""
Cloud Refiner (paper Sec. 4.3): owns the Temporal Buffer, computes the
Hybrid Loss for global refinement, and implements Lazy Synchronization of
GMM parameters back to the edge (Sec. 4.3.3).
"""

from typing import Dict, Optional, Union

import torch

from edge.distributional_memory import DistributionalMemory
from .hybrid_loss import HybridLoss
from .temporal_buffer import TemporalBuffer


class ServerRefiner:
    """Phase 3 server-side refinement: Temporal Buffer + Hybrid Loss."""

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with a `hybrid_loss` section
        """
        hl_config = config['hybrid_loss']
        self.buffer = TemporalBuffer(
            window_size=hl_config['window'],
            k_neighbors=hl_config['k_neighbors'],
        )
        self.hybrid_loss = HybridLoss(config)
        self.sync_every = hl_config['sync_every']
        self._frames_since_sync = 0

    def add_embedding(self, embedding: torch.Tensor,
                       frame_index: Optional[int] = None) -> int:
        """Insert an edge embedding into the Temporal Buffer."""
        return self.buffer.insert(embedding, frame_index)

    def compute_hybrid_loss(
        self,
        task_loss: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Optional[Union[torch.Tensor, Dict]]:
        """Compute the Hybrid Loss (Eq. 13) over the buffer's current graph."""
        embeddings, edges = self.buffer.build_graph()
        if embeddings.numel() == 0:
            return None
        return self.hybrid_loss(
            embeddings, edges, task_loss=task_loss,
            return_components=return_components,
        )

    def should_sync(self) -> bool:
        """
        Whether T_sync frames have elapsed since the last GMM sync
        (Sec. 4.3.3: every T_sync=100 frames).
        """
        self._frames_since_sync += 1
        if self._frames_since_sync >= self.sync_every:
            self._frames_since_sync = 0
            return True
        return False


class LazySync:
    """
    Lazy Synchronization protocol (Sec. 4.3.3): the server periodically
    transmits its updated GMM parameters (<35KB) down to the edge every
    T_sync frames. Encoder weight synchronization is far less frequent and
    gated on device conditions (charging, high-bandwidth WiFi) this repo
    has no real signal for, so it's left as an explicit, caller-triggered
    action rather than simulated here.
    """

    @staticmethod
    def sync_gmm(server_memory: DistributionalMemory,
                 edge_memory: DistributionalMemory):
        """Push server_memory's GMM parameters down to edge_memory."""
        edge_memory.load_sync_payload(server_memory.sync_payload())
