"""
Temporal Buffer for the Cloud Refiner (paper Sec. 4.3.1).

Maintains a sliding window of recent edge embeddings indexed by frame
number (not insertion order), and builds a k-NN graph over temporally
adjacent *available* frames. Frames are "unavailable" when the edge drops
them (network dropout) or the Control Plane chooses full on-device
processing -- the graph naturally bridges across such gaps by connecting
to the nearest available frames, which is the mechanism behind "Manifold
Stitching" (Fig. 5, Sec. 4.3.2).

Edge weights W_ij are left uniform (1.0): Definition 2 (Sec. 3.2) leaves
the weighting scheme for the temporal graph unspecified beyond "connect
temporally adjacent frames with weights W_ij", so this is the simplest
choice consistent with the paper's definition.
"""

from typing import Dict, List, Optional, Tuple

import torch


class TemporalBuffer:
    """Sliding temporal window with a k-NN temporal adjacency graph."""

    def __init__(self, window_size: int = 100, k_neighbors: int = 5):
        """
        Args:
            window_size: W, the buffer's temporal context (paper: 100 frames)
            k_neighbors: k for the temporal k-NN graph (paper: 5)
        """
        self.window_size = window_size
        self.k_neighbors = k_neighbors
        self._buffer: Dict[int, torch.Tensor] = {}
        self._next_index = 0

    def insert(self, embedding: torch.Tensor,
               frame_index: Optional[int] = None) -> int:
        """
        Insert an embedding at a temporal index, evicting indices that fall
        outside the sliding window.

        Args:
            embedding: Embedding tensor [D]
            frame_index: Temporal index; defaults to the next sequential one

        Returns:
            The frame index the embedding was stored at
        """
        if frame_index is None:
            frame_index = self._next_index

        self._buffer[frame_index] = embedding.detach()
        self._next_index = max(self._next_index, frame_index + 1)

        cutoff = self._next_index - self.window_size
        for idx in [i for i in self._buffer if i < cutoff]:
            del self._buffer[idx]

        return frame_index

    def build_graph(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Build the temporal k-NN graph (Eq. 4) over currently available
        frames.

        Returns:
            (embeddings, edges): embeddings stacked in temporal order
            [N, D], and edges as (i, j) index pairs into that stacked
            tensor, connecting each frame to its k_neighbors temporally
            closest available frames.
        """
        indices = sorted(self._buffer.keys())
        if len(indices) < 2:
            return torch.empty(0), []

        embeddings = torch.stack([self._buffer[i] for i in indices])

        edges = []
        for pos, idx in enumerate(indices):
            distances = sorted(
                (abs(idx - other_idx), other_pos)
                for other_pos, other_idx in enumerate(indices)
                if other_pos != pos
            )
            for _, other_pos in distances[:self.k_neighbors]:
                edges.append((pos, other_pos))

        return embeddings, edges

    def __len__(self) -> int:
        return len(self._buffer)
