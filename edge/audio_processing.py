"""
Audio processing module for edge devices.

Implements mel-spectrogram extraction (paper Sec. 5: 128-bin mel
spectrograms, 25ms windows, 10ms hop) and the augmentation pipeline used to
build contrastive positive pairs (Sec. 4.1.3).
"""

from typing import Dict

import numpy as np
import torch


class AudioProcessor:
    """Mel-spectrogram extractor for edge devices."""

    def __init__(self, config: Dict):
        """
        Initialize audio processor.

        Args:
            config: Configuration dictionary with FFT processing parameters
        """
        self.sample_rate = config['data']['sample_rate']
        self.fft_config = config['edge']['fft']

        self.window_size = int(
            self.fft_config['window_size_ms'] * self.sample_rate / 1000
        )
        self.hop_size = int(
            self.fft_config['hop_size_ms'] * self.sample_rate / 1000
        )
        self.n_fft = self.fft_config['n_fft']
        self.n_mels = self.fft_config['n_mels']

        self.window = torch.hann_window(self.window_size, dtype=torch.float32)
        self.mel_filterbank = self._create_mel_filterbank(
            self.n_mels, self.fft_config['fmin'], self.fft_config['fmax']
        )

    def _create_mel_filterbank(self, n_mels: int, fmin: float,
                                fmax: float) -> torch.Tensor:
        """Create mel-scale filterbank."""
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).astype(int)

        filterbank = np.zeros((n_mels, self.n_fft // 2 + 1))
        for i in range(n_mels):
            left, center, right = bin_points[i:i + 3]

            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

        return torch.from_numpy(filterbank).float()

    def process_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process raw audio waveform into a mel-spectrogram.

        Args:
            waveform: Input audio tensor [batch_size, num_samples] or
                [num_samples]

        Returns:
            Mel-spectrogram tensor [batch_size, n_mels, time_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=self.window,
            center=True,
            normalized=False,
            return_complex=True
        )

        mag_spec = torch.abs(spec)  # [batch, freq_bins, time]
        mel_spec = torch.matmul(self.mel_filterbank, mag_spec)
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec


class AudioAugmentor:
    """
    Audio data augmentation for contrastive learning (Sec. 4.1.3): random
    Gaussian noise and frequency masking, applied to produce the positive
    pair (x_tilde_t, x_tilde'_t) consumed by Eq. 10.
    """

    def __init__(self, config: Dict):
        """Initialize augmentor with configuration."""
        self.config = config['edge']['augmentation']
        self.sample_rate = config['data']['sample_rate']

    def gaussian_noise(self, waveform: torch.Tensor,
                        std: float = 0.01) -> torch.Tensor:
        """Add random Gaussian noise to the waveform."""
        return waveform + torch.randn_like(waveform) * std

    def frequency_mask(self, mel_spec: torch.Tensor,
                        mask_param: int = 10) -> torch.Tensor:
        """
        Apply frequency masking.

        Args:
            mel_spec: Mel-spectrogram tensor [batch, n_mels, time]
            mask_param: Maximum number of consecutive mel bins to mask

        Returns:
            Masked mel-spectrogram
        """
        n_mels = mel_spec.shape[1]
        mask_size = np.random.randint(0, mask_param)
        if mask_size < n_mels:
            mask_start = np.random.randint(0, n_mels - mask_size)
        else:
            mask_start = 0

        mel_spec = mel_spec.clone()
        mel_spec[:, mask_start:mask_start + mask_size, :] = 0

        return mel_spec

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to a raw waveform (pre-spectrogram)."""
        return self.gaussian_noise(waveform, self.config['noise_std'])

    def augment_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to a mel-spectrogram (post-FFT)."""
        return self.frequency_mask(mel_spec, self.config['freq_mask_param'])
