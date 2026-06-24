"""
Edge audio dataset loader for on-device continuous audio streams.
Handles real-world edge deployment scenarios.
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, List, Tuple, Optional, Iterator
from pathlib import Path
import json
from tqdm import tqdm
import time


class EdgeAudioDataset(Dataset):
    """Dataset for edge device audio collected from continuous streams."""
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 sample_rate: int = 16000,
                 duration: float = 10.0,
                 num_classes: int = 7,
                 create_synthetic: bool = True,
                 transform=None,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 n_mels: int = 128):
        """
        Initialize Edge Audio dataset.
        
        Args:
            data_dir: Root directory for dataset
            split: 'train', 'val', or 'test'
            sample_rate: Target sample rate
            duration: Audio clip duration in seconds
            num_classes: Number of classes (smart home/urban sounds)
            create_synthetic: Create synthetic data if True
            transform: Optional transform to apply
            n_fft: FFT size for mel-spectrogram
            hop_length: Hop length for mel-spectrogram
            n_mels: Number of mel filterbanks
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_classes = num_classes
        self.transform = transform
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.audio_dir = self.data_dir / 'edge_audio' / 'audio'
        self.metadata_dir = self.data_dir / 'edge_audio' / 'metadata'
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=80,
            f_max=8000
        )
        
        # Load or create dataset
        if create_synthetic and not self._check_exists():
            self._create_synthetic_edge_data()
        
        self.samples = self._load_metadata()
        
        # Filter by split
        split_samples = [s for s in self.samples if s['split'] == split]
        self.samples = split_samples
        
        print(f"Loaded {len(self.samples)} edge audio samples for {split} split")
    
    def _check_exists(self) -> bool:
        """Check if dataset already exists."""
        metadata_file = self.metadata_dir / 'samples.json'
        return (metadata_file.exists() and 
                len(list(self.audio_dir.glob('*.wav'))) > 0)
    
    def _create_synthetic_edge_data(self):
        """Create synthetic edge audio data with realistic noise patterns."""
        print("Creating synthetic edge audio data...")
        
        num_samples_per_split = {
            'train': 600,
            'val': 100,
            'test': 100
        }
        
        # Smart home and urban monitoring classes
        class_names = [
            'silence',
            'speech',
            'door_knock',
            'glass_break',
            'alarm',
            'footsteps',
            'appliance_noise'
        ][:self.num_classes]
        
        samples = []
        sample_id = 0
        
        for split, num_samples in num_samples_per_split.items():
            for i in tqdm(range(num_samples), desc=f"Creating {split} edge data"):
                class_id = i % self.num_classes
                audio_path = self.audio_dir / f"{split}_edge_{sample_id:06d}.wav"
                
                # Create realistic synthetic audio for each class
                waveform = self._generate_class_audio(class_id, class_names[class_id])
                
                # Add realistic edge noise artifacts
                waveform = self._add_edge_noise(waveform)
                
                # Save audio
                torchaudio.save(
                    str(audio_path),
                    torch.from_numpy(waveform).unsqueeze(0),
                    self.sample_rate
                )
                
                # Metadata
                samples.append({
                    'id': sample_id,
                    'path': str(audio_path),
                    'class_id': class_id,
                    'class_name': class_names[class_id],
                    'split': split,
                    'duration': self.duration,
                    'device_id': f'rpi_{(i % 5) + 1}',  # Simulate 5 devices
                    'timestamp': time.time() + i * 10
                })
                
                sample_id += 1
        
        # Save metadata
        metadata_file = self.metadata_dir / 'samples.json'
        with open(metadata_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        # Save class mapping
        class_mapping = {i: name for i, name in enumerate(class_names)}
        with open(self.metadata_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"Created {len(samples)} synthetic edge audio samples")
    
    def _generate_class_audio(self, class_id: int, class_name: str) -> np.ndarray:
        """Generate synthetic audio for a specific class."""
        num_samples = int(self.sample_rate * self.duration)
        t = np.linspace(0, self.duration, num_samples)
        
        if class_name == 'silence':
            # Very low amplitude noise
            waveform = 0.01 * np.random.randn(num_samples)
            
        elif class_name == 'speech':
            # Simulate speech with formants
            f1, f2 = 300 + class_id * 50, 1200 + class_id * 100
            waveform = (0.3 * np.sin(2 * np.pi * f1 * t) +
                       0.2 * np.sin(2 * np.pi * f2 * t))
            # Add amplitude modulation
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
            waveform = waveform * mod
            
        elif class_name == 'door_knock':
            # Simulate knocking (impulsive sounds)
            waveform = np.zeros(num_samples)
            knock_times = [0.5, 1.0, 1.5]
            for knock_t in knock_times:
                idx = int(knock_t * self.sample_rate)
                if idx < num_samples:
                    # Decaying impulse
                    decay = np.exp(-np.arange(2000) / 500)
                    impulse = 0.8 * np.sin(2 * np.pi * 200 * np.arange(2000) / self.sample_rate) * decay
                    end_idx = min(idx + 2000, num_samples)
                    waveform[idx:end_idx] += impulse[:end_idx - idx]
            
        elif class_name == 'glass_break':
            # High frequency noise burst
            waveform = np.random.randn(num_samples) * 0.6
            # Bandpass filter effect
            freq_profile = np.sin(2 * np.pi * np.linspace(2000, 8000, num_samples) * t)
            waveform = waveform * (0.3 + 0.7 * np.abs(freq_profile))
            # Envelope
            envelope = np.exp(-t * 2)
            waveform = waveform * envelope
            
        elif class_name == 'alarm':
            # Periodic beeping
            beep_freq = 1000
            beep_period = 1.0
            beeps = np.sin(2 * np.pi * beep_freq * t)
            # Square wave modulation
            mod = (np.sin(2 * np.pi * (1 / beep_period) * t) > 0).astype(float)
            waveform = 0.5 * beeps * mod
            
        elif class_name == 'footsteps':
            # Rhythmic low frequency impacts
            waveform = np.zeros(num_samples)
            step_times = np.arange(0.5, self.duration, 0.6)
            for step_t in step_times:
                idx = int(step_t * self.sample_rate)
                if idx < num_samples:
                    decay = np.exp(-np.arange(1500) / 400)
                    step = 0.4 * np.sin(2 * np.pi * 100 * np.arange(1500) / self.sample_rate) * decay
                    end_idx = min(idx + 1500, num_samples)
                    waveform[idx:end_idx] += step[:end_idx - idx]
            
        elif class_name == 'appliance_noise':
            # Continuous mechanical hum
            fundamental = 50 + class_id * 10
            waveform = (0.3 * np.sin(2 * np.pi * fundamental * t) +
                       0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +
                       0.1 * np.sin(2 * np.pi * fundamental * 3 * t))
            waveform += 0.15 * np.random.randn(num_samples)
            
        else:
            # Default: white noise
            waveform = 0.2 * np.random.randn(num_samples)
        
        return waveform.astype(np.float32)
    
    def _add_edge_noise(self, waveform: np.ndarray) -> np.ndarray:
        """Add realistic edge device noise and artifacts."""
        # Background noise
        waveform += 0.02 * np.random.randn(len(waveform))
        
        # Occasional clipping (from low-quality ADC)
        if np.random.rand() < 0.1:
            clip_threshold = 0.7
            waveform = np.clip(waveform, -clip_threshold, clip_threshold)
        
        # Quantization noise (simulate 16-bit ADC)
        if np.random.rand() < 0.2:
            quantization_levels = 2**12  # 12-bit
            waveform = np.round(waveform * quantization_levels) / quantization_levels
        
        # Normalize
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val * 0.9
        
        return waveform
    
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata."""
        metadata_file = self.metadata_dir / 'samples.json'
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_file}. "
                "Set create_synthetic=True to create dataset."
            )
        
        with open(metadata_file, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (waveform, label, metadata)
        """
        sample = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(sample['path'])
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure correct duration
        target_length = int(self.sample_rate * self.duration)
        if waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale (dB)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        # Apply additional transform if provided
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        label = sample['class_id']
        
        # Metadata for edge processing
        metadata = {
            'device_id': sample.get('device_id', 'unknown'),
            'timestamp': sample.get('timestamp', 0)
        }
        
        return mel_spec, label, metadata
    
    def get_class_mapping(self) -> Dict[int, str]:
        """Get class ID to name mapping."""
        mapping_file = self.metadata_dir / 'class_mapping.json'
        with open(mapping_file, 'r') as f:
            return json.load(f)


class StreamingEdgeDataset(IterableDataset):
    """Streaming dataset for continuous edge audio processing."""
    
    def __init__(self,
                 data_dir: str,
                 sample_rate: int = 16000,
                 window_size: int = 160000,  # 10 seconds at 16kHz
                 stride: int = 16000):  # 1 second stride
        """
        Initialize streaming edge dataset.
        
        Args:
            data_dir: Directory containing continuous audio streams
            sample_rate: Sample rate
            window_size: Window size in samples
            stride: Stride between windows in samples
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.stride = stride
        
        # Find all audio files
        self.audio_files = sorted(list(self.data_dir.glob('*.wav')))
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over audio windows."""
        for audio_file in self.audio_files:
            # Load full audio
            waveform, sr = torchaudio.load(str(audio_file))
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.squeeze(0)
            
            # Generate windows
            num_windows = (len(waveform) - self.window_size) // self.stride + 1
            
            for i in range(num_windows):
                start = i * self.stride
                end = start + self.window_size
                window = waveform[start:end]
                
                yield window


def create_edge_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for edge audio.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']
    
    # Create datasets
    train_dataset = EdgeAudioDataset(
        data_dir=data_config['data_dir'],
        split='train',
        sample_rate=data_config['sample_rate'],
        duration=data_config['audio_duration'],
        num_classes=7,  # Smart home/urban monitoring classes
        create_synthetic=True
    )
    
    val_dataset = EdgeAudioDataset(
        data_dir=data_config['data_dir'],
        split='val',
        sample_rate=data_config['sample_rate'],
        duration=data_config['audio_duration'],
        num_classes=7,
        create_synthetic=False
    )
    
    test_dataset = EdgeAudioDataset(
        data_dir=data_config['data_dir'],
        split='test',
        sample_rate=data_config['sample_rate'],
        duration=data_config['audio_duration'],
        num_classes=7,
        create_synthetic=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['edge']['batch_size'] // 4,  # Smaller batch for edge
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['edge']['batch_size'] // 4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['edge']['batch_size'] // 4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test edge dataset
    import yaml
    
    with open('../configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing Edge Audio dataset...")
    train_loader, val_loader, test_loader = create_edge_loaders(config)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    waveforms, labels, metadata = next(iter(train_loader))
    print(f"\nBatch waveforms shape: {waveforms.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Sample device IDs: {[m['device_id'] for m in metadata][:5]}")
