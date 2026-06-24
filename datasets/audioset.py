"""
AudioSet dataset loader for StreamSplit.
Downloads and prepares balanced AudioSet subset for training.
"""

import os
import csv
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import urllib.request
from pathlib import Path
from tqdm import tqdm


class AudioSetDataset(Dataset):
    """AudioSet dataset for audio representation learning."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 sample_rate: int = 16000,
                 duration: float = 10.0,
                 num_classes: int = 10,
                 download: bool = True,
                 transform=None,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 n_mels: int = 128):
        """
        Initialize AudioSet dataset.
        
        Args:
            data_dir: Root directory for dataset
            split: 'train', 'val', or 'test'
            sample_rate: Target sample rate
            duration: Audio clip duration in seconds
            num_classes: Number of classes to use
            download: Whether to download if not exists
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
        
        self.audio_dir = self.data_dir / 'audioset' / 'audio'
        self.metadata_dir = self.data_dir / 'audioset' / 'metadata'
        
        # Create mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=80,
            f_max=8000
        )
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        if download and not self._check_exists():
            self._download()
        
        self.samples = self._load_metadata()
        
        # Filter by split
        split_samples = [s for s in self.samples if s['split'] == split]
        self.samples = split_samples
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _check_exists(self) -> bool:
        """Check if dataset already exists."""
        metadata_file = self.metadata_dir / 'samples.json'
        return metadata_file.exists() and len(list(self.audio_dir.glob('*.wav'))) > 0
    
    def _download(self):
        """Download AudioSet balanced subset."""
        print("Downloading AudioSet balanced subset...")
        
        # For demo purposes, we'll create synthetic data
        # In production, you would download from YouTube using yt-dlp
        print("Note: Creating synthetic AudioSet data for demo.")
        print("For production, implement YouTube download with yt-dlp")
        
        self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self):
        """Create synthetic dataset for testing."""
        print("Creating synthetic AudioSet samples...")
        
        num_samples_per_split = {
            'train': 800,
            'val': 100,
            'test': 100
        }
        
        # Class labels (simplified AudioSet categories)
        class_names = [
            'speech', 'music', 'dog', 'car', 'water',
            'bird', 'footsteps', 'door', 'alarm', 'laughter'
        ][:self.num_classes]
        
        samples = []
        sample_id = 0
        
        for split, num_samples in num_samples_per_split.items():
            for i in tqdm(range(num_samples), desc=f"Creating {split} data"):
                # Generate synthetic audio
                class_id = i % self.num_classes
                audio_path = self.audio_dir / f"{split}_{sample_id:06d}.wav"
                
                # Create synthetic waveform
                num_samples_audio = int(self.sample_rate * self.duration)
                
                # Simple synthetic audio: sine wave with noise
                t = np.linspace(0, self.duration, num_samples_audio)
                freq = 220 + class_id * 110  # Different frequency per class
                waveform = 0.3 * np.sin(2 * np.pi * freq * t)
                waveform += 0.1 * np.random.randn(num_samples_audio)
                waveform = waveform.astype(np.float32)
                
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
                    'duration': self.duration
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
        
        print(f"Created {len(samples)} synthetic samples")
    
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata."""
        metadata_file = self.metadata_dir / 'samples.json'
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_file}. "
                "Set download=True to create dataset."
            )
        
        with open(metadata_file, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (waveform, label)
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
            # Pad
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_length:
            # Trim
            waveform = waveform[:, :target_length]
        
        # Convert to mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale (dB)
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize (helps training stability)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        # Apply additional transform if provided
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        label = sample['class_id']
        
        return mel_spec, label
    
    def get_class_mapping(self) -> Dict[int, str]:
        """Get class ID to name mapping."""
        mapping_file = self.metadata_dir / 'class_mapping.json'
        with open(mapping_file, 'r') as f:
            return json.load(f)


def create_audioset_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for AudioSet.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config['data']
    
    # Create datasets
    train_dataset = AudioSetDataset(
        data_dir=data_config['data_dir'],
        split='train',
        sample_rate=data_config['sample_rate'],
        duration=data_config['audio_duration'],
        num_classes=data_config['num_classes'],
        download=True
    )
    
    val_dataset = AudioSetDataset(
        data_dir=data_config['data_dir'],
        split='val',
        sample_rate=data_config['sample_rate'],
        duration=data_config['audio_duration'],
        num_classes=data_config['num_classes'],
        download=False
    )
    
    test_dataset = AudioSetDataset(
        data_dir=data_config['data_dir'],
        split='test',
        sample_rate=data_config['sample_rate'],
        duration=data_config['audio_duration'],
        num_classes=data_config['num_classes'],
        download=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['edge']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['edge']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['edge']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    import yaml
    
    with open('../configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing AudioSet dataset...")
    train_loader, val_loader, test_loader = create_audioset_loaders(config)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    waveforms, labels = next(iter(train_loader))
    print(f"\nBatch waveforms shape: {waveforms.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} to {labels.max()}")
