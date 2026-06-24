#!/usr/bin/env python3
"""
Test script for dataset implementations.
"""

import sys
import yaml
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import (
    AudioSetDataset, 
    create_audioset_loaders,
    EdgeAudioDataset,
    create_edge_loaders
)


def test_audioset():
    """Test AudioSet dataset and dataloaders."""
    print("=" * 60)
    print("Testing AudioSet Dataset")
    print("=" * 60)
    
    # Load config
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_audioset_loaders(config)
    
    # Test sizes
    print(f"\n✓ Created dataloaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test batch loading
    train_batch = next(iter(train_loader))
    waveforms, labels = train_batch
    
    print(f"\n✓ Sample batch:")
    print(f"  Waveform shape: {waveforms.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Waveform dtype: {waveforms.dtype}")
    print(f"  Labels dtype: {labels.dtype}")
    print(f"  Waveform range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")
    print(f"  Unique labels: {torch.unique(labels).tolist()}")
    
    # Test class mapping
    class_mapping = train_loader.dataset.get_class_mapping()
    print(f"\n✓ Class mapping ({len(class_mapping)} classes):")
    for i, name in sorted(class_mapping.items()):
        print(f"  {i}: {name}")
    
    # Test multiple batches
    print("\n✓ Testing batch iteration:")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        waveforms, labels = batch
        print(f"  Batch {i}: {waveforms.shape}, "
              f"labels {labels.shape}, "
              f"mean {waveforms.mean():.3f}")
    
    print("\n✓ AudioSet tests passed!")
    return True


def test_edge_audio():
    """Test edge audio dataset and dataloaders."""
    print("\n" + "=" * 60)
    print("Testing Edge Audio Dataset")
    print("=" * 60)
    
    # Load config
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_edge_loaders(config)
    
    # Test sizes
    print(f"\n✓ Created dataloaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test batch loading
    train_batch = next(iter(train_loader))
    waveforms, labels, metadata = train_batch
    
    print(f"\n✓ Sample batch:")
    print(f"  Waveform shape: {waveforms.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Metadata length: {len(metadata)}")
    print(f"  Waveform dtype: {waveforms.dtype}")
    print(f"  Waveform range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")
    print(f"  Unique labels: {torch.unique(labels).tolist()}")
    
    # Test metadata
    if metadata and len(metadata) > 0:
        print(f"\n✓ Sample metadata:")
        print(f"  Metadata type: {type(metadata)}")
        print(f"  Metadata length: {len(metadata)}")
        if isinstance(metadata, list):
            print(f"  First item type: {type(metadata[0])}")
            if isinstance(metadata[0], dict):
                print(f"  Fields: {list(metadata[0].keys())}")
                print(f"  Device IDs: {[m['device_id'] for m in metadata[:3]]}")
                print(f"  Timestamps: {[m['timestamp'] for m in metadata[:3]]}")
        elif isinstance(metadata, dict):
            print(f"  Dict keys: {list(metadata.keys())}")
            if 'device_id' in metadata:
                print(f"  Device IDs (first 3): {metadata['device_id'][:3]}")
                print(f"  Timestamps (first 3): {metadata['timestamp'][:3]}")
    
    # Test class mapping
    class_mapping = train_loader.dataset.get_class_mapping()
    print(f"\n✓ Class mapping ({len(class_mapping)} classes):")
    for i, name in sorted(class_mapping.items()):
        print(f"  {i}: {name}")
    
    # Test multiple batches
    print("\n✓ Testing batch iteration:")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        waveforms, labels, metadata = batch
        print(f"  Batch {i}: {waveforms.shape}, "
              f"labels {labels.shape}, "
              f"mean {waveforms.mean():.3f}")
    
    print("\n✓ Edge audio tests passed!")
    return True


def test_data_augmentation():
    """Test data augmentation pipeline."""
    print("\n" + "=" * 60)
    print("Testing Data Augmentation")
    print("=" * 60)
    
    # Load config
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from edge.audio_processing import AudioAugmentor
    
    # Create augmentor with full config structure
    augmentor = AudioAugmentor(config)
    
    # Generate sample audio
    sample_rate = 16000
    duration = 10.0
    waveform = torch.randn(int(sample_rate * duration))
    
    print(f"\n✓ Original waveform:")
    print(f"  Shape: {waveform.shape}")
    print(f"  Range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    print(f"  Mean: {waveform.mean():.3f}")
    print(f"  Std: {waveform.std():.3f}")
    
    # Apply augmentation
    augmented = augmentor.augment_waveform(waveform)
    
    print(f"\n✓ Augmented waveform:")
    print(f"  Shape: {augmented.shape}")
    print(f"  Range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    print(f"  Mean: {augmented.mean():.3f}")
    print(f"  Std: {augmented.std():.3f}")
    
    # Test multiple augmentations
    print("\n✓ Testing augmentation variety:")
    for i in range(5):
        aug = augmentor.augment_waveform(waveform)
        print(f"  Aug {i}: range [{aug.min():.3f}, {aug.max():.3f}], "
              f"mean {aug.mean():.3f}")
    
    print("\n✓ Augmentation tests passed!")
    return True


def main():
    """Run all dataset tests."""
    print("\n" + "=" * 60)
    print("StreamSplit Dataset Tests")
    print("=" * 60)
    
    try:
        # Test AudioSet
        audioset_ok = test_audioset()
        
        # Test edge audio
        edge_ok = test_edge_audio()
        
        # Test augmentation
        aug_ok = test_data_augmentation()
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"AudioSet: {'✓ PASS' if audioset_ok else '✗ FAIL'}")
        print(f"Edge Audio: {'✓ PASS' if edge_ok else '✗ FAIL'}")
        print(f"Augmentation: {'✓ PASS' if aug_ok else '✗ FAIL'}")
        
        if audioset_ok and edge_ok and aug_ok:
            print("\n✓ All dataset tests passed!")
            return 0
        else:
            print("\n✗ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
