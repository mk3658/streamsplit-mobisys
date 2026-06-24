"""
Main training script for StreamSplit.

Drives the Edge Learner (Phase 1, Sec. 4.1: streaming contrastive learning
with GMM-sampled virtual negatives) and periodically refines the embedding
distribution through the Cloud Refiner (Phase 3, Sec. 4.3: Hybrid Loss over
the Temporal Buffer), reporting both losses each epoch.

This single-process script simulates both phases for reproducibility;
buffer embeddings are detached before reaching the Cloud Refiner, so the
reported hybrid loss is a diagnostic of representation quality (Diversity/
Affinity drift) rather than a second optimizer step on the encoder -- see
README "Scope & Known Simplifications".
"""

import argparse
import os

import torch
import torch.optim as optim
import yaml

from datasets import create_audioset_loaders
from edge.audio_processing import AudioAugmentor, AudioProcessor
from edge.contrastive_learning import StreamingContrastiveLearning
from edge.distributional_memory import DistributionalMemory
from edge.resource_monitor import ResourceMonitor
from models import AudioResNet18
from server.refiner import ServerRefiner
from utils.device import get_device, optimize_for_device, print_device_info


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_edge(config):
    """Setup Edge Learner components (Sec. 4.1)."""
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start()

    audio_processor = AudioProcessor(config)
    audio_augmentor = AudioAugmentor(config)

    encoder = AudioResNet18(embedding_dim=config['encoder']['embedding_dim'])

    memory = DistributionalMemory(
        num_components=config['edge']['distributional_memory']['num_components'],
        embedding_dim=config['encoder']['embedding_dim'],
        tau=config['edge']['distributional_memory']['tau'],
        cold_start_frames=config['edge']['distributional_memory']['cold_start_frames'],
        ema_lr=config['edge']['distributional_memory']['ema_lr'],
    )

    contrastive_module = StreamingContrastiveLearning(config, encoder, memory)

    return {
        'resource_monitor': resource_monitor,
        'audio_processor': audio_processor,
        'audio_augmentor': audio_augmentor,
        'memory': memory,
        'encoder': encoder,
        'contrastive_module': contrastive_module,
    }


def setup_server(config):
    """Setup Cloud Refiner components (Sec. 4.3)."""
    return {'refiner': ServerRefiner(config)}


def train_epoch(edge_components, server_components, dataloader,
                 optimizer, device, config, epoch):
    """Train for one epoch, exercising Phase 1 + Phase 3 together."""
    contrastive_module = edge_components['contrastive_module']
    audio_augmentor = edge_components['audio_augmentor']
    refiner = server_components['refiner']
    aug_config = config['edge']['augmentation']
    refine_every = config['training']['server']['refine_every']

    contrastive_module.train()

    total_edge_loss = 0.0
    total_hybrid_loss = 0.0
    num_hybrid_steps = 0
    num_batches = 0

    for batch_idx, (mel_specs, _labels) in enumerate(dataloader):
        mel_specs = mel_specs.to(device)

        mel_specs_aug = audio_augmentor.gaussian_noise(
            mel_specs, aug_config['noise_std']
        )
        mel_specs_aug = audio_augmentor.frequency_mask(
            mel_specs_aug, aug_config['freq_mask_param']
        )

        loss, info = contrastive_module(mel_specs, mel_specs_aug)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_edge_loss += loss.item()
        num_batches += 1

        # Phase 3: push embeddings into the Cloud Refiner's Temporal Buffer
        # and periodically compute the Hybrid Loss (Eq. 13) as a diagnostic.
        embeddings = contrastive_module.compute_embedding(mel_specs)
        for embedding in embeddings:
            refiner.add_embedding(embedding)

        if (batch_idx + 1) % refine_every == 0:
            hybrid = refiner.compute_hybrid_loss(return_components=True)
            if hybrid is not None:
                total_hybrid_loss += hybrid['loss_total'].item()
                num_hybrid_steps += 1

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Edge Loss: {loss.item():.4f}, "
                  f"GMM Entropy: {info['gmm_entropy']:.4f}")

    avg_hybrid_loss = (
        total_hybrid_loss / num_hybrid_steps if num_hybrid_steps else float('nan')
    )
    return total_edge_loss / num_batches, avg_hybrid_loss


def main(args):
    """Main training function."""
    config = load_config(args.config)

    device = get_device(force_cpu=args.force_cpu)
    print_device_info(device)
    optimize_for_device(device)
    config['experiment']['device'] = str(device)

    os.makedirs(config['experiment']['log_dir'], exist_ok=True)
    os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)

    print("Setting up Edge Learner...")
    edge_components = setup_edge(config)

    print("Setting up Cloud Refiner...")
    server_components = setup_server(config)

    edge_components['contrastive_module'].to(device)

    optimizer = optim.Adam(
        edge_components['contrastive_module'].parameters(),
        lr=config['training']['edge']['learning_rate'],
    )

    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_audioset_loaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    num_epochs = config['training']['server']['num_epochs']

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 60}")

        avg_edge_loss, avg_hybrid_loss = train_epoch(
            edge_components, server_components, train_loader,
            optimizer, device, config, epoch
        )

        print(f"Epoch {epoch + 1} - Edge Loss: {avg_edge_loss:.4f}, "
              f"Hybrid Loss: {avg_hybrid_loss:.4f}")

        if (epoch + 1) % config['logging']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['experiment']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': edge_components['encoder'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'edge_loss': avg_edge_loss,
                'hybrid_loss': avg_hybrid_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    edge_components['resource_monitor'].stop()
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train StreamSplit')
    parser.add_argument('--config', type=str,
                         default='configs/streamsplit.yaml',
                         help='Path to configuration file')
    parser.add_argument('--force_cpu', action='store_true',
                         help='Force CPU usage even if GPU is available')

    args = parser.parse_args()
    main(args)
