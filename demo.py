"""
Quick demo/example of StreamSplit components.
Run this to verify installation and basic functionality.
"""

import time
import traceback

import torch
import yaml

from edge import (
    AudioAugmentor,
    AudioProcessor,
    DistributionalMemory,
    ResourceMonitor,
    SplitController,
    StreamingContrastiveLearning,
)
from models import AudioResNet18
from server import ServerRefiner


def load_config():
    with open('configs/streamsplit.yaml', 'r') as f:
        return yaml.safe_load(f)


def demo_audio_processing():
    """Demo mel-spectrogram extraction and augmentation (Sec. 4.1.3)."""
    print("\n=== Audio Processing Demo ===")
    config = load_config()

    processor = AudioProcessor(config)
    augmentor = AudioAugmentor(config)

    sample_rate = config['data']['sample_rate']
    duration = config['data']['audio_duration']
    waveform = torch.randn(1, sample_rate * duration)

    mel_spec = processor.process_audio(waveform)
    print(f"Input waveform shape: {waveform.shape}")
    print(f"Mel-spectrogram shape: {mel_spec.shape}")

    mel_spec_aug = augmentor.gaussian_noise(mel_spec, std=0.01)
    mel_spec_aug = augmentor.frequency_mask(mel_spec_aug, mask_param=10)
    print(f"Augmented mel-spectrogram shape: {mel_spec_aug.shape}")

    print("Audio processing works!")


def demo_model():
    """Demo the ResNet-18-1D encoder and split-point execution (Sec. 5)."""
    print("\n=== Model Demo ===")
    config = load_config()

    model = AudioResNet18(embedding_dim=config['encoder']['embedding_dim'])
    num_blocks = config['encoder']['num_blocks']

    batch_size, n_mels, time_frames = 4, 128, 100
    x = torch.randn(batch_size, 1, n_mels, time_frames)

    embeddings = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Full-forward embeddings shape: {embeddings.shape}")

    # Split execution: edge runs blocks[:k], server completes blocks[k:].
    k = num_blocks // 2
    features = model.forward_up_to(x, k)
    embeddings_split = model.forward_from(features, k)
    print(f"Split at k={k}: intermediate shape {features.shape}, "
          f"final embeddings shape {embeddings_split.shape}")

    print("Model works!")


def demo_distributional_memory():
    """Demo the GMM distributional memory (Eq. 7-9, 11)."""
    print("\n=== Distributional Memory Demo ===")
    config = load_config()
    dm_config = config['edge']['distributional_memory']

    memory = DistributionalMemory(
        num_components=dm_config['num_components'],
        embedding_dim=config['encoder']['embedding_dim'],
        tau=dm_config['tau'],
        cold_start_frames=dm_config['cold_start_frames'],
    )

    embeddings = torch.nn.functional.normalize(
        torch.randn(100, config['encoder']['embedding_dim']), dim=1
    )
    for emb in embeddings:
        memory.update(emb.unsqueeze(0))

    print(f"GMM storage footprint: {memory.storage_bytes()} bytes "
          f"(paper: ~33KB for C=64, d=128)")
    print(f"Warm after cold start: {memory.is_warm}")

    anchor = embeddings[0]
    negatives = memory.sample_virtual_negatives(anchor, n_syn=256)
    print(f"Sampled virtual negatives shape: {negatives.shape}")

    entropy = memory.entropy(anchor.unsqueeze(0))
    print(f"Embedding uncertainty U_t: {entropy.item():.4f}")

    print("Distributional memory works!")


def demo_contrastive_learning():
    """Demo streaming InfoNCE with virtual negatives (Eq. 10)."""
    print("\n=== Contrastive Learning Demo ===")
    config = load_config()

    encoder = AudioResNet18(embedding_dim=config['encoder']['embedding_dim'])
    memory = DistributionalMemory(
        num_components=config['edge']['distributional_memory']['num_components'],
        embedding_dim=config['encoder']['embedding_dim'],
        cold_start_frames=2,
    )
    module = StreamingContrastiveLearning(config, encoder, memory)

    x = torch.randn(2, 1, 128, 100)
    x_aug = x + 0.01 * torch.randn_like(x)

    loss, info = module(x, x_aug)
    print(f"Edge contrastive loss: {loss.item():.4f}")
    print(f"Info: {info}")

    print("Contrastive learning works!")


def demo_hybrid_loss():
    """Demo the Cloud Refiner's Temporal Buffer + Hybrid Loss (Sec. 4.3)."""
    print("\n=== Cloud Refiner / Hybrid Loss Demo ===")
    config = load_config()

    refiner = ServerRefiner(config)
    embeddings = torch.nn.functional.normalize(
        torch.randn(50, config['encoder']['embedding_dim']), dim=1
    )
    for emb in embeddings:
        refiner.add_embedding(emb)

    result = refiner.compute_hybrid_loss(return_components=True)
    print(f"Total loss: {result['loss_total']:.4f}")
    print(f"SW loss: {result['loss_sw']:.4f}")
    print(f"Laplacian loss: {result['loss_lap']:.4f}")

    print("Hybrid loss works!")


def demo_resource_monitor():
    """Demo CPU + bandwidth EMA monitoring (Table 8 state: R_cpu, B_net)."""
    print("\n=== Resource Monitor Demo ===")
    config = load_config()

    monitor = ResourceMonitor(config)
    monitor.start()
    time.sleep(0.5)

    monitor.record_transmission(num_bytes=64_000, duration_s=0.05)

    state = monitor.get_state()
    print(f"CPU utilization (R_cpu): {state['cpu_util'] * 100:.1f}%")
    print(f"Bandwidth (B_net): {state['bandwidth_mbps']:.2f} Mbps")

    monitor.stop()
    print("Resource monitor works!")


def demo_control_plane():
    """Demo the Control Plane's runtime split decision (Table 8)."""
    print("\n=== Control Plane Demo ===")
    config = load_config()

    controller = SplitController(
        config, num_blocks=config['encoder']['num_blocks']
    )
    split_layer = controller.get_split_layer(
        uncertainty=0.6, cpu_util=0.4, bandwidth_mbps=15.0
    )
    print(f"Split layer decision: k={split_layer} "
          f"(of L={config['encoder']['num_blocks']})")

    print("Control plane works!")


def main():
    """Run all demos."""
    print("=" * 60)
    print("StreamSplit Demo - Verifying Installation")
    print("=" * 60)

    try:
        demo_audio_processing()
        demo_model()
        demo_distributional_memory()
        demo_contrastive_learning()
        demo_hybrid_loss()
        demo_resource_monitor()
        demo_control_plane()

        print("\n" + "=" * 60)
        print("All demos passed! Installation is working correctly.")
        print("=" * 60)

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
