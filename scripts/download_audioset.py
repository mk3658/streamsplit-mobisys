"""
Download and prepare AudioSet dataset.
Downloads REAL audio from YouTube using yt-dlp or creates synthetic data.
"""

import argparse
import yaml
import sys
import os
import subprocess
import pandas as pd
import urllib.request
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from local datasets module (not HuggingFace datasets)
from datasets.audioset import create_audioset_loaders


def check_yt_dlp():
    """Check if yt-dlp is installed and install if needed."""
    try:
        result = subprocess.run(
            ['yt-dlp', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ yt-dlp version: {result.stdout.strip()}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("⚠️ yt-dlp not found. Installing...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', 'yt-dlp'],
            check=True,
            timeout=60
        )
        print("✅ yt-dlp installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install yt-dlp: {e}")
        return False


def download_youtube_audio(args_tuple):
    """
    Download audio segment from YouTube video using yt-dlp.
    
    Args:
        args_tuple: (video_id, output_path, start_time, duration, index, total)
    
    Returns:
        tuple: (success: bool, video_id: str, filepath: str)
    """
    video_id, output_path, start_time, duration, index, total = args_tuple
    
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        output_template = os.path.join(output_path, f"{video_id}_{int(start_time)}.%(ext)s")
        output_file = os.path.join(output_path, f"{video_id}_{int(start_time)}.wav")
        
        # Skip if already exists
        if os.path.exists(output_file):
            return (True, video_id, output_file)
        
        # yt-dlp command with audio extraction
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio/best',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--postprocessor-args', f'ffmpeg:-ss {start_time} -t {duration} -ar 16000 -ac 1',
            '-o', output_template,
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            '--ignore-errors',
            '--no-check-certificate',
            url
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            text=True
        )
        
        # Check if file was created
        if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
            return (True, video_id, output_file)
        else:
            return (False, video_id, None)
            
    except subprocess.TimeoutExpired:
        return (False, video_id, None)
    except Exception as e:
        return (False, video_id, None)


def download_audioset_metadata(csv_path, subset='balanced'):
    """Download AudioSet metadata CSV if not exists."""
    if os.path.exists(csv_path):
        print(f"✅ Metadata already exists at {csv_path}")
        return True
    
    print(f"Downloading AudioSet {subset} metadata CSV...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # AudioSet CSV URLs
    urls = {
        'balanced': "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv",
        'eval': "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
        'unbalanced': "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
    }
    
    try:
        url = urls.get(subset, urls['balanced'])
        urllib.request.urlretrieve(url, csv_path)
        print(f"✅ Downloaded metadata to {csv_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download metadata: {e}")
        return False


def download_audioset_real(csv_path, output_dir, num_samples=100, num_workers=2, duration=10):
    """
    Download real AudioSet data from YouTube.
    
    Args:
        csv_path: Path to AudioSet CSV metadata
        output_dir: Output directory for audio files
        num_samples: Number of samples to download
        num_workers: Number of parallel download workers
        duration: Audio clip duration in seconds
    
    Returns:
        int: Number of successfully downloaded samples
    """
    print(f"\n{'='*60}")
    print("🌐 DOWNLOADING REAL AUDIOSET FROM YOUTUBE")
    print(f"{'='*60}")
    
    # Read CSV
    print(f"\n📄 Reading AudioSet metadata from {csv_path}...")
    try:
        # AudioSet CSV format: skip first 3 comment lines
        df = pd.read_csv(csv_path, skiprows=3, sep=', ', engine='python')
        df.columns = ['YTID', 'start_seconds', 'end_seconds', 'positive_labels']
        print(f"✅ Found {len(df)} total videos in metadata")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return 0
    
    # Limit samples
    df = df.head(num_samples)
    print(f"📊 Attempting to download {len(df)} samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel download
    download_args = [
        (row['YTID'], output_dir, float(row['start_seconds']), duration, idx, len(df))
        for idx, (_, row) in enumerate(df.iterrows(), 1)
    ]
    
    # Download in parallel with progress bar
    print(f"\n⬇️  Downloading with {num_workers} parallel workers...")
    print("⏳ This may take 5-15 minutes depending on your internet speed...\n")
    
    successful_downloads = []
    failed_downloads = []
    
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(download_args), desc="Downloading", unit="file") as pbar:
            for result in pool.imap_unordered(download_youtube_audio, download_args):
                success, video_id, filepath = result
                if success:
                    successful_downloads.append((video_id, filepath))
                else:
                    failed_downloads.append(video_id)
                pbar.update(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Successfully downloaded: {len(successful_downloads)}/{len(df)} samples")
    print(f"❌ Failed downloads: {len(failed_downloads)}")
    print(f"{'='*60}")
    
    if failed_downloads and len(failed_downloads) <= 10:
        print(f"\n⚠️  Failed video IDs: {', '.join(failed_downloads[:10])}")
    
    return len(successful_downloads)


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare AudioSet dataset (real or synthetic)'
    )
    parser.add_argument('--config', type=str,
                       default='configs/streamsplit.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes to use')
    parser.add_argument('--subset', type=str, default='balanced',
                       choices=['balanced', 'eval', 'unbalanced'],
                       help='AudioSet subset to download')
    parser.add_argument('--use_real', action='store_true',
                       help='Download REAL audio from YouTube (requires yt-dlp)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to download (for real data)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of parallel download workers')
    parser.add_argument('--duration', type=int, default=10,
                       help='Audio clip duration in seconds')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to AudioSet CSV metadata (auto-download if not provided)')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['data']['num_classes'] = args.num_classes
    data_dir = config['data']['data_dir']
    
    print(f"\nDownloading AudioSet ({args.subset} subset)...")
    print(f"Number of classes: {args.num_classes}")
    print(f"Data directory: {data_dir}")
    
    # Handle real AudioSet download
    if args.use_real:
        print("\n🌐 REAL AUDIO DOWNLOAD MODE ENABLED")
        print("="*60)
        
        # Check/install yt-dlp
        if not check_yt_dlp():
            print("❌ Cannot proceed without yt-dlp. Falling back to synthetic data.")
            args.use_real = False
        else:
            # Determine CSV path
            if args.csv_path is None:
                args.csv_path = os.path.join(
                    data_dir, 
                    'audioset', 
                    f'{args.subset}_train_segments.csv'
                )
            
            # Download metadata
            if not download_audioset_metadata(args.csv_path, args.subset):
                print("❌ Cannot download metadata. Falling back to synthetic data.")
                args.use_real = False
            else:
                # Download real audio from YouTube
                audio_dir = os.path.join(data_dir, 'audioset', 'audio')
                num_downloaded = download_audioset_real(
                    args.csv_path,
                    audio_dir,
                    num_samples=args.num_samples,
                    num_workers=args.num_workers,
                    duration=args.duration
                )
                
                if num_downloaded > 0:
                    print(f"\n✅ Successfully downloaded {num_downloaded} real audio samples!")
                    print(f"📁 Audio files saved to: {audio_dir}")
                else:
                    print("\n❌ No samples downloaded. Falling back to synthetic data.")
                    args.use_real = False
    
    if not args.use_real:
        print("\n🎭 Using synthetic AudioSet data for demonstration...")
    
    # Create dataloaders (this will trigger download/creation)
    print("\nCreating datasets...")
    train_loader, val_loader, test_loader = create_audioset_loaders(config)
    
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    total = (len(train_loader.dataset) + 
             len(val_loader.dataset) + 
             len(test_loader.dataset))
    print(f"Total samples: {total}")
    
    # Show class mapping
    class_mapping = train_loader.dataset.get_class_mapping()
    print(f"\nClass Mapping:")
    for class_id, class_name in sorted(class_mapping.items(), 
                                      key=lambda x: int(x[0])):
        print(f"  {class_id}: {class_name}")
    
    # Test loading a batch
    print(f"\nTesting data loading...")
    waveforms, labels = next(iter(train_loader))
    print(f"  Batch shape: {waveforms.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Waveform range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")
    
    print(f"\n{'='*60}")
    print("✓ AudioSet preparation completed successfully!")
    print(f"{'='*60}")
    
    if args.use_real:
        print("\n✅ Using REAL AudioSet data from YouTube!")
    else:
        print("\n🎭 Using synthetic data for demonstration.")
        print("To download real data, run with --use_real flag:")
        print("  python scripts/download_audioset.py --use_real --num_samples 100")


if __name__ == '__main__':
    main()
