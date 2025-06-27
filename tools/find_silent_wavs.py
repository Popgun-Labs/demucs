#!/usr/bin/env python3
"""
Script to recursively find silent WAV files in a directory.
A file is considered silent if its maximum absolute amplitude is below a threshold.
"""

import argparse
import librosa
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm


def is_silent(wav_path, threshold=1e-4, duration_check=None):
    """
    Check if a WAV file is silent.
    
    Args:
        wav_path: Path to the WAV file
        threshold: Amplitude threshold below which file is considered silent
        duration_check: If provided, check only first N seconds of audio
    
    Returns:
        tuple: (is_silent: bool, max_amplitude: float, duration: float)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(wav_path, sr=None)
        
        # If checking only part of the file
        if duration_check is not None:
            max_samples = int(duration_check * sr)
            audio = audio[:max_samples]
        
        # Calculate maximum absolute amplitude
        max_amp = np.max(np.abs(audio))
        duration = len(audio) / sr
        
        return max_amp < threshold, max_amp, duration
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}", file=sys.stderr)
        return False, 0.0, 0.0


def find_silent_wavs(directory, threshold=1e-4, duration_check=None, show_progress=True):
    """
    Recursively find silent WAV files in a directory.
    
    Args:
        directory: Root directory to search
        threshold: Amplitude threshold for silence detection
        duration_check: If provided, check only first N seconds of each file
        show_progress: Whether to show progress bar
    
    Returns:
        list: List of tuples (filepath, max_amplitude, duration)
    """
    directory = Path(directory)
    
    # Find all WAV files recursively
    wav_files = list(directory.rglob("*.wav"))
    wav_files.extend(list(directory.rglob("*.WAV")))  # Include uppercase
    
    if not wav_files:
        print(f"No WAV files found in {directory}")
        return []
    
    print(f"Found {len(wav_files)} WAV files to check...")
    
    silent_files = []
    
    # Process files with optional progress bar
    iterator = tqdm(wav_files, desc="Checking files") if show_progress else wav_files
    
    for wav_path in iterator:
        is_silent_file, max_amp, duration = is_silent(wav_path, threshold, duration_check)
        
        if is_silent_file:
            silent_files.append((str(wav_path), max_amp, duration))
    
    return silent_files


def main():
    parser = argparse.ArgumentParser(
        description="Recursively find silent WAV files in a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "directory",
        help="Directory to search recursively"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=1e-4,
        help="Amplitude threshold below which file is considered silent"
    )
    
    parser.add_argument(
        "-d", "--duration-check",
        type=float,
        help="Check only first N seconds of each file (useful for large files)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Don't show progress bar"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show additional information about each silent file"
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not Path(args.directory).exists():
        print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Find silent files
    silent_files = find_silent_wavs(
        args.directory,
        threshold=args.threshold,
        duration_check=args.duration_check,
        show_progress=not args.no_progress
    )
    
    # Print results
    if silent_files:
        print(f"\n🔇 Found {len(silent_files)} silent WAV files:")
        print("=" * 60)
        
        for filepath, max_amp, duration in silent_files:
            if args.verbose:
                print(f"{filepath}")
                print(f"  Max amplitude: {max_amp:.2e}")
                print(f"  Duration: {duration:.2f}s")
                print()
            else:
                print(filepath)
    else:
        print(f"\n✅ No silent WAV files found (threshold: {args.threshold:.2e})")


if __name__ == "__main__":
    main() 