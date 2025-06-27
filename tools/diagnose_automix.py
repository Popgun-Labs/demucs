#!/usr/bin/env python3
"""
Diagnostic script to analyze automix failures and dataset compatibility.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
sys.path.append('.')

from demucs.pretrained import SOURCES

# Import constants directly
CACHE = Path.home() / '/Volumes/SAMPLES/datasets/tmp/automix_cache'
MAX_TEMPO = 0.15
MAX_PITCH = 3


def analyze_dataset_compatibility(dataset_path):
    """Analyze the dataset for automix compatibility."""
    print(f"🔍 Analyzing dataset: {dataset_path}")
    print(f"Expected sources: {SOURCES}")
    
    # Check if directories exist for each expected source
    root = Path(dataset_path) 
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Sample some tracks to check structure
    sample_tracks = list(root.iterdir())[:5]
    print(f"\n📁 Sample tracks found: {len(sample_tracks)}")
    
    for track_dir in sample_tracks:
        if track_dir.is_dir():
            print(f"\nTrack: {track_dir.name}")
            available_stems = []
            for source in SOURCES:
                stem_file = track_dir / f"{source}.wav"
                if stem_file.exists():
                    available_stems.append(source)
                    print(f"  ✅ {source}.wav")
                else:
                    print(f"  ❌ {source}.wav (missing)")
            
            if len(available_stems) < len(SOURCES):
                print(f"  ⚠️  Only {len(available_stems)}/{len(SOURCES)} stems available")


def analyze_tempo_pitch_distribution():
    """Analyze tempo and pitch distributions from cache."""
    cache_path = CACHE
    if not cache_path.exists():
        print(f"❌ Cache directory not found: {cache_path}")
        return
    
    print(f"🔍 Analyzing cached analysis data from: {cache_path}")
    
    tempos = []
    pitch_data = []
    successful_tracks = 0
    failed_tracks = 0
    
    # Find all cache files
    cache_files = list(cache_path.rglob("*.pkl"))
    
    if not cache_files:
        print("❌ No cache files found. Run automix first to generate analysis data.")
        return
    
    print(f"Found {len(cache_files)} cache files")
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                
            if len(data) == 5:  # New format
                tempo, events, kr_bass, kr_chordal, kr_lead = data
                if tempo is not None:
                    tempos.append(tempo)
                    successful_tracks += 1
                    
                    # Check if any pitch data is available
                    if kr_bass is not None or kr_chordal is not None or kr_lead is not None:
                        pitch_data.append(1)
                    else:
                        pitch_data.append(0)
                else:
                    failed_tracks += 1
            else:
                failed_tracks += 1
                
        except Exception as e:
            print(f"Error reading {cache_file}: {e}")
            failed_tracks += 1
    
    print(f"\n📊 Analysis Results:")
    print(f"Successful tracks: {successful_tracks}")
    print(f"Failed tracks: {failed_tracks}")
    
    if tempos:
        tempos = np.array(tempos)
        print(f"\n🎵 Tempo Analysis:")
        print(f"Mean tempo: {tempos.mean():.1f} BPM")
        print(f"Tempo range: {tempos.min():.1f} - {tempos.max():.1f} BPM")
        print(f"Tempo std: {tempos.std():.1f} BPM")
        
        # Analyze tempo compatibility
        tempo_pairs = []
        for i in range(len(tempos)):
            for j in range(i+1, len(tempos)):
                for scale in [1/4, 1/2, 1, 2, 4]:
                    tempo_scaled = tempos[j] * scale
                    delta = abs(tempos[i] / tempo_scaled - 1)
                    tempo_pairs.append(delta)
        
        tempo_pairs = np.array(tempo_pairs)
        compatible_pairs = (tempo_pairs < MAX_TEMPO).sum()
        total_pairs = len(tempo_pairs)
        compatibility_rate = compatible_pairs / total_pairs * 100
        
        print(f"\n🔗 Tempo Compatibility:")
        print(f"Compatible pairs: {compatible_pairs}/{total_pairs} ({compatibility_rate:.1f}%)")
        print(f"Max allowed tempo delta: {MAX_TEMPO*100:.0f}%")
        
        # Show histogram
        print(f"\n📈 Tempo Distribution:")
        hist, bins = np.histogram(tempos, bins=10)
        for i, (count, start, end) in enumerate(zip(hist, bins[:-1], bins[1:])):
            bar = '█' * int(count * 40 / hist.max())
            print(f"{start:6.1f}-{end:5.1f}: {bar} ({count})")
    
    if pitch_data:
        pitch_available = np.array(pitch_data).mean() * 100
        print(f"\n🎼 Pitch Analysis:")
        print(f"Tracks with pitch data: {pitch_available:.1f}%")


def suggest_improvements():
    """Suggest improvements based on analysis."""
    print(f"\n💡 Suggestions to improve automix success rate:")
    print(f"")
    print(f"1. **Increase tolerance thresholds:**")
    print(f"   - Current MAX_TEMPO: {MAX_TEMPO*100:.0f}% -> try 25-30%")
    print(f"   - Current MAX_PITCH: {MAX_PITCH} semitones -> try 4-5 semitones")
    print(f"")
    print(f"2. **Dataset diversity:**")
    print(f"   - Include more tracks with similar tempos")
    print(f"   - Ensure all required stems are present")
    print(f"   - Check for silent or very quiet stems")
    print(f"")
    print(f"3. **Fallback strategies:**")
    print(f"   - Allow mixing without pitch matching for some stems")
    print(f"   - Use drum-only tempo matching for rhythmic tracks")
    print(f"   - Implement better tempo scaling detection")


def create_compatibility_matrix():
    """Create a compatibility matrix for all tracks."""
    cache_path = CACHE
    cache_files = list(cache_path.rglob("*.pkl"))
    
    if len(cache_files) < 2:
        print("❌ Need at least 2 tracks for compatibility matrix")
        return
    
    # Load all tempo data
    track_data = []
    track_names = []
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            if len(data) == 5:
                tempo, events, kr_bass, kr_chordal, kr_lead = data
                if tempo is not None:
                    track_data.append({
                        'tempo': tempo,
                        'kr_bass': kr_bass,
                        'kr_chordal': kr_chordal,
                        'kr_lead': kr_lead
                    })
                    track_names.append(cache_file.stem)
        except:
            continue
    
    if len(track_data) < 2:
        print("❌ Not enough valid tracks for compatibility matrix")
        return
    
    n_tracks = len(track_data)
    tempo_compat = np.zeros((n_tracks, n_tracks))
    pitch_compat = np.zeros((n_tracks, n_tracks))
    
    print(f"📊 Creating compatibility matrix for {n_tracks} tracks...")
    
    for i in range(n_tracks):
        for j in range(n_tracks):
            if i == j:
                tempo_compat[i, j] = 1.0
                pitch_compat[i, j] = 1.0
                continue
            
            # Check tempo compatibility
            tempo_ok = False
            for scale in [1/4, 1/2, 1, 2, 4]:
                tempo_scaled = track_data[j]['tempo'] * scale
                delta = abs(track_data[i]['tempo'] / tempo_scaled - 1)
                if delta < MAX_TEMPO:
                    tempo_ok = True
                    break
            tempo_compat[i, j] = 1.0 if tempo_ok else 0.0
            
            # Check pitch compatibility (simplified)
            pitch_ok = False
            if (track_data[i]['kr_bass'] is not None and 
                track_data[j]['kr_bass'] is not None):
                # Simplified pitch check - would need full best_pitch_shift logic
                pitch_ok = True  # Placeholder
            pitch_compat[i, j] = 1.0 if pitch_ok else 0.0
    
    print(f"\n🎵 Tempo Compatibility Matrix:")
    overall_tempo_compat = tempo_compat.mean() * 100
    print(f"Overall tempo compatibility: {overall_tempo_compat:.1f}%")
    
    # Show which tracks are most/least compatible
    tempo_scores = tempo_compat.sum(axis=1)
    best_track = np.argmax(tempo_scores)
    worst_track = np.argmin(tempo_scores)
    
    print(f"Most compatible track: {track_names[best_track]} ({tempo_scores[best_track]}/{n_tracks})")
    print(f"Least compatible track: {track_names[worst_track]} ({tempo_scores[worst_track]}/{n_tracks})")


def main():
    parser = argparse.ArgumentParser(description="Diagnose automix failures")
    parser.add_argument("command", choices=['dataset', 'analysis', 'suggestions', 'matrix'], 
                       help="What to analyze")
    parser.add_argument("--dataset-path", default='/Volumes/SAMPLES/datasets/stem_separation_tests',
                       help="Path to dataset")
    
    args = parser.parse_args()
    
    if args.command == 'dataset':
        analyze_dataset_compatibility(args.dataset_path)
    elif args.command == 'analysis':
        analyze_tempo_pitch_distribution()
    elif args.command == 'suggestions':
        suggest_improvements()
    elif args.command == 'matrix':
        create_compatibility_matrix()


if __name__ == '__main__':
    main() 