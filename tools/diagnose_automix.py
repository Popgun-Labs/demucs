#!/usr/bin/env python3
"""
Diagnostic script to analyze automix failures and dataset compatibility.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import torchaudio
import torch
from torch.nn import functional as F
import traceback
import tqdm
import shutil
import json
import scipy.stats
from scipy import signal

# Add librosa imports for tempo/pitch analysis
try:
    from librosa.beat import beat_track
    from librosa.feature import chroma_cqt
    LIBROSA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: librosa not available: {e}")
    LIBROSA_AVAILABLE = False

import sys
sys.path.append('.')

from demucs.pretrained import SOURCES
# from demucs.wav import build_metadata
# from demucs.audio import save_audio
# import hashlib

# Import constants directly
CACHE = Path.home() / '/Volumes/SAMPLES/datasets/tmp/automix_cache'
MAX_TEMPO = 0.15
MAX_PITCH = 3
SR = 44100
CHANNELS = 2


def rms(wav, window=10000):
    """efficient rms computed for each time step over a given window."""
    half = window // 2
    window = 2 * half + 1
    wav = F.pad(wav, (half, half))
    tot = wav.pow(2).cumsum(dim=-1)
    return ((tot[..., window - 1:] - tot[..., :-window + 1]) / window).sqrt()


def best_pitch_shift(kr_a, kr_b):
    """find the best pitch shift between two chroma distributions."""
    deltas = []
    for p in range(12):
        deltas.append((kr_a - kr_b).abs().mean())
        kr_b = kr_b.roll(1, 0)

    ps = np.argmin(deltas)
    if ps > 6:
        ps = ps - 12
    return ps


def spectral_flatness(waveform, sr=44100, hop_length=512, n_fft=2048):
    """Calculate spectral flatness (Wiener entropy) to detect noise-like signals."""
    # Convert to numpy for FFT operations
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    # Compute STFT
    stft = np.abs(signal.stft(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)[2])
    
    # Avoid division by zero
    stft = np.maximum(stft, 1e-10)
    
    # Calculate spectral flatness for each frame
    geometric_mean = np.exp(np.mean(np.log(stft), axis=0))
    arithmetic_mean = np.mean(stft, axis=0)
    
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    
    return np.mean(flatness)  # Average across all frames


def detect_noise_type(waveform, sr=44100, confidence_threshold=0.8):
    """Detect if audio contains noise and classify the type."""
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.numpy()
    else:
        waveform_np = waveform
    
    # Handle stereo by taking mean across channels
    if len(waveform_np.shape) > 1:
        waveform_np = np.mean(waveform_np, axis=0)
    
    # Skip if too short
    if len(waveform_np) < sr // 10:  # Less than 0.1 seconds
        return {"is_noise": False, "confidence": 0.0, "type": "too_short"}
    
    # Calculate various noise indicators
    noise_indicators = {}
    
    # 1. Spectral flatness (0-1, higher = more noise-like)
    flatness = spectral_flatness(waveform_np, sr)
    noise_indicators['spectral_flatness'] = flatness
    
    # 2. Statistical properties
    # Check if amplitude distribution is approximately Gaussian (noise-like)
    try:
        _, p_value = scipy.stats.normaltest(waveform_np)
        noise_indicators['gaussian_p_value'] = p_value
    except Exception:
        noise_indicators['gaussian_p_value'] = 0.0
    
    # 3. Autocorrelation analysis (noise should have low autocorrelation)
    autocorr = np.correlate(waveform_np, waveform_np, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Check autocorrelation at lag 1 (should be low for white noise)
    autocorr_lag1 = abs(autocorr[1]) if len(autocorr) > 1 else 1.0
    noise_indicators['autocorr_lag1'] = autocorr_lag1
    
    # 4. Frequency domain analysis
    fft = np.fft.rfft(waveform_np)
    power_spectrum = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(waveform_np), 1/sr)
    
    # Skip DC component
    power_spectrum = power_spectrum[1:]
    freqs = freqs[1:]
    
    if len(power_spectrum) > 10:  # Need enough frequency bins
        # Log-log slope analysis for noise type classification
        log_freqs = np.log10(freqs + 1e-10)
        log_power = np.log10(power_spectrum + 1e-10)
        
        # Fit line to log-log plot
        try:
            slope, intercept, r_value, _, _ = scipy.stats.linregress(log_freqs, log_power)
            noise_indicators['spectral_slope'] = slope
            noise_indicators['spectral_slope_r2'] = r_value ** 2
        except Exception:
            noise_indicators['spectral_slope'] = 0.0
            noise_indicators['spectral_slope_r2'] = 0.0
        
        # Spectral centroid variability (should be low for noise)
        # Calculate spectral centroid in frames
        frame_size = sr // 20  # 50ms frames
        centroids = []
        for i in range(0, len(waveform_np) - frame_size, frame_size // 2):
            frame = waveform_np[i:i + frame_size]
            frame_fft = np.fft.rfft(frame)
            frame_power = np.abs(frame_fft) ** 2
            frame_freqs = np.fft.rfftfreq(len(frame), 1/sr)
            
            # Calculate weighted centroid
            if np.sum(frame_power) > 0:
                centroid = np.sum(frame_freqs * frame_power) / np.sum(frame_power)
                centroids.append(centroid)
        
        if len(centroids) > 1:
            centroid_std = np.std(centroids)
            centroid_mean = np.mean(centroids)
            noise_indicators['centroid_variability'] = centroid_std / (centroid_mean + 1e-10)
        else:
            noise_indicators['centroid_variability'] = 0.0
    else:
        noise_indicators['spectral_slope'] = 0.0
        noise_indicators['spectral_slope_r2'] = 0.0
        noise_indicators['centroid_variability'] = 0.0
    
    # 5. Zero crossing rate analysis
    zero_crossings = np.where(np.diff(np.sign(waveform_np)))[0]
    zcr = len(zero_crossings) / len(waveform_np) * sr
    noise_indicators['zero_crossing_rate'] = zcr
    
    # Classification logic
    is_noise = False
    confidence = 0.0
    noise_type = "musical"
    
    # Thresholds for noise detection (empirically determined)
    flatness_threshold = 0.8  # Higher = more noise-like
    autocorr_threshold = 0.1  # Lower = more noise-like
    gaussian_threshold = 0.05  # Higher p-value = more Gaussian/noise-like
    
    # Count noise indicators
    noise_score = 0
    max_score = 4
    
    if flatness > flatness_threshold:
        noise_score += 1
    if autocorr_lag1 < autocorr_threshold:
        noise_score += 1
    if noise_indicators['gaussian_p_value'] > gaussian_threshold:
        noise_score += 1
    if noise_indicators['centroid_variability'] < 0.5:  # Low variability suggests noise
        noise_score += 1
    
    confidence = noise_score / max_score
    
    if confidence >= confidence_threshold:  # Use configurable threshold
        is_noise = True
        
        # Classify noise type based on spectral slope
        slope = noise_indicators['spectral_slope']
        slope_r2 = noise_indicators['spectral_slope_r2']
        
        if slope_r2 > 0.7:  # Good linear fit in log-log domain
            if -0.5 <= slope <= 0.5:
                noise_type = "white_noise"
            elif -2.0 <= slope < -0.5:
                noise_type = "pink_noise"
            elif slope < -2.0:
                noise_type = "brown_noise"
            elif slope > 0.5:
                noise_type = "blue_noise"
            else:
                noise_type = "colored_noise"
        else:
            # Poor linear fit, might be more complex noise
            if flatness > 0.9:
                noise_type = "white_noise"  # Very flat spectrum
            else:
                noise_type = "colored_noise"
    
    return {
        "is_noise": is_noise,
        "confidence": confidence,
        "type": noise_type,
        "indicators": noise_indicators
    }


def analyze_audio_chunk_content(chunk, sr=44100, rms_threshold=1e-3, confidence_threshold=0.8):
    """Analyze a chunk of audio to determine if it's silent, noise, or musical content."""
    # Calculate RMS
    chunk_rms = torch.sqrt(torch.mean(chunk**2)).item()
    
    if chunk_rms < rms_threshold:
        return {
            "type": "silent",
            "rms": chunk_rms,
            "noise_analysis": None
        }
    
    # Analyze for noise if above threshold
    noise_result = detect_noise_type(chunk, sr, confidence_threshold)
    
    if noise_result["is_noise"]:
        content_type = f"noise_{noise_result['type']}"
    else:
        content_type = "musical"
    
    return {
        "type": content_type,
        "rms": chunk_rms,
        "noise_analysis": noise_result
    }


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-JSON-serializable types to Python native types."""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def analyse_track_tempo_pitch(track_data, track_name="Unknown"):
    """Analyze track tempo and pitch information using EXACT automix.py methods."""
    try:
        if isinstance(track_data, dict):
            # Convert dict to tensor if needed
            track_tensor = torch.stack([track_data[source] for source in SOURCES])
        else:
            track_tensor = track_data
            
        # EXACT copy of automix.py logic
        mix = track_tensor.sum(0).mean(0)
        ref = mix.std()

        starts = (abs(mix) >= 1e-2 * ref).float().argmax().item()
        track_tensor = track_tensor[..., starts:]

        # Find drums track (should be first in SOURCES list)
        drums_idx = SOURCES.index("drums") if "drums" in SOURCES else 0
        drums = track_tensor[drums_idx].mean(0)
        
        tempo = None
        events = None
        if drums.std() > 1e-2 * ref:
            try:
                tempo, events = beat_track(y=drums.numpy(), units='time', sr=SR)
            except Exception as e:
                print(f"failed drums tempo analysis for {track_name}: {e}")
                return None

        # Analyze pitch-related tracks separately: bass, chordal, and lead
        kr_bass = kr_chordal = kr_lead = None
        
        # Analyze bass - EXACT copy from automix.py
        if "bass" in SOURCES:
            bass_idx = SOURCES.index("bass")
            bass = track_tensor[bass_idx].mean(0)
            r = rms(bass)
            peak = r.max()
            mask = r >= 0.05 * peak
            bass = bass[mask]
            if bass.std() > 1e-2 * ref:
                try:
                    kr = torch.from_numpy(chroma_cqt(y=bass.numpy(), sr=SR))
                    kr_bass = (kr.max(dim=0, keepdim=True)[0] == kr).float().mean(1)
                except Exception as e:
                    print(f"Failed bass pitch analysis for {track_name}: {e}")
        
        # Analyze chordal - EXACT copy from automix.py  
        if "chordal" in SOURCES:
            chordal_idx = SOURCES.index("chordal")
            chordal = track_tensor[chordal_idx].mean(0)
            r = rms(chordal)
            peak = r.max()
            mask = r >= 0.05 * peak
            chordal = chordal[mask]
            if chordal.std() > 1e-2 * ref:
                try:
                    kr = torch.from_numpy(chroma_cqt(y=chordal.numpy(), sr=SR))
                    kr_chordal = (kr.max(dim=0, keepdim=True)[0] == kr).float().mean(1)
                except Exception as e:
                    print(f"Failed chordal pitch analysis for {track_name}: {e}")
        
        # Analyze lead - EXACT copy from automix.py
        if "lead" in SOURCES:
            lead_idx = SOURCES.index("lead")
            lead = track_tensor[lead_idx].mean(0)
            r = rms(lead)
            peak = r.max()
            mask = r >= 0.05 * peak
            lead = lead[mask]
            if lead.std() > 1e-2 * ref:
                try:
                    kr = torch.from_numpy(chroma_cqt(y=lead.numpy(), sr=SR))
                    kr_lead = (kr.max(dim=0, keepdim=True)[0] == kr).float().mean(1)
                except Exception as e:
                    print(f"Failed lead pitch analysis for {track_name}: {e}")
        
        # Fallback to bass if no stems were analyzable - EXACT copy from automix.py
        if kr_bass is None and kr_chordal is None and kr_lead is None:
            bass_idx = SOURCES.index("bass") if "bass" in SOURCES else 1
            bass = track_tensor[bass_idx].mean(0)
            r = rms(bass)
            peak = r.max()
            mask = r >= 0.05 * peak
            bass = bass[mask]
            if bass.std() > 1e-2 * ref:
                try:
                    kr = torch.from_numpy(chroma_cqt(y=bass.numpy(), sr=SR))
                    kr_bass = (kr.max(dim=0, keepdim=True)[0] == kr).float().mean(1)
                except Exception as e:
                    print(f"Failed fallback bass pitch analysis for {track_name}: {e}")
            else:
                print(f"failed pitch analysis on all stems for {track_name}")
                return None

        return {
            'tempo': tempo,
            'events': events,
            'kr_bass': kr_bass,
            'kr_chordal': kr_chordal,
            'kr_lead': kr_lead
        }
    except Exception as e:
        print(f"Failed to analyze {track_name}: {e}")
        return None


def load_track_data(track_path, sources=None):
    """Load track data from directory."""
    if sources is None:
        sources = SOURCES
    
    track_data = {}
    for source in sources:
        stem_path = track_path / f"{source}.wav"
        if stem_path.exists():
            try:
                audio, sr = torchaudio.load(str(stem_path))
                if sr != SR:
                    audio = torchaudio.functional.resample(audio, sr, SR)
                track_data[source] = audio
            except Exception as e:
                print(f"Failed to load {stem_path}: {e}")
                return None
        else:
            print(f"Missing stem: {stem_path}")
            return None
    
    return track_data


def check_tempo_pitch_compatibility(dataset_path, sources=None):
    """Comprehensive tempo and pitch compatibility analysis."""
    if not LIBROSA_AVAILABLE:
        print("❌ Cannot perform tempo/pitch analysis: librosa not available")
        print("This is likely due to scipy compatibility issues.")
        print("Try updating librosa and scipy, or running automix.py directly to see the same errors.")
        return
    
    if sources is None:
        sources = SOURCES
    
    dataset_path = Path(dataset_path)
    
    print(f"🎵 Tempo and Pitch Compatibility Analysis")
    print(f"Dataset: {dataset_path}")
    print(f"Current constraints: MAX_TEMPO={MAX_TEMPO:.1%}, MAX_PITCH={MAX_PITCH} semitones")
    print("=" * 60)
    
    # Find all tracks
    track_dirs = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            # Check if it has all required stems
            has_all_stems = all((item / f"{source}.wav").exists() for source in sources)
            if has_all_stems:
                track_dirs.append(item)
    
    if not track_dirs:
        print("❌ No complete tracks found!")
        return
    
    print(f"📁 Found {len(track_dirs)} complete tracks")
    
    # Analyze all tracks
    track_analyses = {}
    successful_analyses = 0
    failed_analyses = 0
    
    print("\n🔍 Analyzing tracks...")
    for track_dir in tqdm.tqdm(track_dirs, desc="Analyzing tracks"):
        track_name = track_dir.name
        
        # Load track data
        track_data = load_track_data(track_dir, sources)
        if track_data is None:
            failed_analyses += 1
            continue
        
        # Analyze tempo and pitch
        analysis = analyse_track_tempo_pitch(track_data, track_name)
        if analysis is None or analysis['tempo'] is None:
            failed_analyses += 1
            continue
        
        track_analyses[track_name] = analysis
        successful_analyses += 1
    
    print(f"\n📊 Analysis Results:")
    print(f"Successfully analyzed: {successful_analyses}")
    print(f"Failed to analyze: {failed_analyses}")
    
    if not track_analyses:
        print("❌ No tracks could be analyzed!")
        return
    
    # Save tempo data to JSON file for inspection
    tempo_data = {}
    pitch_data = {}
    
    for track_name, analysis in track_analyses.items():
        tempo_data[track_name] = {
            "tempo_bpm": float(analysis['tempo']) if analysis['tempo'] is not None else None,
            "has_kr_bass": analysis['kr_bass'] is not None,
            "has_kr_chordal": analysis['kr_chordal'] is not None,
            "has_kr_lead": analysis['kr_lead'] is not None,
        }
        
        # Add pitch fingerprint info if available
        pitch_info = {}
        if analysis['kr_bass'] is not None:
            pitch_info['bass_fingerprint'] = analysis['kr_bass'].tolist()
        if analysis['kr_chordal'] is not None:
            pitch_info['chordal_fingerprint'] = analysis['kr_chordal'].tolist()
        if analysis['kr_lead'] is not None:
            pitch_info['lead_fingerprint'] = analysis['kr_lead'].tolist()
        
        pitch_data[track_name] = pitch_info
    
    # Save to JSON files
    output_dir = dataset_path / "tempo_analysis"
    output_dir.mkdir(exist_ok=True)
    
    tempo_file = output_dir / "tempo_data.json"
    pitch_file = output_dir / "pitch_data.json"
    
    with open(tempo_file, 'w') as f:
        json.dump(tempo_data, f, indent=2, sort_keys=True)
    
    with open(pitch_file, 'w') as f:
        json.dump(pitch_data, f, indent=2, sort_keys=True)
    
    print(f"💾 Saved tempo data to: {tempo_file}")
    print(f"💾 Saved pitch data to: {pitch_file}")
    
    # Tempo distribution analysis
    tempos = [analysis['tempo'] for analysis in track_analyses.values()]
    tempos = np.array(tempos)
    
    print(f"\n🎵 Tempo Distribution:")
    print(f"Mean tempo: {tempos.mean():.1f} BPM")
    print(f"Range: {tempos.min():.1f} - {tempos.max():.1f} BPM")
    print(f"Std: {tempos.std():.1f} BPM")
    
    # Tempo compatibility analysis - EXACT automix.py logic
    print(f"\n🔄 Tempo Compatibility Analysis:")
    total_pairs = 0
    compatible_pairs = 0
    tempo_failures = []
    
    track_names = list(track_analyses.keys())
    for i, track1 in enumerate(track_names):
        for j, track2 in enumerate(track_names):
            if i >= j:
                continue
            
            total_pairs += 1
            tempo1 = track_analyses[track1]['tempo']
            tempo2 = track_analyses[track2]['tempo']
            
            # EXACT automix.py tempo compatibility logic
            ok = False
            best_delta = float('inf')
            for scale in [1/4, 1/2, 1, 2, 4]:
                tempo = tempo2 * scale
                delta_tempo = tempo1 / tempo - 1
                if abs(delta_tempo) < abs(best_delta):
                    best_delta = delta_tempo
                if abs(delta_tempo) < MAX_TEMPO:
                    ok = True
                    break
            
            if ok:
                compatible_pairs += 1
            else:
                tempo_failures.append((track1, track2, best_delta))
    
    tempo_compat_rate = compatible_pairs / total_pairs if total_pairs > 0 else 0
    print(f"Tempo compatibility rate: {tempo_compat_rate:.1%} ({compatible_pairs}/{total_pairs} pairs)")
    
    # Pitch compatibility analysis - EXACT automix.py logic
    print(f"\n🎼 Pitch Compatibility Analysis:")
    pitch_compatible_pairs = 0
    pitch_failures = []
    no_pitch_data_pairs = 0
    
    for i, track1 in enumerate(track_names):
        for j, track2 in enumerate(track_names):
            if i >= j:
                continue
            
            analysis1 = track_analyses[track1]
            analysis2 = track_analyses[track2]
            
            # EXACT automix.py pitch matching logic
            pitch_matches = []
            if analysis1['kr_bass'] is not None and analysis2['kr_bass'] is not None:
                ps_bass = best_pitch_shift(analysis1['kr_bass'], analysis2['kr_bass'])
                pitch_matches.append(ps_bass)
            if analysis1['kr_chordal'] is not None and analysis2['kr_chordal'] is not None:
                ps_chordal = best_pitch_shift(analysis1['kr_chordal'], analysis2['kr_chordal'])
                pitch_matches.append(ps_chordal)
            if analysis1['kr_lead'] is not None and analysis2['kr_lead'] is not None:
                ps_lead = best_pitch_shift(analysis1['kr_lead'], analysis2['kr_lead'])
                pitch_matches.append(ps_lead)
            
            if pitch_matches:
                # Use the average pitch shift or the most common one
                ps = int(np.mean(pitch_matches))
                if abs(ps) <= MAX_PITCH:
                    pitch_compatible_pairs += 1
                else:
                    pitch_failures.append((track1, track2, ps))
            else:
                no_pitch_data_pairs += 1
                # In automix.py, this would be a failure ("No pitch data available for matching")
    
    pitch_compat_rate = pitch_compatible_pairs / total_pairs if total_pairs > 0 else 0
    print(f"Pitch compatibility rate: {pitch_compat_rate:.1%} ({pitch_compatible_pairs}/{total_pairs} pairs)")
    if no_pitch_data_pairs > 0:
        print(f"No pitch data available for: {no_pitch_data_pairs} pairs")
    
    # Combined compatibility - tracks must pass BOTH tempo and pitch checks
    combined_compatible = 0
    combined_failures = []
    
    for i, track1 in enumerate(track_names):
        for j, track2 in enumerate(track_names):
            if i >= j:
                continue
            
            tempo1 = track_analyses[track1]['tempo']
            tempo2 = track_analyses[track2]['tempo']
            analysis1 = track_analyses[track1]
            analysis2 = track_analyses[track2]
            
            # Check tempo first
            tempo_ok = False
            tempo_delta = float('inf')
            for scale in [1/4, 1/2, 1, 2, 4]:
                tempo = tempo2 * scale
                delta_tempo = tempo1 / tempo - 1
                if abs(delta_tempo) < abs(tempo_delta):
                    tempo_delta = delta_tempo
                if abs(delta_tempo) < MAX_TEMPO:
                    tempo_ok = True
                    break
            
            if not tempo_ok:
                combined_failures.append((track1, track2, f"TEMPO_FAILED: {float(tempo_delta):.1%}"))
                continue
            
            # Check pitch
            pitch_matches = []
            if analysis1['kr_bass'] is not None and analysis2['kr_bass'] is not None:
                ps_bass = best_pitch_shift(analysis1['kr_bass'], analysis2['kr_bass'])
                pitch_matches.append(ps_bass)
            if analysis1['kr_chordal'] is not None and analysis2['kr_chordal'] is not None:
                ps_chordal = best_pitch_shift(analysis1['kr_chordal'], analysis2['kr_chordal'])
                pitch_matches.append(ps_chordal)
            if analysis1['kr_lead'] is not None and analysis2['kr_lead'] is not None:
                ps_lead = best_pitch_shift(analysis1['kr_lead'], analysis2['kr_lead'])
                pitch_matches.append(ps_lead)
            
            if pitch_matches:
                ps = int(np.mean(pitch_matches))
                if abs(ps) <= MAX_PITCH:
                    combined_compatible += 1
                else:
                    combined_failures.append((track1, track2, f"PITCH_FAILED: {ps} semitones"))
            else:
                combined_failures.append((track1, track2, "NO_PITCH_DATA"))
    
    combined_rate = combined_compatible / total_pairs if total_pairs > 0 else 0
    print(f"Combined compatibility rate: {combined_rate:.1%} ({combined_compatible}/{total_pairs} pairs)")
    print(f"This represents tracks that would successfully mix in automix.py")
    
    # Suggest better constraints
    print(f"\n💡 Constraint Suggestions:")
    if tempo_failures:
        tempo_deltas = [abs(delta) for _, _, delta in tempo_failures]
        suggested_tempo = np.percentile(tempo_deltas, 80)  # 80th percentile
        print(f"Current MAX_TEMPO: {float(MAX_TEMPO):.1%}")
        print(f"Suggested MAX_TEMPO: {float(suggested_tempo):.1%} (would allow {80}% of failed pairs)")
    
    if pitch_failures:
        pitch_deltas = [abs(ps) for _, _, ps in pitch_failures]
        suggested_pitch = int(np.percentile(pitch_deltas, 80))  # 80th percentile
        print(f"Current MAX_PITCH: {MAX_PITCH} semitones")
        print(f"Suggested MAX_PITCH: {suggested_pitch} semitones (would allow {80}% of failed pairs)")
    
    # Show worst incompatibilities
    if tempo_failures:
        print(f"\n🚫 Worst Tempo Incompatibilities:")
        worst_tempo = sorted(tempo_failures, key=lambda x: abs(x[2]), reverse=True)[:5]
        for track1, track2, delta in worst_tempo:
            print(f"  {track1} vs {track2}: {float(delta):.1%} tempo difference")
    
    if pitch_failures:
        print(f"\n🚫 Worst Pitch Incompatibilities:")
        worst_pitch = sorted(pitch_failures, key=lambda x: abs(x[2]), reverse=True)[:5]
        for track1, track2, ps in worst_pitch:
            print(f"  {track1} vs {track2}: {ps:.0f} semitone difference")
    
    # Show combined failures breakdown
    if combined_failures:
        print(f"\n📊 Combined Failure Breakdown:")
        failure_types = {}
        for _, _, reason in combined_failures:
            failure_type = reason.split(':')[0]
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        for failure_type, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {failure_type}: {count} pairs ({count/total_pairs:.1%})")
        
        print(f"\n🚫 Example Combined Failures:")
        for i, (track1, track2, reason) in enumerate(combined_failures[:5]):
            print(f"  {track1} vs {track2}: {reason}")
    
    # Save comprehensive analysis to JSON
    analysis_summary = {
        "dataset_info": {
            "path": str(dataset_path),
            "total_tracks": len(track_analyses),
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses
        },
        "tempo_stats": {
            "mean_bpm": float(tempos.mean()),
            "min_bpm": float(tempos.min()),
            "max_bpm": float(tempos.max()),
            "std_bpm": float(tempos.std())
        },
        "current_constraints": {
            "max_tempo_percent": float(MAX_TEMPO) * 100,
            "max_pitch_semitones": int(MAX_PITCH)
        },
        "compatibility_rates": {
            "tempo_percent": tempo_compat_rate * 100,
            "pitch_percent": pitch_compat_rate * 100,
            "combined_percent": combined_rate * 100
        },
        "suggestions": {},
        "failure_breakdown": {}
    }
    
    # Add suggestions if available
    if tempo_failures:
        tempo_deltas = [abs(delta) for _, _, delta in tempo_failures]
        suggested_tempo = np.percentile(tempo_deltas, 80)
        analysis_summary["suggestions"]["suggested_max_tempo_percent"] = float(suggested_tempo) * 100
    
    if pitch_failures:
        pitch_deltas = [abs(ps) for _, _, ps in pitch_failures]
        suggested_pitch = int(np.percentile(pitch_deltas, 80))
        analysis_summary["suggestions"]["suggested_max_pitch_semitones"] = suggested_pitch
    
    # Add failure breakdown
    if combined_failures:
        failure_types = {}
        for _, _, reason in combined_failures:
            failure_type = reason.split(':')[0]
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        analysis_summary["failure_breakdown"] = failure_types
    
    # Save analysis summary
    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"💾 Saved analysis summary to: {summary_file}")
    print(f"\n📁 All analysis files saved to: {output_dir}")


def validate_audio_file(file_path, expected_sr=44100, expected_channels=2):
    """Validate a single audio file for automix compatibility."""
    issues = []
    
    try:
        # Check if file exists
        if not file_path.exists():
            return ["File does not exist"], None
        
        # Try to get file info
        try:
            info = torchaudio.info(str(file_path))
        except Exception as e:
            return [f"Cannot read file metadata: {str(e)}"], None
        
        # Check basic properties
        if info.sample_rate != expected_sr:
            issues.append(f"Sample rate mismatch: {info.sample_rate} (expected {expected_sr})")
        
        if info.num_channels != expected_channels:
            issues.append(f"Channel count mismatch: {info.num_channels} (expected {expected_channels})")
        
        # Try to load the entire file to check for corruption and proper analysis
        try:
            # Load the entire file for accurate level analysis
            waveform, sr = torchaudio.load(str(file_path))
            
            # Check for silence or very low amplitude across the entire file
            rms = torch.sqrt(torch.mean(waveform**2))
            if rms < 1e-4:  # More reasonable threshold than 1e-6
                issues.append("Audio appears to be silent or extremely quiet")
            
            # Check for sufficient non-silent audio content (at least 30% of duration)
            # Calculate RMS in chunks to find non-silent portions
            chunk_size = sr // 10  # 0.1 second chunks
            non_silent_chunks = 0
            total_chunks = 0
            
            # Only show progress for long files (>10 seconds)
            total_samples = waveform.shape[-1]
            show_progress = total_samples > 10 * sr
            
            chunk_range = range(0, total_samples, chunk_size)
            if show_progress:
                chunk_range = tqdm.tqdm(chunk_range, desc=f"Analyzing {file_path.name}", leave=False, unit="chunk")
            
            for i in chunk_range:
                chunk = waveform[..., i:i+chunk_size]
                if chunk.shape[-1] > 0:  # Valid chunk
                    chunk_rms = torch.sqrt(torch.mean(chunk**2))
                    if chunk_rms > 1e-3:  # Threshold for non-silent (higher than overall silence check)
                        non_silent_chunks += 1
                    total_chunks += 1
            
            if total_chunks > 0:
                non_silent_percentage = (non_silent_chunks / total_chunks) * 100
                if non_silent_percentage < 30:
                    issues.append(f"Insufficient audio content: only {non_silent_percentage:.1f}% non-silent (need ≥30%)")
            
            # Check for clipping
            if torch.abs(waveform).max() >= 0.99:
                issues.append("Audio may be clipped (peak >= 0.99)")
            
            # Check for NaN or inf values
            if torch.isnan(waveform).any():
                issues.append("Audio contains NaN values")
            if torch.isinf(waveform).any():
                issues.append("Audio contains infinite values")
                
        except Exception as e:
            issues.append(f"Cannot load audio data: {str(e)}")
            return issues, info
        
        return issues, info
        
    except Exception as e:
        return [f"Unexpected error: {str(e)}"], None


def validate_track_stems(track_dir, sources=None):
    """Validate all stems in a track directory."""
    if sources is None:
        sources = SOURCES
    
    track_issues = []
    stem_info = {}
    
    # Check each required stem
    for source in sources:
        stem_file = track_dir / f"{source}.wav"
        issues, info = validate_audio_file(stem_file)
        
        if issues:
            track_issues.extend([f"{source}: {issue}" for issue in issues])
        else:
            stem_info[source] = info
    
    # Check consistency across stems
    if len(stem_info) > 1:
        # Check if all stems have exactly the same number of samples
        sample_counts = [info.num_frames for info in stem_info.values()]
        if len(set(sample_counts)) > 1:
            # Show both sample counts and durations for debugging
            sample_details = []
            for source, info in stem_info.items():
                duration = info.num_frames / info.sample_rate
                sample_details.append(f"{source}: {info.num_frames} samples ({duration:.3f}s)")
            track_issues.append(f"Inconsistent sample counts: {'; '.join(sample_details)}")
        
        # Check if all stems have same sample rate
        sample_rates = [info.sample_rate for info in stem_info.values()]
        if len(set(sample_rates)) > 1:
            rate_details = []
            for source, info in stem_info.items():
                rate_details.append(f"{source}: {info.sample_rate}Hz")
            track_issues.append(f"Inconsistent sample rates: {'; '.join(rate_details)}")
        
        # Check if all stems have same channel count
        channels = [info.num_channels for info in stem_info.values()]
        if len(set(channels)) > 1:
            channel_details = []
            for source, info in stem_info.items():
                channel_details.append(f"{source}: {info.num_channels}ch")
            track_issues.append(f"Inconsistent channel counts: {'; '.join(channel_details)}")
    
    return track_issues, stem_info


def validate_dataset_structure(dataset_path, sources=None):
    """Validate the overall dataset structure."""
    if sources is None:
        sources = SOURCES
    
    root = Path(dataset_path)
    
    print(f"🔍 Validating dataset structure: {dataset_path}")
    print(f"Expected sources: {sources}")
    
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all potential track directories
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    if not track_dirs:
        print(f"❌ No track directories found in {dataset_path}")
        return False
    
    print(f"📁 Found {len(track_dirs)} track directories")
    
    valid_tracks = 0
    total_issues = 0
    
    with tqdm.tqdm(track_dirs, desc="Validating tracks", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            issues, stem_info = validate_track_stems(track_dir, sources)
            
            if not issues:
                print(f"  ✅ {track_dir.name}: All stems valid")
                valid_tracks += 1
            else:
                print(f"  ❌ {track_dir.name}: Issues found:")
                for issue in issues:
                    print(f"    - {issue}")
                total_issues += len(issues)
    
    print(f"\n📊 Validation Summary:")
    print(f"Valid tracks: {valid_tracks}/{len(track_dirs)}")
    print(f"Total issues: {total_issues}")
    print(f"Success rate: {valid_tracks/len(track_dirs)*100:.1f}%")
    
    return valid_tracks == len(track_dirs)


def analyze_dataset_compatibility(dataset_path):
    """Analyze the dataset for automix compatibility."""
    print(f"🔍 Analyzing dataset: {dataset_path}")
    print(f"Expected sources: {SOURCES}")
    
    # First validate basic structure
    if not validate_dataset_structure(dataset_path):
        print("\n❌ Dataset structure validation failed. Fix issues before proceeding.")
        return
    
    # If structure is valid, analyze compatibility
    root = Path(dataset_path) 
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    print(f"\n🎯 Analyzing automix compatibility...")
    
    # Check for build_metadata compatibility
    try:
        print("Testing build_metadata compatibility...")
        # from demucs.wav import build_metadata
        
        # Try to build metadata for a subset
        sample_dirs = track_dirs[:5]  # Test first 5 tracks
        
        # Create temporary metadata for testing
        temp_metadata = {}
        for track_dir in sample_dirs:
            # Check if all required stems exist
            all_stems_exist = all((track_dir / f"{source}.wav").exists() for source in SOURCES)
            if all_stems_exist:
                temp_metadata[track_dir.name] = {
                    'path': track_dir,
                    'sources': SOURCES
                }
        
        if temp_metadata:
            print(f"✅ Found {len(temp_metadata)} tracks with complete stems")
        else:
            print("❌ No tracks found with all required stems")
            
    except Exception as e:
        print(f"❌ Error testing build_metadata: {e}")
        traceback.print_exc()


def validate_for_automix(dataset_path, quick_check=False):
    """Comprehensive validation for automix compatibility."""
    print(f"🎯 Comprehensive Automix Validation")
    print(f"Dataset: {dataset_path}")
    print(f"Quick check: {quick_check}")
    print(f"=" * 60)
    
    root = Path(dataset_path)
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all track directories
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not track_dirs:
        print(f"❌ No track directories found")
        return False
    
    print(f"📁 Found {len(track_dirs)} potential tracks")
    
    # Validation counters
    valid_tracks = 0
    corrupted_files = 0
    missing_stems = 0
    format_issues = 0
    silent_stems = 0
    
    # If quick check, only validate first 10 tracks
    tracks_to_check = track_dirs[:10] if quick_check else track_dirs
    
    with tqdm.tqdm(tracks_to_check, desc="Validating tracks", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            track_valid = True
            track_issues = []
            
            # Check each required stem
            for source in SOURCES:
                stem_file = track_dir / f"{source}.wav"
                
                if not stem_file.exists():
                    track_issues.append(f"Missing {source}.wav")
                    missing_stems += 1
                    track_valid = False
                    continue
                
                issues, info = validate_audio_file(stem_file)
                
                if issues:
                    for issue in issues:
                        if "Cannot read" in issue or "metadata" in issue:
                            corrupted_files += 1
                        elif "Sample rate" in issue or "Channel count" in issue:
                            format_issues += 1
                        elif "silent" in issue:
                            silent_stems += 1
                    track_issues.extend([f"{source}: {issue}" for issue in issues])
                    track_valid = False
            
            if track_valid:
                valid_tracks += 1
            elif not quick_check:  # Only show details in full check
                print(f"❌ {track_dir.name}: {'; '.join(track_issues)}")
    
    # Summary
    print(f"\n📊 Validation Results:")
    print(f"{'='*60}")
    print(f"Total tracks checked: {len(tracks_to_check)}")
    print(f"Valid tracks: {valid_tracks}")
    print(f"Invalid tracks: {len(tracks_to_check) - valid_tracks}")
    print(f"")
    print(f"Issue breakdown:")
    print(f"  - Corrupted/unreadable files: {corrupted_files}")
    print(f"  - Missing stems: {missing_stems}")
    print(f"  - Format issues: {format_issues}")
    print(f"  - Silent stems: {silent_stems}")
    print(f"")
    
    success_rate = valid_tracks / len(tracks_to_check) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate < 80:
        print(f"⚠️  Low success rate! Consider fixing dataset issues before automix.")
    elif success_rate < 95:
        print(f"🟡 Moderate success rate. Some tracks may fail during automix.")
    else:
        print(f"✅ High success rate. Dataset should work well with automix.")
    
    return success_rate >= 80


def find_problematic_files(dataset_path):
    """Find and list all problematic files in the dataset."""
    print(f"🔍 Finding problematic files in: {dataset_path}")
    
    root = Path(dataset_path)
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    problematic_files = []
    
    with tqdm.tqdm(track_dirs, desc="Checking for problems", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            for source in SOURCES:
                stem_file = track_dir / f"{source}.wav"
                
                if not stem_file.exists():
                    problematic_files.append({
                        'file': str(stem_file),
                        'issue': 'Missing file',
                        'track': track_dir.name,
                        'source': source
                    })
                    continue
                
                issues, _ = validate_audio_file(stem_file)
                
                if issues:
                    for issue in issues:
                        problematic_files.append({
                            'file': str(stem_file),
                            'issue': issue,
                            'track': track_dir.name,
                            'source': source
                        })
    
    if problematic_files:
        print(f"\n❌ Found {len(problematic_files)} problematic files:")
        print(f"{'Track':<20} {'Source':<12} {'Issue':<50} {'File'}")
        print(f"{'-'*100}")
        
        for item in problematic_files:
            print(f"{item['track']:<20} {item['source']:<12} {item['issue']:<50} {Path(item['file']).name}")
    else:
        print(f"✅ No problematic files found!")
    
    return problematic_files


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


def check_missing_files(dataset_path, sources=None):
    """Check for missing stem files in the dataset."""
    if sources is None:
        sources = SOURCES
    
    print(f"🔍 Checking for missing files in: {dataset_path}")
    print(f"Required sources: {sources}")
    print(f"=" * 60)
    
    root = Path(dataset_path)
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all track directories
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not track_dirs:
        print(f"❌ No track directories found")
        return False
    
    print(f"📁 Checking {len(track_dirs)} tracks for missing files...")
    
    complete_tracks = 0
    incomplete_tracks = 0
    missing_files = []
    track_completeness = {}
    
    with tqdm.tqdm(track_dirs, desc="Checking missing files", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            missing_stems = []
            available_stems = []
            
            for source in sources:
                stem_file = track_dir / f"{source}.wav"
                if stem_file.exists():
                    available_stems.append(source)
                else:
                    missing_stems.append(source)
                    missing_files.append({
                        'track': track_dir.name,
                        'source': source,
                        'expected_path': str(stem_file)
                    })
            
            if missing_stems:
                incomplete_tracks += 1
                track_completeness[track_dir.name] = {
                    'missing': missing_stems,
                    'available': available_stems,
                    'completeness': len(available_stems) / len(sources)
                }
            else:
                complete_tracks += 1
    
    # Summary
    print(f"\n📊 Missing Files Summary:")
    print(f"{'='*60}")
    print(f"Total tracks: {len(track_dirs)}")
    print(f"Complete tracks: {complete_tracks}")
    print(f"Incomplete tracks: {incomplete_tracks}")
    print(f"Total missing files: {len(missing_files)}")
    print(f"Completeness rate: {complete_tracks/len(track_dirs)*100:.1f}%")
    
    if incomplete_tracks > 0:
        print(f"\n❌ Incomplete Tracks:")
        print(f"{'Track':<25} {'Missing':<20} {'Available':<20} {'Complete':<10}")
        print(f"{'-'*80}")
        
        # Sort by completeness (worst first)
        sorted_tracks = sorted(track_completeness.items(), 
                             key=lambda x: x[1]['completeness'])
        
        for track_name, info in sorted_tracks:
            missing_str = ', '.join(info['missing'])
            available_str = ', '.join(info['available'])
            completeness = f"{info['completeness']*100:.0f}%"
            
            print(f"{track_name:<25} {missing_str:<20} {available_str:<20} {completeness:<10}")
        
        # Show missing files by source
        print(f"\n📋 Missing Files by Source:")
        missing_by_source = {}
        for item in missing_files:
            source = item['source']
            if source not in missing_by_source:
                missing_by_source[source] = []
            missing_by_source[source].append(item['track'])
        
        for source in sources:
            if source in missing_by_source:
                count = len(missing_by_source[source])
                print(f"  {source}: {count} files missing")
                if count <= 10:  # Show track names if not too many
                    tracks = ', '.join(missing_by_source[source])
                    print(f"    Tracks: {tracks}")
                else:
                    print(f"    (Too many to list - use 'find-problems' for details)")
            else:
                print(f"  {source}: ✅ All files present")
    
    else:
        print(f"\n✅ All tracks are complete! No missing files found.")
    
    return complete_tracks == len(track_dirs)


def check_sample_counts(dataset_path, sources=None):
    """Check for consistent sample counts across stems in each track."""
    if sources is None:
        sources = SOURCES
    
    print(f"🔍 Checking sample count consistency in: {dataset_path}")
    print(f"Required sources: {sources}")
    print(f"=" * 60)
    
    root = Path(dataset_path)
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all track directories
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not track_dirs:
        print(f"❌ No track directories found")
        return False
    
    print(f"📁 Checking {len(track_dirs)} tracks for sample count consistency...")
    
    consistent_tracks = 0
    inconsistent_tracks = 0
    missing_files = 0
    unreadable_files = 0
    
    with tqdm.tqdm(track_dirs, desc="Checking sample counts", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            # Collect sample info for each stem and mix
            stem_info = {}
            track_issues = []
            
            # Check stems
            for source in sources:
                stem_file = track_dir / f"{source}.wav"
                
                if not stem_file.exists():
                    track_issues.append(f"Missing {source}.wav")
                    missing_files += 1
                    continue
                
                try:
                    info = torchaudio.info(str(stem_file))
                    stem_info[source] = {
                        'samples': info.num_frames,
                        'duration': info.num_frames / info.sample_rate,
                        'sample_rate': info.sample_rate
                    }
                except Exception as e:
                    track_issues.append(f"Cannot read {source}.wav: {str(e)}")
                    unreadable_files += 1
            
            # Check mixture.wav
            mix_file = track_dir / "mixture.wav"
            mix_info = None
            if mix_file.exists():
                try:
                    info = torchaudio.info(str(mix_file))
                    mix_info = {
                        'samples': info.num_frames,
                        'duration': info.num_frames / info.sample_rate,
                        'sample_rate': info.sample_rate
                    }
                    stem_info['mixture'] = mix_info  # Add to comparison
                except Exception as e:
                    track_issues.append(f"Cannot read mixture.wav: {str(e)}")
                    unreadable_files += 1
            
            # Check consistency if we have at least 2 readable stems
            if len(stem_info) >= 2:
                sample_counts = [info['samples'] for info in stem_info.values()]
                
                if len(set(sample_counts)) > 1:
                    # Inconsistent sample counts
                    inconsistent_tracks += 1
                    print(f"❌ {track_dir.name}: Inconsistent sample counts")
                    
                    # Show detailed breakdown - stems first, then mix
                    for source in sources:
                        if source in stem_info:
                            info = stem_info[source]
                            print(f"    {source}: {info['samples']:,} samples ({info['duration']:.3f}s)")
                    
                    # Show mixture.wav separately if it exists
                    if 'mixture' in stem_info:
                        mix_info = stem_info['mixture']
                        print(f"    mixture: {mix_info['samples']:,} samples ({mix_info['duration']:.3f}s)")
                    elif mix_file.exists():
                        print(f"    mixture: unreadable")
                    else:
                        print(f"    mixture: missing")
                    
                    # Show differences
                    max_samples = max(sample_counts)
                    min_samples = min(sample_counts)
                    diff_samples = max_samples - min_samples
                    diff_ms = (diff_samples / stem_info[list(stem_info.keys())[0]]['sample_rate']) * 1000
                    print(f"    Difference: {diff_samples:,} samples ({diff_ms:.1f}ms)")
                    
                else:
                    # All sample counts match
                    consistent_tracks += 1
                    sample_count = sample_counts[0]
                    duration = stem_info[list(stem_info.keys())[0]]['duration']
                    mix_status = "✅ included" if 'mixture' in stem_info else ("❌ unreadable" if mix_file.exists() else "⚠️ missing")
                    print(f"✅ {track_dir.name}: All files match ({sample_count:,} samples, {duration:.3f}s) - mixture: {mix_status}")
            
            elif len(stem_info) == 1:
                # Only one readable stem
                source = list(stem_info.keys())[0]
                info = stem_info[source]
                print(f"⚠️  {track_dir.name}: Only {source} readable ({info['samples']:,} samples)")
                
            else:
                # No readable stems
                print(f"❌ {track_dir.name}: No readable stems")
                if track_issues:
                    for issue in track_issues:
                        print(f"    - {issue}")
    
    # Summary
    print(f"\n📊 Sample Count Check Results:")
    print(f"{'='*60}")
    print(f"Total tracks checked: {len(track_dirs)}")
    print(f"Consistent tracks: {consistent_tracks}")
    print(f"Inconsistent tracks: {inconsistent_tracks}")
    print(f"Missing files: {missing_files}")
    print(f"Unreadable files: {unreadable_files}")
    print(f"")
    
    consistency_rate = consistent_tracks / len(track_dirs) * 100 if track_dirs else 0
    print(f"Sample consistency rate: {consistency_rate:.1f}%")
    
    if inconsistent_tracks > 0:
        print(f"⚠️  {inconsistent_tracks} tracks have mismatched sample counts!")
        print(f"   These will likely cause issues during automix processing.")
    else:
        print(f"✅ All tracks have consistent sample counts!")
    
    return inconsistent_tracks == 0


def repair_sample_counts(dataset_path, sources=None, strategy='pad_to_longest', backup=True, dry_run=False):
    """Repair sample count mismatches in tracks by trimming or padding stems.
    
    Args:
        dataset_path: Path to dataset
        sources: List of source names to check
        strategy: 'auto_majority', 'trim_to_shortest', 'pad_to_longest', or 'trim_to_majority'
        backup: Whether to backup original files
        dry_run: If True, only show what would be done without making changes
    """
    if sources is None:
        sources = SOURCES
    
    print(f"🔧 Repairing sample count mismatches in: {dataset_path}")
    print(f"Strategy: {strategy}")
    print(f"Backup: {'Yes' if backup else 'No'}")
    print(f"Dry run: {'Yes' if dry_run else 'No'}")
    print(f"=" * 60)
    
    root = Path(dataset_path)
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all track directories
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not track_dirs:
        print(f"❌ No track directories found")
        return False
    
    tracks_repaired = 0
    tracks_skipped = 0
    global_target_samples = None
    
    # For auto_majority strategy, first scan entire dataset to find most common sample count
    if strategy == 'auto_majority':
        print("🔍 Scanning dataset to find most common sample count...")
        all_sample_counts = []
        
        with tqdm.tqdm(track_dirs, desc="Scanning sample counts", unit="track") as scan_pbar:
            for track_dir in scan_pbar:
                scan_pbar.set_postfix(track=track_dir.name[:20])
                
                for source in sources:
                    stem_file = track_dir / f"{source}.wav"
                    if stem_file.exists():
                        try:
                            info = torchaudio.info(str(stem_file))
                            all_sample_counts.append(info.num_frames)
                        except Exception:
                            continue
                
                # Also check mixture.wav for global analysis
                mix_file = track_dir / "mixture.wav"
                if mix_file.exists():
                    try:
                        info = torchaudio.info(str(mix_file))
                        all_sample_counts.append(info.num_frames)
                    except Exception:
                        continue
        
        if all_sample_counts:
            count_freq = Counter(all_sample_counts)
            global_target_samples = count_freq.most_common(1)[0][0]
            print(f"🎯 Most common sample count: {global_target_samples:,} samples")
            print(f"   Found in {count_freq[global_target_samples]} files across the dataset")
        else:
            print("❌ No readable audio files found in dataset")
            return False
    
    with tqdm.tqdm(track_dirs, desc="Repairing sample counts", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            # Collect sample info for each stem and mixture
            stem_info = {}
            
            for source in sources:
                stem_file = track_dir / f"{source}.wav"
                
                if not stem_file.exists():
                    continue
                
                try:
                    info = torchaudio.info(str(stem_file))
                    stem_info[source] = {
                        'file': stem_file,
                        'samples': info.num_frames,
                        'sample_rate': info.sample_rate,
                        'channels': info.num_channels,
                        'encoding': info.encoding,
                        'bits_per_sample': info.bits_per_sample
                    }
                except Exception as e:
                    print(f"❌ Cannot read {track_dir.name}/{source}.wav: {str(e)}")
                    continue
            
            # Also include mixture.wav in repair process
            mixture_file = track_dir / "mixture.wav"
            if mixture_file.exists():
                try:
                    info = torchaudio.info(str(mixture_file))
                    stem_info['mixture'] = {
                        'file': mixture_file,
                        'samples': info.num_frames,
                        'sample_rate': info.sample_rate,
                        'channels': info.num_channels,
                        'encoding': info.encoding,
                        'bits_per_sample': info.bits_per_sample
                    }
                except Exception as e:
                    print(f"❌ Cannot read {track_dir.name}/mixture.wav: {str(e)}")
                    continue
            
            # Check if repair is needed
            if len(stem_info) < 2:
                continue
            
            sample_counts = [info['samples'] for info in stem_info.values()]
            
            if len(set(sample_counts)) <= 1:
                # Already consistent
                continue
            
            # Determine target sample count based on strategy
            if strategy == 'auto_majority':
                target_samples = global_target_samples
            elif strategy == 'trim_to_shortest':
                target_samples = min(sample_counts)
            elif strategy == 'pad_to_longest':
                target_samples = max(sample_counts)
            elif strategy == 'trim_to_majority':
                # Find the most common sample count within this track
                count_freq = Counter(sample_counts)
                target_samples = count_freq.most_common(1)[0][0]
            else:
                print(f"❌ Unknown strategy: {strategy}")
                continue
            
            print(f"\n🔧 {track_dir.name}: Repairing sample counts")
            print(f"   Target samples: {target_samples:,}")
            
            # Check if all stems can be repaired
            needs_repair = []
            for source, info in stem_info.items():
                if info['samples'] != target_samples:
                    action = 'trim' if info['samples'] > target_samples else 'pad'
                    diff = abs(info['samples'] - target_samples)
                    needs_repair.append({
                        'source': source,
                        'info': info,
                        'action': action,
                        'diff': diff
                    })
            
            if not needs_repair:
                continue
            
            # Show repair plan
            for item in needs_repair:
                action_word = "Trim" if item['action'] == 'trim' else "Pad"
                print(f"   {action_word} {item['source']}: {item['info']['samples']:,} -> {target_samples:,} ({item['diff']:,} samples)")
            
            if dry_run:
                print(f"   [DRY RUN] Would repair {len(needs_repair)} stems")
                tracks_repaired += 1
                continue
            
            # Backup original files if requested
            if backup:
                backup_dir = track_dir / 'backup_original'
                backup_dir.mkdir(exist_ok=True)
                
                for item in needs_repair:
                    source = item['source']
                    original_file = item['info']['file']
                    backup_file = backup_dir / f"{source}.wav"
                    
                    if not backup_file.exists():
                        try:
                            shutil.copy2(original_file, backup_file)
                            print(f"   📦 Backed up {source}.wav")
                        except Exception as e:
                            print(f"   ❌ Failed to backup {source}.wav: {e}")
                            continue
            
            # Perform repairs
            repair_success = True
            for item in needs_repair:
                source = item['source']
                info = item['info']
                
                try:
                    # Load the audio file
                    waveform, sr = torchaudio.load(str(info['file']))
                    
                    if item['action'] == 'trim':
                        # Trim to target length
                        waveform = waveform[..., :target_samples]
                    elif item['action'] == 'pad':
                        # Pad with zeros to target length
                        padding_needed = target_samples - waveform.shape[-1]
                        waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
                    
                    # Save the repaired file with original format parameters
                    save_kwargs = {
                        'sample_rate': sr,
                        'encoding': info['encoding'],
                        'bits_per_sample': info['bits_per_sample']
                    }
                    torchaudio.save(str(info['file']), waveform, **save_kwargs)
                    print(f"   ✅ Repaired {source}.wav (preserved {info['bits_per_sample']}-bit {info['encoding']})")
                    
                except Exception as e:
                    print(f"   ❌ Failed to repair {source}.wav: {e}")
                    repair_success = False
            
            if repair_success:
                tracks_repaired += 1
                print(f"   ✅ Successfully repaired {track_dir.name}")
            else:
                tracks_skipped += 1
                print(f"   ❌ Failed to repair {track_dir.name}")
    
    # Summary
    print(f"\n📊 Repair Results:")
    print(f"{'='*60}")
    print(f"Tracks repaired: {tracks_repaired}")
    print(f"Tracks skipped: {tracks_skipped}")
    
    if dry_run:
        print(f"🔍 This was a dry run - no files were modified")
        print(f"Run without --dry-run to perform actual repairs")
    elif tracks_repaired > 0:
        print(f"✅ Repair completed!")
        if backup:
            print(f"📦 Original files backed up in each track's 'backup_original' folder")
    
    return tracks_repaired > 0


def check_audio_content_percentages(dataset_path, sources=None, threshold=30, sensitivity=0.01, include_noise_analysis=False, confidence_threshold=0.8):
    """Check percentage of significant audio content in each stem."""
    if sources is None:
        sources = SOURCES
    
    print(f"🔍 Checking audio content percentages in: {dataset_path}")
    print(f"Threshold: {threshold}% significant audio")
    print(f"Sensitivity: {sensitivity*100:.1f}% of peak RMS")
    print(f"Window size: 50ms (DC-removed)")
    print(f"Noise analysis: {'Enabled' if include_noise_analysis else 'Disabled'}")
    if include_noise_analysis:
        print(f"Noise confidence threshold: {confidence_threshold*100:.0f}%")
    print(f"Sources: {sources}")
    print(f"=" * 60)
    
    root = Path(dataset_path)
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all track directories
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not track_dirs:
        print(f"❌ No track directories found")
        return False
    
    print(f"📁 Analyzing {len(track_dirs)} tracks for audio content...")
    
    # Results tracking
    track_results = {}
    problematic_stems = []
    total_stems_analyzed = 0
    stems_below_threshold = 0
    
    # Additional noise tracking if enabled
    noise_chunks_found = 0
    total_chunks_analyzed = 0
    noise_type_counts = Counter()
    
    def analyze_stem_content(stem_file):
        """Analyze audio content percentage in a single stem file."""
        nonlocal noise_chunks_found, total_chunks_analyzed, noise_type_counts
        
        try:
            # Load the entire file
            waveform, sr = torchaudio.load(str(stem_file))
            total_samples = waveform.shape[-1]
            
            if total_samples == 0:
                return 0.0, "Empty file", {}, {}
            
            # Remove DC offset first
            waveform_dc_removed = waveform - torch.mean(waveform, dim=-1, keepdim=True)
            
            # Calculate overall RMS to set adaptive threshold (using DC-removed signal)
            overall_rms = torch.sqrt(torch.mean(waveform_dc_removed**2))
            peak_rms = torch.sqrt(torch.mean(waveform_dc_removed**2, dim=0).max())
            
            # Use adaptive threshold: user-defined % of peak RMS, but at least 1e-4
            adaptive_threshold = max(peak_rms * sensitivity, 1e-4)
            
            # Calculate RMS in chunks to find non-silent portions
            chunk_size = sr // 20  # 0.05 second (50ms) chunks
            non_silent_chunks = 0
            total_chunks = 0
            chunk_rms_values = []
            
            # Noise analysis tracking
            noise_analysis_results = {}
            chunk_noise_types = Counter()
            noise_chunks_in_stem = 0
            
            # Process in chunks
            for i in range(0, total_samples, chunk_size):
                chunk = waveform_dc_removed[..., i:i+chunk_size]
                if chunk.shape[-1] > 0:  # Valid chunk
                    chunk_rms = torch.sqrt(torch.mean(chunk**2))
                    chunk_rms_values.append(float(chunk_rms))
                    total_chunks += 1
                    
                    if include_noise_analysis:
                        total_chunks_analyzed += 1
                    
                    is_significant = chunk_rms > adaptive_threshold
                    if is_significant:
                        non_silent_chunks += 1
                        
                        # Perform noise analysis on significant chunks if requested
                        if include_noise_analysis and chunk.shape[-1] >= sr // 10:  # At least 0.1s for noise analysis
                            noise_result = detect_noise_type(chunk, sr, confidence_threshold)
                            if noise_result["is_noise"]:
                                noise_chunks_in_stem += 1
                                noise_chunks_found += 1
                                chunk_noise_types[noise_result["type"]] += 1
                                noise_type_counts[noise_result["type"]] += 1
            
            if total_chunks > 0:
                percentage = (non_silent_chunks / total_chunks) * 100
                
                # Return debug info
                debug_info = {
                    'overall_rms': float(overall_rms),
                    'peak_rms': float(peak_rms),
                    'adaptive_threshold': float(adaptive_threshold),
                    'total_chunks': total_chunks,
                    'non_silent_chunks': non_silent_chunks,
                    'chunk_rms_mean': float(np.mean(chunk_rms_values)) if chunk_rms_values else 0,
                    'chunk_rms_max': float(np.max(chunk_rms_values)) if chunk_rms_values else 0
                }
                
                # Noise analysis results
                if include_noise_analysis:
                    noise_analysis_results = {
                        'noise_chunks': noise_chunks_in_stem,
                        'noise_percentage': (noise_chunks_in_stem / non_silent_chunks * 100) if non_silent_chunks > 0 else 0,
                        'noise_types': dict(chunk_noise_types)
                    }
                
                return percentage, None, debug_info, noise_analysis_results
            else:
                return 0.0, "No valid chunks", {}, {}
                
        except Exception as e:
            return 0.0, f"Error: {str(e)}", {}, {}
    
    with tqdm.tqdm(track_dirs, desc="Analyzing audio content", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            track_results[track_dir.name] = {}
            
            # Analyze each stem
            for source in sources:
                stem_file = track_dir / f"{source}.wav"
                
                if not stem_file.exists():
                    track_results[track_dir.name][source] = {
                        'percentage': 0.0,
                        'status': 'missing',
                        'error': 'File not found',
                        'noise_analysis': {} if include_noise_analysis else None
                    }
                    continue
                
                total_stems_analyzed += 1
                percentage, error, debug_info, noise_analysis_results = analyze_stem_content(stem_file)
                
                track_results[track_dir.name][source] = {
                    'percentage': percentage,
                    'status': 'ok' if error is None else 'error',
                    'error': error,
                    'debug': debug_info,
                    'noise_analysis': noise_analysis_results if include_noise_analysis else None
                }
                
                # Check if below threshold
                if error is None and percentage < threshold:
                    stems_below_threshold += 1
                    problematic_stems.append({
                        'track': track_dir.name,
                        'source': source,
                        'percentage': percentage
                    })
    
    # Generate report
    print(f"\n📊 Audio Content Analysis Results:")
    print(f"{'='*60}")
    print(f"Total tracks analyzed: {len(track_dirs)}")
    print(f"Total stems analyzed: {total_stems_analyzed}")
    print(f"Stems below {threshold}% threshold: {stems_below_threshold}")
    print(f"Problem rate: {stems_below_threshold/total_stems_analyzed*100:.1f}%")
    
    # Add noise analysis summary if enabled
    if include_noise_analysis and total_chunks_analyzed > 0:
        noise_rate = noise_chunks_found / total_chunks_analyzed * 100
        print(f"\n🎭 Noise Analysis Summary:")
        print(f"Total chunks analyzed: {total_chunks_analyzed}")
        print(f"Chunks containing noise: {noise_chunks_found}")
        print(f"Overall noise rate: {noise_rate:.1f}%")
        
        if noise_type_counts:
            print(f"Noise type distribution:")
            for noise_type, count in noise_type_counts.most_common():
                percentage = count / sum(noise_type_counts.values()) * 100
                print(f"  {noise_type}: {count} chunks ({percentage:.1f}%)")
    
    if problematic_stems:
        print(f"\n🚫 Stems with < {threshold}% significant audio:")
        print(f"{'='*60}")
        
        # Group by track
        tracks_with_problems = {}
        for stem in problematic_stems:
            track = stem['track']
            if track not in tracks_with_problems:
                tracks_with_problems[track] = []
            tracks_with_problems[track].append(stem)
        
        # Sort tracks by number of problematic stems (worst first)
        sorted_tracks = sorted(tracks_with_problems.items(), 
                             key=lambda x: len(x[1]), reverse=True)
        
        for track_name, stems in sorted_tracks:
            print(f"\n📁 {track_name}:")
            for stem in sorted(stems, key=lambda x: x['percentage']):
                noise_info = ""
                if include_noise_analysis:
                    stem_result = track_results[track_name][stem['source']]
                    if stem_result.get('noise_analysis') and stem_result['noise_analysis'].get('noise_chunks', 0) > 0:
                        noise_pct = stem_result['noise_analysis']['noise_percentage']
                        main_noise_type = max(stem_result['noise_analysis']['noise_types'].items(), key=lambda x: x[1])[0] if stem_result['noise_analysis']['noise_types'] else 'unknown'
                        noise_info = f" (🎭 {noise_pct:.1f}% {main_noise_type})"
                
                print(f"   {stem['source']}: {stem['percentage']:.1f}% significant audio{noise_info}")
        
    
    else:
        print(f"\n✅ All stems have sufficient audio content (≥{threshold}%)!")
    
    # Create visualization grouped by source type
    source_data = {}
    for track_name, track_results_dict in track_results.items():
        for source, source_result in track_results_dict.items():
            if source_result['status'] == 'ok':
                if source not in source_data:
                    source_data[source] = []
                source_data[source].append(source_result['percentage'])
    
    if source_data:
        print(f"\n📊 Creating distribution plots for each source type...")
        
        # Create separate plots for each source type
        available_sources = [source for source in sources if source in source_data]
        
        # Calculate subplot layout - add extra plot for noise if noise analysis is enabled
        n_plots = len(available_sources)
        if include_noise_analysis and noise_type_counts:
            n_plots += 1
        
        cols = min(3, n_plots)  # Max 3 columns
        rows = (n_plots + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = axes if n_plots > 1 else [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]
        
        # Create histogram for each source
        for i, source in enumerate(available_sources):
            ax = axes[i]
            values = np.array(source_data[source])
            
            # Create histogram
            n_bins = min(20, len(values))  # Adjust bins based on data size
            counts, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, color='skyblue', 
                                          edgecolor='black', density=False)
            
            # Color bars below threshold in red
            for j, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
                if bin_right <= threshold:
                    patch.set_facecolor('red')
                elif bin_left < threshold < bin_right:
                    patch.set_facecolor('orange')  # Bins that cross threshold
            
            # Add threshold line
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold ({threshold}%)')
            
            # Add statistics lines
            median_val = np.median(values)
            mean_val = np.mean(values)
            ax.axvline(x=median_val, color='green', linestyle='-', linewidth=2, 
                      label=f'Median ({median_val:.1f}%)')
            ax.axvline(x=mean_val, color='purple', linestyle=':', linewidth=2, 
                      label=f'Mean ({mean_val:.1f}%)')
            
            # Customize subplot
            ax.set_xlabel('Audio Content Percentage (%)')
            ax.set_ylabel('Number of Stems')
            ax.set_title(f'{source.capitalize()} Distribution (n={len(values)})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            
            # Add statistics text
            below_threshold = (values < threshold).sum()
            at_100_percent = (values >= 99.9).sum()
            stats_text = f'Below threshold: {below_threshold}\nAt 100%: {at_100_percent}\nStd: {values.std():.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add noise type distribution plot if noise analysis was performed
        if include_noise_analysis and noise_type_counts and len(available_sources) < n_plots:
            ax = axes[len(available_sources)]
            
            types = list(noise_type_counts.keys())
            counts = list(noise_type_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            
            wedges, texts, autotexts = ax.pie(counts, labels=types, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            ax.set_title(f"Noise Type Distribution\n(Total: {sum(counts)} chunks)")
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save individual distribution plots
        output_dir = root / "audio_content_analysis"
        output_dir.mkdir(exist_ok=True)
        
        plot_filename = "audio_content_distributions_with_noise.png" if include_noise_analysis else "audio_content_distributions.png"
        plot_file = output_dir / plot_filename
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Distribution plots saved to: {plot_file}")
        
        # Show statistics by source type
        print(f"\n📊 Statistics by Source Type:")
        for source in available_sources:
            values = np.array(source_data[source])
            below_threshold = (values < threshold).sum()
            at_100_percent = (values >= 99.9).sum()
            print(f"   {source}: {len(values)} stems, min={values.min():.1f}%, med={np.median(values):.1f}%, max={values.max():.1f}%, std={values.std():.1f}%, {below_threshold} below threshold, {at_100_percent} at 100%")
        
        # Show debug info for some 100% stems to understand why
        print(f"\n🔍 Debug Info for 100% Stems (to understand threshold calculation):")
        hundred_percent_count = 0
        for track_name, track_results_dict in track_results.items():
            for source, source_result in track_results_dict.items():
                if source_result['status'] == 'ok' and source_result['percentage'] >= 99.9:
                    if hundred_percent_count < 3:  # Show first 3 examples
                        debug = source_result['debug']
                        print(f"   {track_name}/{source}: {source_result['percentage']:.1f}%")
                        print(f"     Overall RMS: {debug['overall_rms']:.6f}")
                        print(f"     Peak RMS: {debug['peak_rms']:.6f}")
                        print(f"     Adaptive threshold: {debug['adaptive_threshold']:.6f}")
                        print(f"     Non-silent chunks: {debug['non_silent_chunks']}/{debug['total_chunks']}")
                        print(f"     Chunk RMS mean: {debug['chunk_rms_mean']:.6f}")
                        print(f"     Chunk RMS max: {debug['chunk_rms_max']:.6f}")
                        
                        if include_noise_analysis and source_result.get('noise_analysis'):
                            noise_info = source_result['noise_analysis']
                            print(f"     Noise chunks: {noise_info.get('noise_chunks', 0)} ({noise_info.get('noise_percentage', 0):.1f}%)")
                            if noise_info.get('noise_types'):
                                print(f"     Noise types: {noise_info['noise_types']}")
                        
                        hundred_percent_count += 1
                    else:
                        break
            if hundred_percent_count >= 3:
                break
        
        plt.show()
    
    # Save detailed results to JSON
    results_file = output_dir / "audio_content_results.json"
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable(track_results), f, indent=2, sort_keys=True)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    return stems_below_threshold == 0


def analyze_noise_content(dataset_path, sources=None, chunk_duration=1.0, analysis_limit=None, confidence_threshold=0.8):
    """Analyze dataset for noise content in stems."""
    if sources is None:
        sources = SOURCES
    
    print(f"🔍 Noise Content Analysis")
    print(f"Dataset: {dataset_path}")
    print(f"Chunk duration: {chunk_duration}s")
    print(f"Analysis limit: {analysis_limit or 'None'}")
    print(f"Confidence threshold: {confidence_threshold*100:.0f}%")
    print(f"Sources: {sources}")
    print(f"=" * 60)
    
    root = Path(dataset_path)
    if not root.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find all track directories
    track_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not track_dirs:
        print(f"❌ No track directories found")
        return False
    
    # Limit analysis if requested
    if analysis_limit:
        track_dirs = track_dirs[:analysis_limit]
        print(f"📊 Limiting analysis to first {analysis_limit} tracks")
    
    print(f"📁 Analyzing noise content in {len(track_dirs)} tracks...")
    
    # Results tracking
    noise_results = {}
    total_chunks_analyzed = 0
    noise_chunks_found = 0
    noise_type_counts = Counter()
    track_noise_summaries = {}
    
    chunk_size = int(SR * chunk_duration)  # samples per chunk
    
    with tqdm.tqdm(track_dirs, desc="Analyzing noise content", unit="track") as pbar:
        for track_dir in pbar:
            pbar.set_postfix(track=track_dir.name[:20])
            
            track_noise_results = {}
            track_chunk_count = 0
            track_noise_count = 0
            track_noise_types = Counter()
            
            # Analyze each stem
            for source in sources:
                stem_file = track_dir / f"{source}.wav"
                
                if not stem_file.exists():
                    track_noise_results[source] = {
                        "status": "missing",
                        "chunks_analyzed": 0,
                        "noise_chunks": 0,
                        "noise_types": {},
                        "error": "File not found"
                    }
                    continue
                
                try:
                    # Load audio file
                    waveform, sr = torchaudio.load(str(stem_file))
                    
                    if sr != SR:
                        waveform = torchaudio.functional.resample(waveform, sr, SR)
                        sr = SR
                    
                    # Remove DC offset
                    waveform = waveform - torch.mean(waveform, dim=-1, keepdim=True)
                    
                    # Analyze in chunks
                    total_samples = waveform.shape[-1]
                    source_chunks = 0
                    source_noise_chunks = 0
                    source_noise_types = Counter()
                    chunk_details = []
                    
                    # Process chunks with overlap
                    hop_size = chunk_size // 2  # 50% overlap
                    
                    for start_idx in range(0, total_samples - chunk_size, hop_size):
                        chunk = waveform[..., start_idx:start_idx + chunk_size]
                        
                        # Analyze this chunk
                        chunk_analysis = analyze_audio_chunk_content(chunk, sr, confidence_threshold=confidence_threshold)
                        source_chunks += 1
                        track_chunk_count += 1
                        total_chunks_analyzed += 1
                        
                        if chunk_analysis["type"].startswith("noise_"):
                            source_noise_chunks += 1
                            track_noise_count += 1
                            noise_chunks_found += 1
                            
                            noise_type = chunk_analysis["noise_analysis"]["type"]
                            source_noise_types[noise_type] += 1
                            track_noise_types[noise_type] += 1
                            noise_type_counts[noise_type] += 1
                            
                            # Store details for this chunk
                            chunk_details.append({
                                "start_time": start_idx / sr,
                                "duration": chunk_duration,
                                "noise_type": noise_type,
                                "confidence": chunk_analysis["noise_analysis"]["confidence"],
                                "rms": chunk_analysis["rms"],
                                "indicators": chunk_analysis["noise_analysis"]["indicators"]
                            })
                        
                        # Limit chunks per stem to avoid excessive processing
                        if source_chunks >= 100:  # Max 100 chunks per stem
                            break
                    
                    # Calculate percentages for this source
                    noise_percentage = (source_noise_chunks / source_chunks * 100) if source_chunks > 0 else 0
                    
                    track_noise_results[source] = {
                        "status": "ok",
                        "chunks_analyzed": source_chunks,
                        "noise_chunks": source_noise_chunks,
                        "noise_percentage": noise_percentage,
                        "noise_types": dict(source_noise_types),
                        "chunk_details": chunk_details[:10]  # Keep first 10 noise chunks for details
                    }
                    
                except Exception as e:
                    track_noise_results[source] = {
                        "status": "error",
                        "chunks_analyzed": 0,
                        "noise_chunks": 0,
                        "noise_types": {},
                        "error": str(e)
                    }
            
            # Track summary
            track_noise_percentage = (track_noise_count / track_chunk_count * 100) if track_chunk_count > 0 else 0
            track_noise_summaries[track_dir.name] = {
                "total_chunks": track_chunk_count,
                "noise_chunks": track_noise_count,
                "noise_percentage": track_noise_percentage,
                "noise_types": dict(track_noise_types)
            }
            
            noise_results[track_dir.name] = track_noise_results
    
    # Generate comprehensive report
    print(f"\n📊 Noise Analysis Results:")
    print(f"{'='*60}")
    print(f"Total tracks analyzed: {len(track_dirs)}")
    print(f"Total chunks analyzed: {total_chunks_analyzed}")
    print(f"Chunks containing noise: {noise_chunks_found}")
    print(f"Overall noise rate: {noise_chunks_found/total_chunks_analyzed*100:.1f}%")
    
    # Noise type distribution
    if noise_type_counts:
        print(f"\n🎭 Noise Type Distribution:")
        total_noise_chunks = sum(noise_type_counts.values())
        for noise_type, count in noise_type_counts.most_common():
            percentage = count / total_noise_chunks * 100
            print(f"  {noise_type}: {count} chunks ({percentage:.1f}%)")
    
    # Track-level analysis
    tracks_with_noise = [name for name, summary in track_noise_summaries.items() 
                        if summary["noise_chunks"] > 0]
    
    print(f"\n📁 Track-Level Summary:")
    print(f"Tracks with noise content: {len(tracks_with_noise)}/{len(track_dirs)} ({len(tracks_with_noise)/len(track_dirs)*100:.1f}%)")
    
    if tracks_with_noise:
        print(f"\n🚫 Tracks with significant noise content (>10%):")
        print(f"{'Track':<25} {'Noise %':<10} {'Main Types'}")
        print(f"{'-'*60}")
        
        significant_noise_tracks = []
        for track_name, summary in track_noise_summaries.items():
            if summary["noise_percentage"] > 10:
                significant_noise_tracks.append((track_name, summary))
        
        # Sort by noise percentage (worst first)
        significant_noise_tracks.sort(key=lambda x: x[1]["noise_percentage"], reverse=True)
        
        for track_name, summary in significant_noise_tracks[:20]:  # Show top 20
            main_types = ", ".join([f"{ntype}({count})" for ntype, count in 
                                  Counter(summary["noise_types"]).most_common(3)])
            print(f"{track_name:<25} {summary['noise_percentage']:<9.1f}% {main_types}")
        
        if len(significant_noise_tracks) > 20:
            print(f"  ... and {len(significant_noise_tracks) - 20} more tracks")
    
    # Source-level analysis
    print(f"\n🎵 Noise by Source Type:")
    source_noise_stats = {}
    for source in sources:
        total_source_chunks = 0
        noise_source_chunks = 0
        source_type_counts = Counter()
        
        for track_results in noise_results.values():
            if source in track_results and track_results[source]["status"] == "ok":
                total_source_chunks += track_results[source]["chunks_analyzed"]
                noise_source_chunks += track_results[source]["noise_chunks"]
                for ntype, count in track_results[source]["noise_types"].items():
                    source_type_counts[ntype] += count
        
        if total_source_chunks > 0:
            noise_rate = noise_source_chunks / total_source_chunks * 100
            source_noise_stats[source] = {
                "noise_rate": noise_rate,
                "total_chunks": total_source_chunks,
                "noise_chunks": noise_source_chunks,
                "top_noise_types": source_type_counts.most_common(3)
            }
            
            top_types_str = ", ".join([f"{ntype}({count})" for ntype, count in source_type_counts.most_common(3)])
            print(f"  {source}: {noise_rate:.1f}% noise ({noise_source_chunks}/{total_source_chunks} chunks) - {top_types_str}")
    
    # Save detailed results
    output_dir = root / "noise_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Save main results
    results_file = output_dir / "noise_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable(noise_results), f, indent=2, sort_keys=True)
    
    # Save summary
    summary_data = {
        "analysis_info": {
            "total_tracks": len(track_dirs),
            "total_chunks": total_chunks_analyzed,
            "chunk_duration_seconds": chunk_duration,
            "analysis_limit": analysis_limit
        },
        "overall_stats": {
            "noise_chunks": noise_chunks_found,
            "noise_rate_percent": noise_chunks_found/total_chunks_analyzed*100 if total_chunks_analyzed > 0 else 0,
            "tracks_with_noise": len(tracks_with_noise),
            "tracks_with_noise_percent": len(tracks_with_noise)/len(track_dirs)*100 if track_dirs else 0
        },
        "noise_type_distribution": dict(noise_type_counts),
        "source_stats": source_noise_stats,
        "track_summaries": track_noise_summaries
    }
    
    summary_file = output_dir / "noise_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(convert_to_json_serializable(summary_data), f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"  Detailed results: {results_file}")
    print(f"  Summary: {summary_file}")
    
    # Create visualization if matplotlib is available
    try:
        # Create noise distribution plot
        if noise_type_counts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Noise type distribution
            types = list(noise_type_counts.keys())
            counts = list(noise_type_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            
            wedges, texts, autotexts = ax1.pie(counts, labels=types, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            ax1.set_title("Noise Type Distribution")
            
            # Plot 2: Source noise rates
            if source_noise_stats:
                sources_list = list(source_noise_stats.keys())
                rates = [source_noise_stats[s]["noise_rate"] for s in sources_list]
                
                bars = ax2.bar(sources_list, rates, color='lightcoral', alpha=0.7)
                ax2.set_title("Noise Rate by Source Type")
                ax2.set_ylabel("Noise Rate (%)")
                ax2.set_xlabel("Source")
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, rate in zip(bars, rates):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{rate:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            plot_file = output_dir / "noise_distribution_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"  Plots: {plot_file}")
            
            plt.show()
            
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    overall_noise_rate = noise_chunks_found/total_chunks_analyzed*100 if total_chunks_analyzed > 0 else 0
    
    if overall_noise_rate > 20:
        print(f"  ❌ High noise content ({overall_noise_rate:.1f}%) detected!")
        print(f"     This may significantly impact automix quality.")
        print(f"     Consider noise reduction preprocessing or dataset cleanup.")
    elif overall_noise_rate > 5:
        print(f"  ⚠️  Moderate noise content ({overall_noise_rate:.1f}%) detected.")
        print(f"     Monitor automix results for quality issues.")
    else:
        print(f"  ✅ Low noise content ({overall_noise_rate:.1f}%). Dataset appears clean.")
    
    # Source-specific recommendations
    if source_noise_stats:
        problematic_sources = [s for s, stats in source_noise_stats.items() 
                             if stats["noise_rate"] > 15]
        if problematic_sources:
            print(f"  🎵 Sources with high noise: {', '.join(problematic_sources)}")
            print(f"     Focus noise reduction efforts on these stem types.")
    
    return overall_noise_rate < 10  # Return True if acceptable noise level


def main():
    parser = argparse.ArgumentParser(description="Diagnose automix failures and validate datasets")
    parser.add_argument("command", choices=[
        'dataset', 'analysis', 'suggestions', 'matrix', 
        'validate', 'quick-validate', 'find-problems', 'missing', 'samples', 'repair', 'tempo-pitch', 'audio-content', 'noise'
    ], help="What to analyze")
    parser.add_argument("--dataset-path", default='/Volumes/SAMPLES/datasets/musdb18hq/train',
                       help="Path to dataset")
    parser.add_argument("--strategy", choices=['auto_majority', 'trim_to_shortest', 'pad_to_longest', 'trim_to_majority'], 
                       default='auto_majority', help="Repair strategy for sample count mismatches")
    parser.add_argument("--no-backup", action='store_true', help="Don't backup original files")
    parser.add_argument("--dry-run", action='store_true', help="Show what would be done without making changes")
    parser.add_argument("--threshold", type=float, default=30.0, help="Threshold percentage for significant audio content (default: 30.0)")
    parser.add_argument("--sensitivity", type=float, default=0.01, help="Sensitivity for detecting significant audio (0.01 = 1%% of peak RMS, default: 0.01)")
    parser.add_argument("--chunk-duration", type=float, default=1.0, help="Duration of audio chunks for noise analysis in seconds (default: 1.0)")
    parser.add_argument("--analysis-limit", type=int, help="Limit noise analysis to first N tracks (for faster testing)")
    parser.add_argument("--include-noise", action='store_true', help="Include noise analysis in audio-content command")
    parser.add_argument("--confidence-threshold", type=float, default=0.8, help="Confidence threshold for noise detection (0.0-1.0, default: 0.8)")
    
    args = parser.parse_args()
    
    if args.command == 'dataset':
        analyze_dataset_compatibility(args.dataset_path)
    elif args.command == 'analysis':
        analyze_tempo_pitch_distribution()
    elif args.command == 'suggestions':
        suggest_improvements()
    elif args.command == 'matrix':
        create_compatibility_matrix()
    elif args.command == 'validate':
        validate_for_automix(args.dataset_path, quick_check=False)
    elif args.command == 'quick-validate':
        validate_for_automix(args.dataset_path, quick_check=True)
    elif args.command == 'find-problems':
        find_problematic_files(args.dataset_path)
    elif args.command == 'missing':
        check_missing_files(args.dataset_path)
    elif args.command == 'samples':
        check_sample_counts(args.dataset_path)
    elif args.command == 'repair':
        repair_sample_counts(args.dataset_path, strategy=args.strategy, 
                           backup=not args.no_backup, dry_run=args.dry_run)
    elif args.command == 'tempo-pitch':
        check_tempo_pitch_compatibility(args.dataset_path)
    elif args.command == 'audio-content':
        check_audio_content_percentages(args.dataset_path, threshold=args.threshold, sensitivity=args.sensitivity, include_noise_analysis=args.include_noise, confidence_threshold=args.confidence_threshold)
    elif args.command == 'noise':
        analyze_noise_content(args.dataset_path, chunk_duration=args.chunk_duration, analysis_limit=args.analysis_limit, confidence_threshold=args.confidence_threshold)


if __name__ == '__main__':
    main() 