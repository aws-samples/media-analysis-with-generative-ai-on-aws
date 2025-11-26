"""
Audio Spectrogram Analyzer for Audio Understanding

This module provides the AudioSpectrogramAnalyzer class for generating and analyzing
audio spectrograms with waveform visualization, feature extraction, and comprehensive
audio analysis capabilities.

Features:
- Mel-spectrogram generation
- Waveform visualization
- Audio feature extraction (MFCC, spectral centroid, tempo, RMS energy)
- Multi-panel visualization with timeline synchronization
- Audio characteristic description

Author: Audio Understanding Team
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal
import base64
from io import BytesIO


class AudioSpectrogramAnalyzer:
    """Generates and analyzes audio spectrograms with timeline information"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.spectrogram_data = None
        self.audio_features = {}
        self.audio_data = None  # Store original audio for waveform display
    
    def extract_audio_from_video(self, video_path, start_time=0, duration=120):
        """Extract audio from video file for spectrogram analysis"""
        try:
            print(f"🎵 Extracting audio for spectrogram analysis...")
            
            # Use librosa to load audio directly from video
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress librosa warnings
                audio_data, sr = librosa.load(
                    video_path, 
                    sr=self.sample_rate, 
                    offset=start_time, 
                    duration=duration
                )
            
            # Store audio data for waveform visualization
            self.audio_data = audio_data
            
            print(f"✅ Audio extracted: {len(audio_data)/sr:.1f}s at {sr}Hz")
            return audio_data, sr
            
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
            return None, None
    
    def generate_spectrogram(self, audio_data, sr):
        """Generate spectrogram with detailed frequency analysis"""
        try:
            print("📊 Generating spectrogram...")
            
            # Generate mel-spectrogram for better visualization
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=sr, 
                n_mels=128, 
                fmax=8000,
                hop_length=512
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Store spectrogram data
            self.spectrogram_data = {
                'mel_spec_db': mel_spec_db,
                'sr': sr,
                'hop_length': 512,
                'duration': len(audio_data) / sr
            }
            
            print(f"✅ Spectrogram generated: {mel_spec_db.shape[1]} time frames")
            return mel_spec_db
            
        except Exception as e:
            print(f"❌ Error generating spectrogram: {e}")
            return None
    
    def analyze_audio_features(self, audio_data, sr):
        """Extract audio features for enhanced analysis"""
        try:
            print("🔍 Analyzing audio features...")
            
            # Extract various audio features
            features = {}
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # MFCC features (important for speech)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            
            # Tempo and rhythm - FIX: Convert to scalar
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
            tempo = tempo.item() if hasattr(tempo, 'item') else float(tempo)  # Convert numpy array to scalar
            
            # RMS energy (loudness)
            rms = librosa.feature.rms(y=audio_data)[0]
            
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'mfcc_means': np.mean(mfccs, axis=1).tolist(),
                'tempo': tempo,
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),
                'duration': len(audio_data) / sr
            }
            
            self.audio_features = features
            print(f"✅ Audio features extracted: {len(features)} features")
            return features
            
        except Exception as e:
            print(f"❌ Error analyzing audio features: {e}")
            return {}
    
    def create_spectrogram_visualization(self, start_time_offset=0):
        """Create a detailed spectrogram and waveform visualization with timeline"""
        if self.spectrogram_data is None or self.audio_data is None:
            print("❌ No spectrogram or audio data available")
            return None
        
        try:
            print("🎨 Creating spectrogram and waveform visualization...")
            
            # Suppress warnings
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            
            # Create figure with 3 subplots: waveform, spectrogram, and RMS energy
            fig, (ax_wave, ax_spec, ax_rms) = plt.subplots(3, 1, figsize=(16, 12))
            
            # Get data
            mel_spec_db = self.spectrogram_data['mel_spec_db']
            sr = self.spectrogram_data['sr']
            hop_length = self.spectrogram_data['hop_length']
            audio_data = self.audio_data
            
            # Create time axes
            time_audio = np.linspace(0, len(audio_data) / sr, len(audio_data)) + start_time_offset
            time_frames = librosa.frames_to_time(
                np.arange(mel_spec_db.shape[1]), 
                sr=sr, 
                hop_length=hop_length
            ) + start_time_offset
            
            # 1. Waveform visualization
            ax_wave.plot(time_audio, audio_data, color='steelblue', linewidth=0.5, alpha=0.8)
            ax_wave.set_title('Audio Waveform (Amplitude over Time)', fontsize=14, fontweight='bold')
            ax_wave.set_ylabel('Amplitude', fontsize=12)
            ax_wave.grid(True, alpha=0.3)
            ax_wave.set_xlim(time_audio[0], time_audio[-1])
            
            # 2. Spectrogram visualization
            img = librosa.display.specshow(
                mel_spec_db, 
                x_axis='time', 
                y_axis='mel', 
                sr=sr, 
                hop_length=hop_length,
                ax=ax_spec,
                cmap='viridis'
            )
            
            ax_spec.set_title('Audio Spectrogram (Frequency over Time)', fontsize=14, fontweight='bold')
            ax_spec.set_ylabel('Mel Frequency (Hz)', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(img, ax=ax_spec, format='%+2.0f dB')
            cbar.set_label('Power (dB)', fontsize=12)
            
            # 3. RMS Energy visualization
            rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
            rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) + start_time_offset
            
            ax_rms.plot(rms_times, rms, color='red', linewidth=2, alpha=0.8)
            ax_rms.fill_between(rms_times, rms, alpha=0.3, color='red')
            ax_rms.set_title('Audio Energy (RMS over Time)', fontsize=14, fontweight='bold')
            ax_rms.set_ylabel('RMS Energy', fontsize=12)
            ax_rms.set_xlabel('Time (seconds)', fontsize=12)
            ax_rms.grid(True, alpha=0.3)
            ax_rms.set_xlim(rms_times[0], rms_times[-1])
            
            # Add time markers every 10 seconds across all plots
            max_time = max(time_audio[-1], time_frames[-1], rms_times[-1])
            for t in range(0, int(max_time) + 1, 10):
                ax_wave.axvline(x=t, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax_spec.axvline(x=t, color='white', linestyle='--', alpha=0.5, linewidth=1)
                ax_rms.axvline(x=t, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add overall title
            fig.suptitle('Audio Analysis: Waveform, Spectrogram & Energy', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Waveform and spectrogram visualization created")
            return fig
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return None
    
    def get_audio_description(self):
        """Generate a textual description of audio characteristics"""
        if not self.audio_features:
            return "No audio analysis available"
        
        features = self.audio_features
        description = []
        
        # Analyze spectral characteristics
        centroid = features.get('spectral_centroid_mean', 0)
        if centroid > 3000:
            description.append("High-frequency content (bright/sharp audio)")
        elif centroid > 1500:
            description.append("Mid-frequency content (typical speech)")
        else:
            description.append("Low-frequency content (deep/bass-heavy audio)")
        
        # Analyze energy
        rms_mean = features.get('rms_mean', 0)
        if rms_mean > 0.1:
            description.append("High energy audio (loud/dynamic)")
        elif rms_mean > 0.05:
            description.append("Moderate energy audio (normal speech levels)")
        else:
            description.append("Low energy audio (quiet/whispered)")
        
        # Analyze tempo - FIX: Handle numpy array conversion
        tempo = features.get('tempo', 0)
        if isinstance(tempo, (list, tuple, np.ndarray)):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0
        elif hasattr(tempo, 'item'):  # numpy scalar
            tempo = float(tempo.item())
        else:
            tempo = float(tempo) if tempo else 0
        
        if tempo > 0:
            description.append(f"Detected rhythm/tempo: {tempo:.1f} BPM")
        
        # Analyze zero crossing rate (speech vs music indicator)
        zcr = features.get('zero_crossing_rate_mean', 0)
        if zcr > 0.1:
            description.append("High zero-crossing rate (likely speech/vocals)")
        else:
            description.append("Low zero-crossing rate (likely music/tonal content)")
        
        return "; ".join(description)