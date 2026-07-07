"""
Audio Utilities
Helper functions for audio processing
"""

import numpy as np
import os
from pathlib import Path


class AudioUtils:
    """
    Utility functions for audio processing
    """
    
    @staticmethod
    def convert_audio_format(input_path, output_path, target_sample_rate=16000):
        """
        Convert audio to target format
        Note: Requires librosa or ffmpeg
        
        Args:
            input_path (str): Input audio file path
            output_path (str): Output audio file path
            target_sample_rate (int): Target sample rate
            
        Returns:
            dict: Conversion result
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            y, sr = librosa.load(input_path, sr=target_sample_rate)
            
            # Save converted audio
            sf.write(output_path, y, sr)
            
            return {
                'success': True,
                'output_path': output_path,
                'sample_rate': sr,
                'duration': len(y) / sr
            }
        except ImportError:
            return {
                'success': False,
                'error': 'librosa or soundfile not installed. Install with: pip install librosa soundfile'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_audio_info(file_path):
        """
        Get audio file information
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            dict: Audio information
        """
        try:
            import librosa
            
            y, sr = librosa.load(file_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                'success': True,
                'sample_rate': sr,
                'duration': duration,
                'file_size': os.path.getsize(file_path),
                'format': Path(file_path).suffix
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def normalize_audio(audio_data, target_level=-20.0):
        """
        Normalize audio level
        
        Args:
            audio_data (np.ndarray): Audio data
            target_level (float): Target level in dB
            
        Returns:
            np.ndarray: Normalized audio
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Avoid division by zero
        if rms == 0:
            return audio_data
        
        # Convert target level from dB to linear
        target_rms = 10 ** (target_level / 20.0)
        
        # Calculate normalization factor
        factor = target_rms / rms
        
        return audio_data * factor
    
    @staticmethod
    def apply_noise_reduction(audio_data, threshold=0.02):
        """
        Simple noise reduction using threshold
        
        Args:
            audio_data (np.ndarray): Audio data
            threshold (float): Noise threshold
            
        Returns:
            np.ndarray: Denoised audio
        """
        # Create binary mask
        mask = np.abs(audio_data) > threshold
        
        # Apply mask
        denoised = audio_data * mask
        
        return denoised
    
    @staticmethod
    def split_audio_chunks(audio_data, chunk_duration=30, sample_rate=16000):
        """
        Split audio into chunks
        
        Args:
            audio_data (np.ndarray): Audio data
            chunk_duration (int): Duration of each chunk in seconds
            sample_rate (int): Sample rate
            
        Returns:
            list: List of audio chunks
        """
        chunk_size = chunk_duration * sample_rate
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def detect_silence(audio_data, threshold=0.01, min_duration=0.5, sample_rate=16000):
        """
        Detect silent sections in audio
        
        Args:
            audio_data (np.ndarray): Audio data
            threshold (float): Silence threshold
            min_duration (float): Minimum silence duration in seconds
            sample_rate (int): Sample rate
            
        Returns:
            list: List of (start, end) tuples for silent sections
        """
        # Simple energy-based detection
        rms = np.sqrt(np.convolve(np.square(audio_data), np.ones(1024) / 1024, mode='valid'))
        
        silent_frames = rms < threshold
        min_frames = int(min_duration * sample_rate / 1024)
        
        silent_sections = []
        in_silence = False
        start = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_silence:
                start = i
                in_silence = True
            elif not is_silent and in_silence:
                if i - start >= min_frames:
                    silent_sections.append((start, i))
                in_silence = False
        
        return silent_sections
    
    @staticmethod
    def estimate_processing_time(audio_duration, real_time_factor=0.1):
        """
        Estimate processing time
        
        Args:
            audio_duration (float): Duration of audio in seconds
            real_time_factor (float): Processing speed relative to real-time
            
        Returns:
            float: Estimated processing time in seconds
        """
        return audio_duration * real_time_factor
    
    @staticmethod
    def validate_audio_data(audio_data):
        """
        Validate audio data
        
        Args:
            audio_data (np.ndarray): Audio data to validate
            
        Returns:
            dict: Validation result
        """
        if not isinstance(audio_data, np.ndarray):
            return {'valid': False, 'error': 'Audio data must be numpy array'}
        
        if audio_data.size == 0:
            return {'valid': False, 'error': 'Audio data is empty'}
        
        if np.all(audio_data == 0):
            return {'valid': False, 'error': 'Audio data is silent'}
        
        if np.any(np.isnan(audio_data)):
            return {'valid': False, 'error': 'Audio data contains NaN values'}
        
        return {'valid': True, 'size': audio_data.size, 'dtype': str(audio_data.dtype)}
