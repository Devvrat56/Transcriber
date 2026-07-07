"""
Real-time Audio Recorder
Captures audio from microphone in real-time
"""

try:
    import pyaudio
    _HAS_PYAUDIO = True
except ImportError:
    pyaudio = None
    _HAS_PYAUDIO = False

import threading
import queue
from collections import deque
import numpy as np


class AudioRecorder:
    """
    Handles real-time audio capture from microphone
    """
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        """
        Initialize audio recorder
        
        Args:
            sample_rate (int): Audio sample rate in Hz (default: 16000)
            chunk_size (int): Size of audio chunks to record (default: 1024)
            channels (int): Number of audio channels (default: 1 for mono)
        """
        if not _HAS_PYAUDIO:
            raise RuntimeError(
                "pyaudio is not installed. Live recording is unavailable."
            )

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = pyaudio.paFloat32
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=sample_rate * 10)  # 10 second buffer
        self.recording_thread = None
        
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
    
    def start_recording(self):
        """Start recording audio in background thread"""
        if self.is_recording:
            print("Already recording...")
            return
        
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
        self.recording_thread.start()
        print("Recording started...")
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        print("Recording stopped...")
    
    def _record_audio(self):
        """Internal method to record audio stream"""
        try:
            self.stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Add to queue and buffer
                    self.audio_queue.put(audio_data)
                    self.audio_buffer.extend(audio_data)
                    
                except Exception as e:
                    print(f"Error reading audio data: {e}")
                    break
        
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def get_audio_chunk(self, timeout=1):
        """
        Get next audio chunk from queue
        
        Args:
            timeout (int): Queue timeout in seconds
            
        Returns:
            np.ndarray: Audio chunk or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_buffer_data(self):
        """
        Get all buffered audio data
        
        Returns:
            np.ndarray: Buffered audio data
        """
        return np.array(list(self.audio_buffer), dtype=np.float32)
    
    def clear_buffer(self):
        """Clear audio buffer"""
        self.audio_buffer.clear()
    
    def is_audio_available(self):
        """
        Check if audio data is available
        
        Returns:
            bool: True if audio queue has data
        """
        return not self.audio_queue.empty()
    
    def cleanup(self):
        """Cleanup and release resources"""
        self.stop_recording()
        self.pyaudio_instance.terminate()
