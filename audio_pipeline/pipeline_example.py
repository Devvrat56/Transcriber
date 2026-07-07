"""
Integration Example: Audio Pipeline with Groq Transcription
Shows how to use the audio pipeline for real-time transcription
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from groq import Groq
from audio_pipeline import AudioUploadHandler, AudioWorker, AudioUtils
from audio_pipeline.audio_catcher import AudioRecorder
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class TranscriptionPipeline:
    """
    Complete audio transcription pipeline
    Combines upload, recording, and real-time processing
    """
    
    def __init__(self, api_key=None):
        """
        Initialize transcription pipeline
        
        Args:
            api_key (str): Groq API key
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        
        # Initialize components
        self.upload_handler = AudioUploadHandler(upload_dir='./uploads')
        self.audio_recorder = AudioRecorder()
        self.audio_worker = AudioWorker(
            transcriber_func=self.transcribe_audio,
            max_workers=2
        )
    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio using Groq Whisper API
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            str: Transcription text
        """
        try:
            with open(audio_file_path, 'rb') as f:
                transcript = self.client.audio.transcriptions.create(
                    file=(Path(audio_file_path).name, f, 'audio/mpeg'),
                    model='whisper-large-v3-turbo',
                    language='en'
                )
            return transcript.text
        except Exception as e:
            print(f"Transcription error: {e}")
            raise
    
    def process_uploaded_file(self, file_path):
        """
        Process an uploaded audio file
        
        Args:
            file_path (str): Path to uploaded file
            
        Returns:
            dict: Processing result
        """
        # Validate file
        validation = self.upload_handler.validate_audio_file(file_path)
        if not validation['valid']:
            return {'success': False, 'error': validation['error']}
        
        # Save upload
        save_result = self.upload_handler.save_upload(file_path)
        if not save_result['success']:
            return save_result
        
        saved_path = save_result['path']
        
        # Submit for transcription
        job_id = self.audio_worker.submit_task(saved_path)
        
        return {
            'success': True,
            'job_id': job_id,
            'path': saved_path,
            'filename': save_result['filename']
        }
    
    def get_transcription_result(self, timeout=60):
        """
        Get transcription result
        
        Args:
            timeout (int): Timeout in seconds
            
        Returns:
            dict: Transcription result
        """
        result = self.audio_worker.get_result(timeout=timeout)
        
        if result:
            job_id = result['job_id']
            if result['status'] == 'completed':
                # Move file to processed
                self.upload_handler.move_to_processed(result.get('path', ''))
                return {
                    'success': True,
                    'job_id': job_id,
                    'transcription': result['result']
                }
            else:
                # Move file to failed
                self.upload_handler.move_to_failed(
                    result.get('path', ''),
                    reason=result.get('error', 'Unknown error')
                )
                return {
                    'success': False,
                    'job_id': job_id,
                    'error': result.get('error', 'Transcription failed')
                }
        
        return {'success': False, 'error': 'No result available'}
    
    def start_live_transcription(self):
        """Start live audio recording and transcription"""
        print("Starting live transcription...")
        self.audio_recorder.start_recording()
        return True
    
    def stop_live_transcription(self):
        """Stop live audio recording"""
        print("Stopping live transcription...")
        self.audio_recorder.stop_recording()
    
    def get_stats(self):
        """
        Get pipeline statistics
        
        Returns:
            dict: Pipeline stats
        """
        return {
            'worker_stats': self.audio_worker.get_stats(),
            'buffer_status': self.audio_recorder.is_audio_available()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.audio_recorder.cleanup()
        self.audio_worker.stop()


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TranscriptionPipeline()
    
    # Example 1: Process uploaded file
    print("\n=== Example 1: Upload and Transcribe ===")
    # Replace with actual file path
    # result = pipeline.process_uploaded_file('/path/to/audio.mp3')
    # print(f"Upload result: {result}")
    
    # Example 2: Live transcription
    print("\n=== Example 2: Live Transcription ===")
    # pipeline.start_live_transcription()
    # ... record for some time ...
    # pipeline.stop_live_transcription()
    
    print("\nPipeline initialized successfully!")
    print(f"Stats: {pipeline.get_stats()}")
    
    # Cleanup
    pipeline.cleanup()
