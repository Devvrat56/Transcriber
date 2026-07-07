"""
Audio Upload Handler
Manages file uploads and validation
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import hashlib


class AudioUploadHandler:
    """
    Handles audio file uploads, validation, and storage
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
    
    def __init__(self, upload_dir='./uploads'):
        """
        Initialize upload handler
        
        Args:
            upload_dir (str): Directory to store uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.temp_dir = self.upload_dir / 'temp'
        self.processed_dir = self.upload_dir / 'processed'
        self.failed_dir = self.upload_dir / 'failed'
        
        for directory in [self.temp_dir, self.processed_dir, self.failed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_audio_file(self, file_path):
        """
        Validate audio file
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            dict: Validation result with status and message
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            return {'valid': False, 'error': 'File does not exist'}
        
        # Check file extension
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return {
                'valid': False,
                'error': f'Unsupported format. Supported: {", ".join(self.SUPPORTED_FORMATS)}'
            }
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return {
                'valid': False,
                'error': f'File too large. Max: {self.MAX_FILE_SIZE / (1024*1024):.0f}MB'
            }
        
        if file_size == 0:
            return {'valid': False, 'error': 'File is empty'}
        
        return {'valid': True, 'file_size': file_size}
    
    def save_upload(self, file_path, custom_name=None):
        """
        Save uploaded file to storage
        
        Args:
            file_path (str): Path to uploaded file
            custom_name (str): Custom name for saved file
            
        Returns:
            dict: Save result with status and new file path
        """
        file_path = Path(file_path)
        
        # Validate file
        validation = self.validate_audio_file(file_path)
        if not validation['valid']:
            return {'success': False, 'error': validation['error']}
        
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_hash = self._get_file_hash(file_path)[:8]
            
            if custom_name:
                base_name = Path(custom_name).stem
            else:
                base_name = file_path.stem
            
            new_filename = f"{timestamp}_{file_hash}_{base_name}{file_path.suffix}"
            new_path = self.temp_dir / new_filename
            
            # Copy file
            shutil.copy2(file_path, new_path)
            
            return {
                'success': True,
                'path': str(new_path),
                'filename': new_filename,
                'size': validation['file_size']
            }
        
        except Exception as e:
            return {'success': False, 'error': f'Upload failed: {str(e)}'}
    
    def move_to_processed(self, file_path):
        """
        Move file from temp to processed directory
        
        Args:
            file_path (str): Path to file
            
        Returns:
            dict: Move result
        """
        try:
            file_path = Path(file_path)
            new_path = self.processed_dir / file_path.name
            shutil.move(str(file_path), str(new_path))
            return {'success': True, 'path': str(new_path)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def move_to_failed(self, file_path, reason=''):
        """
        Move file from temp to failed directory
        
        Args:
            file_path (str): Path to file
            reason (str): Reason for failure
            
        Returns:
            dict: Move result
        """
        try:
            file_path = Path(file_path)
            new_path = self.failed_dir / file_path.name
            shutil.move(str(file_path), str(new_path))
            
            # Save failure log
            log_path = self.failed_dir / f"{file_path.stem}_error.txt"
            with open(log_path, 'w') as f:
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Original file: {file_path.name}\n")
                f.write(f"Reason: {reason}\n")
            
            return {'success': True, 'path': str(new_path)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def cleanup_temp(self, days_old=7):
        """
        Cleanup old temporary files
        
        Args:
            days_old (int): Delete files older than this many days
            
        Returns:
            dict: Cleanup result
        """
        from time import time
        current_time = time()
        cutoff_time = current_time - (days_old * 86400)
        
        deleted_count = 0
        try:
            for file in self.temp_dir.glob('*'):
                if file.stat().st_mtime < cutoff_time:
                    file.unlink()
                    deleted_count += 1
            
            return {'success': True, 'deleted_count': deleted_count}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _get_file_hash(file_path, chunk_size=8192):
        """
        Generate MD5 hash of file
        
        Args:
            file_path (str): Path to file
            chunk_size (int): Chunk size for reading
            
        Returns:
            str: File hash
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
