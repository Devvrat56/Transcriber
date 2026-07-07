"""
Audio Worker
Handles audio processing and transcription in real-time
"""

import threading
import queue
from datetime import datetime
import logging
from typing import Callable, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioWorker:
    """
    Real-time audio processing worker
    Handles audio transcription and processing tasks
    """
    
    def __init__(self, transcriber_func: Callable, max_workers=1):
        """
        Initialize audio worker
        
        Args:
            transcriber_func: Function to process audio chunks
            max_workers (int): Number of worker threads
        """
        self.transcriber_func = transcriber_func
        self.max_workers = max_workers
        
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_threads = []
        
        self.is_running = False
        self.jobs = {}  # Track job status
        
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"AudioWorker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
        
        self.is_running = True
        logger.info(f"Started {self.max_workers} audio workers")
    
    def _worker_loop(self):
        """Main worker loop"""
        while self.is_running:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Poison pill
                    break
                
                job_id, audio_data, callback = task
                
                try:
                    # Process audio
                    result = self.transcriber_func(audio_data)
                    
                    # Send result
                    self.result_queue.put({
                        'job_id': job_id,
                        'status': 'completed',
                        'result': result,
                        'timestamp': datetime.now(),
                        'callback': callback
                    })
                    
                    self.jobs[job_id] = 'completed'
                    logger.info(f"Job {job_id} completed")
                
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {str(e)}")
                    self.result_queue.put({
                        'job_id': job_id,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now(),
                        'callback': callback
                    })
                    self.jobs[job_id] = 'failed'
                
                finally:
                    self.task_queue.task_done()
            
            except queue.Empty:
                continue
    
    def submit_task(self, audio_data, job_id=None, callback=None):
        """
        Submit audio for processing
        
        Args:
            audio_data: Audio data to process
            job_id (str): Optional job identifier
            callback: Optional callback function
            
        Returns:
            str: Job ID
        """
        import uuid
        
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = 'queued'
        self.task_queue.put((job_id, audio_data, callback))
        
        logger.info(f"Task submitted with job_id: {job_id}")
        return job_id
    
    def get_result(self, timeout=5):
        """
        Get processing result
        
        Args:
            timeout (int): Timeout in seconds
            
        Returns:
            dict: Result or None
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_job_status(self, job_id):
        """
        Get status of a job
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            str: Job status (queued, processing, completed, failed)
        """
        return self.jobs.get(job_id, 'unknown')
    
    def wait_for_completion(self, timeout=None):
        """
        Wait for all tasks to complete
        
        Args:
            timeout (float): Timeout in seconds
            
        Returns:
            bool: True if all tasks completed
        """
        try:
            self.task_queue.join()
            return True
        except Exception as e:
            logger.error(f"Error waiting for completion: {e}")
            return False
    
    def get_stats(self):
        """
        Get worker statistics
        
        Returns:
            dict: Worker stats
        """
        completed = sum(1 for s in self.jobs.values() if s == 'completed')
        failed = sum(1 for s in self.jobs.values() if s == 'failed')
        queued = sum(1 for s in self.jobs.values() if s == 'queued')
        
        return {
            'total_jobs': len(self.jobs),
            'completed': completed,
            'failed': failed,
            'queued': queued,
            'queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }
    
    def stop(self):
        """Stop all worker threads"""
        self.is_running = False
        
        # Send poison pills
        for _ in range(self.max_workers):
            self.task_queue.put(None)
        
        # Wait for threads
        for thread in self.worker_threads:
            thread.join(timeout=2)
        
        logger.info("Audio workers stopped")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
