"""
Stream Monitor - Centralized stream-end detection
Module-specific component for Modality Fusion Understanding
"""

import time

# Import shared components
try:
    from src.shared import log_component
except ImportError:
    def log_component(component, message, level="INFO"):
        print(f"[{component}] {message}")


class StreamMonitor:
    """
    Monitors component activity to detect stream end centrally.
    
    This monitor:
    - Tracks chunk processor activity
    - Tracks transcription processor activity
    - Detects stream end based on inactivity
    - Provides centralized stream status
    
    Key Features:
    - Activity-based stream detection
    - Configurable timeout
    - Tracks multiple component states
    - Prevents premature shutdown
    """
    
    def __init__(self, chunk_processor, transcription_processor, stream_timeout=60):
        """
        Initialize stream monitor.
        
        Args:
            chunk_processor: ChunkProcessor instance to monitor
            transcription_processor: TranscriptionProcessor instance to monitor
            stream_timeout: Seconds of inactivity before considering stream ended
        """
        self.chunk_processor = chunk_processor
        self.transcription_processor = transcription_processor
        self.stream_timeout = stream_timeout
        self.last_activity_time = time.time()
        self.last_chunk_count = 0
        self.transcription_was_running = False  # Track if transcription was ever running
        
    def update_activity(self):
        """
        Check for component activity and update last activity time.
        
        Returns:
            bool: True if activity detected, False otherwise
        """
        current_time = time.time()
        
        # Check if new chunks are being processed
        current_chunk_count = getattr(self.chunk_processor, 'chunk_count', 0)
        if current_chunk_count > self.last_chunk_count:
            self.last_activity_time = current_time
            self.last_chunk_count = current_chunk_count
            return True
            
        # Note: We don't check transcription is_running here because:
        # - Transcription can timeout (15s) while video continues
        # - We want to keep processing video even if audio stops
        # - Only chunk activity indicates the video stream is alive
            
        return False
        
    def stream_appears_ended(self):
        """
        Check if stream appears to have ended based on component activity.
        
        This method checks:
        1. If transcription has stopped (after it was running)
        2. If activity timeout has been exceeded
        
        Returns:
            bool: True if stream appears ended, False otherwise
        """
        # First check if transcription has stopped (only if it was previously running)
        if (self.transcription_was_running and
            hasattr(self.transcription_processor, 'is_running') and 
            not self.transcription_processor.is_running):
            return True
            
        # Then check activity timeout
        self.update_activity()
        time_since_activity = time.time() - self.last_activity_time
        return time_since_activity >= self.stream_timeout
    
    def get_status(self):
        """
        Get current stream status information.
        
        Returns:
            dict: Status information including:
                - stream_active: Whether stream appears active
                - time_since_activity: Seconds since last activity
                - chunk_count: Number of chunks processed
                - transcription_running: Whether transcription is active
        """
        self.update_activity()
        
        return {
            'stream_active': not self.stream_appears_ended(),
            'time_since_activity': time.time() - self.last_activity_time,
            'chunk_count': self.last_chunk_count,
            'transcription_running': (
                hasattr(self.transcription_processor, 'is_running') and 
                self.transcription_processor.is_running
            ),
            'transcription_was_running': self.transcription_was_running
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Stream Monitor - Example Usage")
    print("=" * 80)
    
    print("\nExample: Monitor stream activity and detect end")
    print("-" * 80)
    print("""
from components import StreamMonitor

monitor = StreamMonitor(
    chunk_processor=chunk_processor,
    transcription_processor=transcription_processor,
    stream_timeout=60
)

# Check if stream has ended
if monitor.stream_appears_ended():
    print("Stream has ended, shutting down...")
    
# Get detailed status
status = monitor.get_status()
print(f"Stream active: {status['stream_active']}")
print(f"Chunks processed: {status['chunk_count']}")
    """)
    
    print("\n" + "=" * 80)
    print("✅ Stream Monitor module ready for use!")
    print("=" * 80)
