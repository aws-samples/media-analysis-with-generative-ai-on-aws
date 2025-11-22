"""
Chunk Monitor - Monitors and processes video chunks independently
Separate from ChunkProcessor for better Jupyter compatibility
"""

import os
import sys
import time
import glob
import re
import threading
import subprocess
import json

# Import shared components
try:
    from src.shared import (
        log_component,
        FilmstripProcessor,
        create_fusion_filmstrip_processor,
        create_fusion_detector
    )
except ImportError:
    def log_component(component, message, level="INFO"):
        print(f"[{component}] {message}")
    FilmstripProcessor = None
    create_fusion_filmstrip_processor = None
    create_fusion_detector = None

# Import Jupyter compatibility
try:
    from .jupyter_compat import create_daemon_thread, is_jupyter
except ImportError:
    def create_daemon_thread(target, name=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return threading.Thread(target=target, name=name, args=args, kwargs=kwargs, daemon=True)
    
    def is_jupyter():
        return False


class ChunkMonitor:
    """
    Monitors a directory for new video chunks and processes them.
    
    This is a standalone component that can run independently of ChunkProcessor.
    It watches for new chunk files, creates filmstrips, and triggers analysis.
    
    Designed to work reliably in Jupyter notebooks with proper async handling.
    """
    
    def __init__(
        self,
        output_dir,
        chunk_duration=20,
        fusion_analyzer=None,
        shot_detector=None,
        filmstrip_processor=None,
        check_interval=0.5
    ):
        """
        Initialize chunk monitor.
        
        Args:
            output_dir: Directory to monitor for chunks
            chunk_duration: Expected duration of chunks (for filename pattern)
            fusion_analyzer: Optional FusionAnalyzer for automatic analysis
            shot_detector: Optional ShotChangeDetector
            filmstrip_processor: Optional FilmstripProcessor
            check_interval: How often to check for new chunks (seconds)
        """
        log_component("ChunkMonitor", f"🔍 Initializing ChunkMonitor", "DEBUG")
        
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.fusion_analyzer = fusion_analyzer
        self.check_interval = check_interval
        self.is_running = False
        self.chunk_count = 0
        self.processed_chunks = set()
        
        # Initialize shot detector
        if shot_detector is None and create_fusion_detector:
            shot_detector = create_fusion_detector(threshold=0.7)
        
        # Initialize filmstrip processor
        if filmstrip_processor is None:
            if create_fusion_filmstrip_processor:
                try:
                    self.filmstrip_processor = create_fusion_filmstrip_processor(shot_detector=shot_detector)
                    log_component("ChunkMonitor", "✅ FilmstripProcessor initialized")
                except Exception as e:
                    log_component("ChunkMonitor", f"❌ Failed to initialize FilmstripProcessor: {e}", "ERROR")
                    self.filmstrip_processor = None
            else:
                log_component("ChunkMonitor", "⚠️ FilmstripProcessor not available", "WARNING")
                self.filmstrip_processor = None
        else:
            self.filmstrip_processor = filmstrip_processor
            log_component("ChunkMonitor", "✅ Using provided FilmstripProcessor", "DEBUG")
        
        log_component("ChunkMonitor", "✅ ChunkMonitor initialization complete", "DEBUG")
    
    def start_monitoring(self):
        """Start monitoring for new chunks"""
        if self.is_running:
            log_component("ChunkMonitor", "⚠️ Already running", "WARNING")
            return
        
        self.is_running = True
        
        if is_jupyter():
            log_component("ChunkMonitor", "🔧 Detected Jupyter environment - using compatible threading", "DEBUG")
        
        # Start monitoring thread
        self.monitor_thread = create_daemon_thread(
            target=self._monitor_loop,
            name="ChunkMonitor-Main"
        )
        self.monitor_thread.start()
        
        log_component("ChunkMonitor", f"✅ Started monitoring for chunks: {self.output_dir}/chunks/")
        
        if is_jupyter():
            log_component("ChunkMonitor", f"   Monitor thread: {self.monitor_thread.name} (alive={self.monitor_thread.is_alive()})", "DEBUG")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        log_component("ChunkMonitor", "🔄 Monitoring loop started", "DEBUG")
        log_component("ChunkMonitor", f"Thread ID={threading.current_thread().ident}, Name={threading.current_thread().name}", "DEBUG")
        
        try:
            while self.is_running:
                # Check for new chunk files
                search_pattern = f"{self.output_dir}/chunks/chunk_*_{self.chunk_duration}s.mp4"
                chunk_files = glob.glob(search_pattern)
                # Heartbeat logging every 5 seconds
                current_time = time.time()
                if not hasattr(self, '_last_heartbeat') or (current_time - self._last_heartbeat) > 5:
                    log_component("ChunkMonitor", f"💓 Heartbeat: {len(chunk_files)} files found, {len(self.processed_chunks)} processed", "DEBUG")
                    log_component("ChunkMonitor", f"❤️ Thread alive, is_running={self.is_running}", "DEBUG")
                    self._last_heartbeat = current_time
                # Process new chunks
                for chunk_file in sorted(chunk_files):
                    chunk_match = re.search(r'chunk_(\d+)_', chunk_file)
                    if chunk_match:
                        chunk_id = int(chunk_match.group(1))
                        
                        if chunk_id not in self.processed_chunks and os.path.exists(chunk_file):
                            if self._verify_chunk_ready(chunk_file):
                                log_component("ChunkMonitor", f"📁 Processing chunk {chunk_id}: {chunk_file}")
                                self._process_chunk(chunk_file, chunk_id)
                                self.processed_chunks.add(chunk_id)
                                self.chunk_count = max(self.chunk_count, chunk_id + 1)
                
                # Sleep with Jupyter-compatible approach
                self._jupyter_safe_sleep(self.check_interval)
                
            log_component("ChunkMonitor", "🛑 Monitoring loop exited")
            
        except Exception as e:
            log_component("ChunkMonitor", f"❌ Monitoring error: {e}", "ERROR")
            import traceback
            log_component("ChunkMonitor", f"   Traceback: {traceback.format_exc()}", "ERROR")
    
    def _jupyter_safe_sleep(self, duration):
        """Sleep in a way that's safe for Jupyter notebooks"""
        if is_jupyter():
            # Break sleep into small chunks with I/O operations
            num_chunks = int(duration / 0.1)
            for _ in range(num_chunks):
                time.sleep(0.1)
                sys.stdout.flush()
            # Handle remainder
            remainder = duration - (num_chunks * 0.1)
            if remainder > 0:
                time.sleep(remainder)
                sys.stdout.flush()
        else:
            time.sleep(duration)
    
    def _verify_chunk_ready(self, file_path, is_final=False):
        """Verify chunk is ready by checking file stability and duration"""
        if not os.path.exists(file_path):
            return False
        
        try:
            # Check minimum file size
            if os.path.getsize(file_path) < 50000:
                return False
            
            # Use ffprobe to check if file is readable and get duration
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', file_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
            
            if probe_result.returncode != 0:
                return False
            
            # Parse duration from ffprobe output
            probe_data = json.loads(probe_result.stdout)
            duration = float(probe_data.get('format', {}).get('duration', 0))
            
            if is_final:
                # For final chunks, accept any reasonable duration (minimum 1 second)
                return duration >= 1.0
            else:
                # For regular chunks, check if duration matches expected
                expected_duration = self.chunk_duration
                duration_diff = abs(duration - expected_duration)
                return duration_diff <= 1.0  # Allow 1 second tolerance
        
        except Exception as e:
            log_component("ChunkMonitor", f"⚠️ Error verifying chunk: {e}", "WARNING")
            return False
    
    def _process_chunk(self, video_file, chunk_id):
        """Process a chunk: create filmstrip and trigger analysis"""
        try:
            # Create filmstrip
            filmstrip_path = f"{self.output_dir}/filmstrips/filmstrip_{chunk_id:04d}_4x5.jpg"
            
            if self.filmstrip_processor:
                shot_change_frames = self.filmstrip_processor.create_filmstrip_from_video(
                    video_file=video_file,
                    output_path=filmstrip_path,
                    start_time=chunk_id * self.chunk_duration,
                    num_frames=20,
                    interval=1.0,
                    detect_shot_changes=True
                )
                log_component("ChunkMonitor", f"   ✅ Filmstrip created: {filmstrip_path}", "DEBUG")
            else:
                log_component("ChunkMonitor", "   ⚠️ No filmstrip processor - skipping filmstrip", "WARNING")
                shot_change_frames = []
            
            # Trigger analysis
            if self.fusion_analyzer:
                start_time = chunk_id * self.chunk_duration
                end_time = start_time + self.chunk_duration
                self.fusion_analyzer.queue_analysis(chunk_id, filmstrip_path, start_time, end_time, shot_change_frames)
                log_component("ChunkMonitor", f"   🔄 Queued fusion analysis for chunk {chunk_id}", "DEBUG")
            
        except Exception as e:
            log_component("ChunkMonitor", f"❌ Error processing chunk {chunk_id}: {e}", "ERROR")
            import traceback
            log_component("ChunkMonitor", f"   Traceback: {traceback.format_exc()}", "ERROR")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        log_component("ChunkMonitor", "🛑 Stopping chunk monitoring...", "DEBUG")
        
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            log_component("ChunkMonitor", "   ⏳ Waiting for monitor thread...")
            self.monitor_thread.join(timeout=5)
            if self.monitor_thread.is_alive():
                log_component("ChunkMonitor", "   ⚠️ Thread still running (daemon will exit)", "WARNING")
            else:
                log_component("ChunkMonitor", "   ✅ Thread stopped")
        
        log_component("ChunkMonitor", f"🛑 Monitoring stopped ({self.chunk_count} chunks processed)", "DEBUG")
    
    def get_status(self):
        """Get current monitoring status"""
        return {
            'is_running': self.is_running,
            'chunk_count': self.chunk_count,
            'processed_chunks': len(self.processed_chunks),
            'thread_alive': hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive()
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Chunk Monitor - Example Usage")
    print("=" * 80)
    
    print("\nExample: Monitor directory for new chunks")
    print("-" * 80)
    print("""
from components import ChunkMonitor

monitor = ChunkMonitor(
    output_dir="output",
    chunk_duration=20,
    fusion_analyzer=analyzer
)

monitor.start_monitoring()
# ... chunks are processed automatically as they appear ...
monitor.stop_monitoring()
    """)
    
    print("\n" + "=" * 80)
    print("✅ Chunk Monitor module ready for use!")
    print("=" * 80)
