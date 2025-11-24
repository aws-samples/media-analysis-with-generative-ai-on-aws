"""
Chunk Processor - Processes video into 20-second chunks with filmstrips
Module-specific component for Modality Fusion Understanding
"""

import subprocess
import time
import os
import sys
import queue
import threading
import json
import glob
import re
from .audio_spectrogram_analyzer import AudioSpectrogramAnalyzer

# Import shared components
try:
    from src.shared import (
        log_component, 
        ShotChangeDetector, 
        create_fusion_detector,
        FilmstripProcessor,
        create_fusion_filmstrip_processor
    )
except ImportError:
    # Fallback if shared components not available
    def log_component(component, message, level="INFO"):
        print(f"[{component}] {message}")
    ShotChangeDetector = None
    create_fusion_detector = None
    FilmstripProcessor = None
    create_fusion_filmstrip_processor = None

# Import Jupyter compatibility utilities
try:
    from .jupyter_compat import create_daemon_thread, keep_thread_alive_wrapper, is_jupyter
except ImportError:
    # Fallback if not available
    def create_daemon_thread(target, name=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return threading.Thread(target=target, name=name, args=args, kwargs=kwargs, daemon=True)
    
    def keep_thread_alive_wrapper(func):
        return func
    
    def is_jupyter():
        return False


class ChunkProcessor:
    """
    Processes video into 20-second chunks with audio and filmstrips.
    
    This processor:
    - Captures UDP video stream using FFmpeg
    - Segments into 20-second chunks
    - Extracts 20 frames per chunk (1 fps)
    - Creates enhanced 4×5 filmstrip grids
    - Detects shot changes within and across chunks
    - Triggers fusion analysis automatically
    
    Key Features:
    - Buffered continuous processing
    - Cross-chunk shot detection
    - Enhanced filmstrip with borders and labels
    - Automatic analysis triggering
    - Graceful shutdown handling
    """
    
    def __init__(
        self, 
        udp_port, 
        output_dir, 
        chunk_duration=20, 
        stream_timeout=60, 
        fusion_analyzer=None,
        shot_detector=None,
        filmstrip_processor=None
    ):
        """
        Initialize chunk processor.
        
        Args:
            udp_port: UDP port for video stream
            output_dir: Output directory for chunks and filmstrips
            chunk_duration: Duration of each chunk in seconds (default: 20)
            stream_timeout: Stop if no data for this many seconds
            fusion_analyzer: Optional FusionAnalyzer for automatic analysis
            shot_detector: Optional ShotChangeDetector (creates default if None)
            filmstrip_processor: Optional FilmstripProcessor (creates default if None)
        """
        log_component("ChunkProcessor", f"🎬 Initializing ChunkProcessor (UDP port: {udp_port})")
        
        self.udp_port = udp_port
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.chunk_count = 0
        self.is_running = False
        self.stream_timeout = stream_timeout
        self.last_successful_chunk_time = None
        self.fusion_analyzer = fusion_analyzer
        self._stop_event = threading.Event()  # Use Event instead of sleep for better thread control
        
        # Initialize shot detector
        if shot_detector is None and create_fusion_detector:
            shot_detector = create_fusion_detector(threshold=0.7)
        
        # Initialize filmstrip processor
        if filmstrip_processor is None:
            if create_fusion_filmstrip_processor:
                try:
                    self.filmstrip_processor = create_fusion_filmstrip_processor(shot_detector=shot_detector)
                    log_component("ChunkProcessor", "✅ FilmstripProcessor initialized", "DEBUG")
                except Exception as e:
                    log_component("ChunkProcessor", f"❌ Failed to initialize FilmstripProcessor: {e}", "ERROR")
                    self.filmstrip_processor = None
            else:
                log_component("ChunkProcessor", "⚠️ create_fusion_filmstrip_processor not available - filmstrips will not be created", "WARNING")
                self.filmstrip_processor = None
        else:
            self.filmstrip_processor = filmstrip_processor
            log_component("ChunkProcessor", "✅ Using provided FilmstripProcessor", "DEBUG")
        
        # Initialize spectrogram analyzer
        try:
            self.spectrogram_analyzer = AudioSpectrogramAnalyzer()
            log_component("ChunkProcessor", "✅ AudioSpectrogramAnalyzer initialized", "DEBUG")
        except Exception as e:
            log_component("ChunkProcessor", f"⚠️ Failed to initialize AudioSpectrogramAnalyzer: {e}", "WARNING")
            self.spectrogram_analyzer = None
        
        log_component("ChunkProcessor", "✅ ChunkProcessor initialization complete", "DEBUG")
        
    def start_processing(self):
        """Start FFmpeg chunking process"""
        if self.is_running:
            return
            
        self.is_running = True
        
        if is_jupyter():
            log_component("ChunkProcessor", "🔧 Detected Jupyter environment", "DEBUG")
        
        # Start FFmpeg in its own thread
        self.ffmpeg_thread = create_daemon_thread(
            target=self._run_ffmpeg,
            name="ChunkProcessor-FFmpeg"
        )
        self.ffmpeg_thread.start()
        
        log_component("ChunkProcessor", f"✅ Started chunk processor on UDP port {self.udp_port}")
        log_component("ChunkProcessor", f"   💡 Use ChunkMonitor separately to process chunks", "DEBUG")
    
    def _run_ffmpeg(self):
        """Run FFmpeg process to chunk the UDP stream"""
        udp_url = f"udp://127.0.0.1:{self.udp_port}"
        
        log_component("ChunkProcessor", "🎥 Starting FFmpeg chunking process...", "DEBUG")
        
        # Optimized segment approach with low-latency and buffer flushing
        cmd = [
            'ffmpeg', '-i', udp_url,
            '-fflags', '+flush_packets',  # Flush packets immediately
            '-flush_packets', '1',        # Enable packet flushing
            '-max_delay', '0',            # Minimize delay
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-c:a', 'aac', '-ac', '2', '-ar', '48000',
            '-force_key_frames', f'expr:gte(t,n_forced*{self.chunk_duration})',
            '-f', 'segment',
            '-segment_time', str(self.chunk_duration),
            '-segment_format', 'mp4',
            '-segment_start_number', '0',
            '-reset_timestamps', '1',
            '-avoid_negative_ts', 'make_zero',
            '-movflags', '+faststart+frag_keyframe+empty_moov',
            '-segment_format_options', 'movflags=+faststart+frag_keyframe+empty_moov',
            f'{self.output_dir}/chunks/chunk_%04d_{self.chunk_duration}s.mp4'
        ]
        
        try:
            # Start FFmpeg process for continuous chunking
            self.ffmpeg_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            log_component("ChunkProcessor", f"FFmpeg process started (PID: {self.ffmpeg_process.pid})", "DEBUG")
            log_component("ChunkProcessor", f"✅ FFmpeg process running (PID: {self.ffmpeg_process.pid})", "DEBUG")
            
            # Wait for FFmpeg to complete or be stopped
            self.ffmpeg_process.wait()
            
            log_component("ChunkProcessor", f"⚠️ FFmpeg process ended with return code: {self.ffmpeg_process.returncode}", "WARNING")
            
        except Exception as e:
            log_component("ChunkProcessor", f"❌ FFmpeg error: {e}", "ERROR")
            import traceback
            log_component("ChunkProcessor", f"   Traceback: {traceback.format_exc()}", "ERROR")
        finally:
            if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process.poll() is None:
                log_component("ChunkProcessor", "🛑 Terminating FFmpeg process...")
                try:
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=5)
                except Exception as e:
                    log_component("ChunkProcessor", f"⚠️ Error terminating FFmpeg: {e}", "WARNING")
    
    def _monitor_chunks_wrapper(self):
        """Wrapper for monitoring with Jupyter compatibility"""
        try:
            self._monitor_chunks()
        except Exception as e:
            log_component("ChunkProcessor", f"❌ Monitor wrapper caught exception: {e}", "ERROR")
            import traceback
            log_component("ChunkProcessor", f"   Traceback: {traceback.format_exc()}", "ERROR")
    
    def _monitor_chunks(self):
        """Monitor for new chunk files and process them"""
        processed_chunks = set()
        log_component("ChunkProcessor", f"🔄 Starting chunk monitoring loop", "DEBUG")
        log_component("ChunkProcessor", f"Monitoring loop started, looking in {self.output_dir}/chunks/", "DEBUG")
        log_component("ChunkProcessor", f"Thread ID={threading.current_thread().ident}, Name={threading.current_thread().name}", "DEBUG")
        
        try:
            while self.is_running:
                # Check for new chunk files
                search_pattern = f"{self.output_dir}/chunks/chunk_*_{self.chunk_duration}s.mp4"
                chunk_files = glob.glob(search_pattern)
                print(f"chunk files : {chunk_files}")
                # Debug: Log every 5 seconds to show loop is running
                # In Jupyter, this heartbeat helps prevent thread suspension
                import time as time_module
                current_time = time_module.time()
                if not hasattr(self, '_last_loop_log') or (current_time - self._last_loop_log) > 5:
                    log_component("ChunkProcessor", f"🔄 Monitoring... (found {len(chunk_files)} files, processed {len(processed_chunks)})", "DEBUG")
                    log_component("ChunkProcessor", f"❤️ Heartbeat - thread alive, is_running={self.is_running}", "DEBUG")
                    self._last_loop_log = current_time
                    
                    # Force a small I/O operation to keep thread active in Jupyter
                    if is_jupyter():
                        sys.stdout.flush()
                
                # Process new chunks
                for chunk_file in sorted(chunk_files):
                    chunk_match = re.search(r'chunk_(\d+)_', chunk_file)
                    if chunk_match:
                        chunk_id = int(chunk_match.group(1))
                        
                        if chunk_id not in processed_chunks and os.path.exists(chunk_file):
                            print(f"Chunk Ready : {self._verify_chunk_ready(chunk_file)}")
                            if self._verify_chunk_ready(chunk_file):
                                log_component("ChunkProcessor", f"📁 New chunk ready: {chunk_file}")
                                self._create_filmstrip(chunk_file, chunk_id)
                                processed_chunks.add(chunk_id)
                                self.chunk_count = max(self.chunk_count, chunk_id + 1)
                
                # Sleep briefly to avoid busy loop
                # In Jupyter, use multiple short sleeps with I/O to prevent thread suspension
                if is_jupyter():
                    # Break 1 second into 10x 100ms sleeps with I/O operations
                    for _ in range(10):
                        time.sleep(0.1)
                        # Touch stdout to keep thread active in Jupyter's event loop
                        sys.stdout.flush()
                else:
                    time.sleep(1)
            
            # Process any remaining chunks after stopping
            log_component("ChunkProcessor", "🔍 Processing any remaining chunks...")
            chunk_files = glob.glob(f"{self.output_dir}/chunks/chunk_*_{self.chunk_duration}s.mp4")
            
            for chunk_file in sorted(chunk_files):
                chunk_match = re.search(r'chunk_(\d+)_', chunk_file)
                if chunk_match:
                    chunk_id = int(chunk_match.group(1))
                    if chunk_id not in processed_chunks and os.path.exists(chunk_file):
                        if self._verify_chunk_ready(chunk_file, is_final=True):
                            log_component("ChunkProcessor", f"📁 Final chunk ready: {chunk_file}")
                            self._create_filmstrip(chunk_file, chunk_id)
                            processed_chunks.add(chunk_id)
                            self.chunk_count = max(self.chunk_count, chunk_id + 1)
                            
        except Exception as e:
            log_component("ChunkProcessor", f"❌ Monitoring error: {e}", "ERROR")
            import traceback
            log_component("ChunkProcessor", f"   Traceback: {traceback.format_exc()}", "ERROR")
    
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
                if duration >= 1.0:
                    log_component("ChunkProcessor", f"✅ Final chunk ready: duration {duration:.1f}s")
                    return True
                else:
                    log_component("ChunkProcessor", f"⏳ Final chunk too short: {duration:.1f}s", "DEBUG")
                    return False
            else:
                # For regular chunks, check if duration matches expected
                expected_duration = self.chunk_duration
                duration_diff = abs(duration - expected_duration)
                
                if duration_diff <= 1.0:  # Allow 1 second tolerance
                    log_component("ChunkProcessor", f"✅ Chunk ready: duration {duration:.1f}s (expected {expected_duration}s)")
                    return True
                else:
                    log_component("ChunkProcessor", f"⏳ Chunk duration {duration:.1f}s, waiting for {expected_duration}s", "DEBUG")
                    return False
                
        except Exception as e:
            log_component("ChunkProcessor", f"⚠️ Error verifying chunk: {e}", "WARNING")
            return False
    
    def _create_filmstrip(self, video_file, chunk_id):
        """Create enhanced filmstrip from video chunk using shared FilmstripProcessor"""
        try:
            log_component("ChunkProcessor", f"🎬 Creating filmstrip for chunk {chunk_id} from {video_file}", "DEBUG")
            
            # Wait for chunk to be ready
            max_wait = 30
            for attempt in range(max_wait):
                if self._verify_chunk_ready(video_file):
                    break
                time.sleep(1)
            else:
                log_component("ChunkProcessor", f"❌ Chunk {chunk_id} not ready after {max_wait}s, proceeding anyway", "WARNING")
            
            # Create filmstrip using shared processor
            filmstrip_path = f"{self.output_dir}/filmstrips/filmstrip_{chunk_id:04d}_4x5.jpg"
            
            if self.filmstrip_processor:
                log_component("ChunkProcessor", f"   📸 Using FilmstripProcessor to create filmstrip")
                shot_change_frames = self.filmstrip_processor.create_filmstrip_from_video(
                    video_file=video_file,
                    output_path=filmstrip_path,
                    start_time=chunk_id * self.chunk_duration,
                    num_frames=20,
                    interval=1.0,
                    detect_shot_changes=True
                )
                log_component("ChunkProcessor", f"   ✅ Filmstrip created: {filmstrip_path}", "DEBUG")
            else:
                log_component("ChunkProcessor", "⚠️ No filmstrip processor available - skipping filmstrip creation", "WARNING")
                shot_change_frames = []
            
            # Trigger multimodal analysis
            self._trigger_analysis(chunk_id, filmstrip_path, video_file, shot_change_frames)
                
        except Exception as e:
            log_component("ChunkProcessor", f"❌ Filmstrip creation error: {e}", "ERROR")
            import traceback
            log_component("ChunkProcessor", f"   Traceback: {traceback.format_exc()}", "ERROR")
    

    
    def _trigger_analysis(self, chunk_id, filmstrip_path, video_file, shot_change_frames=None):
        """Trigger multimodal analysis for the chunk with spectrogram data"""
        if self.fusion_analyzer:
            start_time = chunk_id * self.chunk_duration
            end_time = start_time + self.chunk_duration
            
            # Generate spectrogram analysis data
            spectrogram_data = None
            try:
                spectrogram_data = self.spectrogram_analyzer.analyze_audio_file(video_file)
                log_component("ChunkProcessor", f"✅ Generated spectrogram data for chunk {chunk_id}", "DEBUG")
            except Exception as e:
                log_component("ChunkProcessor", f"⚠️ Failed to generate spectrogram for chunk {chunk_id}: {e}", "WARNING")
            
            self.fusion_analyzer.queue_analysis(chunk_id, filmstrip_path, start_time, end_time, shot_change_frames, spectrogram_data)
            log_component("ChunkProcessor", f"🔄 Queued fusion analysis for chunk {chunk_id}", "DEBUG")
        else:
            if shot_change_frames:
                log_component("ChunkProcessor", f"🔄 Analysis ready for chunk {chunk_id} with {len(shot_change_frames)} shot changes", "DEBUG")
            else:
                log_component("ChunkProcessor", f"🔄 Analysis ready for chunk {chunk_id}", "DEBUG")
    
    def stop_processing(self):
        """Stop chunk processing"""
        self.is_running = False
        self._stop_event.set()  # Signal the thread to wake up
        log_component("ChunkProcessor", "🛑 Stopping chunk processing...")
        
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            log_component("ChunkProcessor", "   ⏳ Waiting for chunk processor thread...")
            self.processing_thread.join(timeout=10)
            if self.processing_thread.is_alive():
                log_component("ChunkProcessor", "   ⚠️ Thread still running (daemon will exit)", "WARNING")
            else:
                log_component("ChunkProcessor", "   ✅ Thread stopped")
        
        log_component("ChunkProcessor", "🛑 Chunk processing stopped")


if __name__ == "__main__":
    print("=" * 80)
    print("Chunk Processor - Example Usage")
    print("=" * 80)
    
    print("\nExample: Process video into 20-second chunks with filmstrips")
    print("-" * 80)
    print("""
from components import ChunkProcessor

processor = ChunkProcessor(
    udp_port="1234",
    output_dir="output",
    chunk_duration=20,
    fusion_analyzer=analyzer
)

processor.start_processing()
# ... chunks are processed automatically ...
processor.stop_processing()
    """)
    
    print("\n" + "=" * 80)
    print("✅ Chunk Processor module ready for use!")
    print("=" * 80)
