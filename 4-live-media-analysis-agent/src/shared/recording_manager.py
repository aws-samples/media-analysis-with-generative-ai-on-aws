"""
Recording Manager - Manages continuous video recording from UDP stream
Shared component used across Visual Understanding and Modality Fusion modules
"""

import subprocess
import time


class RecordingManager:
    """
    Manages continuous recording from UDP stream to MXF format.
    
    This component captures live UDP video streams and records them to MXF
    (Material Exchange Format) for later clip extraction and archival.
    
    Used in:
    - 20. visual_understanding
    - 40. modality_fused_understanding
    
    Usage:
        manager = RecordingManager(udp_port="1234", output_dir="output")
        manager.start_recording()
        # ... processing ...
        manager.stop_recording()
    """
    
    def __init__(self, udp_port, output_dir):
        """
        Initialize Recording Manager.
        
        Args:
            udp_port: UDP port to capture stream from (e.g., "1234")
            output_dir: Output directory for recording file
        """
        self.udp_port = udp_port
        self.output_dir = output_dir
        self.recording_file = f"{output_dir}/recording/continuous_recording_{int(time.time())}.mxf"
        self.recording_process = None
        self.recording_start_time = None
        self.is_recording = False
    
    def start_recording(self):
        """
        Start continuous recording from UDP stream.
        
        Creates an MXF file with MPEG-2 video and PCM audio.
        The recording runs in a background process.
        """
        try:
            from component_monitor import log_component
        except ImportError:
            # Fallback if component_monitor not available
            def log_component(component, message, level="INFO"):
                print(f"[{component}] {message}")
        
        log_component("Recording", f"📹 Starting continuous recording from UDP port {self.udp_port}", "DEBUG")
        
        udp_url = f"udp://127.0.0.1:{self.udp_port}?overrun_nonfatal=1"
        
        cmd = [
            'ffmpeg', '-i', udp_url,
            '-c:v', 'mpeg2video', '-b:v', '5M',
            '-c:a', 'pcm_s16le', '-ar', '48000',
            '-f', 'mxf', '-y', self.recording_file
        ]
        
        try:
            self.recording_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.recording_start_time = time.time()
            self.is_recording = True
            log_component("Recording", f"✅ Recording started on UDP port {self.udp_port} to file: {self.recording_file}")
            
        except Exception as e:
            log_component("Recording", f"❌ Failed to start recording: {e}", "ERROR")
    
    def stop_recording(self):
        """
        Stop continuous recording.
        
        Gracefully terminates the FFmpeg recording process.
        If termination fails, force kills the process.
        """
        try:
            from component_monitor import log_component
        except ImportError:
            def log_component(component, message, level="INFO"):
                print(f"[{component}] {message}")
        
        if self.recording_process and self.recording_process.poll() is None:
            log_component("Recording", "🛑 Stopping continuous recording...")
            try:
                self.recording_process.terminate()
                self.recording_process.wait(timeout=5)  # 5 second timeout
                log_component("Recording", "✅ Recording stopped")
            except subprocess.TimeoutExpired:
                log_component("Recording", "⚠️ Recording process didn't terminate, force killing...", "WARNING")
                try:
                    self.recording_process.kill()
                    self.recording_process.wait(timeout=2)  # 2 second timeout for kill
                    log_component("Recording", "✅ Recording stopped (process killed)")
                except Exception as e:
                    log_component("Recording", f"❌ Error killing recording process: {e}", "ERROR")
            except Exception as e:
                log_component("Recording", f"⚠️ Error stopping recording: {e}", "WARNING")
            finally:
                self.is_recording = False
        else:
            log_component("Recording", "✅ Recording already stopped")
    
    def get_recording_duration(self) -> float:
        """
        Get current recording duration in seconds.
        
        Returns:
            Duration in seconds, or 0 if not recording
        """
        if self.is_recording and self.recording_start_time:
            return time.time() - self.recording_start_time
        return 0.0
    
    def is_active(self) -> bool:
        """Check if recording is currently active"""
        return self.is_recording and self.recording_process and self.recording_process.poll() is None


if __name__ == "__main__":
    print("=" * 80)
    print("Recording Manager - Example Usage")
    print("=" * 80)
    
    # Example configuration
    print("\nExample: Recording from UDP port 1234")
    print("-" * 80)
    print("""
manager = RecordingManager(udp_port="1234", output_dir="output")
manager.start_recording()

# ... processing happens ...

duration = manager.get_recording_duration()
print(f"Recording duration: {duration:.1f}s")

manager.stop_recording()
    """)
    
    print("\n" + "=" * 80)
    print("✅ Recording Manager module ready for use!")
    print("=" * 80)
