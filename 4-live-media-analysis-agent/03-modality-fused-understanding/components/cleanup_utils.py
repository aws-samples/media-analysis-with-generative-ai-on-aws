"""
Cleanup Utilities for Modality Fused Understanding

This module provides utility functions for cleaning up directories and FFmpeg processes
to ensure clean shutdown and prevent conflicts between notebook runs.
"""

import os
import shutil
import subprocess
import signal
from pathlib import Path
from typing import Optional, List


class CleanupUtils:
    """Utility class for cleanup operations"""
    
    @staticmethod
    def cleanup_directory(directory_path: str, create_subdirs: Optional[dict] = None) -> bool:
        """
        Clean up existing directory and optionally create subdirectories
        
        Args:
            directory_path: Path to directory to clean up
            create_subdirs: Dict of {name: path} for subdirectories to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(directory_path):
                print(f"🗑️  Cleaning up existing output directory: {directory_path}")
                shutil.rmtree(directory_path)
                print("✅ Cleanup complete")
            else:
                print("📁 No existing output directory found")
            
            # Create subdirectories if specified
            if create_subdirs:
                for name, path in create_subdirs.items():
                    os.makedirs(path, exist_ok=True)
                    print(f"✅ Created: {path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during directory cleanup: {e}")
            return False
    
    @staticmethod
    def cleanup_ffmpeg_processes(skip_cleanup: bool = False) -> bool:
        """
        Kill all FFmpeg processes to ensure clean shutdown
        
        Args:
            skip_cleanup: If True, skip the cleanup process
            
        Returns:
            bool: True if successful, False otherwise
        """
        if skip_cleanup:
            print("⏭️  Skipping FFmpeg cleanup (skip_cleanup=True)")
            return True
            
        try:
            # Find all FFmpeg processes
            result = subprocess.run(
                ['pgrep', '-f', 'ffmpeg'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"🔍 Found {len(pids)} FFmpeg process(es)")
                
                for pid in pids:
                    try:
                        pid_int = int(pid)
                        os.kill(pid_int, signal.SIGTERM)
                        print(f"✅ Terminated FFmpeg process (PID: {pid_int})")
                    except (ValueError, ProcessLookupError) as e:
                        print(f"⚠️  Could not terminate PID {pid}: {e}")
                
                print("\n🧹 Cleanup complete! All FFmpeg processes terminated.")
                print("✅ UDP ports (1234, 1235, 1236) are now free.")
                return True
            else:
                print("✅ No FFmpeg processes found running.")
                return True
                
        except Exception as e:
            print(f"❌ Error during FFmpeg cleanup: {e}")
            print("💡 You can manually kill FFmpeg processes using: pkill -f ffmpeg")
            return False
    
    @staticmethod
    def get_ffmpeg_processes() -> List[int]:
        """
        Get list of running FFmpeg process PIDs
        
        Returns:
            List[int]: List of FFmpeg process PIDs
        """
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'ffmpeg'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = [int(pid) for pid in result.stdout.strip().split('\n')]
                return pids
            else:
                return []
                
        except Exception:
            return []
    
    @staticmethod
    def cleanup_all(output_dir: str, create_subdirs: Optional[dict] = None, 
                   skip_ffmpeg: bool = False) -> bool:
        """
        Perform complete cleanup: directories and FFmpeg processes
        
        Args:
            output_dir: Output directory to clean up
            create_subdirs: Subdirectories to create after cleanup
            skip_ffmpeg: Skip FFmpeg process cleanup
            
        Returns:
            bool: True if all cleanup successful, False otherwise
        """
        print("🧹 Starting complete cleanup...")
        
        # Clean up directory
        dir_success = CleanupUtils.cleanup_directory(output_dir, create_subdirs)
        
        # Clean up FFmpeg processes
        ffmpeg_success = CleanupUtils.cleanup_ffmpeg_processes(skip_cleanup=skip_ffmpeg)
        
        if dir_success and ffmpeg_success:
            print("✅ Complete cleanup successful!")
            return True
        else:
            print("⚠️  Some cleanup operations failed")
            return False


# Convenience functions for direct import
def cleanup_directory(directory_path: str, create_subdirs: Optional[dict] = None) -> bool:
    """Convenience function for directory cleanup"""
    return CleanupUtils.cleanup_directory(directory_path, create_subdirs)


def cleanup_ffmpeg_processes(skip_cleanup: bool = False) -> bool:
    """Convenience function for FFmpeg cleanup"""
    return CleanupUtils.cleanup_ffmpeg_processes(skip_cleanup)


def cleanup_all(output_dir: str, create_subdirs: Optional[dict] = None, 
               skip_ffmpeg: bool = False) -> bool:
    """Convenience function for complete cleanup"""
    return CleanupUtils.cleanup_all(output_dir, create_subdirs, skip_ffmpeg)
