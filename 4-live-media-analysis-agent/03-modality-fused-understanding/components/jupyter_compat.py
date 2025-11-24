"""
Jupyter Notebook Compatibility Utilities
Handles threading issues specific to Jupyter environments
"""

import sys
import threading
import time

# Import logging from shared component monitor
import os
from pathlib import Path

# Add parent directory to path to import from src.shared
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.shared.component_monitor import log_component

def is_jupyter():
    """Check if running in Jupyter notebook"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except:
        pass
    return False

def ensure_thread_alive(thread, name="Thread", check_interval=5):
    """
    Monitor a thread and log if it dies unexpectedly.
    This is a debugging utility for Jupyter threading issues.
    
    Args:
        thread: The thread to monitor
        name: Name for logging
        check_interval: How often to check (seconds)
    """
    def monitor():
        while thread.is_alive():
            time.sleep(check_interval)
        log_component("JupyterCompat", f"{name} has stopped!", "WARNING")
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

def create_daemon_thread(target, name=None, args=(), kwargs=None):
    """
    Create a daemon thread with Jupyter-compatible settings.
    
    In Jupyter, daemon threads can be problematic. This function
    creates threads with proper settings for Jupyter environments.
    
    Args:
        target: The function to run in the thread
        name: Optional thread name
        args: Positional arguments for target
        kwargs: Keyword arguments for target
    
    Returns:
        threading.Thread: Configured thread (not started)
    """
    if kwargs is None:
        kwargs = {}
    
    thread = threading.Thread(
        target=target,
        name=name,
        args=args,
        kwargs=kwargs,
        daemon=True
    )
    
    # In Jupyter, we want to ensure threads are tracked
    if is_jupyter():
        log_component("JupyterCompat", f"Creating Jupyter-compatible thread: {name or 'unnamed'}", "DEBUG")
    
    return thread

def keep_thread_alive_wrapper(func):
    """
    Decorator to wrap thread functions with keep-alive mechanism.
    Helps prevent Jupyter from suspending threads.
    
    Usage:
        @keep_thread_alive_wrapper
        def my_thread_function():
            while running:
                # do work
                pass
    """
    def wrapper(*args, **kwargs):
        if is_jupyter():
            log_component("JupyterCompat", f"Thread {threading.current_thread().name} starting with keep-alive wrapper", "DEBUG")
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_component("JupyterCompat", f"Thread {threading.current_thread().name} crashed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if is_jupyter():
                log_component("JupyterCompat", f"Thread {threading.current_thread().name} exiting", "DEBUG")
    
    return wrapper

class JupyterThreadManager:
    """
    Manages threads in Jupyter notebooks to prevent suspension issues.
    
    Usage:
        manager = JupyterThreadManager()
        thread = manager.create_thread(target=my_function, name="MyThread")
        thread.start()
        
        # Later, check status
        manager.print_status()
    """
    
    def __init__(self):
        self.threads = {}
        self.is_jupyter = is_jupyter()
        
        if self.is_jupyter:
            log_component("JupyterCompat", "JupyterThreadManager initialized for Jupyter environment", "DEBUG")
    
    def create_thread(self, target, name=None, args=(), kwargs=None):
        """Create and register a thread"""
        thread = create_daemon_thread(target, name, args, kwargs)
        
        if name:
            self.threads[name] = thread
        
        return thread
    
    def start_thread(self, thread, name=None):
        """Start a thread and optionally monitor it"""
        thread.start()
        
        if self.is_jupyter and name:
            # Add monitoring in Jupyter
            ensure_thread_alive(thread, name)
        
        return thread
    
    def print_status(self):
        """Print status of all managed threads"""
        log_component("JupyterCompat", "=" * 60, "DEBUG")
        log_component("JupyterCompat", "Thread Status:", "DEBUG")
        log_component("JupyterCompat", "=" * 60, "DEBUG")
        
        for name, thread in self.threads.items():
            status = "ALIVE" if thread.is_alive() else "DEAD"
            log_level = "DEBUG" if status == "ALIVE" else "WARNING"
            log_component("JupyterCompat", f"  {name}: {status}", log_level)
        
        log_component("JupyterCompat", "=" * 60, "DEBUG")
    
    def stop_all(self):
        """Helper to check all threads (doesn't actually stop them)"""
        self.print_status()


# Global thread manager instance
_thread_manager = None

def get_thread_manager():
    """Get or create global thread manager"""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = JupyterThreadManager()
    return _thread_manager
