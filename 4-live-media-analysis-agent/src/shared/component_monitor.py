"""
Component Activity Monitor for Multimodal Fusion System
Enhanced stdout approach with visual formatting and comprehensive log level control
"""

import threading
from datetime import datetime
from collections import defaultdict
from enum import IntEnum

class LogLevel(IntEnum):
    """Log level enumeration for consistent level handling"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    DISABLED = 5

class ComponentMonitor:
    """Thread-safe monitor with enhanced visual stdout formatting and log level control"""
    
    def __init__(self, default_level=LogLevel.INFO):
        self.lock = threading.Lock()
        self.component_logs = defaultdict(list)
        self.last_component = None  # Track component changes for grouping
        
        # Enhanced logging level control
        self.current_level = default_level
        self.debug_enabled = (default_level == LogLevel.DEBUG)
        
        # Per-component log level overrides (optional)
        self.component_levels = {}
        
        # Pre-define all expected components with colors and icons
        self.components = {
            "Main": {"color": "\033[96m", "icon": "🚀", "width": 15},
            "Recording": {"color": "\033[92m", "icon": "📹", "width": 15},
            "ChunkProcessor": {"color": "\033[93m", "icon": "🎬", "width": 15},
            "Transcription": {"color": "\033[95m", "icon": "🎧", "width": 15},
            "FusionAnalyzer": {"color": "\033[94m", "icon": "🧠", "width": 15}
        }
        self.reset = "\033[0m"
        
        # Level-specific styling
        self.level_styles = {
            LogLevel.DEBUG: {"color": "\033[90m", "icon": "🔍", "label": "DEBUG"},
            LogLevel.INFO: {"color": "", "icon": "ℹ️", "label": "INFO"},
            LogLevel.WARNING: {"color": "\033[93m", "icon": "⚠️", "label": "WARNING"},
            LogLevel.ERROR: {"color": "\033[91m", "icon": "❌", "label": "ERROR"},
            LogLevel.CRITICAL: {"color": "\033[91m\033[1m", "icon": "🔥", "label": "CRITICAL"}
        }
        
        # Initialize empty logs for all components
        for component in self.components.keys():
            self.component_logs[component] = []
        
        # Statistics tracking
        self.log_counts = defaultdict(lambda: defaultdict(int))
        
    def set_level(self, level):
        """
        Set the global logging level
        
        Args:
            level: Can be LogLevel enum, string ("DEBUG", "INFO", etc.), or int (0-5)
        """
        with self.lock:
            if isinstance(level, str):
                level = level.upper()
                level_map = {
                    "DEBUG": LogLevel.DEBUG,
                    "INFO": LogLevel.INFO,
                    "WARNING": LogLevel.WARNING,
                    "ERROR": LogLevel.ERROR,
                    "CRITICAL": LogLevel.CRITICAL,
                    "DISABLED": LogLevel.DISABLED
                }
                level = level_map.get(level, LogLevel.INFO)
            elif isinstance(level, int):
                level = LogLevel(level)
            
            self.current_level = level
            self.debug_enabled = (level == LogLevel.DEBUG)
            
            if level != LogLevel.DISABLED:
                level_name = self.level_styles.get(level, {}).get("label", str(level))
                print(f"🔧 Logging level set to: {level_name}")
    
    def set_component_level(self, component_name, level):
        """
        Set logging level for a specific component (overrides global level)
        
        Args:
            component_name: Name of the component
            level: Can be LogLevel enum, string, or int
        """
        with self.lock:
            if isinstance(level, str):
                level = level.upper()
                level_map = {
                    "DEBUG": LogLevel.DEBUG,
                    "INFO": LogLevel.INFO,
                    "WARNING": LogLevel.WARNING,
                    "ERROR": LogLevel.ERROR,
                    "CRITICAL": LogLevel.CRITICAL,
                    "DISABLED": LogLevel.DISABLED
                }
                level = level_map.get(level, LogLevel.INFO)
            elif isinstance(level, int):
                level = LogLevel(level)
            
            self.component_levels[component_name] = level
            level_name = self.level_styles.get(level, {}).get("label", str(level))
            print(f"🔧 {component_name} logging level set to: {level_name}")
    
    def get_effective_level(self, component_name):
        """Get the effective logging level for a component"""
        return self.component_levels.get(component_name, self.current_level)
    
    def set_debug_mode(self, enabled):
        """Enable or disable debug logging (legacy compatibility)"""
        self.set_level(LogLevel.DEBUG if enabled else LogLevel.INFO)
        
    def set_logging_level(self, level):
        """Set minimum logging level (legacy compatibility)"""
        self.set_level(level)
        
    def log(self, component_name, message, level="INFO"):
        """
        Log a message for a specific component
        
        Args:
            component_name: Name of the component logging the message
            message: The log message
            level: Log level (string, LogLevel enum, or int)
        """
        # Convert level to LogLevel enum
        if isinstance(level, str):
            level = level.upper()
            level_map = {
                "DEBUG": LogLevel.DEBUG,
                "INFO": LogLevel.INFO,
                "WARNING": LogLevel.WARNING,
                "ERROR": LogLevel.ERROR,
                "CRITICAL": LogLevel.CRITICAL
            }
            log_level = level_map.get(level, LogLevel.INFO)
        elif isinstance(level, int):
            log_level = LogLevel(level)
        else:
            log_level = level
        
        # Check if we should display this log level
        effective_level = self.get_effective_level(component_name)
        if log_level < effective_level:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_label = self.level_styles.get(log_level, {}).get("label", str(log_level))
        formatted_msg = f"[{timestamp}] {level_label}: {message}"
        
        with self.lock:
            self.component_logs[component_name].append(formatted_msg)
            self.log_counts[component_name][log_level] += 1
            
        # Enhanced visual formatting
        self._print_formatted_log(component_name, formatted_msg, log_level)
        self.last_component = component_name
    
    def _print_formatted_log(self, component_name, message, level):
        """
        Print log with enhanced visual formatting
        
        Args:
            component_name: Name of the component
            message: Formatted message with timestamp
            level: LogLevel enum value
        """
        # Get component info
        comp_info = self.components.get(component_name, {
            "color": "", "icon": "📋", "width": 15
        })
        
        color = comp_info["color"]
        icon = comp_info["icon"]
        width = comp_info["width"]
        
        # Get level-specific styling
        level_style = self.level_styles.get(level, {"color": "", "icon": "•"})
        level_color = level_style["color"]
        level_icon = level_style.get("icon", "•")
        
        # Add visual separator when switching components
        if self.last_component and self.last_component != component_name:
            print(f"{color}{'─' * 80}{self.reset}")
        
        # Format component name with consistent width and styling
        component_display = f"{icon} {component_name}".ljust(width + 2)
        
        # Print with enhanced formatting and level coloring
        print(f"{color}┃{self.reset} {color}{component_display}{self.reset} │ {level_color}{level_icon} {message}{self.reset}")
    
    def show_table(self):
        """Display monitor header and legend"""
        print("\n" + "═" * 80)
        print("🔄 COMPONENT ACTIVITY MONITOR")
        print("═" * 80)
        
        # Show current logging level
        level_name = self.level_styles.get(self.current_level, {}).get("label", str(self.current_level))
        print(f"🔧 Global Log Level: {level_name}")
        
        # Show component-specific levels if any
        if self.component_levels:
            print("🔧 Component-Specific Levels:")
            for comp, lvl in self.component_levels.items():
                lvl_name = self.level_styles.get(lvl, {}).get("label", str(lvl))
                print(f"   • {comp}: {lvl_name}")
        
        print("─" * 80)
        
        # Show component legend
        print("📊 Components:")
        for name, info in self.components.items():
            color = info["color"]
            icon = info["icon"]
            print(f"   {color}{icon} {name}{self.reset}")
        
        print("─" * 80)
        
        # Show log level legend
        print("📊 Log Levels:")
        for level in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]:
            style = self.level_styles[level]
            print(f"   {style['color']}{style['icon']} {style['label']}{self.reset}")
        
        print("─" * 80)
        print("📋 Activity Log:")
        print()
    
    def get_statistics(self):
        """Get logging statistics for all components"""
        stats = {}
        with self.lock:
            for component in self.components.keys():
                stats[component] = {
                    "total": sum(self.log_counts[component].values()),
                    "by_level": dict(self.log_counts[component])
                }
        return stats
    
    def print_statistics(self):
        """Print logging statistics"""
        stats = self.get_statistics()
        print("\n" + "═" * 80)
        print("📊 LOGGING STATISTICS")
        print("═" * 80)
        
        for component, data in stats.items():
            if data["total"] > 0:
                print(f"\n{component}: {data['total']} total logs")
                for level, count in data["by_level"].items():
                    level_name = self.level_styles.get(level, {}).get("label", str(level))
                    print(f"  • {level_name}: {count}")
        print("═" * 80)

# Global monitor instance
component_monitor = ComponentMonitor()

def set_debug_logging(level="INFO"):
    """
    Set logging level globally for both component monitor and Python logging
    
    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", or "DISABLED"
    
    Examples:
        set_debug_logging("DEBUG")     # Show all logs including debug
        set_debug_logging("INFO")      # Show info and above (default)
        set_debug_logging("WARNING")   # Show only warnings and errors
        set_debug_logging("ERROR")     # Show only errors
        set_debug_logging("DISABLED")  # Disable all logging
    """
    import logging
    import sys
    import os
    
    level = level.upper()
    
    if level == "DISABLED":
        # Disable all loggers
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        logging.getLogger('boto3').setLevel(logging.CRITICAL + 1)
        logging.getLogger('botocore').setLevel(logging.CRITICAL + 1)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL + 1)
        logging.disable(logging.CRITICAL)
        component_monitor.set_level(LogLevel.DISABLED)
        
        # Redirect stdout to suppress print statements
        sys.stdout = open(os.devnull, 'w')
        
    elif level == "DEBUG":
        # Re-enable stdout if it was disabled
        if hasattr(sys.stdout, 'name') and sys.stdout.name == os.devnull:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        
        logging.disable(logging.NOTSET)
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('boto3').setLevel(logging.DEBUG)
        logging.getLogger('botocore').setLevel(logging.DEBUG)
        logging.getLogger('urllib3').setLevel(logging.DEBUG)
        component_monitor.set_level(LogLevel.DEBUG)
        
    else:
        # Re-enable stdout if it was disabled
        if hasattr(sys.stdout, 'name') and sys.stdout.name == os.devnull:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            
        logging.disable(logging.NOTSET)
        
        # Map string levels to Python logging levels
        python_level_map = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING, 
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        python_level = python_level_map.get(level, logging.INFO)
        
        # Set Python logging levels
        logging.getLogger().setLevel(python_level)
        logging.getLogger('boto3').setLevel(logging.CRITICAL + 1)
        logging.getLogger('botocore').setLevel(logging.CRITICAL + 1)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL + 1)
        
        # Set component monitor logging level
        component_monitor.set_level(level)
        
        # Create a specific logger for our message only
        if level == "WARNING":
            stream_logger = logging.getLogger('stream_instruction')
            stream_logger.setLevel(logging.WARNING)
            if not stream_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                stream_logger.addHandler(handler)
            stream_logger.warning("Please follow the instruction to ingest the live stream")

def set_component_logging_level(component_name, level):
    """
    Set logging level for a specific component
    
    Args:
        component_name: Name of the component
        level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", or "DISABLED"
    
    Example:
        set_component_logging_level("Transcription", "DEBUG")
        set_component_logging_level("Recording", "WARNING")
    """
    component_monitor.set_component_level(component_name, level)

def log_component(component_name, message, level="INFO"):
    """Convenience function for logging"""
    component_monitor.log(component_name, message, level)

def show_component_table():
    """Convenience function to show component activity table"""
    component_monitor.show_table()

def get_component_summary():
    """Convenience function to get component activity summary"""
    return {
        'components': list(component_monitor.components.keys()),
        'total_logs': sum(len(logs) for logs in component_monitor.component_logs.values()),
        'statistics': component_monitor.get_statistics()
    }

def print_logging_statistics():
    """Convenience function to print logging statistics"""
    component_monitor.print_statistics()
