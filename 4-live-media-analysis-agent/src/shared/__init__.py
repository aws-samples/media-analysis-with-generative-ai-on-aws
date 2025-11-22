"""
Shared Components for Live Video Understanding
These components are reusable across all modules (visual, audio, and modality fusion)
"""

from .shot_change_detector import (
    ShotChangeDetector,
    create_visual_detector,
    create_fusion_detector
)

from .recording_manager import RecordingManager

from .component_monitor import (
    ComponentMonitor,
    LogLevel,
    log_component,
    set_debug_logging,
    set_component_logging_level,
    show_component_table,
    get_component_summary,
    print_logging_statistics
)

from .transcription_handler import TranscriptionHandler
from .transcription_processor import TranscriptionProcessor
from .filmstrip_processor import (
    FilmstripProcessor,
    create_fusion_filmstrip_processor,
    create_visual_filmstrip_processor
)

#from .clip_creator import ClipCreator
#from .bedrock_visual_analyzer import BedrockVisualAnalyzer
#from .visual_processor import VisualProcessor

__all__ = [
    'ShotChangeDetector',
    'create_visual_detector',
    'create_fusion_detector',
    'RecordingManager',
    'ComponentMonitor',
    'LogLevel',
    'log_component',
    'set_debug_logging',
    'set_component_logging_level',
    'show_component_table',
    'get_component_summary',
    'print_logging_statistics',
    'TranscriptionHandler',
    'TranscriptionProcessor',
    'FilmstripProcessor',
    'create_fusion_filmstrip_processor',
    'create_visual_filmstrip_processor',
    #'ClipCreator',
    #'BedrockVisualAnalyzer',
    #'VisualProcessor',
]

__version__ = '1.0.0'
