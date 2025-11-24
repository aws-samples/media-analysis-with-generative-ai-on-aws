"""
Modality Fusion Understanding Components
Module-specific components for combining visual and audio analysis
"""

# Module-specific components
from .fusion_analyzer import FusionAnalyzer
from .chunk_processor import ChunkProcessor
from .chunk_monitor import ChunkMonitor
from .stream_monitor import StreamMonitor
from .audio_spectrogram_analyzer import AudioSpectrogramAnalyzer
from .jupyter_compat import JupyterThreadManager, get_thread_manager, is_jupyter
from .cleanup_utils import CleanupUtils, cleanup_directory, cleanup_ffmpeg_processes, cleanup_all
from .processing_utils import ProcessingUtils, start_fusion_processing

# Import shared components for convenience
try:
    from src.shared import (
        ShotChangeDetector,
        create_visual_detector,
        create_fusion_detector,
        RecordingManager,
        ComponentMonitor,
        log_component,
        show_component_table,
        get_component_summary,
        TranscriptionHandler,
        TranscriptionProcessor,
        AdaptiveFilmstripProcessor,
        create_fusion_filmstrip_processor,
        create_visual_filmstrip_processor
    )
    # Alias for backward compatibility
    FilmstripProcessor = AdaptiveFilmstripProcessor
except ImportError:
    # Fallback if shared components not available
    print("⚠️ Warning: Shared components not found. Please ensure src/shared is in Python path.")
    ShotChangeDetector = None
    create_visual_detector = None
    create_fusion_detector = None
    RecordingManager = None
    ComponentMonitor = None
    log_component = lambda c, m, l="INFO": print(f"[{c}] {m}")
    show_component_table = None
    get_component_summary = None
    TranscriptionHandler = None
    TranscriptionProcessor = None
    AdaptiveFilmstripProcessor = None
    FilmstripProcessor = None
    create_fusion_filmstrip_processor = None
    create_visual_filmstrip_processor = None

__all__ = [
    # Module-specific components
    'FusionAnalyzer',
    'ChunkProcessor',
    'ChunkMonitor',
    'StreamMonitor',
    'AudioSpectrogramAnalyzer',
    'JupyterThreadManager',
    'get_thread_manager',
    'is_jupyter',
    
    # Cleanup utilities
    'CleanupUtils',
    'cleanup_directory',
    'cleanup_ffmpeg_processes',
    'cleanup_all',
    
    # Processing utilities
    'ProcessingUtils',
    'start_fusion_processing',
    
    # Shared components (re-exported for convenience)
    'ShotChangeDetector',
    'create_visual_detector',
    'create_fusion_detector', 
    'RecordingManager',
    'ComponentMonitor',
    'log_component',
    'show_component_table',
    'get_component_summary',
    'TranscriptionHandler',
    'TranscriptionProcessor',
    'AdaptiveFilmstripProcessor',
    'FilmstripProcessor',  # Backward compatibility alias
    'create_fusion_filmstrip_processor',
    'create_visual_filmstrip_processor',
]

__version__ = '1.0.0'
