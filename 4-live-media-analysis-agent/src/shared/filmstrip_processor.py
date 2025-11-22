"""
Filmstrip Processor - Creates enhanced filmstrip grids from video frames
Shared component used across Visual Understanding and Modality Fusion modules
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

# Import shared components
try:
    from .component_monitor import log_component
    from .shot_change_detector import ShotChangeDetector
except ImportError:
    # Fallback if component_monitor not available
    def log_component(component, message, level="INFO"):
        print(f"[{component}] {message}")
    ShotChangeDetector = None


class FilmstripProcessor:
    """
    Creates enhanced filmstrip grids from video frames.
    
    This processor:
    - Extracts frames from video files
    - Creates labeled grid layouts (4×5, 5×4, etc.)
    - Adds borders and timestamps
    - Integrates with shot change detection
    - Supports cross-chunk detection
    - Generates high-quality filmstrip images
    
    Key Features:
    - Flexible grid configurations
    - Enhanced visual design with borders and labels
    - Shot change detection integration
    - Cross-chunk frame tracking
    - Customizable styling
    - Error handling and recovery
    """
    
    def __init__(
        self,
        grid_rows: int = 4,
        grid_cols: int = 5,
        cell_size: int = 512,
        cell_width: Optional[int] = None,
        cell_height: Optional[int] = None,
        border_thickness: int = 8,
        label_height: int = 40,
        border_color: str = 'red',
        label_bg_color: str = 'black',
        label_text_color: str = 'white',
        shot_detector: Optional[ShotChangeDetector] = None
    ):
        """
        Initialize filmstrip processor.
        
        Args:
            grid_rows: Number of rows in grid (default: 4)
            grid_cols: Number of columns in grid (default: 5)
            cell_size: Size of each cell in pixels (default: 512) - used if cell_width/height not specified
            cell_width: Width of each cell in pixels (overrides cell_size)
            cell_height: Height of each cell in pixels (overrides cell_size)
            border_thickness: Thickness of grid borders (default: 8)
            label_height: Height of label area below each frame (default: 40)
            border_color: Color of grid borders (default: 'red')
            label_bg_color: Background color of labels (default: 'black')
            label_text_color: Text color of labels (default: 'white')
            shot_detector: Optional ShotChangeDetector for detecting scene changes
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Use separate width/height if provided, otherwise use square cell_size
        self.cell_width = cell_width if cell_width is not None else cell_size
        self.cell_height = cell_height if cell_height is not None else cell_size
        self.cell_size = cell_size  # Keep for backward compatibility
        
        self.border_thickness = border_thickness
        self.label_height = label_height
        self.border_color = border_color
        self.label_bg_color = label_bg_color
        self.label_text_color = label_text_color
        self.shot_detector = shot_detector
        
        # Cross-chunk tracking
        self.last_frame = None
        
        # Try to load font
        try:
            self.label_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                52
            )
        except:
            self.label_font = ImageFont.load_default()
    
    def extract_frames_from_video(
        self, 
        video_file: str, 
        start_time: float = 0.0,
        num_frames: int = 20,
        interval: float = 1.0
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames from video file with timestamps.
        
        Args:
            video_file: Path to video file
            start_time: Start time offset in seconds
            num_frames: Number of frames to extract
            interval: Time interval between frames in seconds (default: 1.0 for 1 fps)
        
        Returns:
            List of tuples (frame, timestamp)
        """
        frames_with_timestamps = []
        
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            log_component("FilmstripProcessor", f"❌ Cannot open video file: {video_file}", "ERROR")
            # Retry once
            import time
            time.sleep(2)
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                log_component("FilmstripProcessor", f"❌ Still cannot open video file: {video_file}", "ERROR")
                return frames_with_timestamps
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 20
        
        log_component("FilmstripProcessor", f"   📊 Video: {video_duration:.1f}s, {fps:.1f}fps, {total_frames} frames", "DEBUG")
        
        # Extract frames at specified intervals
        for i in range(num_frames):
            # Seek to middle of each interval (e.g., 0.5s, 1.5s, 2.5s for 1s intervals)
            time_offset = i * interval + (interval / 2)
            time_ms = time_offset * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            
            ret, frame = cap.read()
            
            if ret and frame is not None:
                timestamp = start_time + time_offset
                frames_with_timestamps.append((frame, timestamp))
            else:
                log_component("FilmstripProcessor", f"   ⚠️ Could not read frame at {time_offset:.1f}s", "WARNING")
                # Add black frame as placeholder
                if frames_with_timestamps:
                    black_frame = np.zeros_like(frames_with_timestamps[0][0])
                else:
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                timestamp = start_time + time_offset
                frames_with_timestamps.append((black_frame, timestamp))
        
        cap.release()
        
        log_component("FilmstripProcessor", f"   📸 Extracted {len(frames_with_timestamps)} frames", "DEBUG")
        return frames_with_timestamps
    
    def create_filmstrip(
        self,
        frames_with_timestamps: List[Tuple[np.ndarray, float]],
        output_path: str,
        detect_shot_changes: bool = True
    ) -> List[int]:
        """
        Create enhanced filmstrip grid from frames.
        
        Args:
            frames_with_timestamps: List of (frame, timestamp) tuples
            output_path: Path to save filmstrip image
            detect_shot_changes: Whether to detect shot changes (default: True)
        
        Returns:
            List of frame indices where shot changes were detected
        """
        if not frames_with_timestamps:
            log_component("FilmstripProcessor", "❌ No frames provided for filmstrip", "ERROR")
            return []
        
        # Extract frames for shot detection
        frames = [frame for frame, _ in frames_with_timestamps]
        
        # Detect shot changes
        shot_change_frames = []
        if detect_shot_changes and self.shot_detector:
            log_component("FilmstripProcessor", "   🔍 Detecting shot changes...", "DEBUG")
            # Note: detect_batch uses self.last_frame internally for cross-chunk detection
            shot_changes = self.shot_detector.detect_batch(frames)
            shot_change_frames = [i for i, is_change in enumerate(shot_changes) if is_change]
            log_component("FilmstripProcessor", f"Found {len(shot_change_frames)} shot changes at frames: {shot_change_frames}")
        
        # Create grid
        self._create_grid(frames_with_timestamps, output_path)
        
        # Store last frame for cross-chunk detection
        if len(frames) > 0:
            self.last_frame = frames[-1].copy()
        
        return shot_change_frames
    
    def _create_grid(
        self,
        frames_with_timestamps: List[Tuple[np.ndarray, float]],
        output_path: str
    ):
        """
        Create the actual grid image with borders and labels.
        
        Args:
            frames_with_timestamps: List of (frame, timestamp) tuples
            output_path: Path to save filmstrip image
        """
        # Calculate total dimensions
        total_width = (
            self.grid_cols * self.cell_width + 
            (self.grid_cols + 1) * self.border_thickness
        )
        total_height = (
            self.grid_rows * (self.cell_height + self.label_height) + 
            (self.grid_rows + 1) * self.border_thickness
        )
        
        # Create white background
        grid_image = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(grid_image)
        
        # Place frames in grid
        max_frames = self.grid_rows * self.grid_cols
        for idx, (frame, timestamp) in enumerate(frames_with_timestamps[:max_frames]):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            
            # Calculate position
            x = col * (self.cell_width + self.border_thickness) + self.border_thickness
            y = row * (self.cell_height + self.label_height + self.border_thickness) + self.border_thickness
            
            # Convert OpenCV frame (BGR) to PIL (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize frame
            try:
                frame_resized = frame_pil.resize(
                    (self.cell_width, self.cell_height), 
                    Image.Resampling.LANCZOS
                )
            except AttributeError:
                # Fallback for older Pillow versions
                frame_resized = frame_pil.resize(
                    (self.cell_width, self.cell_height), 
                    Image.LANCZOS
                )
            
            # Paste frame
            grid_image.paste(frame_resized, (x, y))
            
            # Draw label below frame
            label_y = y + self.cell_height
            label_box = [
                (x, label_y), 
                (x + self.cell_width, label_y + self.label_height)
            ]
            draw.rectangle(label_box, fill=self.label_bg_color)
            
            # Create label text with grid position and timestamp
            label_text = f"[{row+1}×{col+1}] | {timestamp:.1f}s"
            
            # Center text in label
            bbox = draw.textbbox((0, 0), label_text, font=self.label_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x + (self.cell_width - text_width) // 2
            text_y = label_y + (self.label_height - text_height) // 2
            
            draw.text(
                (text_x, text_y), 
                label_text, 
                fill=self.label_text_color, 
                font=self.label_font
            )
        
        # Draw grid borders
        self._draw_borders(draw, total_width, total_height)
        
        # Save image
        grid_image.save(output_path, quality=95)
        
        log_component("FilmstripProcessor", f"🎞️ Filmstrip created: {output_path}")
        log_component("FilmstripProcessor", f"   📊 Grid: {self.grid_rows}×{self.grid_cols}, Cell: {self.cell_width}×{self.cell_height}px", "DEBUG")
    
    def _draw_borders(self, draw: ImageDraw.Draw, total_width: int, total_height: int):
        """
        Draw grid borders.
        
        Args:
            draw: PIL ImageDraw object
            total_width: Total width of grid
            total_height: Total height of grid
        """
        # Draw horizontal borders
        for i in range(self.grid_rows + 1):
            y = i * (self.cell_height + self.label_height + self.border_thickness)
            draw.rectangle(
                [(0, y), (total_width, y + self.border_thickness)], 
                fill=self.border_color
            )
        
        # Draw vertical borders
        for i in range(self.grid_cols + 1):
            x = i * (self.cell_width + self.border_thickness)
            draw.rectangle(
                [(x, 0), (x + self.border_thickness, total_height)], 
                fill=self.border_color
            )
    
    def create_filmstrip_from_video(
        self,
        video_file: str,
        output_path: str,
        start_time: float = 0.0,
        num_frames: int = 20,
        interval: float = 1.0,
        detect_shot_changes: bool = True
    ) -> List[int]:
        """
        Convenience method: Extract frames and create filmstrip in one call.
        
        Args:
            video_file: Path to video file
            output_path: Path to save filmstrip image
            start_time: Start time offset in seconds
            num_frames: Number of frames to extract
            interval: Time interval between frames in seconds
            detect_shot_changes: Whether to detect shot changes
        
        Returns:
            List of frame indices where shot changes were detected
        """
        log_component("FilmstripProcessor", f"🎬 Creating filmstrip from {video_file}", "DEBUG")
        
        # Extract frames
        frames_with_timestamps = self.extract_frames_from_video(
            video_file, 
            start_time, 
            num_frames, 
            interval
        )
        
        if not frames_with_timestamps:
            log_component("FilmstripProcessor", "❌ No frames extracted", "ERROR")
            return []
        
        # Create filmstrip
        shot_change_frames = self.create_filmstrip(
            frames_with_timestamps,
            output_path,
            detect_shot_changes
        )
        
        return shot_change_frames
    
    def reset_cross_chunk_tracking(self):
        """Reset cross-chunk frame tracking (useful when starting new video)"""
        self.last_frame = None
        log_component("FilmstripProcessor", "🔄 Cross-chunk tracking reset", "DEBUG")


# Convenience functions for common configurations

def create_fusion_filmstrip_processor(shot_detector=None) -> FilmstripProcessor:
    """
    Create filmstrip processor configured for Modality Fusion (4×5 grid).
    
    Args:
        shot_detector: Optional ShotChangeDetector instance
    
    Returns:
        FilmstripProcessor configured for fusion analysis
    """
    return FilmstripProcessor(
        grid_rows=4,
        grid_cols=5,
        cell_size=512,
        border_thickness=8,
        label_height=40,
        shot_detector=shot_detector
    )


def create_visual_filmstrip_processor(shot_detector=None) -> FilmstripProcessor:
    """
    Create filmstrip processor configured for Visual Understanding (5×4 grid).
    
    Args:
        shot_detector: Optional ShotChangeDetector instance
    
    Returns:
        FilmstripProcessor configured for visual analysis
    """
    return FilmstripProcessor(
        grid_rows=5,
        grid_cols=4,
        cell_size=512,
        border_thickness=8,
        label_height=40,
        shot_detector=shot_detector
    )


class AdaptiveFilmstripProcessor:
    """
    Adaptive filmstrip processor that automatically calculates optimal frame extraction
    and packing based on video properties and constraints.
    
    Given:
    - Video duration, FPS, and resolution
    - Maximum grid image size (e.g., 8000×8000)
    - Maximum number of grid images (e.g., 20)
    
    Calculates:
    - How many frames fit in each grid
    - Optimal sampling rate
    - Grid layout (rows × cols)
    - Whether frame sampling is needed
    """
    
    def __init__(
        self,
        max_grid_size: Tuple[int, int] = (8000, 8000),
        max_grid_images: int = 20,
        border_thickness: int = 8,
        label_height: int = 40,
        border_color: str = 'red',
        label_bg_color: str = 'black',
        label_text_color: str = 'white',
        preserve_source_resolution: bool = False,
        fixed_grid_layout: Optional[Tuple[int, int]] = None,
        max_file_size_mb: Optional[float] = None,
        shot_detector: Optional[ShotChangeDetector] = None
    ):
        """
        Initialize adaptive filmstrip processor.
        
        Args:
            max_grid_size: Maximum size of each grid image (width, height)
            max_grid_images: Maximum number of grid images to create
            border_thickness: Thickness of grid borders
            label_height: Height of label area below each frame
            border_color: Color of grid borders
            label_bg_color: Background color of labels
            label_text_color: Text color of labels
            preserve_source_resolution: If True, cell size matches source resolution exactly
            fixed_grid_layout: If provided (rows, cols), uses this fixed grid layout instead of computing optimal
            max_file_size_mb: If provided, downscale cell resolution to fit within this file size limit (in MB)
            shot_detector: Optional ShotChangeDetector for detecting scene changes
        """
        self.max_grid_width, self.max_grid_height = max_grid_size
        self.max_grid_images = max_grid_images
        self.border_thickness = border_thickness
        self.label_height = label_height
        self.border_color = border_color
        self.label_bg_color = label_bg_color
        self.label_text_color = label_text_color
        self.preserve_source_resolution = preserve_source_resolution
        self.fixed_grid_layout = fixed_grid_layout
        self.max_file_size_mb = max_file_size_mb
        self.shot_detector = shot_detector
        
        # Try to load font
        try:
            self.label_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                36
            )
        except:
            self.label_font = ImageFont.load_default()
    
    def calculate_optimal_layout(
        self,
        video_duration: float,
        source_fps: float,
        source_resolution: Tuple[int, int]
    ) -> dict:
        """
        Calculate optimal frame extraction and grid layout.
        
        Args:
            video_duration: Duration of video in seconds
            source_fps: Source video FPS
            source_resolution: Source video resolution (width, height)
        
        Returns:
            Dictionary with layout parameters:
            - total_frames: Total frames in video
            - frames_per_grid: Frames that fit in each grid
            - grid_rows: Number of rows per grid
            - grid_cols: Number of columns per grid
            - cell_size: Size of each cell (width, height)
            - num_grids_needed: Number of grid images needed
            - sampling_rate: Frame sampling rate (1 = no sampling, 2 = every 2nd frame, etc.)
            - frames_to_extract: Actual number of frames to extract
            - extraction_interval: Time interval between extracted frames
        """
        source_width, source_height = source_resolution
        
        # Calculate total frames in video
        total_frames = int(video_duration * source_fps)
        
        log_component("AdaptiveFilmstripProcessor", f"📊 Video Analysis:", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Duration: {video_duration:.1f}s", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   FPS: {source_fps}", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Resolution: {source_width}×{source_height}", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Total frames: {total_frames}", "DEBUG")
        
        # Calculate cell size maintaining aspect ratio
        aspect_ratio = source_width / source_height
        
        # Start with a reasonable cell size and adjust
        # We need to fit cells in the grid considering borders and labels
        cell_width, cell_height, grid_rows, grid_cols = self._calculate_grid_dimensions(
            aspect_ratio, source_resolution
        )
        
        # Calculate how many frames fit in one grid
        frames_per_grid = grid_rows * grid_cols
        
        # Calculate maximum frames we can handle across all grids
        max_total_frames = frames_per_grid * self.max_grid_images
        
        # Determine if sampling is needed
        if total_frames <= max_total_frames:
            # No sampling needed - we can fit all frames
            sampling_rate = 1
            frames_to_extract = total_frames
            num_grids_needed = (frames_to_extract + frames_per_grid - 1) // frames_per_grid
        else:
            # Sampling needed - calculate sampling rate
            sampling_rate = (total_frames + max_total_frames - 1) // max_total_frames
            frames_to_extract = (total_frames + sampling_rate - 1) // sampling_rate
            num_grids_needed = (frames_to_extract + frames_per_grid - 1) // frames_per_grid
        
        # Calculate extraction interval in seconds
        extraction_interval = (video_duration / frames_to_extract) if frames_to_extract > 0 else 1.0
        
        layout = {
            'total_frames': total_frames,
            'frames_per_grid': frames_per_grid,
            'grid_rows': grid_rows,
            'grid_cols': grid_cols,
            'cell_size': (cell_width, cell_height),
            'num_grids_needed': num_grids_needed,
            'sampling_rate': sampling_rate,
            'frames_to_extract': frames_to_extract,
            'extraction_interval': extraction_interval,
            'max_frames_capacity': max_total_frames
        }
        
        log_component("AdaptiveFilmstripProcessor", f"🎯 Optimal Layout Calculated:", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Grid layout: {grid_rows}×{grid_cols} ({frames_per_grid} frames/grid)", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Cell size: {cell_width}×{cell_height}px", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Grids needed: {num_grids_needed}/{self.max_grid_images}", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Sampling: {'None' if sampling_rate == 1 else f'Every {sampling_rate} frames'}", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Frames to extract: {frames_to_extract}/{total_frames}", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Extraction interval: {extraction_interval:.3f}s", "DEBUG")
        
        return layout
    
    def _estimate_file_size(
        self,
        grid_width: int,
        grid_height: int,
        quality: int = 95
    ) -> float:
        """
        Estimate JPEG file size in MB based on image dimensions.
        
        Uses empirical formula: size ≈ (width × height × bits_per_pixel) / (8 × 1024 × 1024)
        For JPEG quality=95, bits_per_pixel ≈ 2.5-3.0 (varies with content complexity)
        
        Args:
            grid_width: Width of grid image in pixels
            grid_height: Height of grid image in pixels
            quality: JPEG quality (default: 95)
        
        Returns:
            Estimated file size in MB
        """
        # Empirical bits per pixel for JPEG at different quality levels
        # Quality 95: ~2.8 bpp (high quality, moderate compression)
        # Quality 85: ~2.0 bpp (good quality, better compression)
        # Quality 75: ~1.5 bpp (acceptable quality, high compression)
        
        if quality >= 90:
            bits_per_pixel = 2.8
        elif quality >= 80:
            bits_per_pixel = 2.0
        else:
            bits_per_pixel = 1.5
        
        # Calculate size in MB
        total_pixels = grid_width * grid_height
        size_mb = (total_pixels * bits_per_pixel) / (8 * 1024 * 1024)
        
        return size_mb
    
    def _apply_file_size_constraint(
        self,
        cell_width: int,
        cell_height: int,
        grid_rows: int,
        grid_cols: int,
        aspect_ratio: float
    ) -> Tuple[int, int]:
        """
        Downscale cell resolution to fit within max file size constraint.
        
        Args:
            cell_width: Current cell width
            cell_height: Current cell height
            grid_rows: Number of rows in grid
            grid_cols: Number of columns in grid
            aspect_ratio: Width/height ratio to maintain
        
        Returns:
            Tuple of (adjusted_cell_width, adjusted_cell_height)
        """
        # Calculate current grid dimensions
        total_width = grid_cols * cell_width + (grid_cols + 1) * self.border_thickness
        total_height = grid_rows * (cell_height + self.label_height) + (grid_rows + 1) * self.border_thickness
        
        # Estimate current file size
        current_size_mb = self._estimate_file_size(total_width, total_height)
        
        log_component("AdaptiveFilmstripProcessor", f"📏 File Size Constraint:", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Current grid: {total_width}×{total_height}px", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Estimated size: {current_size_mb:.2f}MB", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"   Target size: {self.max_file_size_mb:.2f}MB", "DEBUG")
        
        if current_size_mb <= self.max_file_size_mb:
            log_component("AdaptiveFilmstripProcessor", f"   ✅ Within size limit, no downscaling needed", "DEBUG")
            return (cell_width, cell_height)
        
        # Calculate scaling factor needed
        # size ∝ width × height, so scale = sqrt(target_size / current_size)
        scale_factor = (self.max_file_size_mb / current_size_mb) ** 0.5
        
        # Apply scaling to cell dimensions (maintaining aspect ratio)
        new_cell_width = int(cell_width * scale_factor)
        new_cell_height = int(cell_height * scale_factor)
        
        # Ensure minimum cell size
        min_cell_size = 100
        if new_cell_width < min_cell_size or new_cell_height < min_cell_size:
            log_component("AdaptiveFilmstripProcessor", f"   ⚠️ Downscaling would make cells too small ({new_cell_width}×{new_cell_height}px)", "WARNING")
            log_component("AdaptiveFilmstripProcessor", f"   Using minimum cell size: {min_cell_size}px", "DEBUG")
            new_cell_width = int(min_cell_size * aspect_ratio)
            new_cell_height = min_cell_size
        
        # Calculate new grid dimensions and verify size
        new_total_width = grid_cols * new_cell_width + (grid_cols + 1) * self.border_thickness
        new_total_height = grid_rows * (new_cell_height + self.label_height) + (grid_rows + 1) * self.border_thickness
        new_size_mb = self._estimate_file_size(new_total_width, new_total_height)
        
        log_component("AdaptiveFilmstripProcessor", f"   🔽 Downscaling cells:", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"      Scale factor: {scale_factor:.3f}", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"      Cell: {cell_width}×{cell_height}px → {new_cell_width}×{new_cell_height}px", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"      Grid: {total_width}×{total_height}px → {new_total_width}×{new_total_height}px", "DEBUG")
        log_component("AdaptiveFilmstripProcessor", f"      Size: {current_size_mb:.2f}MB → {new_size_mb:.2f}MB", "DEBUG")
        
        return (new_cell_width, new_cell_height)
    
    def _calculate_grid_dimensions(
        self, 
        aspect_ratio: float, 
        source_resolution: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate optimal grid dimensions that fit within max_grid_size.
        
        Args:
            aspect_ratio: Width/height ratio of source video
            source_resolution: Source video resolution (width, height)
        
        Returns:
            Tuple of (cell_width, cell_height, grid_rows, grid_cols)
        """
        source_width, source_height = source_resolution
        
        # If fixed_grid_layout is provided, use it
        if self.fixed_grid_layout is not None:
            fixed_rows, fixed_cols = self.fixed_grid_layout
            log_component("AdaptiveFilmstripProcessor", f"📌 Using fixed grid layout: {fixed_rows}×{fixed_cols}", "DEBUG")
            
            # Calculate cell size based on fixed layout
            if self.preserve_source_resolution:
                # Use exact source resolution
                cell_width, cell_height = source_width, source_height
                log_component("AdaptiveFilmstripProcessor", f"   Cell size: {cell_width}×{cell_height}px (source resolution)", "DEBUG")
            else:
                # Calculate cell size that fits in the grid
                available_width = self.max_grid_width - (fixed_cols + 1) * self.border_thickness
                available_height = self.max_grid_height - (fixed_rows + 1) * self.border_thickness - fixed_rows * self.label_height
                
                cell_width = available_width // fixed_cols
                cell_height = available_height // fixed_rows
                
                # Adjust to maintain aspect ratio
                if cell_width / cell_height > aspect_ratio:
                    cell_width = int(cell_height * aspect_ratio)
                else:
                    cell_height = int(cell_width / aspect_ratio)
                
                log_component("AdaptiveFilmstripProcessor", f"   Cell size: {cell_width}×{cell_height}px (calculated)", "DEBUG")
            
            # Verify the layout fits
            total_width = fixed_cols * cell_width + (fixed_cols + 1) * self.border_thickness
            total_height = fixed_rows * (cell_height + self.label_height) + (fixed_rows + 1) * self.border_thickness
            
            if total_width > self.max_grid_width or total_height > self.max_grid_height:
                log_component("AdaptiveFilmstripProcessor", f"⚠️ Fixed layout {fixed_rows}×{fixed_cols} doesn't fit in {self.max_grid_width}×{self.max_grid_height}, using fallback", "WARNING")
                return (512, 512, 4, 5)
            
            log_component("AdaptiveFilmstripProcessor", f"   Grid dimensions: {total_width}×{total_height}px", "DEBUG")
            log_component("AdaptiveFilmstripProcessor", f"   Frames per grid: {fixed_rows * fixed_cols}", "DEBUG")
            
            # Apply file size constraint if specified
            if self.max_file_size_mb is not None:
                cell_width, cell_height = self._apply_file_size_constraint(
                    cell_width, cell_height, fixed_rows, fixed_cols, aspect_ratio
                )
            
            return (cell_width, cell_height, fixed_rows, fixed_cols)
        
        # If preserve_source_resolution is True, use exact source dimensions
        if self.preserve_source_resolution:
            log_component("AdaptiveFilmstripProcessor", f"🔒 Preserving source resolution: {source_width}×{source_height}", "DEBUG")
            
            # Calculate how many cells fit with exact source resolution
            cell_width, cell_height = source_width, source_height
            
            # Calculate max rows and cols that fit
            max_cols = (self.max_grid_width - self.border_thickness) // (cell_width + self.border_thickness)
            max_rows = (self.max_grid_height - self.border_thickness) // (cell_height + self.label_height + self.border_thickness)
            
            if max_cols < 1 or max_rows < 1:
                log_component("AdaptiveFilmstripProcessor", "⚠️ Source resolution too large for grid, using fallback", "WARNING")
                return (512, 512, 4, 5)
            
            log_component("AdaptiveFilmstripProcessor", f"   Grid: {max_rows}×{max_cols} ({max_rows * max_cols} frames/grid)", "DEBUG")
            
            # Apply file size constraint if specified
            if self.max_file_size_mb is not None:
                cell_width, cell_height = self._apply_file_size_constraint(
                    cell_width, cell_height, max_rows, max_cols, aspect_ratio
                )
            
            return (cell_width, cell_height, max_rows, max_cols)
        
        # Otherwise, optimize for maximum frames
        best_layout = None
        max_frames = 0
        
        # Try different grid configurations
        for rows in range(1, 21):  # Try up to 20 rows
            for cols in range(1, 21):  # Try up to 20 cols
                # Calculate cell dimensions for this layout
                available_width = self.max_grid_width - (cols + 1) * self.border_thickness
                available_height = self.max_grid_height - (rows + 1) * self.border_thickness - rows * self.label_height
                
                cell_width = available_width // cols
                cell_height = available_height // rows
                
                # Check if cells are too small
                if cell_width < 100 or cell_height < 100:
                    continue
                
                # Adjust cell size to maintain aspect ratio
                if cell_width / cell_height > aspect_ratio:
                    # Width is limiting factor
                    cell_width = int(cell_height * aspect_ratio)
                else:
                    # Height is limiting factor
                    cell_height = int(cell_width / aspect_ratio)
                
                # Verify the layout fits
                total_width = cols * cell_width + (cols + 1) * self.border_thickness
                total_height = rows * (cell_height + self.label_height) + (rows + 1) * self.border_thickness
                
                if total_width <= self.max_grid_width and total_height <= self.max_grid_height:
                    frames = rows * cols
                    if frames > max_frames:
                        max_frames = frames
                        best_layout = (cell_width, cell_height, rows, cols)
        
        if best_layout is None:
            # Fallback to a simple layout
            log_component("AdaptiveFilmstripProcessor", "⚠️ Using fallback layout", "WARNING")
            return (512, 512, 4, 5)
        
        cell_width, cell_height, rows, cols = best_layout
        
        # Apply file size constraint if specified
        if self.max_file_size_mb is not None:
            cell_width, cell_height = self._apply_file_size_constraint(
                cell_width, cell_height, rows, cols, aspect_ratio
            )
        
        return (cell_width, cell_height, rows, cols)
    
    def create_adaptive_filmstrips(
        self,
        video_file: str,
        output_prefix: str,
        video_duration: float = None,
        source_fps: float = None,
        source_resolution: Tuple[int, int] = None,
        detect_shot_changes: bool = True,
        start_time: float = 0.0,
        process_duration: float = None
    ) -> dict:
        """
        Create adaptive filmstrips from video file.
        
        Args:
            video_file: Path to video file
            output_prefix: Prefix for output files (e.g., 'filmstrip' -> 'filmstrip_0000.jpg', 'filmstrip_0001.jpg')
            video_duration: Video duration in seconds (auto-detected if None)
            source_fps: Source FPS (auto-detected if None)
            source_resolution: Source resolution (auto-detected if None)
            detect_shot_changes: Whether to detect shot changes
            start_time: Start time in seconds (default: 0.0)
            process_duration: Duration to process in seconds (default: None = process entire video)
        
        Returns:
            Dictionary with processing results:
            - layout: Layout parameters
            - output_files: List of created filmstrip files
            - shot_changes: List of shot change information per grid
        """
        log_component("AdaptiveFilmstripProcessor", f"🎬 Creating adaptive filmstrips from {video_file}")
        
        # Auto-detect video properties if not provided
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            log_component("AdaptiveFilmstripProcessor", f"❌ Cannot open video file: {video_file}", "ERROR")
            return {'layout': None, 'output_files': [], 'shot_changes': []}
        
        if source_fps is None:
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        if source_resolution is None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_resolution = (width, height)
        
        if video_duration is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / source_fps if source_fps > 0 else 0
        
        cap.release()
        
        # Apply start_time and process_duration constraints
        actual_video_duration = video_duration
        if process_duration is not None:
            # Limit to specified duration
            video_duration = min(process_duration, video_duration - start_time)
            log_component("AdaptiveFilmstripProcessor", f"⏱️ Processing {video_duration:.1f}s from {start_time:.1f}s (total video: {actual_video_duration:.1f}s)")
        elif start_time > 0:
            # Start from specified time, process to end
            video_duration = video_duration - start_time
            log_component("AdaptiveFilmstripProcessor", f"⏱️ Processing from {start_time:.1f}s to end ({video_duration:.1f}s)")
        else:
            # Default: process entire video
            log_component("AdaptiveFilmstripProcessor", f"⏱️ Processing entire video ({video_duration:.1f}s)")
        
        # Calculate optimal layout based on the duration to process
        layout = self.calculate_optimal_layout(video_duration, source_fps, source_resolution)
        
        # Create filmstrip processor with calculated dimensions
        processor = FilmstripProcessor(
            grid_rows=layout['grid_rows'],
            grid_cols=layout['grid_cols'],
            cell_width=layout['cell_size'][0],
            cell_height=layout['cell_size'][1],
            border_thickness=self.border_thickness,
            label_height=self.label_height,
            border_color=self.border_color,
            label_bg_color=self.label_bg_color,
            label_text_color=self.label_text_color,
            shot_detector=self.shot_detector
        )
        
        # Extract all frames at once
        log_component("AdaptiveFilmstripProcessor", f"📸 Extracting {layout['frames_to_extract']} frames...")
        all_frames = processor.extract_frames_from_video(
            video_file=video_file,
            start_time=start_time,  # Use the specified start time
            num_frames=layout['frames_to_extract'],
            interval=layout['extraction_interval']
        )
        
        # Create multiple grids
        output_files = []
        shot_changes_per_grid = []
        frames_per_grid = layout['frames_per_grid']
        grid_rows = layout['grid_rows']
        grid_cols = layout['grid_cols']
        
        for grid_idx in range(layout['num_grids_needed']):
            start_idx = grid_idx * frames_per_grid
            end_idx = min(start_idx + frames_per_grid, len(all_frames))
            grid_frames = all_frames[start_idx:end_idx]
            
            if not grid_frames:
                break
            
            # Generate output filename
            output_file = f"{output_prefix}_{grid_idx:04d}.jpg"
            
            log_component("AdaptiveFilmstripProcessor", f"🎞️ Creating grid {grid_idx + 1}/{layout['num_grids_needed']}...")
            
            # Create filmstrip for this grid
            shot_changes = processor.create_filmstrip(
                frames_with_timestamps=grid_frames,
                output_path=output_file,
                detect_shot_changes=detect_shot_changes
            )
            
            # Process shot changes with timestamps and grid positions
            shot_segments = []
            for shot_idx in shot_changes:
                # Calculate position in grid
                row = (shot_idx // grid_cols) + 1
                col = (shot_idx % grid_cols) + 1
                
                # Calculate timestamp from the frame data
                global_frame_idx = start_idx + shot_idx
                if global_frame_idx < len(all_frames):
                    _, timestamp = all_frames[global_frame_idx]
                    
                    shot_segments.append({
                        'frame_index': shot_idx,
                        'global_frame_index': global_frame_idx,
                        'row': row,
                        'col': col,
                        'timestamp': timestamp
                    })
            
            output_files.append(output_file)
            shot_changes_per_grid.append({
                'grid_index': grid_idx,
                'output_file': output_file,
                'shot_changes': shot_changes,  # Keep original for backward compatibility
                'shot_segments': shot_segments,  # New: detailed shot info with timestamps
                'frame_range': (start_idx, end_idx - 1)
            })
        
        log_component("AdaptiveFilmstripProcessor", f"✅ Created {len(output_files)} filmstrip grids")
        
        return {
            'layout': layout,
            'output_files': output_files,
            'shot_changes': shot_changes_per_grid
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Filmstrip Processor - Example Usage")
    print("=" * 80)
    
    print("\nExample 1: Create filmstrip from video file")
    print("-" * 80)
    print("""
from src.shared import FilmstripProcessor, create_fusion_detector

# Create processor with shot detection
detector = create_fusion_detector(threshold=0.7)
processor = FilmstripProcessor(
    grid_rows=4,
    grid_cols=5,
    shot_detector=detector
)

# Create filmstrip from video
shot_changes = processor.create_filmstrip_from_video(
    video_file='chunk_0000.mp4',
    output_path='filmstrip_0000.jpg',
    start_time=0.0,
    num_frames=20,
    interval=1.0
)

print(f"Shot changes detected at frames: {shot_changes}")
    """)
    
    print("\nExample 2: Use convenience functions")
    print("-" * 80)
    print("""
from src.shared import (
    create_fusion_filmstrip_processor,
    create_visual_filmstrip_processor,
    create_fusion_detector
)

# For Modality Fusion (4×5 grid)
detector = create_fusion_detector()
fusion_processor = create_fusion_filmstrip_processor(shot_detector=detector)

# For Visual Understanding (5×4 grid)
visual_processor = create_visual_filmstrip_processor(shot_detector=detector)

# Use them
fusion_processor.create_filmstrip_from_video('video.mp4', 'filmstrip.jpg')
    """)
    
    print("\nExample 3: Two-step process (extract then create)")
    print("-" * 80)
    print("""
from src.shared import FilmstripProcessor

processor = FilmstripProcessor()

# Step 1: Extract frames
frames = processor.extract_frames_from_video(
    video_file='video.mp4',
    start_time=0.0,
    num_frames=20
)

# Step 2: Create filmstrip
shot_changes = processor.create_filmstrip(
    frames_with_timestamps=frames,
    output_path='filmstrip.jpg',
    detect_shot_changes=True
)
    """)
    
    print("\nExample 4: Adaptive Filmstrip Processor")
    print("-" * 80)
    print("""
from src.shared import AdaptiveFilmstripProcessor, create_fusion_detector

# Create adaptive processor (optimized mode - default)
detector = create_fusion_detector()
adaptive_processor = AdaptiveFilmstripProcessor(
    max_grid_size=(8000, 8000),
    max_grid_images=20,
    preserve_source_resolution=False,  # Optimize for max frames
    shot_detector=detector
)

# Process video - automatically calculates optimal layout
result = adaptive_processor.create_adaptive_filmstrips(
    video_file='long_video.mp4',
    output_prefix='filmstrip',
    video_duration=300.0,  # 5 minutes
    source_fps=30.0,
    source_resolution=(1024, 720)
)

# Results
print(f"Layout: {result['layout']['grid_rows']}×{result['layout']['grid_cols']}")
print(f"Created {len(result['output_files'])} grids")
print(f"Sampling rate: {result['layout']['sampling_rate']}")
print(f"Files: {result['output_files']}")

# Example with 1024×720 source, 8000×8000 grid:
# OPTIMIZED MODE (preserve_source_resolution=False):
# - Calculates 15×15 grid (225 frames per grid)
# - Cell size: 524×368px (resized from source)
# - Max capacity: 225 × 20 = 4500 frames
# - Sampling: Every 2nd frame
# - Coverage: 50% of video
#
# PRESERVED MODE (preserve_source_resolution=True):
# - Calculates 10×7 grid (70 frames per grid)
# - Cell size: 1024×720px (exact source resolution)
# - Max capacity: 70 × 20 = 1400 frames
# - Sampling: Every 7th frame
# - Coverage: 14.3% of video
# - No quality loss from resizing
    """)
    
    print("\nExample 5: Adaptive Filmstrip with Preserved Resolution")
    print("-" * 80)
    print("""
from src.shared import AdaptiveFilmstripProcessor

# Create processor with preserved resolution (pixel-perfect quality)
processor = AdaptiveFilmstripProcessor(
    max_grid_size=(8000, 8000),
    max_grid_images=20,
    preserve_source_resolution=True  # Use exact source resolution
)

# Process video - cells will be exact source resolution
result = processor.create_adaptive_filmstrips(
    video_file='video.mp4',
    output_prefix='filmstrip_hq',
    source_resolution=(1024, 720)  # Cells will be exactly 1024×720
)

# Benefits:
# - Zero quality loss (no resizing)
# - Pixel-perfect frames
# - Ideal for detailed inspection
# Trade-off:
# - Fewer frames per grid
# - More sampling needed
# - Lower overall coverage
    """)
    
    print("\nExample 6: Fixed Grid Layout")
    print("-" * 80)
    print("""
from src.shared import AdaptiveFilmstripProcessor

# Create processor with fixed 4×5 grid layout
processor = AdaptiveFilmstripProcessor(
    max_grid_size=(8000, 8000),
    max_grid_images=20,
    fixed_grid_layout=(4, 5),  # Always use 4 rows × 5 cols = 20 frames/grid
    preserve_source_resolution=False  # Cell size will be calculated
)

# Process video - will use fixed 4×5 grid
result = processor.create_adaptive_filmstrips(
    video_file='video.mp4',
    output_prefix='filmstrip_fixed'
)

# Benefits:
# - Consistent grid layout across all videos
# - Predictable number of frames per grid (4×5 = 20)
# - Useful for standardized processing pipelines
# - Can combine with preserve_source_resolution=True

# Example combinations:
# 1. fixed_grid_layout=(4, 5) + preserve_source_resolution=False
#    → 4×5 grid with optimized cell size
# 2. fixed_grid_layout=(4, 5) + preserve_source_resolution=True
#    → 4×5 grid with exact source resolution cells
# 3. fixed_grid_layout=None (default)
#    → Auto-compute optimal grid layout
    """)
    
    print("\n" + "=" * 80)
    print("✅ Filmstrip Processor module ready for use!")
    print("=" * 80)
