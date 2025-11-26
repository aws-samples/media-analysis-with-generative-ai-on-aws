"""
Unified Shot Change Detector
Detects shot transitions in video using histogram correlation or MSE methods.
Supports both single-frame and batch processing modes.

This module is shared across:
- 20. visual_understanding (single frame detection)
- 40. modality_fused_understanding (batch frame detection with cross-chunk support)
"""

import cv2
import numpy as np
import time
from typing import Tuple, List, Optional


class ShotChangeDetector:
    """
    Unified shot change detection for video analysis.
    
    Supports:
    - Histogram correlation method (default)
    - MSE (Mean Squared Error) method
    - Single frame detection
    - Batch frame detection
    - Cross-chunk detection (optional)
    
    Usage Examples:
    
    # Single frame detection (Visual Understanding)
    detector = ShotChangeDetector(method='histogram', threshold=0.3)
    is_shot_change, detection_time = detector.detect_single(frame, frame_number)
    
    # Batch detection (Modality Fusion)
    detector = ShotChangeDetector(method='histogram', threshold=0.7, enable_cross_chunk=True)
    shot_changes = detector.detect_batch(frames)  # Returns list of booleans
    """
    
    def __init__(
        self, 
        method: str = 'histogram',
        threshold: float = 0.7,
        enable_cross_chunk: bool = False,
        hist_bins: Tuple[int, int, int] = (8, 8, 8)
    ):
        """
        Initialize shot change detector.
        
        Args:
            method: Detection method - 'histogram' or 'mse'
            threshold: Detection threshold
                - For histogram: correlation < threshold indicates shot change (0.0-1.0)
                - For MSE: mse > threshold indicates shot change
            enable_cross_chunk: Enable cross-chunk detection (compares with previous batch's last frame)
            hist_bins: Histogram bins for each channel (H, S, V) - default (8, 8, 8)
        """
        if method not in ['histogram', 'mse']:
            raise ValueError(f"Invalid method: {method}. Must be 'histogram' or 'mse'")
        
        self.method = method
        self.threshold = threshold
        self.enable_cross_chunk = enable_cross_chunk
        self.hist_bins = hist_bins
        
        # State for cross-chunk detection
        self.last_frame = None
        self.previous_histogram = None
    
    def detect_single(self, frame: np.ndarray, frame_number: int) -> Tuple[bool, float]:
        """
        Detect shot change for a single frame (used in Visual Understanding).
        
        Args:
            frame: Current video frame (BGR format)
            frame_number: Frame number in sequence
            
        Returns:
            Tuple of (is_shot_change: bool, detection_time: float)
        """
        detection_start = time.time()
        shot_change = False
        
        if self.method == 'histogram':
            shot_change = self._detect_histogram_single(frame)
        elif self.method == 'mse':
            shot_change = self._detect_mse_single(frame)
        
        detection_time = time.time() - detection_start
        return shot_change, detection_time
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[bool]:
        """
        Detect shot changes for a batch of frames (used in Modality Fusion).
        
        Args:
            frames: List of video frames (BGR format)
            
        Returns:
            List of booleans indicating shot changes for each frame
        """
        if not frames:
            return []
        
        shot_changes = [False] * len(frames)  # First frame is never a shot change by default
        
        if len(frames) < 2:
            return shot_changes
        
        # Cross-chunk detection: compare first frame with previous batch's last frame
        if self.enable_cross_chunk and self.last_frame is not None:
            if self.method == 'histogram':
                is_change = self._compare_frames_histogram(self.last_frame, frames[0])
            else:  # mse
                is_change = self._compare_frames_mse(self.last_frame, frames[0])
            
            if is_change:
                shot_changes[0] = True
        
        # Detect shot changes within the batch
        for i in range(1, len(frames)):
            if self.method == 'histogram':
                is_change = self._compare_frames_histogram(frames[i-1], frames[i])
            else:  # mse
                is_change = self._compare_frames_mse(frames[i-1], frames[i])
            
            if is_change:
                shot_changes[i] = True
        
        # Store last frame for next batch (if cross-chunk enabled)
        if self.enable_cross_chunk and frames:
            self.last_frame = frames[-1].copy()
        
        return shot_changes
    
    def reset(self):
        """Reset detector state (clears previous frame/histogram for cross-chunk detection)"""
        self.last_frame = None
        self.previous_histogram = None
    
    # ========================================================================
    # Private Methods - Histogram Detection
    # ========================================================================
    
    def _detect_histogram_single(self, frame: np.ndarray) -> bool:
        """Detect shot change using histogram correlation (single frame mode)"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist(
            [hsv], 
            [0, 1, 2], 
            None, 
            [self.hist_bins[0], self.hist_bins[1], self.hist_bins[2]], 
            [0, 180, 0, 256, 0, 256]
        )
        
        # Compare with previous histogram
        if self.previous_histogram is not None:
            correlation = cv2.compareHist(self.previous_histogram, hist, cv2.HISTCMP_CORREL)
            self.previous_histogram = hist
            return correlation < self.threshold
        
        self.previous_histogram = hist
        return False
    
    def _compare_frames_histogram(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Compare two frames using histogram correlation"""
        # Convert both frames to HSV
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist1 = cv2.calcHist(
            [hsv1], 
            [0, 1, 2], 
            None, 
            [self.hist_bins[0], self.hist_bins[1], self.hist_bins[2]], 
            [0, 180, 0, 256, 0, 256]
        )
        hist2 = cv2.calcHist(
            [hsv2], 
            [0, 1, 2], 
            None, 
            [self.hist_bins[0], self.hist_bins[1], self.hist_bins[2]], 
            [0, 180, 0, 256, 0, 256]
        )
        
        # Compare histograms
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation < self.threshold
    
    # ========================================================================
    # Private Methods - MSE Detection
    # ========================================================================
    
    def _detect_mse_single(self, frame: np.ndarray) -> bool:
        """Detect shot change using MSE (single frame mode)"""
        if self.last_frame is not None:
            mse = self._calculate_mse(self.last_frame, frame)
            self.last_frame = frame.copy()
            return mse > self.threshold
        
        self.last_frame = frame.copy()
        return False
    
    def _compare_frames_mse(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Compare two frames using MSE"""
        mse = self._calculate_mse(frame1, frame2)
        return mse > self.threshold
    
    def _calculate_mse(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate Mean Squared Error between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Resize if shapes don't match
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Calculate MSE
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        return mse
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_config(self) -> dict:
        """Get current detector configuration"""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'enable_cross_chunk': self.enable_cross_chunk,
            'hist_bins': self.hist_bins
        }
    
    def __repr__(self) -> str:
        return (f"ShotChangeDetector(method='{self.method}', threshold={self.threshold}, "
                f"enable_cross_chunk={self.enable_cross_chunk})")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_visual_detector(threshold: float = 0.3, method: str = 'histogram') -> ShotChangeDetector:
    """
    Create detector configured for Visual Understanding module.
    
    Args:
        threshold: Detection threshold (default 0.3 for histogram)
        method: 'histogram' or 'mse'
    
    Returns:
        Configured ShotChangeDetector for single-frame detection
    """
    return ShotChangeDetector(
        method=method,
        threshold=threshold,
        enable_cross_chunk=False
    )


def create_fusion_detector(threshold: float = 0.7) -> ShotChangeDetector:
    """
    Create detector configured for Modality Fusion module.
    
    Args:
        threshold: Detection threshold (default 0.7 for histogram)
    
    Returns:
        Configured ShotChangeDetector for batch detection with cross-chunk support
    """
    return ShotChangeDetector(
        method='histogram',
        threshold=threshold,
        enable_cross_chunk=True
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Shot Change Detector - Example Usage")
    print("=" * 80)
    
    # Example 1: Visual Understanding (single frame detection)
    print("\n1. Visual Understanding Mode (single frame):")
    print("-" * 80)
    detector_visual = create_visual_detector(threshold=0.3, method='histogram')
    print(f"Detector: {detector_visual}")
    print(f"Config: {detector_visual.get_config()}")
    
    # Example 2: Modality Fusion (batch detection)
    print("\n2. Modality Fusion Mode (batch with cross-chunk):")
    print("-" * 80)
    detector_fusion = create_fusion_detector(threshold=0.7)
    print(f"Detector: {detector_fusion}")
    print(f"Config: {detector_fusion.get_config()}")
    
    # Example 3: Custom configuration
    print("\n3. Custom Configuration:")
    print("-" * 80)
    detector_custom = ShotChangeDetector(
        method='mse',
        threshold=1000.0,
        enable_cross_chunk=True,
        hist_bins=(16, 16, 16)
    )
    print(f"Detector: {detector_custom}")
    print(f"Config: {detector_custom.get_config()}")
    
    print("\n" + "=" * 80)
    print("✅ Shot Change Detector module ready for use!")
    print("=" * 80)
