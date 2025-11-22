# Shared Components

This directory contains reusable components shared across all video understanding modules.

## Components

### 1. ShotChangeDetector

Unified shot change detection for video analysis.

**Features:**
- Histogram correlation method (default)
- MSE (Mean Squared Error) method
- Single frame detection mode
- Batch frame detection mode
- Cross-chunk detection support

**Used In:**
- `20. visual_understanding` - Single frame detection
- `40. modality_fused_understanding` - Batch detection with cross-chunk support

**Usage:**

```python
from src.shared import ShotChangeDetector, create_visual_detector, create_fusion_detector

# For Visual Understanding (single frame)
detector = create_visual_detector(threshold=0.3, method='histogram')
is_shot_change, detection_time = detector.detect_single(frame, frame_number)

# For Modality Fusion (batch with cross-chunk)
detector = create_fusion_detector(threshold=0.7)
shot_changes = detector.detect_batch(frames)  # Returns list of booleans

# Custom configuration
detector = ShotChangeDetector(
    method='histogram',  # or 'mse'
    threshold=0.7,
    enable_cross_chunk=True,
    hist_bins=(8, 8, 8)
)
```

**Methods:**
- `detect_single(frame, frame_number)` - Single frame detection
- `detect_batch(frames)` - Batch frame detection
- `reset()` - Reset detector state
- `get_config()` - Get current configuration

**Parameters:**
- `method`: 'histogram' or 'mse'
- `threshold`: Detection threshold
  - Histogram: correlation < threshold indicates shot change (0.0-1.0)
  - MSE: mse > threshold indicates shot change
- `enable_cross_chunk`: Enable cross-chunk detection
- `hist_bins`: Histogram bins for each channel (H, S, V)

## Future Shared Components

The following components are planned to be moved here:

### 2. RecordingManager (Planned)
- Manages continuous video recording to MXF format
- Used in: Visual Understanding + Modality Fusion

### 3. TranscriptionHandler (Planned)
- Processes Amazon Transcribe streaming results
- Used in: Audio Understanding + Modality Fusion

### 4. TranscriptionProcessor (Planned)
- Manages Amazon Transcribe streaming client
- Used in: Audio Understanding + Modality Fusion

### 5. ComponentMonitor (Planned)
- Provides logging and activity tracking
- Used in: All modules

## Benefits of Shared Components

1. **Code Deduplication**: Single source of truth eliminates duplicate code
2. **Consistency**: Same behavior across all modules
3. **Maintainability**: Fix bugs once, benefit everywhere
4. **Testability**: Test shared components once
5. **Extensibility**: Easy to add new features (e.g., deep learning-based detection)

## Installation

The shared components are automatically available when you import from `src.shared`:

```python
from src.shared import ShotChangeDetector
```

Make sure the `src/` directory is in your Python path.

## Testing

Run the module directly to see example usage:

```bash
python src/shared/shot_change_detector.py
```

## Version History

- **v1.0.0** (2024-11-12)
  - Initial release with ShotChangeDetector
  - Unified histogram and MSE detection methods
  - Support for single and batch detection modes
  - Cross-chunk detection capability
