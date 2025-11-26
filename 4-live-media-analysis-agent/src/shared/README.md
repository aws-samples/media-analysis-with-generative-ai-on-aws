# Shared Components

This directory contains reusable components shared across all video understanding modules for building live video understanding solutions using Agentic AI on AWS.

## Workshop Overview

This workshop demonstrates how to build a streaming companion agent that provides viewers with real-time interactive assistance during live streamed shows. The agent offers immediate context about current events, generates ongoing summaries, and enables quick access to key moments within the stream.

![Use Case](../../static/introduction/usecase.png)

The workshop is structured in two parts:
1. **Live Video Understanding**: Implement solutions using AWS services and Amazon Bedrock foundation models to process visual, audio, and multi-modal content in real-time
2. **Intelligent Agent Development**: Develop and deploy an agent using AWS Strands Agents and Amazon Bedrock AgentCore that leverages video understanding to interact with viewers

## Key Technologies

- **[Amazon Bedrock](https://aws.amazon.com/bedrock/)**: Flexible platform for building generative AI applications and agents
- **[Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/)**: Agentic platform to build, deploy and operate agents securely at scale
- **[AWS Strands Agents](https://strandsagents.com/latest/)**: Simple-to-use, code-first framework for building AI agents
- **[Amazon Transcribe](https://aws.amazon.com/transcribe/)**: Fully managed, automatic speech recognition (ASR) service

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

## Workshop Modules

- **Module 1: Live Visual Understanding (30 minutes):** Analyze live video frames and detect shot changes using Amazon Bedrock with Nova Lite model for real-time visual understanding
- **Module 2: Live Audio Understanding (30 minutes):** Convert speech to text using Amazon Transcribe and analyze content using Nova Lite model for comprehensive audio understanding
- **Module 3: Live Multi-Modal Understanding (30 minutes):** Combine visual and audio insights using Claude Sonnet model to create contextual understanding of the live content
- **Module 4: Live Streaming Companion Agent (30 minutes):** Build and deploy an interactive AWS Strands agent on Amazon Bedrock AgentCore with integrated memory systems

## Key Learning Points

By using these shared components, you will learn to:

- **Live Video Processing:** Implement real-time video understanding solutions that address latency, incremental context, and resource optimization challenges
- **Foundation Model Integration:** Leverage Amazon Bedrock models for visual analysis, audio processing, and multi-modal understanding of streaming content
- **Agent Development:** Build an interactive agent using AWS Strands Agents that maintains video context and handles real-time viewer queries
- **Production Deployment:** Deploy and optimize streaming agents on Amazon Bedrock AgentCore with efficient memory management and cost controls

## Version History

- **v1.0.0** (2024-11-12)
  - Initial release with ShotChangeDetector
  - Unified histogram and MSE detection methods
  - Support for single and batch detection modes
  - Cross-chunk detection capability
