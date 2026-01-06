# Project Organization & Structure

## Root Level Files

### Setup & Configuration
- `00-prerequisites.ipynb` - **REQUIRED FIRST** - Environment setup, package installation, AWS resource deployment
- `requirements.txt` - Python dependencies for entire workshop
- `workshop-customer.yaml` - Minimal CloudFormation stack (S3, MediaConvert, OpenSearch)
- `media-operations-agents.yaml` - Agent infrastructure (Knowledge Bases, DynamoDB, Lambda, Gateway)
- `live-vu-lab.yaml` - Live video understanding resources (AgentCore Memory, Knowledge Base)

### Documentation
- `README.md` - Workshop overview and getting started guide
- `09-resources.ipynb` - Additional resources and reference links
- `10-cleanup.ipynb` - Resource cleanup procedures

## Workshop Modules

### 1. Media Analysis using BDA (`1-media-analysis-using-bda/`)
Bedrock Data Automation for automated video content extraction.

```
1-media-analysis-using-bda/
├── 01-extract-analyze-a-movie.ipynb    # BDA video analysis workflow
├── 02-contextual-ad-overlay.ipynb     # Ad targeting use case
├── utils.py                           # BDA utility functions
└── static/                            # Ad images and examples
    ├── ads/                           # Sample advertisement images
    └── images/                        # Documentation images
```

### 2. Media Analysis using Amazon Nova (`2-media-analysis-using-amazon-nova/`)
Foundation model-based video understanding with visual and audio segmentation.

```
2-media-analysis-using-amazon-nova/
├── 01A-visual-segments-frames-shots-scenes.ipynb  # Visual segmentation (REQUIRED)
├── 01B-audio-segments.ipynb                       # Audio segmentation (REQUIRED)
├── 02-ad-breaks-and-contextual-ad-targeting.ipynb # Ad break detection
├── 03-semantic-video-search.ipynb                 # Multi-modal search
├── 04-video-summarization.ipynb                   # Video summarization
├── lib/                                            # Core processing libraries
│   ├── bedrock_helper.py              # Bedrock API interactions
│   ├── frame_utils.py                 # Frame processing utilities
│   ├── scenes.py                      # Scene detection logic
│   ├── shots.py                       # Shot boundary detection
│   ├── transcript.py                  # Audio transcription handling
│   └── util.py                        # Common utilities
├── Netflix_Open_Content_Meridian/     # Processed video data
│   ├── frames/                        # Extracted video frames
│   ├── chapters/                      # Chapter representative images
│   ├── scenes/                        # Scene boundary data
│   └── *.json                         # Metadata files
└── static/                            # UI assets and icons
```

### 3. Media Operations Agents (`3-media-operations-agent/`)
AI agents for sports analysis, compliance, and content operations.

```
3-media-operations-agent/
├── 1-create-an-sports-agent/          # Basic sports agent
├── 2-sports-agent-with-gateway/       # Agent with MCP Gateway
├── 3-sports-agent-on-runtime/         # AgentCore Runtime deployment
├── 4-multi-agent-for-sports-analysis/ # Multi-agent orchestration
│   ├── agentcore_orchestrator_agent.py # Main orchestrator
│   ├── agents/                        # Individual agent implementations
│   ├── prompts/                       # Agent prompt templates
│   └── .bedrock_agentcore.yaml        # Runtime configuration
├── helper/                            # Infrastructure helpers
│   ├── agentcore_helper.py           # AgentCore operations
│   ├── bedrock_agent_helper.py       # Bedrock agent utilities
│   ├── gateway_helper.py             # MCP Gateway management
│   └── lambda_helper.py              # Lambda function utilities
└── resources/                         # Data and infrastructure
    ├── knowledge_base/                # KB data (sports, compliance, news, films)
    ├── dynamodb/                      # Sample data (players, cast)
    └── lambda/                        # Lambda function code
```

### 4. Live Media Analysis Agent (`4-live-media-analysis-agent/`)
Real-time streaming video analysis and companion agents.

```
4-live-media-analysis-agent/
├── 01-visual-understanding/           # Real-time visual analysis
├── 02-audio-understanding/            # Real-time audio processing
├── 03-modality-fused-understanding/   # Multi-modal fusion
├── 04-Live-companion-agent/           # Streaming companion
├── sample_videos/                     # Test video content
│   ├── Netflix_Open_Content_Meridian.mp4  # Full video
│   └── netflix-2mins.mp4              # 2-minute sample
└── src/shared/                        # Shared components
```

## Code Organization Patterns

### Library Structure
- **`lib/` directories** - Reusable processing modules (frames, scenes, transcripts)
- **`helper/` directories** - AWS service interaction helpers
- **`components/` directories** - UI and visualization components

### Data Flow
1. **Input**: Video files in `sample_videos/` or S3 buckets
2. **Processing**: Frame extraction → Scene detection → Analysis
3. **Storage**: Processed data in JSON files and S3
4. **Output**: Insights, summaries, and search indices

### Configuration Management
- **Notebook variables** - Stored using `%store` magic for cross-notebook sharing
- **CloudFormation outputs** - AWS resource IDs and endpoints
- **Model configurations** - Bedrock model IDs and parameters

## Key Conventions

### File Naming
- Notebooks: `##-descriptive-name.ipynb` (numbered sequence)
- Data files: `lowercase_with_underscores.json`
- Helper modules: `service_helper.py` pattern
- Static assets: Organized by type (`images/`, `ads/`, `icons/`)

### Import Patterns
```python
# AWS services
import boto3
from botocore.exceptions import ClientError

# Workshop libraries
from lib import bedrock_helper, frame_utils
from helper import agentcore_helper

# Agent frameworks
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
```

### Error Handling
- Graceful fallbacks for missing dependencies (FFmpeg, model access)
- CloudFormation stack existence checks before deployment
- Resource cleanup procedures in dedicated notebooks