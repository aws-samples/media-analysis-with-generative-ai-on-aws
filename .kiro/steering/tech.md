# Technology Stack & Build System

## Core Technologies

### AWS Services
- **Amazon Bedrock** - Foundation models (Claude, Nova) for video/audio analysis
- **Bedrock Data Automation (BDA)** - Automated video content extraction
- **Bedrock AgentCore** - Agent runtime, memory, and gateway services
- **Amazon S3** - Video storage and vector data buckets
- **OpenSearch Serverless** - Vector search and indexing
- **DynamoDB** - Structured data storage (players, cast members)
- **Lambda** - Serverless functions for agent tools
- **MediaConvert** - Video processing and transcoding
- **Cognito** - Authentication for agent gateways

### AI/ML Frameworks
- **Strands Agent SDK** - Multi-agent orchestration and tool integration
- **Amazon Nova** - Multimodal foundation models for video understanding
- **Claude Models** - Text analysis and reasoning (Sonnet variants)

### Python Libraries
- **Video Processing**: OpenCV, MoviePy, FFmpeg-python, ImageIO
- **Audio Processing**: Librosa, SciPy, SoundFile
- **Data Science**: NumPy, Pandas, Matplotlib, FAISS (vector search)
- **AWS SDKs**: Boto3, SageMaker SDK
- **Notebook Environment**: Jupyter, IPython, IPywidgets

## Development Environment

### Prerequisites
- Python 3.8+ with pip
- FFmpeg (video/audio processing)
- AWS CLI configured with appropriate permissions
- SageMaker Studio (recommended) or local Jupyter environment

### Installation Commands
```bash
# Install system dependencies (Linux/SageMaker)
sudo apt update -y && sudo apt-get -y install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Verify FFmpeg installation
ffmpeg -version
```

### CloudFormation Deployment
```bash
# Deploy minimal workshop resources
aws cloudformation create-stack \
  --stack-name workshop \
  --template-body file://workshop-customer.yaml \
  --capabilities CAPABILITY_NAMED_IAM

# Deploy media operations agents infrastructure
aws cloudformation create-stack \
  --stack-name media-operations-agents \
  --template-body file://media-operations-agents.yaml \
  --capabilities CAPABILITY_NAMED_IAM

# Deploy live video understanding resources
aws cloudformation create-stack \
  --stack-name live-vu \
  --template-body file://live-vu-lab.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## Common Commands

### Notebook Execution
```bash
# Start Jupyter (if running locally)
jupyter notebook

# Run prerequisites setup
jupyter nbconvert --execute 00-prerequisites.ipynb
```

### Video Processing
```bash
# Create video clips with FFmpeg
ffmpeg -i input.mp4 -t 120 -c copy output-2min.mp4

# Extract frames at intervals
ffmpeg -i input.mp4 -vf fps=1/30 frame_%04d.jpg
```

### AWS Resource Management
```bash
# List CloudFormation stacks
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE

# Check Bedrock model access
aws bedrock list-foundation-models --region us-east-1

# Sync Knowledge Base data sources
aws bedrock-agent start-ingestion-job \
  --knowledge-base-id <kb-id> \
  --data-source-id <ds-id>
```

## Model Configuration

### Default Model IDs by module

**Module 2 (Nova — visual/audio segmentation, ad breaks, semantic search, summarization):**
- Generation: `global.amazon.nova-2-lite-v1:0` (Nova 2 Lite — unified multimodal)
- Embeddings: `amazon.nova-2-multimodal-embeddings-v1:0` (Nova Multimodal Embeddings — us-east-1 only)
- Reranking: `amazon.rerank-v1:0` (hard-coded to us-west-2 in `03-semantic-video-search.ipynb`; supports `rank_fields` in `additionalModelRequestFields`)

**Module 3 (Media Operations Agents):**
- Agent reasoning: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Knowledge Base embeddings: `amazon.titan-embed-text-v2:0` (KBs don't support Nova MM Embed; Titan stays)

**Module 4 (Live Media Analysis Agent):**
- Visual/Audio/Audiovisual analysis: `global.anthropic.claude-sonnet-4-20250514-v1:0`
- Live visual/audio understanding fallback: `us.amazon.nova-lite-v1:0`
- Companion agent Knowledge Base embeddings: `amazon.titan-embed-text-v2:0`

### Alternatives available in Bedrock
- Claude Sonnet 4.6 (`global.anthropic.claude-sonnet-4-6`) — next-gen Sonnet
- Claude Haiku 4.5 (`global.anthropic.claude-haiku-4-5-20251001-v1:0`) — fast, cheap
- Nova 2 Sonic — speech
- Cohere Rerank 3.5 (`cohere.rerank-v3-5:0`) — alternative reranker; only accepts `max_tokens_per_doc` in `additionalModelRequestFields` (rejects `rank_fields`)

### Agent Runtime
- **Strands Agents**: Version 1.16.0+
- **AgentCore Runtime**: Containerized deployment with Docker
- **MCP Tools**: Model Context Protocol for tool integration