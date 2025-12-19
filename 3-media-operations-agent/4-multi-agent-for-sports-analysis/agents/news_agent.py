"""
Intelligent Video Metadata Agent using Strands Agents

This agent analyzes video content and generates structured news metadata.
Now uses AgentCore Gateway for tool access instead of local tools.
"""

import json
import logging
import sys
from pathlib import Path

# Add helper and prompts directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'helper'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'prompts'))

from strands import Agent
from strands.models import BedrockModel
from news_prompts import NewsPrompts
from gateway_helper import GatewayConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize gateway configuration
gateway_config = GatewayConfig()

# Configure Bedrock model
bedrock_model = BedrockModel(
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 4096
    }
)

def _create_agent():
    """Create agent with gateway tools"""
    access_token = gateway_config.get_access_token()
    mcp_client = gateway_config.create_mcp_client(access_token)
    
    with mcp_client:
        gateway_tools = mcp_client.list_tools_sync()
        logger.info(f"Loaded {len(gateway_tools)} tools from gateway")
        
        return Agent(
            model=bedrock_model,
            tools=gateway_tools,
            system_prompt=NewsPrompts.SYSTEM_PROMPT
        ), mcp_client

# Create agent instance
news_agent, _mcp_client = _create_agent()


def process_news(video_data: dict, s3_url: str = "") -> dict:
    """
    Main function to process video metadata using the Strands agent.
    
    Args:
        video_data: Dictionary containing video summaries
        s3_url: S3 URL of the video file (optional)
        
    Returns:
        Processed metadata in structured format
    """
    try:
        # Extract video summaries
        video_summaries = json.dumps(video_data.get('video_summaries', []))
        
        # Create prompt
        prompt = f"""Extract news metadata from this video:

VIDEO SUMMARIES: {video_summaries}
S3 URL: {s3_url if s3_url else 'Not provided'}"""
        
        # Run the agent with MCP client context
        with _mcp_client:
            return news_agent(prompt)
        
    except Exception as e:
        return json.dumps({"error": str(e), "processing_complete": False})