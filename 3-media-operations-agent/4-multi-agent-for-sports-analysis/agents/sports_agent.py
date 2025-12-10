"""
Intelligent Sports Video Metadata Agent using Strands Agents

This agent analyzes sports video content and generates structured sports metadata
including match information, player details, and team data.

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
from sports_prompts import SportsPrompts
from gateway_helper import GatewayConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = 'global.anthropic.claude-sonnet-4-5-20250929-v1:0'


def process_sports(video_data: dict, s3_url: str = "") -> dict:
    """
    Main function to process sports video metadata using the Strands agent.
    
    Args:
        video_data: Dictionary containing video summaries
        s3_url: S3 URL of the video file (optional)
        
    Returns:
        Processed sports metadata in structured format
    """
    try:
        # Extract video summaries
        video_summaries = json.dumps(video_data.get('video_summaries', []))
        
        # Create prompt
        prompt = f"""Extract complete sports metadata from this video.

VIDEO SUMMARIES: {video_summaries}
S3 URL: {s3_url if s3_url else 'Not provided'}

REQUIRED INFORMATION:
1. Search the match database to identify the actual team names (not just colors)
2. Look up player information from the database using team names and player numbers
3. Provide complete metadata including:
   - Team names (actual names, not colors)
   - Player names and details
   - Match score and context
   - Key plays and events

Use the available tools to gather this information from the database."""
        
        # Initialize gateway configuration
        gateway_config = GatewayConfig()
        
        # Get access token from Cognito
        access_token = gateway_config.get_access_token()
        logger.info("Successfully obtained access token from Cognito")
        
        # Create MCP client
        mcp_client = gateway_config.create_mcp_client(access_token)
        
        # Use MCP client within context manager
        with mcp_client:
            # Get tools from gateway
            gateway_tools = mcp_client.list_tools_sync()
            logger.info(f"Loaded {len(gateway_tools)} tools from gateway")
            
            # Configure Bedrock model
            bedrock_model = BedrockModel(
                model_id=MODEL_ID,
                temperature=0.3
            )
            
            # Create agent with gateway tools
            sports_agent = Agent(
                model=bedrock_model,
                tools=gateway_tools,
                system_prompt=SportsPrompts.SYSTEM_PROMPT
            )
            
            logger.info("Invoking sports agent...")
            response = sports_agent(prompt)
            return response
        
    except Exception as e:
        logger.error(f"Error in process_sports: {e}")
        return json.dumps({"error": str(e), "processing_complete": False})