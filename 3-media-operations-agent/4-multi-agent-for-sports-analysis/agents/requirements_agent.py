#!/usr/bin/env python3
"""
Simple Requirements Agent - Analyzes video summaries and extracts compliance requirements.
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
from requirements_prompts import RequirementsPrompts
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

def _get_agent_and_client():
    """Create agent with gateway tools - creates new client each time"""
    access_token = gateway_config.get_access_token()
    mcp_client = gateway_config.create_mcp_client(access_token)
    
    with mcp_client:
        gateway_tools = mcp_client.list_tools_sync()
        logger.info(f"Loaded {len(gateway_tools)} tools from gateway")
        
        agent = Agent(
            model=bedrock_model,
            tools=gateway_tools,
            system_prompt=RequirementsPrompts.SYSTEM_PROMPT
        )
        
        return agent, mcp_client


def get_requirements(content_type: str, distribution_channel: str) -> str:
    """
    Get compliance requirements (rules and guidelines) for content type and distribution channel.
    
    Args:
        content_type: Type of content (e.g., "news", "sports", "film")
        distribution_channel: Distribution channel (e.g., "social media", "website")
    
    Returns:
        List of compliance requirements as text
    """
    try:
        query = f"Retrieve the compliance requirements for {content_type} content on {distribution_channel}. Return ONLY the requirements list, not the schema."
        agent, mcp_client = _get_agent_and_client()
        with mcp_client:
            result = agent(query)
            return json.dumps(result) if isinstance(result, dict) else result
    except Exception as e:
        logger.error(f"Error in get_requirements: {e}")
        return json.dumps({"error": str(e)})


def get_output_schema(content_type: str, distribution_channel: str) -> str:
    """
    Get the output schema format for the content type and distribution channel.
    
    Args:
        content_type: Type of content (e.g., "news", "sports", "film")
        distribution_channel: Distribution channel (e.g., "social media", "website")
    
    Returns:
        JSON schema example showing the expected output format
    """
    try:
        query = f"Retrieve the output schema example for {content_type} content on {distribution_channel}. Return ONLY the JSON schema example, not the requirements."
        agent, mcp_client = _get_agent_and_client()
        with mcp_client:
            result = agent(query)
            return json.dumps(result) if isinstance(result, dict) else result
    except Exception as e:
        logger.error(f"Error in get_output_schema: {e}")
        return json.dumps({"error": str(e)})

def main():
    """Test compliance requirements extraction"""
    
    # Test different content types and channels
    test_cases = [
        ("news", "social media"),
        ("film", "website"),
        ("sports", "social media")
    ]
    
    for content_type, channel in test_cases:
        print(f"\n🔍 Testing: {content_type} on {channel}")
        print("=" * 60)
        
        print("\n📋 Requirements:")
        requirements = get_requirements(content_type, channel)
        print(requirements)
        
        print("\n📐 Output Schema:")
        schema = get_output_schema(content_type, channel)
        print(schema)

if __name__ == "__main__":
    main()