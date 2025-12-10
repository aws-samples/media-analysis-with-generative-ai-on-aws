"""
Simple Metadata Generation Agent using Strands Agents

This agent generates and enhances compliant video metadata based on 
requirements, feedback, and defined schemas.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add prompts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'prompts'))

from strands import Agent, tool
from strands.models import BedrockModel
from metadata_prompts import MetadataPrompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Bedrock model
bedrock_model = BedrockModel(
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 4096
    }
)

@tool
def create_metadata(video_analysis: str, additional_info: str, requirements: str, output_format: str) -> str:
    """Generate compliant video metadata from video analysis and requirements."""
    print("✨ Creating new metadata...")
    
    agent = Agent(model=bedrock_model, system_prompt=MetadataPrompts.CREATE_SYSTEM_PROMPT)
    
    prompt = MetadataPrompts.get_create_prompt(
        video_analysis=video_analysis,
        additional_info=additional_info,
        requirements=requirements,
        output_format=output_format
    )
    
    return agent(prompt)


@tool
def enhance_metadata(current_metadata: str, feedback: str, video_analysis: str, 
                    additional_info: str, requirements: str, output_format: str) -> str:
    """Enhance existing video metadata based on feedback."""
    print("✨ Enhancing existing metadata...")
    
    agent = Agent(model=bedrock_model, system_prompt=MetadataPrompts.ENHANCE_SYSTEM_PROMPT)
    
    prompt = MetadataPrompts.get_enhance_prompt(
        current_metadata=current_metadata,
        feedback=feedback,
        video_analysis=video_analysis,
        additional_info=additional_info,
        requirements=requirements,
        output_format=output_format
    )
    
    return agent(prompt)

# Create Metadata Agent
metadata_agent = Agent(
    model=bedrock_model,
    tools=[create_metadata, enhance_metadata],
    system_prompt=MetadataPrompts.MAIN_SYSTEM_PROMPT
)


def process_metadata(video_analysis: str = "", additional_info: str = "", 
                    requirements: str = "", output_format: str = "",
                    current_metadata: str = "", feedback: str = "") -> dict:
    """
    Process video metadata - either generate new or enhance existing based on whether feedback is provided.
    
    Args:
        video_analysis: Raw video analysis data
        additional_info: Additional context about the video
        requirements: Compliance requirements to follow
        output_format: Expected output format/schema
        current_metadata: Existing metadata to enhance (used with feedback)
        feedback: Feedback for improvements (triggers enhancement mode)
        
    Returns:
        Processed metadata response
    """
    try:
        # Set defaults for empty parameters
        video_analysis = video_analysis or "No video analysis provided"
        additional_info = additional_info or "No additional information"
        requirements = requirements or "Generate professional metadata"
        output_format = output_format or "JSON format with title and description"
        
        # Simple decision: if feedback is provided, enhance; otherwise create
        if feedback and feedback.strip():
            # Enhancement mode
            current_metadata = current_metadata or "No current metadata provided"
            query = MetadataPrompts.get_main_agent_enhance_prompt(
                current_metadata=current_metadata,
                feedback=feedback,
                video_analysis=video_analysis,
                additional_info=additional_info,
                requirements=requirements,
                output_format=output_format
            )
        else:
            # Creation mode
            query = MetadataPrompts.get_main_agent_create_prompt(
                video_analysis=video_analysis,
                additional_info=additional_info,
                requirements=requirements,
                output_format=output_format
            )
        
        return metadata_agent(query)
        
    except Exception as e:
        logger.error(f"Error in process_metadata: {e}")
        return json.dumps({"error": str(e), "processing_complete": False})