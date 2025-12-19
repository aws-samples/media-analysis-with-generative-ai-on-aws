#!/usr/bin/env python3
"""
AgentCore Orchestrator Agent - Multi-Agent Orchestration for Media Operations

This agent orchestrates all specialized agents to:
1. Identify content type (sports, news, film)
2. Retrieve compliance requirements
3. Extract content-specific information
4. Generate compliant metadata
5. Validate metadata against requirements

Compatible with Bedrock AgentCore Runtime for deployment.
"""

import json
import logging
import sys
from pathlib import Path

from strands import Agent, tool
from strands.models import BedrockModel
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Import prompts and agent functions using proper package imports
from prompts.orchestrator_prompts import OrchestratorPrompts
from agents.sports_agent import process_sports
from agents.news_agent import process_news
from agents.film_agent import process_film
from agents.requirements_agent import get_requirements, get_output_schema
from agents.metadata_agent import process_metadata as metadata_process
from agents.qc_agent import qc_check

app = BedrockAgentCoreApp()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Bedrock model
bedrock_model = BedrockModel(
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    temperature=0.3,
    max_tokens=4096
)


@tool
def extract_sports_content(video_data: str, s3_url: str = "") -> str:
    """
    Extract sports information from video using the sports agent.
    Use this when the content is identified as sports-related.
    
    Args:
        video_data: JSON string with video_summaries, audio_transcripts, screen_texts
        s3_url: S3 URL of the video (optional)
    
    Returns:
        Sports analysis with teams, players, scores, and match details
    """
    try:
        if isinstance(video_data, str):
            video_dict = json.loads(video_data)
        else:
            video_dict = video_data
        
        result = process_sports(video_dict, s3_url)
        
        if isinstance(result, dict):
            return json.dumps(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Error in extract_sports_content: {e}")
        return json.dumps({"error": str(e)})


@tool
def extract_news_content(video_data: str, s3_url: str = "") -> str:
    """
    Extract news information from video using the news agent.
    Use this when the content is identified as news-related.
    
    Args:
        video_data: JSON string with video_summaries, audio_transcripts, screen_texts
        s3_url: S3 URL of the video (optional)
    
    Returns:
        News analysis with WHO, WHAT, WHEN, WHERE, WHY
    """
    try:
        if isinstance(video_data, str):
            video_dict = json.loads(video_data)
        else:
            video_dict = video_data
        
        result = process_news(video_dict, s3_url)
        
        if isinstance(result, dict):
            return json.dumps(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Error in extract_news_content: {e}")
        return json.dumps({"error": str(e)})


@tool
def extract_film_content(video_data: str, s3_url: str = "") -> str:
    """
    Extract film information from video using the film agent.
    Use this when the content is identified as film/movie-related.
    
    Args:
        video_data: JSON string with video_summaries, audio_transcripts, screen_texts
        s3_url: S3 URL of the video (optional)
    
    Returns:
        Film analysis with cast, characters, and scene details
    """
    try:
        if isinstance(video_data, str):
            video_dict = json.loads(video_data)
        else:
            video_dict = video_data
        
        result = process_film(video_dict, s3_url)
        
        if isinstance(result, dict):
            return json.dumps(result)
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    except Exception as e:
        logger.error(f"Error in extract_film_content: {e}")
        return json.dumps({"error": str(e)})


@tool
def get_compliance_requirements(content_type: str, distribution_channel: str) -> str:
    """
    Retrieve compliance requirements (rules and guidelines) for the content type and distribution channel.
    
    Args:
        content_type: Type of content (e.g., "sports", "news", "film")
        distribution_channel: Where content will be distributed (e.g., "social media", "website")
    
    Returns:
        List of compliance requirements as text
    """
    try:
        result = get_requirements(content_type, distribution_channel)
        return str(result) if not isinstance(result, str) else result
    except Exception as e:
        logger.error(f"Error in get_compliance_requirements: {e}")
        return json.dumps({"error": str(e)})


@tool
def get_output_schema_tool(content_type: str, distribution_channel: str) -> str:
    """
    Retrieve the output schema format for the content type and distribution channel.
    This shows the exact JSON structure that the final metadata must follow.
    
    Args:
        content_type: Type of content (e.g., "sports", "news", "film")
        distribution_channel: Where content will be distributed (e.g., "social media", "website")
    
    Returns:
        JSON schema example showing the expected output format
    """
    try:
        result = get_output_schema(content_type, distribution_channel)
        return str(result) if not isinstance(result, str) else result
    except Exception as e:
        logger.error(f"Error in get_output_schema: {e}")
        return json.dumps({"error": str(e)})


@tool
def generate_metadata(video_analysis: str, additional_info: str, 
                     requirements: str, output_format: str) -> str:
    """
    Generate compliant metadata based on video analysis and requirements.
    
    Args:
        video_analysis: Extracted content information from specialized agents
        additional_info: Additional context about the video
        requirements: Compliance requirements to follow
        output_format: Expected JSON schema format
    
    Returns:
        Generated metadata in the specified format
    """
    try:
        result = metadata_process(
            video_analysis=video_analysis,
            additional_info=additional_info,
            requirements=requirements,
            output_format=output_format
        )
        return json.dumps(result) if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error in generate_metadata: {e}")
        return json.dumps({"error": str(e)})


@tool
def enhance_metadata(current_metadata: str, feedback: str, video_analysis: str,
                    additional_info: str, requirements: str, output_format: str) -> str:
    """
    Enhance existing metadata based on feedback.
    
    Args:
        current_metadata: Current metadata that needs improvement
        feedback: Specific feedback on what to improve
        video_analysis: Original video analysis for reference
        additional_info: Additional context
        requirements: Compliance requirements
        output_format: Expected JSON schema format
    
    Returns:
        Enhanced metadata
    """
    try:
        result = metadata_process(
            video_analysis=video_analysis,
            additional_info=additional_info,
            requirements=requirements,
            output_format=output_format,
            current_metadata=current_metadata,
            feedback=feedback
        )
        return json.dumps(result) if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error in enhance_metadata: {e}")
        return json.dumps({"error": str(e)})


@tool
def validate_metadata(metadata: str, requirements: str, schema: str) -> str:
    """
    Validate metadata against compliance requirements and schema format.
    
    Args:
        metadata: Metadata to validate
        requirements: List of compliance requirements
        schema: Expected JSON schema format
    
    Returns:
        "Pass" if compliant, or list of specific violations
    """
    try:
        result = qc_check(metadata, requirements, schema)
        return json.dumps(result) if isinstance(result, dict) else str(result)
    except Exception as e:
        logger.error(f"Error in validate_metadata: {e}")
        return json.dumps({"error": str(e)})


# Create orchestrator agent
orchestrator_agent = Agent(
    model=bedrock_model,
    tools=[
        extract_sports_content,
        extract_news_content,
        extract_film_content,
        get_compliance_requirements,
        get_output_schema_tool,
        generate_metadata,
        enhance_metadata,
        validate_metadata
    ],
    system_prompt=OrchestratorPrompts.get_system_prompt()
)


@app.entrypoint
async def process_video_metadata(payload):
    """
    Process video metadata with intelligent routing and orchestration.
    
    The orchestrator automatically:
    - Detects query type (simple question, compliance, generation, validation)
    - Routes to appropriate specialized agents
    - Coordinates multi-step workflows
    - Validates outputs against compliance requirements
    
    Args:
        payload: Dict with:
            - video_summaries: List of video summary strings
            - s3_url: S3 URL of video (optional)
            - task_description: What to do
            - existing_metadata: JSON string of metadata to validate (optional)
    
    Yields:
        Streaming response chunks
    """
    try:
        video_summaries = payload.get("video_summaries", [])
        s3_url = payload.get("s3_url", "")
        task_description = payload.get("task_description", "")
        existing_metadata = payload.get("existing_metadata", "")
        
        # Create user prompt
        user_input = f"""Video Summaries: {json.dumps(video_summaries)}
S3 URL: {s3_url if s3_url else 'Not provided'}

Task: {task_description}"""
        
        if existing_metadata:
            user_input += f"\n\nExisting Metadata to Validate: {existing_metadata}"
        
        user_input += "\n\nComplete the requested task."
        
        # Stream the response
        stream = orchestrator_agent.stream_async(user_input)
        
        async for event in stream:
            # Stream regular text data
            if "data" in event:
                yield event["data"]
            
            # Include tool usage information for transparency
            if "message" in event and "content" in event["message"]:
                for content_item in event['message']['content']:
                    if "toolUse" in content_item:
                        tool_name = content_item["toolUse"].get("name", "Unknown")
                        # Optionally yield tool call information
                        # yield f"\n[Using tool: {tool_name}]\n"
                        
    except Exception as e:
        error_message = f"Error in process_video_metadata: {e}"
        logger.error(error_message)
        yield error_message


if __name__ == "__main__":
    app.run()
