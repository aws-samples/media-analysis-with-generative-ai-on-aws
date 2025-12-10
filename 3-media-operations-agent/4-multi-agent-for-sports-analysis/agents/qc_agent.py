"""
Generic Quality Check (QC) Agent using Strands Agents

This agent intelligently evaluates metadata against any requirements,
using specific tools only when needed.
"""

import json
import logging
import re
import sys
from pathlib import Path

# Add prompts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'prompts'))

from strands import Agent, tool
from strands.models import BedrockModel
from qc_prompts import QCPrompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def word_count(text: str) -> int:
    """
    Count words in text.
    Use this tool when requirements mention word limits, word count, or maximum/minimum words.
    
    Args:
        text: Text to count words in
        
    Returns:
        Number of words in the text
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


@tool
def hashtag_count(text: str) -> int:
    """
    Count hashtags in text.
    Use this tool when requirements mention hashtags, #tags, or hashtag requirements.
    
    Args:
        text: Text to count hashtags in
        
    Returns:
        Number of hashtags found in the text
    """
    if not text or not isinstance(text, str):
        return 0
    hashtags = re.findall(r'#\w+', text)
    return len(hashtags)


@tool
def mention_count(text: str) -> int:
    """
    Count mentions in text.
    Use this tool when requirements mention mentions, @handles, or @ symbols.
    
    Args:
        text: Text to count mentions in
        
    Returns:
        Number of mentions found in the text
    """
    if not text or not isinstance(text, str):
        return 0
    mentions = re.findall(r'@\w+', text)
    return len(mentions)


@tool
def character_count(text: str) -> int:
    """
    Count characters in text.
    Use this tool when requirements mention character limits, character count, or text length.
    
    Args:
        text: Text to count characters in
        
    Returns:
        Number of characters in the text
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text)

# Configure Bedrock model
bedrock_model = BedrockModel(
    model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 4096
    }
)
# Generic QC agent
qc_agent = Agent(
    model=bedrock_model,
    tools=[word_count, hashtag_count, mention_count, character_count],
    system_prompt=QCPrompts.SYSTEM_PROMPT
)


def qc_check(metadata: str, requirements: str, schema: str) -> dict:
    """
    Generic QC check that evaluates metadata against requirements and schema format.
    
    Args:
        metadata: Metadata to evaluate (string)
        requirements: List of requirements to check against (string)
        schema: Expected schema format (string)
        
    Returns:
        QC evaluation result
    """
    prompt = QCPrompts.get_evaluation_prompt(
        metadata=metadata,
        requirements=requirements,
        schema=schema
    )
    try:
        return qc_agent(prompt)
    except Exception as e:
        logger.error(f"Error in qc_check: {e}")
        return json.dumps({"error": str(e), "processing_complete": False})