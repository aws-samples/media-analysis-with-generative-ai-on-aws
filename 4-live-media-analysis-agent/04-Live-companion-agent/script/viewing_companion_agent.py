#!/usr/bin/env python3
"""
Viewing Companion Agent for AgentCore Runtime
Provides real-time video understanding with memory and summarization
"""
import json
import logging
import boto3
from typing import List, Dict

from strands import Agent, tool
from strands.models import BedrockModel
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Initialize app
app = BedrockAgentCoreApp()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
ssm_client = boto3.client('ssm')
agentcore_client = boto3.client('bedrock-agentcore')
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')


def get_ssm_parameter(parameter_name: str) -> str:
    """Get parameter value from AWS Systems Manager Parameter Store."""
    try:
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=True
        )
        return response['Parameter']['Value']
    except Exception as e:
        logger.error(f"Error getting SSM parameter {parameter_name}: {e}")
        return None


# Load configuration from SSM Parameter Store
MODEL_ID = get_ssm_parameter("/viewing-companion/model_id")
MEMORY_ID = get_ssm_parameter("/viewing-companion/memory_id")
ACTOR_ID = get_ssm_parameter("/viewing-companion/actor_id")
SESSION_ID = get_ssm_parameter("/viewing-companion/session_id")
ROLLING_SUMMARY_NAMESPACE = get_ssm_parameter("/viewing-companion/rolling_summary_namespace")
KB_ID = get_ssm_parameter("/viewing-companion/kb_id")

logger.info(f"Configuration loaded - Memory: {MEMORY_ID}, KB: {KB_ID}")


# System prompt
SYSTEM_PROMPT = """You are a companion agent watching a live streaming show with the viewer. you receive events when topic changes in the show.
Guideline:
- Only respond to the user question about the show
- If provided, closely follow the output format user asked for. No extra explanations
- Don't ask questions back
- Respond in clear and professional manner
- Summarize response in one complete paragraphs, long response need to follow chronological order
- DO NOT respond to questions outside of the shows you are watching.
"""


# Memory Hook to load recent events
class MemoryHook(HookProvider):
    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load recent events and metadata from short-term memory"""
        try:
            events_resp = agentcore_client.list_events(
                memoryId=MEMORY_ID,
                actorId=ACTOR_ID,
                sessionId=SESSION_ID,
                maxResults=1,
                includePayloads=True
            )
            
            if events_resp.get('events'):
                latest_event = events_resp['events'][0]
                
                # Extract event content
                events = []
                for payload in latest_event.get('payload', []):
                    if 'conversational' in payload:
                        conv = payload['conversational']
                        events.append(f"{conv.get('role')}: {conv.get('content', {}).get('text', '')}")
                
                # Extract metadata
                metadata = latest_event.get('metadata', {})
                metadata_info = []
                if 'title' in metadata:
                    metadata_info.append(f"Show: {metadata['title'].get('stringValue', '')}")
                if 'genre' in metadata:
                    metadata_info.append(f"Genre: {metadata['genre'].get('stringValue', '')}")
                if 'start_ms' in metadata and 'end_ms' in metadata:
                    start = metadata['start_ms'].get('stringValue', '0')
                    end = metadata['end_ms'].get('stringValue', '0')
                    metadata_info.append(f"Timestamp: {start}ms - {end}ms")
                
                # Build context
                context_parts = []
                if metadata_info:
                    context_parts.append("Show Information:\n" + "\n".join(metadata_info))
                if events:
                    context_parts.append("\nMost Recent Event:\n" + "\n".join(events))
                
                if context_parts:
                    event.agent.system_prompt += "\n\n" + "\n\n".join(context_parts)
                
        except Exception as e:
            logger.error(f"Memory load error: {e}")
    
    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)


# Tool definitions
@tool
def get_show_summary() -> str:
    """
    Retrieve a comprehensive summary of what has happened in the current live video show so far.
    Use this when users ask about show content, current topics, or what they've missed.
    """
    try:
        # Get latest summary from memory records
        summary_resp = agentcore_client.list_memory_records(
            memoryId=MEMORY_ID,
            namespace=ROLLING_SUMMARY_NAMESPACE,
            maxResults=1
        )
        summary = summary_resp['memoryRecordSummaries'][0]['content']['text'] if summary_resp.get('memoryRecordSummaries') else ""
        
        # Get latest event
        events_resp = agentcore_client.list_events(
            memoryId=MEMORY_ID,
            actorId=ACTOR_ID,
            sessionId=SESSION_ID,
            maxResults=1,
            includePayloads=True
        )
        
        latest_event = ""
        if events_resp.get('events'):
            for payload in events_resp['events'][0].get('payload', []):
                if 'conversational' in payload:
                    latest_event = payload['conversational']['content']['text']
        
        # Combine summary and latest event
        context = []
        if summary:
            context.append(f"Show Summary: {summary}")
        if latest_event:
            context.append(f"Latest Update: {latest_event}")
            
        return "\n\n".join(context) if context else "No show information available yet."
        
    except Exception as e:
        logger.error(f"Error in get_show_summary: {e}")
        return f"Unable to retrieve show summary: {str(e)}"


@tool
def search_key_moments(query: str, max_results: int = 2) -> List[Dict]:
    """
    Search key moments of the video from Knowledge Base.
    
    Args:
        query: Search query text describing what you're looking for
        max_results: Maximum number of results to return (default: 2)
        
    Returns:
        List of chapters with content, start_ms, and end_ms
    """
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )
        
        results = []
        for result in response['retrievalResults']:
            metadata = result.get('metadata', {})
            results.append({
                'content': result['content']['text'],
                'start_ms': metadata.get('start_ms', 0),
                'end_ms': metadata.get('end_ms', 0)
            })
        
        return results
    except Exception as e:
        logger.error(f"Error in search_key_moments: {e}")
        return [{"error": f"Unable to search key moments: {str(e)}"}]


@app.entrypoint
async def invoke(payload, context=None):
    """
    AgentCore Runtime entrypoint function.
    
    Args:
        payload: Dict with:
            - prompt: User query about the show
    
    Yields:
        Streaming response chunks
    """
    try:
        user_input = payload.get("prompt", "")
        
        # Validate configuration
        if not all([MODEL_ID, MEMORY_ID, ACTOR_ID, SESSION_ID, ROLLING_SUMMARY_NAMESPACE, KB_ID]):
            yield "Error: Missing required SSM parameters"
            return
        
        logger.info(f"Processing request - Memory: {MEMORY_ID}, KB: {KB_ID}")
        
        # Initialize model
        model = BedrockModel(model_id=MODEL_ID, temperature=0.3)
        
        # Create agent with memory hook and tools
        agent = Agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=[get_show_summary, search_key_moments],
            hooks=[MemoryHook()]
        )
        
        # Stream the response
        stream = agent.stream_async(user_input)
        
        async for event in stream:
            # Stream regular text data
            if "data" in event:
                yield event["data"]
            
            # Include tool usage information for transparency (optional)
            if "message" in event and "content" in event["message"]:
                for content_item in event['message']['content']:
                    if "toolUse" in content_item:
                        tool_name = content_item["toolUse"].get("name", "Unknown")
                        # Optionally yield tool call information
                        # yield f"\n[Using tool: {tool_name}]\n"
                        
    except Exception as e:
        error_message = f"Error in invoke: {e}"
        logger.error(error_message)
        yield error_message


if __name__ == "__main__":
    app.run()
