"""
Lambda function for Sports Agent Tools

Exposes sports tools as Lambda functions for AgentCore Gateway.
"""

import json
import os
import boto3
import time
from urllib.parse import urlparse

# Initialize AWS clients
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')
rekognition = boto3.client('rekognition')


def get_tool_name(event, context):
    """Extract tool name from event or context.
    
    Handles both direct Lambda invocation and AgentCore Gateway invocation.
    """
    # AgentCore Gateway format - tool name is in context.client_context
    if context and hasattr(context, 'client_context') and context.client_context:
        try:
            tool_name = context.client_context.custom.get('bedrockAgentCoreToolName', '')
            # Remove target prefix if present (format: "target-name___tool-name")
            delimiter = "___"
            if delimiter in tool_name:
                tool_name = tool_name[tool_name.index(delimiter) + len(delimiter):]
            return tool_name
        except (AttributeError, KeyError):
            pass
    
    # Direct Lambda invocation format (for testing)
    if 'tool_name' in event:
        return event.get('tool_name', '')
    
    return ''


def get_named_parameter(event, name):
    """Extract named parameter from event.
    
    Handles both direct Lambda invocation and AgentCore Gateway invocation.
    """
    # AgentCore Gateway format - parameters are directly in the event
    if name in event:
        return event.get(name)
    
    # Direct Lambda invocation format (for testing)
    if 'parameters' in event:
        parameters = event.get('parameters', {})
        return parameters.get(name)
    
    return None


def retrieve_match_info(query: str, max_results: int = 1) -> dict:
    """
    Retrieve relevant match information from sports knowledge base.
    
    Args:
        query: Search query for relevant match/game information
        max_results: Maximum number of results to return (default: 1)
        
    Returns:
        Dictionary with match retrieval results
    """
    try:
        kb_id = os.environ.get('SPORTS_KB_ID')
        if not kb_id:
            return {
                "error": "SPORTS_KB_ID environment variable not set",
                "note": "Falling back to video analysis only"
            }
        
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )
        return {"results": response.get('retrievalResults', [])}
    except Exception as e:
        return {"error": str(e), "note": "Falling back to video analysis only"}


def retrieve_compliance_requirements(query: str, max_results: int = 3) -> dict:
    """
    Retrieve compliance requirements from compliance knowledge base.
    
    Args:
        query: Search query for compliance requirements (e.g., "social media", "website")
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        Dictionary with compliance requirements retrieval results
    """
    try:
        kb_id = os.environ.get('COMPLIANCE_KB_ID')
        if not kb_id:
            return {
                "error": "COMPLIANCE_KB_ID environment variable not set",
                "note": "No compliance requirements available"
            }
        
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )
        return {"results": response.get('retrievalResults', [])}
    except Exception as e:
        return {"error": str(e), "note": "No compliance requirements available"}


def retrieve_news_articles(query: str, max_results: int = 2) -> dict:
    """
    Retrieve news articles from news knowledge base.
    
    Args:
        query: Search query for news articles
        max_results: Maximum number of results to return (default: 2)
        
    Returns:
        Dictionary with news article retrieval results
    """
    try:
        kb_id = os.environ.get('NEWS_KB_ID')
        if not kb_id:
            return {
                "error": "NEWS_KB_ID environment variable not set",
                "note": "No news articles available"
            }
        
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )
        return {"results": response.get('retrievalResults', [])}
    except Exception as e:
        return {"error": str(e), "note": "No news articles available"}


def retrieve_film_info(query: str, max_results: int = 2) -> dict:
    """
    Retrieve film information from films knowledge base.
    
    Args:
        query: Search query for film information
        max_results: Maximum number of results to return (default: 2)
        
    Returns:
        Dictionary with film information retrieval results
    """
    try:
        kb_id = os.environ.get('FILMS_KB_ID')
        if not kb_id:
            return {
                "error": "FILMS_KB_ID environment variable not set",
                "note": "No film information available"
            }
        
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )
        return {"results": response.get('retrievalResults', [])}
    except Exception as e:
        return {"error": str(e), "note": "No film information available"}


def lookup_player_info(team_name: str, player_number: str) -> dict:
    """
    Look up player information from DynamoDB table using team and player number.
    
    Args:
        team_name: Name of the team (e.g., "Lakers", "Warriors")
        player_number: Player's jersey number (e.g., "14", "8")
        
    Returns:
        Dictionary with player information
    """
    try:
        table_name = os.environ.get('DYNAMODB_TABLE', 'sports_players')
        table = dynamodb.Table(table_name)
        
        response = table.get_item(
            Key={
                'team_name': team_name,
                'player_number': player_number
            }
        )
        
        if 'Item' in response:
            return response['Item']
        else:
            return {
                "team_name": team_name,
                "player_number": player_number,
                "player_name": f"Player #{player_number}",
                "position": "Unknown",
                "status": "not_found"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "team_name": team_name,
            "player_number": player_number,
            "status": "lookup_failed"
        }


def get_cast_member(name: str, film: str) -> dict:
    """
    Look up cast member information from DynamoDB table using name and film.
    
    Args:
        name: Name of the cast member
        film: Name of the film
        
    Returns:
        Dictionary with cast member information
    """
    try:
        table_name = os.environ.get('CAST_TABLE', 'film-cast-members')
        table = dynamodb.Table(table_name)
        
        response = table.get_item(
            Key={
                'name': name,
                'film': film
            }
        )
        
        if 'Item' in response:
            return response['Item']
        else:
            return {
                "name": name,
                "film": film,
                "role": "Unknown",
                "status": "not_found"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "name": name,
            "film": film,
            "status": "lookup_failed"
        }


def identify_people_with_rekognition(s3_url: str, min_confidence: float = 95.0) -> dict:
    """
    Use AWS Rekognition to identify people/celebrities in video.
    Uses Rekognition's async video analysis API.
    
    Args:
        s3_url: S3 URL of the video file
        min_confidence: Minimum confidence threshold (default: 95.0)
        
    Returns:
        Dictionary with people identification results
    """
    try:
        # Parse S3 URL
        parsed = urlparse(s3_url)
        if parsed.scheme != 's3':
            return {"error": "Invalid S3 URL", "identified_people": []}
        
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        # Start celebrity recognition job
        response = rekognition.start_celebrity_recognition(
            Video={'S3Object': {'Bucket': bucket, 'Name': key}},
            NotificationChannel={
                'SNSTopicArn': os.environ.get('SNS_TOPIC_ARN', ''),
                'RoleArn': os.environ.get('REKOGNITION_ROLE_ARN', '')
            } if os.environ.get('SNS_TOPIC_ARN') else None
        )
        
        job_id = response['JobId']
        
        # Poll for completion (max 5 minutes)
        max_attempts = 60
        attempt = 0
        
        while attempt < max_attempts:
            result = rekognition.get_celebrity_recognition(JobId=job_id)
            status = result['JobStatus']
            
            if status == 'SUCCEEDED':
                # Extract unique celebrities
                celebrities = {}
                for celebrity_data in result.get('Celebrities', []):
                    celebrity = celebrity_data.get('Celebrity', {})
                    confidence = celebrity.get('Confidence', 0)
                    
                    if confidence >= min_confidence:
                        name = celebrity.get('Name', 'Unknown')
                        if name not in celebrities or confidence > celebrities[name]['confidence']:
                            celebrities[name] = {
                                'name': name,
                                'confidence': confidence
                            }
                
                celebrity_list = list(celebrities.values())
                
                return {
                    "identified_people": [c['name'] for c in celebrity_list],
                    "confidence_scores": [c['confidence'] for c in celebrity_list],
                    "total_found": len(celebrity_list)
                }
                
            elif status == 'FAILED':
                return {"error": "Rekognition job failed", "identified_people": []}
            
            # Wait before next poll
            time.sleep(5)
            attempt += 1
        
        return {"error": "Timeout waiting for Rekognition", "identified_people": []}
        
    except Exception as e:
        return {"error": str(e), "identified_people": []}


def lambda_handler(event, context):
    """
    Lambda handler for sports agent tools.
    
    Handles both direct Lambda invocation and AgentCore Gateway invocation.
    - AgentCore Gateway: tool name in context.client_context, parameters directly in event
    - Direct invocation: tool name and parameters in event
    """
    try:
        # Log the incoming event and context for debugging
        print(f"Received event: {json.dumps(event)}")
        if context and hasattr(context, 'client_context') and context.client_context:
            print(f"Context client_context: {context.client_context.custom}")
        
        tool_name = get_tool_name(event, context)
        print(f"Extracted tool name: {tool_name}")
        
        if tool_name == "retrieve_match_info":
            query = get_named_parameter(event, "query")
            max_results = get_named_parameter(event, "max_results") or 1
            
            result = retrieve_match_info(query, max_results)
            return {
                "statusCode": 200,
                "body": json.dumps(result)
            }
            
        elif tool_name == "retrieve_compliance_requirements":
            query = get_named_parameter(event, "query")
            max_results = get_named_parameter(event, "max_results") or 3
            
            result = retrieve_compliance_requirements(query, max_results)
            return {
                "statusCode": 200,
                "body": json.dumps(result)
            }
            
        elif tool_name == "retrieve_news_articles":
            query = get_named_parameter(event, "query")
            max_results = get_named_parameter(event, "max_results") or 2
            
            result = retrieve_news_articles(query, max_results)
            return {
                "statusCode": 200,
                "body": json.dumps(result)
            }
            
        elif tool_name == "retrieve_film_info":
            query = get_named_parameter(event, "query")
            max_results = get_named_parameter(event, "max_results") or 2
            
            result = retrieve_film_info(query, max_results)
            return {
                "statusCode": 200,
                "body": json.dumps(result)
            }
            
        elif tool_name == "lookup_player_info":
            team_name = get_named_parameter(event, "team_name")
            player_number = get_named_parameter(event, "player_number")
            
            result = lookup_player_info(team_name, player_number)
            return {
                "statusCode": 200,
                "body": json.dumps(result)
            }
            
        elif tool_name == "get_cast_member":
            name = get_named_parameter(event, "name")
            film = get_named_parameter(event, "film")
            
            result = get_cast_member(name, film)
            return {
                "statusCode": 200,
                "body": json.dumps(result)
            }
            
        elif tool_name == "identify_people_with_rekognition":
            s3_url = get_named_parameter(event, "s3_url")
            
            result = identify_people_with_rekognition(s3_url)
            return {
                "statusCode": 200,
                "body": json.dumps(result)
            }
        
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Unknown tool: {tool_name}"})
            }
            
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
