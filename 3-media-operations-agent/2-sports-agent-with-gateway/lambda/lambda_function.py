"""
Lambda function for Sports Agent Tools

Exposes sports tools as Lambda functions for AgentCore Gateway.
"""

import json
import os
import boto3

# Initialize AWS clients
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')


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
            
        elif tool_name == "lookup_player_info":
            team_name = get_named_parameter(event, "team_name")
            player_number = get_named_parameter(event, "player_number")
            
            result = lookup_player_info(team_name, player_number)
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
