import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client
from strands.models import BedrockModel

# Initialize AWS clients
ssm_client = boto3.client('ssm')

# Model configuration
MODEL_ID = 'global.anthropic.claude-sonnet-4-5-20250929-v1:0'


def get_ssm_parameter(parameter_name: str) -> str:
    """Get parameter value from AWS Systems Manager Parameter Store."""
    try:
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=True
        )
        return response['Parameter']['Value']
    except Exception as e:
        print(f"Error getting SSM parameter {parameter_name}: {e}")
        return None

# System prompt for the sports video analysis agent
SYSTEM_PROMPT = """You are a sports video analysis assistant that answers user queries about the provided sports video.
You have video metadata, as well as additional tools to get more information from match reports and player table.
Base on the information you obtained from metadata and tools, generate a clear and accurate answer to the user query. 
DO NOT answer queries beyond the game you are reviewing.
"""


# Initialize the Bedrock model
model = BedrockModel(model_id=MODEL_ID, temperature=0.3)

# Initialize the AgentCore Runtime App
app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload, context=None):
    """AgentCore Runtime entrypoint function"""
    user_input = payload.get("prompt", "")
    
    # Access request headers - handle None case
    request_headers = context.request_headers or {}
    
    # Get Client JWT token
    auth_header = request_headers.get('Authorization', '')
    print(f"Authorization header present: {bool(auth_header)}")
    
    # Get Gateway URL from SSM Parameter Store
    gateway_url = get_ssm_parameter("/sports-agent/lab_gateway_url")
    
    if not gateway_url:
        return "Error: Gateway URL not found in SSM Parameter Store (/sports-agent/lab_gateway_url)"
    
    print(f"Using Gateway URL: {gateway_url}")
    
    # Create MCP client and agent within context manager if JWT token available
    if gateway_url and auth_header:
        try:
            mcp_client = MCPClient(lambda: streamablehttp_client(
                url=gateway_url,
                headers={"Authorization": auth_header}
            ))
            
            with mcp_client:
                # Get tools from MCP gateway
                tools = mcp_client.list_tools_sync()
                print(f"Loaded {len(tools)} tools from gateway")
                
                # Create the agent with gateway tools
                agent = Agent(
                    model=model,
                    tools=tools,
                    system_prompt=SYSTEM_PROMPT,
                )
                
                # Invoke the agent
                response = agent(user_input)
                
                # Extract response text
                if isinstance(response, dict):
                    return response.get('output', str(response))
                elif hasattr(response, 'message'):
                    return response.message["content"][0]["text"]
                else:
                    return str(response)
                    
        except Exception as e:
            print(f"MCP client error: {str(e)}")
            return f"Error: {str(e)}"
    else:
        error_msg = []
        if not gateway_url:
            error_msg.append("Missing gateway URL")
        if not auth_header:
            error_msg.append("Missing authorization header")
        return f"Error: {', '.join(error_msg)}"


if __name__ == "__main__":
    app.run()
