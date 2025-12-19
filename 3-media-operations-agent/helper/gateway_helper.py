"""
Gateway Helper for AgentCore Gateway Integration

This module provides utilities for connecting agents to the AgentCore Gateway
with Cognito authentication. Configuration is read from AWS SSM Parameter Store.
"""

import json
import boto3
import requests
from typing import Optional
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client


class GatewayConfig:
    """Configuration for AgentCore Gateway connection"""
    
    def __init__(self, region_name: str = None):
        """
        Initialize gateway configuration from SSM Parameter Store
        
        Args:
            region_name: AWS region name. If not provided, uses default region
        """
        self.ssm_client = boto3.client('ssm', region_name=region_name)
        self.region = region_name or boto3.session.Session().region_name
        
        # Load configuration from SSM Parameter Store
        self.gateway_url = self._get_parameter('/sports-agent/gateway_url')
        self.user_pool_id = self._get_parameter('/sports-agent/cognito_user_pool_id')
        self.client_id = self._get_parameter('/sports-agent/cognito_client_id')
        self.client_secret = self._get_parameter('/sports-agent/cognito_client_secret', decrypt=True)
    
    def _get_parameter(self, parameter_name: str, decrypt: bool = False) -> str:
        """
        Get parameter value from SSM Parameter Store
        
        Args:
            parameter_name: Name of the SSM parameter
            decrypt: Whether to decrypt the parameter value
            
        Returns:
            Parameter value
        """
        try:
            response = self.ssm_client.get_parameter(
                Name=parameter_name,
                WithDecryption=decrypt
            )
            return response['Parameter']['Value']
        except Exception as e:
            raise ValueError(f"Failed to get SSM parameter {parameter_name}: {e}")
        
    def get_access_token(self) -> str:
        """
        Get OAuth2 access token from Cognito using client credentials flow
        
        Returns:
            Access token string
        """
        user_pool_id_without_underscore = self.user_pool_id.replace("_", "").lower()
        url = f"https://{user_pool_id_without_underscore}.auth.{self.region}.amazoncognito.com/oauth2/token"
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "media-agents-gateway-id/gateway:read media-agents-gateway-id/gateway:write",
        }
        
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        
        return response.json()["access_token"]
    
    def create_mcp_client(self, token: Optional[str] = None) -> MCPClient:
        """
        Create an MCP client configured for the gateway
        
        Args:
            token: Optional access token. If not provided, will fetch a new one
            
        Returns:
            Configured MCPClient instance
        """
        if token is None:
            token = self.get_access_token()
        
        def create_transport():
            return streamablehttp_client(
                url=self.gateway_url,
                headers={"Authorization": f"Bearer {token}"}
            )
        
        return MCPClient(create_transport)
