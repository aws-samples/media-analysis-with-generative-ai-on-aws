# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission

"""
This module contains a helper class for working with AWS Cognito.
The CognitoHelper class provides a convenient interface for creating and managing
Cognito User Pools, App Clients, Resource Servers, and authentication.
"""

import boto3
import json
import requests
from boto3.session import Session
from typing import Dict, List, Optional, Tuple


class CognitoHelper:
    """Provides an easy to use wrapper for AWS Cognito operations."""

    def __init__(self, region_name: str = None):
        """Constructs an instance.
        
        Args:
            region_name (str, optional): AWS region name. If not provided, uses default region.
        """
        self._session = Session()
        self._region = region_name or self._session.region_name
        self._cognito_client = boto3.client('cognito-idp', region_name=self._region)

    def setup_user_pool(
        self,
        pool_name: str = 'MCPServerPool',
        client_name: str = 'MCPServerPoolClient',
        username: str = 'testuser',
        password: str = 'MyPassword123!'
    ) -> Dict[str, str]:
        """Creates a complete Cognito User Pool setup with user and authentication.

        Args:
            pool_name (str): Name of the user pool to create
            client_name (str): Name of the app client to create
            username (str): Username for the test user
            password (str): Password for the test user

        Returns:
            Dict containing pool_id, client_id, bearer_token, and discovery_url
        """
        try:
            # Create User Pool
            user_pool_response = self._cognito_client.create_user_pool(
                PoolName=pool_name,
                Policies={
                    'PasswordPolicy': {
                        'MinimumLength': 8
                    }
                }
            )
            pool_id = user_pool_response['UserPool']['Id']
            
            # Create App Client
            app_client_response = self._cognito_client.create_user_pool_client(
                UserPoolId=pool_id,
                ClientName=client_name,
                GenerateSecret=False,
                ExplicitAuthFlows=[
                    'ALLOW_USER_PASSWORD_AUTH',
                    'ALLOW_REFRESH_TOKEN_AUTH'
                ]
            )
            client_id = app_client_response['UserPoolClient']['ClientId']
            
            # Create User
            self._cognito_client.admin_create_user(
                UserPoolId=pool_id,
                Username=username,
                TemporaryPassword='Temp123!',
                MessageAction='SUPPRESS'
            )
            
            # Set Permanent Password
            self._cognito_client.admin_set_user_password(
                UserPoolId=pool_id,
                Username=username,
                Password=password,
                Permanent=True
            )
            
            # Authenticate User and get Access Token
            auth_response = self._cognito_client.initiate_auth(
                ClientId=client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                }
            )
            bearer_token = auth_response['AuthenticationResult']['AccessToken']
            
            discovery_url = f"https://cognito-idp.{self._region}.amazonaws.com/{pool_id}/.well-known/openid-configuration"
            
            # Output the required values
            print(f"Pool id: {pool_id}")
            print(f"Discovery URL: {discovery_url}")
            print(f"Client ID: {client_id}")
            print(f"Bearer Token: {bearer_token}")
            
            return {
                'pool_id': pool_id,
                'client_id': client_id,
                'bearer_token': bearer_token,
                'discovery_url': discovery_url
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_or_create_user_pool(self, user_pool_name: str) -> str:
        """Gets an existing user pool by name or creates a new one.

        Args:
            user_pool_name (str): Name of the user pool

        Returns:
            str: User pool ID
        """
        response = self._cognito_client.list_user_pools(MaxResults=60)
        
        for pool in response["UserPools"]:
            if pool["Name"] == user_pool_name:
                user_pool_id = pool["Id"]
                response = self._cognito_client.describe_user_pool(
                    UserPoolId=user_pool_id
                )
            
                # Get the domain from user pool description
                user_pool = response.get('UserPool', {})
                domain = user_pool.get('Domain')
            
                if domain:
                    region = user_pool_id.split('_')[0] if '_' in user_pool_id else self._region
                    domain_url = f"https://{domain}.auth.{region}.amazoncognito.com"
                    print(f"Found domain for user pool {user_pool_id}: {domain} ({domain_url})")
                else:
                    print(f"No domains found for user pool {user_pool_id}")
                return pool["Id"]
        
        print('Creating new user pool')
        created = self._cognito_client.create_user_pool(PoolName=user_pool_name)
        user_pool_id = created["UserPool"]["Id"]
        user_pool_id_without_underscore_lc = user_pool_id.replace("_", "").lower()
        
        self._cognito_client.create_user_pool_domain(
            Domain=user_pool_id_without_underscore_lc,
            UserPoolId=user_pool_id
        )
        print("Domain created as well")
        return created["UserPool"]["Id"]

    def get_or_create_resource_server(
        self,
        user_pool_id: str,
        resource_server_id: str,
        resource_server_name: str,
        scopes: List[Dict[str, str]]
    ) -> str:
        """Gets an existing resource server or creates a new one.

        Args:
            user_pool_id (str): User pool ID
            resource_server_id (str): Identifier for the resource server
            resource_server_name (str): Name of the resource server
            scopes (List[Dict]): List of scope definitions

        Returns:
            str: Resource server identifier
        """
        try:
            existing = self._cognito_client.describe_resource_server(
                UserPoolId=user_pool_id,
                Identifier=resource_server_id
            )
            return resource_server_id
        except self._cognito_client.exceptions.ResourceNotFoundException:
            print('Creating new resource server')
            self._cognito_client.create_resource_server(
                UserPoolId=user_pool_id,
                Identifier=resource_server_id,
                Name=resource_server_name,
                Scopes=scopes
            )
            return resource_server_id

    def get_or_create_m2m_client(
        self,
        user_pool_id: str,
        client_name: str,
        resource_server_id: str,
        scopes: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """Gets an existing machine-to-machine client or creates a new one.

        Args:
            user_pool_id (str): User pool ID
            client_name (str): Name of the client
            resource_server_id (str): Resource server identifier
            scopes (List[str], optional): List of OAuth scopes

        Returns:
            Tuple[str, str]: Client ID and Client Secret
        """
        response = self._cognito_client.list_user_pool_clients(
            UserPoolId=user_pool_id,
            MaxResults=60
        )
        
        for client in response["UserPoolClients"]:
            if client["ClientName"] == client_name:
                describe = self._cognito_client.describe_user_pool_client(
                    UserPoolId=user_pool_id,
                    ClientId=client["ClientId"]
                )
                return client["ClientId"], describe["UserPoolClient"]["ClientSecret"]
        
        print('Creating new m2m client')

        # Default scopes if not provided
        if scopes is None:
            scopes = [
                f"{resource_server_id}/gateway:read",
                f"{resource_server_id}/gateway:write"
            ]

        created = self._cognito_client.create_user_pool_client(
            UserPoolId=user_pool_id,
            ClientName=client_name,
            GenerateSecret=True,
            AllowedOAuthFlows=["client_credentials"],
            AllowedOAuthScopes=scopes,
            AllowedOAuthFlowsUserPoolClient=True,
            SupportedIdentityProviders=["COGNITO"],
            ExplicitAuthFlows=["ALLOW_REFRESH_TOKEN_AUTH"]
        )
        return created["UserPoolClient"]["ClientId"], created["UserPoolClient"]["ClientSecret"]

    def get_token(
        self,
        user_pool_id: str,
        client_id: str,
        client_secret: str,
        scope_string: str
    ) -> Dict:
        """Gets an OAuth token using client credentials flow.

        Args:
            user_pool_id (str): User pool ID
            client_id (str): Client ID
            client_secret (str): Client secret
            scope_string (str): Space-separated scope string

        Returns:
            Dict: Token response containing access_token, token_type, expires_in
        """
        try:
            user_pool_id_without_underscore = user_pool_id.replace("_", "")
            url = f"https://{user_pool_id_without_underscore}.auth.{self._region}.amazoncognito.com/oauth2/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": scope_string,
            }
            print(client_id)
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as err:
            return {"error": str(err)}

    def delete_user_pool(self, user_pool_id: str) -> bool:
        """Deletes a Cognito User Pool and all its resources.

        Args:
            user_pool_id (str): User pool ID to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Deleting user pool: {user_pool_id}")
            
            # Delete domain if exists
            try:
                user_pool_id_without_underscore = user_pool_id.replace("_", "").lower()
                self._cognito_client.delete_user_pool_domain(
                    Domain=user_pool_id_without_underscore,
                    UserPoolId=user_pool_id
                )
                print(f"  Deleted domain: {user_pool_id_without_underscore}")
            except Exception as e:
                print(f"  Note: {e}")
            
            # Delete all app clients
            try:
                clients = self._cognito_client.list_user_pool_clients(
                    UserPoolId=user_pool_id,
                    MaxResults=60
                )
                for client in clients['UserPoolClients']:
                    self._cognito_client.delete_user_pool_client(
                        UserPoolId=user_pool_id,
                        ClientId=client['ClientId']
                    )
                    print(f"  Deleted client: {client['ClientName']}")
            except Exception as e:
                print(f"  Note: {e}")
            
            # Delete all resource servers
            try:
                resource_servers = self._cognito_client.list_resource_servers(
                    UserPoolId=user_pool_id,
                    MaxResults=50
                )
                for server in resource_servers['ResourceServers']:
                    self._cognito_client.delete_resource_server(
                        UserPoolId=user_pool_id,
                        Identifier=server['Identifier']
                    )
                    print(f"  Deleted resource server: {server['Name']}")
            except Exception as e:
                print(f"  Note: {e}")
            
            # Finally, delete the user pool
            self._cognito_client.delete_user_pool(UserPoolId=user_pool_id)
            print(f"✅ User pool {user_pool_id} deleted successfully")
            return True
            
        except self._cognito_client.exceptions.ResourceNotFoundException:
            print(f"User pool {user_pool_id} does not exist")
            return True
        except Exception as e:
            print(f"Error deleting user pool: {e}")
            return False

    def delete_user_pool_by_name(self, user_pool_name: str) -> bool:
        """Deletes a Cognito User Pool by name.

        Args:
            user_pool_name (str): Name of the user pool to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self._cognito_client.list_user_pools(MaxResults=60)
            for pool in response["UserPools"]:
                if pool["Name"] == user_pool_name:
                    return self.delete_user_pool(pool["Id"])
            
            print(f"User pool '{user_pool_name}' not found")
            return True
        except Exception as e:
            print(f"Error finding user pool: {e}")
            return False
