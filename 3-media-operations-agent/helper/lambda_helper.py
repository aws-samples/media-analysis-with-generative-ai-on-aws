# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission

"""
This module contains a helper class for working with AWS Lambda.
The LambdaHelper class provides a convenient interface for creating and managing
Lambda functions with proper IAM roles.
"""

import boto3
import json
import time
import botocore
from boto3.session import Session
from typing import Dict


class LambdaHelper:
    """Provides an easy to use wrapper for AWS Lambda operations."""

    def __init__(self, region_name: str = None):
        """Constructs an instance.
        
        Args:
            region_name (str, optional): AWS region name. If not provided, uses default region.
        """
        self._session = Session()
        self._region = region_name or self._session.region_name
        self._lambda_client = boto3.client('lambda', region_name=self._region)
        self._iam_client = boto3.client('iam', region_name=self._region)

    def create_gateway_lambda(
        self,
        lambda_function_code_path: str,
        function_name: str = 'gateway_lambda',
        role_name: str = 'gateway_lambda_iamrole',
        additional_policies: Dict = None,
        environment_variables: Dict = None,
        timeout: int = 60,
        memory_size: int = 256
    ) -> Dict[str, any]:
        """Creates a Lambda function for use with Bedrock AgentCore Gateway.

        Args:
            lambda_function_code_path (str): Path to the zip file containing Lambda code
            function_name (str): Name of the Lambda function to create
            role_name (str): Name of the IAM role to create for the Lambda
            additional_policies (Dict, optional): Additional IAM policy document to attach
            environment_variables (Dict, optional): Environment variables for the Lambda
            timeout (int): Lambda timeout in seconds (default: 60)
            memory_size (int): Lambda memory in MB (default: 256)

        Returns:
            Dict containing lambda_function_arn and exit_code
        """
        return_resp = {"lambda_function_arn": "Pending", "exit_code": 1}
        role_arn = ''

        print("Reading code from zip file")
        with open(lambda_function_code_path, 'rb') as f:
            lambda_function_code = f.read()

        try:
            print("Creating IAM role for lambda function")

            response = self._iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "lambda.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }),
                Description="IAM role to be assumed by lambda function"
            )

            role_arn = response['Role']['Arn']

            print("Attaching policy to the IAM role")

            response = self._iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )

            print(f"Role '{role_name}' created successfully: {role_arn}")
            
            # Attach additional policies if provided
            if additional_policies:
                print("Attaching additional IAM policies")
                self._iam_client.put_role_policy(
                    RoleName=role_name,
                    PolicyName='AdditionalLambdaPolicy',
                    PolicyDocument=json.dumps(additional_policies)
                )
                print("Additional policies attached")
            
            time.sleep(10)  # Wait for IAM role to propagate
            
        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == "EntityAlreadyExists":
                response = self._iam_client.get_role(RoleName=role_name)
                role_arn = response['Role']['Arn']
                print(f"IAM role {role_name} already exists. Using the same ARN {role_arn}")
            else:
                error_message = error.response['Error']['Code'] + "-" + error.response['Error']['Message']
                print(f"Error creating role: {error_message}")
                return_resp['lambda_function_arn'] = error_message
                return return_resp

        if role_arn != "":
            print("Creating lambda function")
            try:
                # Build Lambda configuration
                lambda_config = {
                    'FunctionName': function_name,
                    'Role': role_arn,
                    'Runtime': 'python3.12',
                    'Handler': 'lambda_function.lambda_handler',
                    'Code': {'ZipFile': lambda_function_code},
                    'Description': 'Lambda function for Bedrock AgentCore Gateway',
                    'PackageType': 'Zip',
                    'Timeout': timeout,
                    'MemorySize': memory_size
                }
                
                # Add environment variables if provided
                if environment_variables:
                    lambda_config['Environment'] = {'Variables': environment_variables}
                
                lambda_response = self._lambda_client.create_function(**lambda_config)

                return_resp['lambda_function_arn'] = lambda_response['FunctionArn']
                return_resp['exit_code'] = 0
                print(f"Lambda function created successfully: {lambda_response['FunctionArn']}")
                
            except botocore.exceptions.ClientError as error:
                if error.response['Error']['Code'] == "ResourceConflictException":
                    response = self._lambda_client.get_function(FunctionName=function_name)
                    lambda_arn = response['Configuration']['FunctionArn']
                    print(f"AWS Lambda function {function_name} already exists. Using the same ARN {lambda_arn}")
                    return_resp['lambda_function_arn'] = lambda_arn
                    return_resp['exit_code'] = 0
                else:
                    error_message = error.response['Error']['Code'] + "-" + error.response['Error']['Message']
                    print(f"Error creating lambda function: {error_message}")
                    return_resp['lambda_function_arn'] = error_message

        return return_resp

    def create_lambda_role(
        self,
        role_name: str,
        additional_policies: Dict = None
    ) -> str:
        """Creates an IAM role for Lambda function.

        Args:
            role_name (str): Name of the IAM role to create
            additional_policies (Dict, optional): Additional IAM policy document to attach

        Returns:
            str: Role ARN
        """
        try:
            print(f"Creating IAM role: {role_name}")
            
            response = self._iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }),
                Description="IAM role for Lambda function"
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach basic execution role
            self._iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            print("Attached AWSLambdaBasicExecutionRole")
            
            # Attach additional policies if provided
            if additional_policies:
                self._iam_client.put_role_policy(
                    RoleName=role_name,
                    PolicyName='AdditionalLambdaPolicy',
                    PolicyDocument=json.dumps(additional_policies)
                )
                print("Attached additional policies")
            
            print(f"✅ Role created: {role_arn}")
            time.sleep(10)  # Wait for IAM role to propagate
            return role_arn
            
        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == "EntityAlreadyExists":
                response = self._iam_client.get_role(RoleName=role_name)
                role_arn = response['Role']['Arn']
                print(f"✅ Using existing role: {role_arn}")
                return role_arn
            else:
                raise

    def delete_lambda_role(self, role_name: str) -> bool:
        """Deletes an IAM role and all its attached policies.

        Args:
            role_name (str): Name of the IAM role to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Deleting IAM role: {role_name}")
            
            # First, detach all managed policies
            try:
                attached_policies = self._iam_client.list_attached_role_policies(
                    RoleName=role_name
                )
                for policy in attached_policies['AttachedPolicies']:
                    self._iam_client.detach_role_policy(
                        RoleName=role_name,
                        PolicyArn=policy['PolicyArn']
                    )
                    print(f"  Detached managed policy: {policy['PolicyName']}")
            except Exception as e:
                print(f"  Note: {e}")
            
            # Delete all inline policies
            try:
                inline_policies = self._iam_client.list_role_policies(
                    RoleName=role_name
                )
                for policy_name in inline_policies['PolicyNames']:
                    self._iam_client.delete_role_policy(
                        RoleName=role_name,
                        PolicyName=policy_name
                    )
                    print(f"  Deleted inline policy: {policy_name}")
            except Exception as e:
                print(f"  Note: {e}")
            
            # Finally, delete the role
            self._iam_client.delete_role(RoleName=role_name)
            print(f"✅ IAM role {role_name} deleted successfully")
            return True
            
        except self._iam_client.exceptions.NoSuchEntityException:
            print(f"IAM role {role_name} does not exist")
            return True
        except Exception as e:
            print(f"Error deleting IAM role: {e}")
            return False

    def delete_lambda_function(self, function_name: str) -> bool:
        """Deletes a Lambda function.

        Args:
            function_name (str): Name of the Lambda function to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._lambda_client.delete_function(FunctionName=function_name)
            print(f"✅ Lambda function {function_name} deleted successfully")
            return True
        except self._lambda_client.exceptions.ResourceNotFoundException:
            print(f"Lambda function {function_name} does not exist")
            return True
        except Exception as e:
            print(f"Error deleting Lambda function: {e}")
            return False

    def delete_lambda_and_role(
        self,
        function_name: str,
        role_name: str
    ) -> bool:
        """Deletes both Lambda function and its IAM role.

        Args:
            function_name (str): Name of the Lambda function to delete
            role_name (str): Name of the IAM role to delete

        Returns:
            bool: True if both deletions successful, False otherwise
        """
        lambda_deleted = self.delete_lambda_function(function_name)
        role_deleted = self.delete_lambda_role(role_name)
        
        if lambda_deleted and role_deleted:
            print(f"✅ Successfully deleted Lambda function and IAM role")
            return True
        else:
            print(f"⚠️ Some resources may not have been deleted")
            return False

    def update_lambda_code(
        self,
        function_name: str,
        lambda_function_code_path: str
    ) -> bool:
        """Updates the code of an existing Lambda function.

        Args:
            function_name (str): Name of the Lambda function to update
            lambda_function_code_path (str): Path to the zip file containing new Lambda code

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Reading code from zip file: {lambda_function_code_path}")
            with open(lambda_function_code_path, 'rb') as f:
                lambda_function_code = f.read()

            response = self._lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=lambda_function_code
            )
            print(f"Lambda function {function_name} code updated successfully")
            return True
        except Exception as e:
            print(f"Error updating Lambda function code: {e}")
            return False

    def invoke_lambda(
        self,
        function_name: str,
        payload: Dict = None,
        invocation_type: str = 'RequestResponse'
    ) -> Dict:
        """Invokes a Lambda function.

        Args:
            function_name (str): Name of the Lambda function to invoke
            payload (Dict, optional): Payload to send to the Lambda function
            invocation_type (str): Type of invocation ('RequestResponse', 'Event', 'DryRun')

        Returns:
            Dict: Response from the Lambda invocation
        """
        try:
            if payload is None:
                payload = {}

            response = self._lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=json.dumps(payload)
            )

            if invocation_type == 'RequestResponse':
                response_payload = json.loads(response['Payload'].read())
                return {
                    'StatusCode': response['StatusCode'],
                    'Payload': response_payload
                }
            else:
                return {'StatusCode': response['StatusCode']}

        except Exception as e:
            print(f"Error invoking Lambda function: {e}")
            return {'error': str(e)}

    def get_lambda_function(self, function_name: str) -> Dict:
        """Gets information about a Lambda function.

        Args:
            function_name (str): Name of the Lambda function

        Returns:
            Dict: Lambda function configuration
        """
        try:
            response = self._lambda_client.get_function(FunctionName=function_name)
            return response
        except Exception as e:
            print(f"Error getting Lambda function: {e}")
            return None
