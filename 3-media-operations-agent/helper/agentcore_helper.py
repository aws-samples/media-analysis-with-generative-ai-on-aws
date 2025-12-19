# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission

"""
This module contains a helper class for working with Amazon Bedrock AgentCore.
The AgentCoreHelper class provides a convenient interface for creating and managing
AgentCore resources including gateways, roles, and workload identities.
"""

import boto3
import json
import time
from boto3.session import Session
from botocore.exceptions import ClientError
from typing import Dict, List, Optional


class AgentCoreHelper:
    """Provides an easy to use wrapper for Amazon Bedrock AgentCore operations."""

    def __init__(self, region_name: str = None):
        """Constructs an instance.
        
        Args:
            region_name (str, optional): AWS region name. If not provided, uses default region.
        """
        self._session = Session()
        self._region = region_name or self._session.region_name
        self._account_id = boto3.client("sts").get_caller_identity()["Account"]
        self._iam_client = boto3.client('iam', region_name=self._region)
        self._gateway_client = boto3.client('bedrock-agentcore-control', region_name=self._region)

    def create_agentcore_role(self, agent_name: str) -> Dict:
        """Creates an IAM role for AgentCore runtime.

        Args:
            agent_name (str): Name of the agent

        Returns:
            Dict: IAM role response with Role ARN
        """
        iam_client = self._iam_client
        agentcore_role_name = f'agentcore-{agent_name}-role'
        
        role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "BedrockPermissions",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:DescribeLogStreams",
                        "logs:CreateLogGroup"
                    ],
                    "Resource": [
                        f"arn:aws:logs:{self._region}:{self._account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:DescribeLogGroups"],
                    "Resource": [f"arn:aws:logs:{self._region}:{self._account_id}:log-group:*"]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": [
                        f"arn:aws:logs:{self._region}:{self._account_id}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "xray:PutTraceSegments",
                        "xray:PutTelemetryRecords",
                        "xray:GetSamplingRules",
                        "xray:GetSamplingTargets"
                    ],
                    "Resource": ["*"]
                },
                {
                    "Effect": "Allow",
                    "Resource": "*",
                    "Action": "cloudwatch:PutMetricData",
                    "Condition": {
                        "StringEquals": {
                            "cloudwatch:namespace": "bedrock-agentcore"
                        }
                    }
                },
                {
                    "Effect": "Allow",
                    "Resource": "*",
                    "Action": "s3:GetObject"
                },
                {
                    "Effect": "Allow",
                    "Resource": "*",
                    "Action": "lambda:InvokeFunction"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock-agentcore:*",
                        "iam:PassRole"
                    ],
                    "Resource": "*"
                },
                {
                    "Sid": "GetAgentAccessToken",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock-agentcore:GetWorkloadAccessToken",
                        "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                        "bedrock-agentcore:GetWorkloadAccessTokenForUserId"
                    ],
                    "Resource": [
                        f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:workload-identity-directory/default",
                        f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:workload-identity-directory/default/workload-identity/{agent_name}-*"
                    ]
                }
            ]
        }

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AssumeRolePolicy",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock-agentcore.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {
                            "aws:SourceAccount": f"{self._account_id}"
                        },
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:*"
                        }
                    }
                }
            ]
        }

        assume_role_policy_document_json = json.dumps(assume_role_policy_document)
        role_policy_document = json.dumps(role_policy)

        try:
            agentcore_iam_role = iam_client.create_role(
                RoleName=agentcore_role_name,
                AssumeRolePolicyDocument=assume_role_policy_document_json
            )
            time.sleep(10)
        except iam_client.exceptions.EntityAlreadyExistsException:
            print(f"Role {agentcore_role_name} already exists -- updating policies")
            policies = iam_client.list_role_policies(
                RoleName=agentcore_role_name,
                MaxItems=100
            )
            for policy_name in policies['PolicyNames']:
                iam_client.delete_role_policy(
                    RoleName=agentcore_role_name,
                    PolicyName=policy_name
                )
            agentcore_iam_role = iam_client.get_role(RoleName=agentcore_role_name)

        try:
            iam_client.put_role_policy(
                PolicyDocument=role_policy_document,
                PolicyName="AgentCorePolicy",
                RoleName=agentcore_role_name
            )
        except Exception as e:
            print(f"Error attaching policy: {e}")

        return agentcore_iam_role

    def create_agentcore_gateway_role(
        self,
        gateway_name: str,
        include_s3_permissions: bool = False
    ) -> Dict:
        """Creates an IAM role for AgentCore Gateway.

        Args:
            gateway_name (str): Name of the gateway
            include_s3_permissions (bool): Whether to include S3 permissions for Smithy models

        Returns:
            Dict: IAM role response with Role ARN
        """
        iam_client = self._iam_client
        agentcore_gateway_role_name = f'agentcore-{gateway_name}-role'
        
        actions = [
            "bedrock-agentcore:*",
            "bedrock:*",
            "agent-credential-provider:*",
            "iam:PassRole",
            "secretsmanager:GetSecretValue",
            "lambda:InvokeFunction"
        ]
        
        if include_s3_permissions:
            actions.append("s3:*")

        role_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": actions,
                "Resource": "*"
            }]
        }

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AssumeRolePolicy",
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock-agentcore.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {
                            "aws:SourceAccount": f"{self._account_id}"
                        },
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:*"
                        }
                    }
                }
            ]
        }

        assume_role_policy_document_json = json.dumps(assume_role_policy_document)
        role_policy_document = json.dumps(role_policy)

        try:
            agentcore_iam_role = iam_client.create_role(
                RoleName=agentcore_gateway_role_name,
                AssumeRolePolicyDocument=assume_role_policy_document_json
            )
            time.sleep(10)
        except iam_client.exceptions.EntityAlreadyExistsException:
            print(f"Role {agentcore_gateway_role_name} already exists -- updating policies")
            policies = iam_client.list_role_policies(
                RoleName=agentcore_gateway_role_name,
                MaxItems=100
            )
            for policy_name in policies['PolicyNames']:
                iam_client.delete_role_policy(
                    RoleName=agentcore_gateway_role_name,
                    PolicyName=policy_name
                )
            agentcore_iam_role = iam_client.get_role(RoleName=agentcore_gateway_role_name)

        try:
            iam_client.put_role_policy(
                PolicyDocument=role_policy_document,
                PolicyName="AgentCorePolicy",
                RoleName=agentcore_gateway_role_name
            )
        except Exception as e:
            print(f"Error attaching policy: {e}")

        return agentcore_iam_role

    def delete_gateway(self, gateway_id: str) -> bool:
        """Deletes a gateway and all its targets.

        Args:
            gateway_id (str): Gateway identifier

        Returns:
            bool: True if successful
        """
        try:
            print(f"Deleting all targets for gateway {gateway_id}")
            list_response = self._gateway_client.list_gateway_targets(
                gatewayIdentifier=gateway_id,
                maxResults=100
            )
            
            for item in list_response['items']:
                target_id = item["targetId"]
                print(f"  Deleting target {target_id}")
                self._gateway_client.delete_gateway_target(
                    gatewayIdentifier=gateway_id,
                    targetId=target_id
                )
                time.sleep(5)
            
            print(f"Deleting gateway {gateway_id}")
            self._gateway_client.delete_gateway(gatewayIdentifier=gateway_id)
            print(f"✅ Gateway {gateway_id} deleted successfully")
            return True
        except Exception as e:
            print(f"Error deleting gateway: {e}")
            return False

    def delete_all_gateways(self) -> bool:
        """Deletes all gateways in the region.

        Returns:
            bool: True if successful
        """
        try:
            list_response = self._gateway_client.list_gateways(maxResults=100)
            for item in list_response['items']:
                gateway_id = item["gatewayId"]
                self.delete_gateway(gateway_id)
            return True
        except Exception as e:
            print(f"Error deleting all gateways: {e}")
            return False

    def create_gateway_invoke_role(
        self,
        role_name: str,
        gateway_id: str,
        current_arn: str
    ) -> Dict:
        """Creates an IAM role that can invoke a gateway.

        Args:
            role_name (str): Name of the role to create
            gateway_id (str): Gateway identifier
            current_arn (str): ARN of the current user/role that should be able to assume this role

        Returns:
            Dict: IAM role response
        """
        # Normalize current_arn
        if isinstance(current_arn, (list, set, tuple)):
            current_arn = list(current_arn)[0]
        current_arn = str(current_arn)

        # Trust policy
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AssumeRoleByAgentCore",
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                    "Action": ["sts:AssumeRole"]
                },
                {
                    "Sid": "AllowCallerToAssume",
                    "Effect": "Allow",
                    "Principal": {"AWS": [current_arn]},
                    "Action": ["sts:AssumeRole"]
                }
            ]
        }
        assume_role_policy_json = json.dumps(assume_role_policy_document)

        # Inline role policy
        role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["bedrock-agentcore:InvokeGateway"],
                    "Resource": f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:gateway/{gateway_id}"
                }
            ]
        }
        role_policy_json = json.dumps(role_policy)

        try:
            agentcoregw_iam_role = self._iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=assume_role_policy_json
            )
            print(f"✅ Created role: {role_name}")
            time.sleep(3)
        except self._iam_client.exceptions.EntityAlreadyExistsException:
            print(f"Role '{role_name}' already exists — updating policies")
            self._iam_client.update_assume_role_policy(
                RoleName=role_name,
                PolicyDocument=assume_role_policy_json
            )
            for policy_name in self._iam_client.list_role_policies(RoleName=role_name).get('PolicyNames', []):
                self._iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            agentcoregw_iam_role = self._iam_client.get_role(RoleName=role_name)

        # Attach inline role policy
        self._iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName="AgentCoreGatewayInvokePolicy",
            PolicyDocument=role_policy_json
        )

        role_arn = agentcoregw_iam_role['Role']['Arn']
        print(f"✅ Role ready: {role_arn}")
        
        return agentcoregw_iam_role

    def get_current_role_arn(self) -> str:
        """Gets the ARN of the current IAM user or role.

        Returns:
            str: Current IAM ARN
        """
        sts_client = boto3.client("sts")
        role_arn = sts_client.get_caller_identity()["Arn"]
        return role_arn

    def create_agentcore_runtime_execution_role(
        self,
        role_name: str = "AgentCoreRuntimeExecutionRole",
        policy_name: str = "AgentCoreRuntimeExecutionPolicy"
    ) -> str:
        """Creates an IAM role for AgentCore Runtime execution.

        Args:
            role_name (str): Name of the IAM role to create
            policy_name (str): Name of the IAM policy to create

        Returns:
            str: Role ARN
        """
        # Trust relationship policy
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AssumeRolePolicy",
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": self._account_id},
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:*"
                        },
                    },
                }
            ],
        }

        # IAM policy document
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ECRImageAccess",
                    "Effect": "Allow",
                    "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                    "Resource": [f"arn:aws:ecr:{self._region}:{self._account_id}:repository/*"],
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:DescribeLogStreams", "logs:CreateLogGroup"],
                    "Resource": [
                        f"arn:aws:logs:{self._region}:{self._account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"
                    ],
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:DescribeLogGroups"],
                    "Resource": [f"arn:aws:logs:{self._region}:{self._account_id}:log-group:*"],
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                    "Resource": [
                        f"arn:aws:logs:{self._region}:{self._account_id}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
                    ],
                },
                {
                    "Sid": "ECRTokenAccess",
                    "Effect": "Allow",
                    "Action": ["ecr:GetAuthorizationToken"],
                    "Resource": "*",
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "xray:PutTraceSegments",
                        "xray:PutTelemetryRecords",
                        "xray:GetSamplingRules",
                        "xray:GetSamplingTargets",
                    ],
                    "Resource": ["*"],
                },
                {
                    "Effect": "Allow",
                    "Resource": "*",
                    "Action": "cloudwatch:PutMetricData",
                    "Condition": {"StringEquals": {"cloudwatch:namespace": "bedrock-agentcore"}},
                },
                {
                    "Sid": "GetAgentAccessToken",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock-agentcore:GetWorkloadAccessToken",
                        "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                        "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
                    ],
                    "Resource": [
                        f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:workload-identity-directory/default",
                        f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:workload-identity-directory/default/workload-identity/*",
                    ],
                },
                {
                    "Sid": "BedrockModelInvocation",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream",
                        "bedrock:ApplyGuardrail",
                        "bedrock:Retrieve",
                    ],
                    "Resource": [
                        "arn:aws:bedrock:*::foundation-model/*",
                        f"arn:aws:bedrock:{self._region}:{self._account_id}:*",
                    ],
                },
                {
                    "Sid": "AllowAgentToUseMemory",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock-agentcore:CreateEvent",
                        "bedrock-agentcore:GetMemoryRecord",
                        "bedrock-agentcore:GetMemory",
                        "bedrock-agentcore:RetrieveMemoryRecords",
                        "bedrock-agentcore:ListMemoryRecords",
                    ],
                    "Resource": [f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:*"],
                },
                {
                    "Sid": "GetMemoryId",
                    "Effect": "Allow",
                    "Action": ["ssm:GetParameter"],
                    "Resource": [f"arn:aws:ssm:{self._region}:{self._account_id}:parameter/*"],
                },
                {
                    "Sid": "GatewayAccess",
                    "Effect": "Allow",
                    "Action": ["bedrock-agentcore:GetGateway", "bedrock-agentcore:InvokeGateway"],
                    "Resource": [f"arn:aws:bedrock-agentcore:{self._region}:{self._account_id}:gateway/*"],
                },
            ],
        }

        try:
            # Check if role already exists
            try:
                existing_role = self._iam_client.get_role(RoleName=role_name)
                print(f"✅ Role {role_name} already exists")
                print(f"   Role ARN: {existing_role['Role']['Arn']}")
                return existing_role["Role"]["Arn"]
            except self._iam_client.exceptions.NoSuchEntityException:
                pass

            # Create IAM role
            role_response = self._iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="IAM role for Amazon Bedrock AgentCore Runtime with required permissions",
            )
            print(f"✅ Created IAM role: {role_name}")
            print(f"   Role ARN: {role_response['Role']['Arn']}")

            # Check if policy already exists
            policy_arn = f"arn:aws:iam::{self._account_id}:policy/{policy_name}"
            try:
                self._iam_client.get_policy(PolicyArn=policy_arn)
                print(f"✅ Policy {policy_name} already exists")
            except self._iam_client.exceptions.NoSuchEntityException:
                # Create policy
                policy_response = self._iam_client.create_policy(
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(policy_document),
                    Description="Policy for Amazon Bedrock AgentCore Runtime permissions",
                )
                print(f"✅ Created policy: {policy_name}")
                policy_arn = policy_response["Policy"]["Arn"]

            # Attach policy to role
            try:
                self._iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
                print("✅ Attached policy to role")
            except Exception as e:
                if "already attached" in str(e).lower():
                    print("✅ Policy already attached to role")
                else:
                    raise

            print(f"   Policy ARN: {policy_arn}")
            return role_response["Role"]["Arn"]

        except Exception as e:
            print(f"❌ Error creating IAM role: {str(e)}")
            return None

    def delete_agentcore_runtime_execution_role(
        self,
        role_name: str = "AgentCoreRuntimeExecutionRole",
        policy_name: str = "AgentCoreRuntimeExecutionPolicy"
    ) -> bool:
        """Deletes the AgentCore Runtime execution role and policy.

        Args:
            role_name (str): Name of the IAM role to delete
            policy_name (str): Name of the IAM policy to delete

        Returns:
            bool: True if successful
        """
        try:
            policy_arn = f"arn:aws:iam::{self._account_id}:policy/{policy_name}"

            # Detach policy from role
            try:
                self._iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
                print("✅ Detached policy from role")
            except Exception:
                pass

            # Delete role
            try:
                self._iam_client.delete_role(RoleName=role_name)
                print(f"✅ Deleted role: {role_name}")
            except Exception:
                pass

            # Delete policy
            try:
                self._iam_client.delete_policy(PolicyArn=policy_arn)
                print(f"✅ Deleted policy: {policy_name}")
            except Exception:
                pass

            return True

        except Exception as e:
            print(f"❌ Error during cleanup: {str(e)}")
            return False


    def runtime_resource_cleanup(self, runtime_arn: str = None, agent_name: str = None) -> bool:
        """Deletes AgentCore Runtime and associated ECR repositories.

        Args:
            runtime_arn (str, optional): ARN of specific runtime to delete. If None, deletes all runtimes.
            agent_name (str, optional): Agent name to filter ECR repositories. If None, deletes all AgentCore repos.

        Returns:
            bool: True if successful
        """
        try:
            # Initialize clients
            agentcore_control_client = boto3.client("bedrock-agentcore-control", region_name=self._region)
            ecr_client = boto3.client("ecr", region_name=self._region)

            # Delete specific runtime or all runtimes
            if runtime_arn:
                runtime_id = runtime_arn.split(":")[-1].split("/")[-1]
                print(f"Deleting runtime: {runtime_id}")
                response = agentcore_control_client.delete_agent_runtime(agentRuntimeId=runtime_id)
                print(f"✅ Agent runtime deleted: {response['status']}")
            else:
                print("Deleting all AgentCore runtimes...")
                runtimes = agentcore_control_client.list_agent_runtimes()
                for runtime in runtimes.get("agentRuntimes", []):
                    runtime_id = runtime["agentRuntimeId"]
                    response = agentcore_control_client.delete_agent_runtime(agentRuntimeId=runtime_id)
                    print(f"✅ Agent runtime deleted: {runtime_id} - {response['status']}")

            # Delete ECR repositories
            print("Deleting ECR repositories...")
            try:
                repositories = ecr_client.describe_repositories()
                for repo in repositories.get("repositories", []):
                    repo_name = repo["repositoryName"]
                    # Filter by agent name if provided, otherwise delete all bedrock-agentcore repos
                    if agent_name:
                        if f"bedrock-agentcore-{agent_name}" in repo_name:
                            ecr_client.delete_repository(repositoryName=repo_name, force=True)
                            print(f"✅ ECR repository deleted: {repo_name}")
                    elif "bedrock-agentcore" in repo_name:
                        ecr_client.delete_repository(repositoryName=repo_name, force=True)
                        print(f"✅ ECR repository deleted: {repo_name}")
            except ecr_client.exceptions.RepositoryNotFoundException:
                print("No ECR repositories found")
            except Exception as e:
                print(f"Note: Error deleting ECR repositories: {e}")

            return True

        except Exception as e:
            print(f"⚠️ Error during runtime cleanup: {e}")
            return False

    def delete_ssm_parameter(self, parameter_name: str) -> bool:
        """Deletes an SSM parameter.

        Args:
            parameter_name (str): Name of the SSM parameter to delete

        Returns:
            bool: True if successful
        """
        try:
            ssm_client = boto3.client('ssm', region_name=self._region)
            ssm_client.delete_parameter(Name=parameter_name)
            print(f"✅ Deleted SSM parameter: {parameter_name}")
            return True
        except ssm_client.exceptions.ParameterNotFound:
            print(f"SSM parameter {parameter_name} not found")
            return True
        except Exception as e:
            print(f"Error deleting SSM parameter: {e}")
            return False
