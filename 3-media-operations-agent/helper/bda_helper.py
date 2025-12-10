"""
BDA Video Processing Module

This module provides utilities for uploading videos to S3, 
running BDA jobs, monitoring status, and retrieving results.
"""

import boto3
import json
import time
import uuid
from typing import Dict, Optional, Tuple
from pathlib import Path


class BDAVideoProcessor:
    """Handle BDA video processing operations"""
    
    def __init__(self, region: str = None, bucket: str = None):
        """
        Initialize BDA Video Processor
        
        Args:
            region: AWS region (defaults to session region)
            bucket: S3 bucket name (required for upload/output)
        """
        self.region = region or boto3.Session().region_name
        self.bucket = bucket
        
        self.bda_client = boto3.client('bedrock-data-automation', region_name=self.region)
        self.bda_runtime_client = boto3.client('bedrock-data-automation-runtime', region_name=self.region)
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.sts_client = boto3.client('sts', region_name=self.region)
        
        self.account_id = self.sts_client.get_caller_identity()["Account"]
        self.default_profile_arn = f"arn:aws:bedrock:{self.region}:{self.account_id}:data-automation-profile/us.data-automation-v1"
    
    def create_project(
        self,
        project_name: Optional[str] = None,
        enable_video_summary: bool = True,
        enable_scene_summary: bool = True,
        enable_iab_categories: bool = True,
        enable_transcript: bool = True,
        enable_text_detection: bool = True,
        enable_bounding_boxes: bool = True
    ) -> str:
        """
        Create a BDA project with video configuration
        
        Args:
            project_name: Name for the project (auto-generated if None)
            enable_video_summary: Enable full video summary
            enable_scene_summary: Enable scene-level summaries
            enable_iab_categories: Enable IAB category classification
            enable_transcript: Enable audio transcription
            enable_text_detection: Enable text detection in video
            enable_bounding_boxes: Enable bounding boxes for detected text
            
        Returns:
            Project ARN
        """
        if not project_name:
            project_name = f'bda-video-project-{str(uuid.uuid4())[0:8]}'
        
        # Build extraction types
        extraction_types = []
        if enable_text_detection:
            extraction_types.append('TEXT_DETECTION')
        if enable_transcript:
            extraction_types.append('TRANSCRIPT')
        
        # Build generative field types
        generative_types = []
        if enable_video_summary:
            generative_types.append('VIDEO_SUMMARY')
        if enable_scene_summary:
            generative_types.append('CHAPTER_SUMMARY')
        if enable_iab_categories:
            generative_types.append('IAB')
        
        response = self.bda_client.create_data_automation_project(
            projectName=project_name,
            projectDescription='BDA video processing project',
            projectStage='DEVELOPMENT',
            standardOutputConfiguration={
                'video': {
                    'extraction': {
                        'category': {
                            'state': 'ENABLED' if extraction_types else 'DISABLED',
                            'types': extraction_types,
                        },
                        'boundingBox': {
                            'state': 'ENABLED' if enable_bounding_boxes else 'DISABLED',
                        }
                    },
                    'generativeField': {
                        'state': 'ENABLED' if generative_types else 'DISABLED',
                        'types': generative_types,
                    }
                }
            }
        )
        
        project_arn = response.get("projectArn")
        print(f"✓ Created BDA project: {project_arn}")
        return project_arn
    
    def upload_video(self, video_path: str, s3_prefix: str = "bda/video") -> str:
        """
        Upload video to S3
        
        Args:
            video_path: Local path to video file
            s3_prefix: S3 prefix for upload
            
        Returns:
            S3 URI of uploaded video
        """
        if not self.bucket:
            raise ValueError("Bucket name is required for upload")
        
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        s3_key = f'{s3_prefix}/{video_file.name}'
        print(f"Uploading {video_file.name} to s3://{self.bucket}/{s3_key}...")
        
        self.s3_client.upload_file(str(video_file), self.bucket, s3_key)
        s3_uri = f's3://{self.bucket}/{s3_key}'
        print(f"✓ Uploaded to {s3_uri}")
        
        return s3_uri

    def start_job(
        self,
        input_s3_uri: str,
        project_arn: str,
        output_prefix: str = "bda/video/output"
    ) -> str:
        """
        Start BDA video processing job
        
        Args:
            input_s3_uri: S3 URI of input video
            project_arn: BDA project ARN
            output_prefix: S3 prefix for output
            
        Returns:
            Invocation ARN
        """
        if not self.bucket:
            raise ValueError("Bucket name is required for output")
        
        output_s3_uri = f's3://{self.bucket}/{output_prefix}'
        
        response = self.bda_runtime_client.invoke_data_automation_async(
            inputConfiguration={
                's3Uri': input_s3_uri
            },
            outputConfiguration={
                's3Uri': output_s3_uri
            },
            dataAutomationConfiguration={
                'dataAutomationProjectArn': project_arn,
                'stage': 'DEVELOPMENT'
            },
            notificationConfiguration={
                'eventBridgeConfiguration': {
                    'eventBridgeEnabled': False
                }
            },
            dataAutomationProfileArn=self.default_profile_arn
        )
        
        invocation_arn = response.get("invocationArn")
        print(f"✓ Started BDA job: {invocation_arn}")
        return invocation_arn
    
    def check_status(self, invocation_arn: str) -> Tuple[str, Dict]:
        """
        Check BDA job status
        
        Args:
            invocation_arn: Invocation ARN from start_job
            
        Returns:
            Tuple of (status, full_response)
        """
        response = self.bda_runtime_client.get_data_automation_status(
            invocationArn=invocation_arn
        )
        status = response.get("status")
        return status, response
    
    def wait_for_completion(
        self,
        invocation_arn: str,
        poll_interval: int = 5,
        verbose: bool = True
    ) -> Dict:
        """
        Wait for BDA job to complete
        
        Args:
            invocation_arn: Invocation ARN from start_job
            poll_interval: Seconds between status checks
            verbose: Print status updates
            
        Returns:
            Final status response
        """
        status = None
        status_response = None
        
        while status not in ["Success", "ServiceError", "ClientError"]:
            status, status_response = self.check_status(invocation_arn)
            
            if verbose:
                print(f"Status: {status}", end='\r')
            
            if status not in ["Success", "ServiceError", "ClientError"]:
                time.sleep(poll_interval)
        
        if verbose:
            print(f"\n✓ Job completed with status: {status}")
        
        if status != "Success":
            raise RuntimeError(f"BDA job failed with status: {status}")
        
        return status_response
    
    def get_results(self, status_response: Dict) -> Dict:
        """
        Get BDA job results from S3
        
        Args:
            status_response: Response from wait_for_completion or check_status
            
        Returns:
            Results JSON as dictionary
        """
        output_config_uri = status_response.get("outputConfiguration", {}).get("s3Uri")
        if not output_config_uri:
            raise ValueError("No output configuration found in status response")
        
        print(f"Reading output config from {output_config_uri}")
        config_data = self._read_s3_json(output_config_uri)
        
        result_uri = config_data["output_metadata"][0]["segment_metadata"][0]["standard_output_path"]
        print(f"Reading results from {result_uri}")
        result_data = self._read_s3_json(result_uri)
        
        print("✓ Results retrieved successfully")
        return result_data
    
    def _read_s3_json(self, s3_uri: str) -> Dict:
        """Read JSON file from S3"""
        # Parse S3 URI
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
        parts = s3_uri[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        # Read from S3
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    
    def delete_project(self, project_arn: str):
        """Delete BDA project"""
        self.bda_client.delete_data_automation_project(projectArn=project_arn)
        print(f"✓ Deleted project: {project_arn}")
    
    def process_video(
        self,
        video_path: str,
        project_arn: Optional[str] = None,
        create_project_if_none: bool = True,
        wait: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        End-to-end video processing
        
        Args:
            video_path: Local path to video file
            project_arn: Existing project ARN (creates new if None)
            create_project_if_none: Create project if project_arn is None
            wait: Wait for job completion
            verbose: Print progress updates
            
        Returns:
            Dictionary with job info and results (if wait=True)
        """
        # Create project if needed
        created_project = False
        if not project_arn:
            if not create_project_if_none:
                raise ValueError("project_arn is required or set create_project_if_none=True")
            project_arn = self.create_project()
            created_project = True
        
        # Upload video
        s3_uri = self.upload_video(video_path)
        
        # Start job
        invocation_arn = self.start_job(s3_uri, project_arn)
        
        result = {
            'project_arn': project_arn,
            'invocation_arn': invocation_arn,
            's3_uri': s3_uri,
            'created_project': created_project
        }
        
        # Wait for completion if requested
        if wait:
            status_response = self.wait_for_completion(invocation_arn, verbose=verbose)
            results = self.get_results(status_response)
            result['status_response'] = status_response
            result['results'] = results
        
        return result
