#!/usr/bin/env python

import asyncio
import boto3
import json
import time
import os
from boto3.dynamodb.conditions import Key, Attr
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("celebrity-detection-server")

region = os.getenv('AWS_DEFAULT_REGION')
# Initialize AWS clients
rek_client = boto3.client('rekognition', region_name=region)
dynamodb_resource = boto3.resource('dynamodb', region_name=region)

# Load configuration from environment variables
CAST_TABLE = os.environ.get('CAST_TABLE')
CAST_PK = os.environ.get('CAST_PK')

# Validate required environment variables
if not CAST_TABLE or not CAST_PK:
    raise ValueError(
        "Missing required environment variables. "
        "Please set CAST_TABLE and CAST_PK."
    )

# Helper functions
def get_cast_member(cast_id):
    try:
        table = dynamodb_resource.Table(CAST_TABLE)
        key_expression = Key(CAST_PK).eq(cast_id)
        query_data = table.query(
            KeyConditionExpression=key_expression
        )
        return query_data['Items']
    except Exception as e:
        print(f'Error querying table: {CAST_TABLE}.')
        print(str(e))
        return []

def start_celebrity_detection(bucket, video_key):
    response = rek_client.start_celebrity_recognition(
        Video={
            'S3Object': {
                'Bucket': bucket,
                'Name': video_key
            }
        }
    )
    return response['JobId']

def get_celebrity_detection_results(job_id):
    response = rek_client.get_celebrity_recognition(JobId=job_id)
    return response

def extract_bucket_key(video_s3_path):
    path = video_s3_path[5:]  # Remove 's3://'
    bucket, key = path.split('/', 1)  # Split into bucket and key   
    return bucket, key

# MCP Tool
@mcp.tool()
async def detect_key_figures(video_s3_path: str) -> str:
    """
    Detect key figures and their cast role in a video using AWS Rekognition.

    Parameters:
        video_s3_path: S3 path to the video (format: s3://bucket/prefix/video_file)

    Returns:
        JSON string containing a list of detected celebrities with their information:
        - name: Celebrity name
        - confidence: Detection confidence score
        - id: Celebrity ID
        - first_appearance: Timestamp of first appearance
        - Additional cast information if available in DynamoDB

    Environment Variables Required:
        CAST_TABLE: DynamoDB table name for cast information
        CAST_PK: Primary key name for the cast table
    """
    def process_detection():
        try:
            bucket, key = extract_bucket_key(video_s3_path)
            
            job_id = start_celebrity_detection(bucket, key)
            print(f"Started celebrity detection job: {job_id}")

            while True:
                response = get_celebrity_detection_results(job_id)
                status = response['JobStatus']
                
                if status in ['SUCCEEDED', 'FAILED']:
                    print("Job complete")
                    break
                
                print("Job in progress...")
                time.sleep(10)

            unique_celebrities = {}

            if status == 'SUCCEEDED':
                for celebrity in response['Celebrities']:
                    celeb = celebrity['Celebrity']
                    
                    # Only process celebrities with 95%+ confidence
                    if celeb['Confidence'] >= 95.0:
                        celeb_id = celeb['Id']
                        
                        if celeb_id not in unique_celebrities:
                            celebrity_info = {
                                'name': celeb['Name'],
                                'confidence': celeb['Confidence'],
                                'id': celeb_id,
                                'first_appearance': celebrity['Timestamp']
                            }
                            
                            try:
                                query_items = get_cast_member(celeb_id)
                                if query_items:
                                    celebrity_info.update(query_items[0])
                            except Exception as e:
                                print(f"Error fetching additional celebrity info: {str(e)}")
                            
                            unique_celebrities[celeb_id] = celebrity_info
                
                return json.dumps(list(unique_celebrities.values()))
            else:
                print("Detection failed")
                return json.dumps({"error": "Celebrity detection job failed"})
                
        except Exception as e:
            error_message = f"Error processing video: {str(e)}"
            print(error_message)
            return json.dumps({"error": error_message})

    return await asyncio.to_thread(process_detection)

# Run the server
if __name__ == "__main__":
    print(f"Starting celebrity detection server with:")
    print(f"CAST_TABLE: {CAST_TABLE}")
    print(f"CAST_PK: {CAST_PK}")
    mcp.run()