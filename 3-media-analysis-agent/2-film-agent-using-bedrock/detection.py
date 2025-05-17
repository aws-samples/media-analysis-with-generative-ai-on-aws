import boto3
import json
import os
import time
from boto3.dynamodb.conditions import Key, Attr

## DynamoDB parameters
dynamodb_resource = boto3.resource('dynamodb')
cast_table = os.getenv('cast_table')
cast_pk = os.getenv('cast_pk')

## Rekognition parameters
rek_client = boto3.client('rekognition')

def get_named_parameter(event, name):
    try:
        return next(item for item in event['parameters'] if item['name'] == name)['value']
    except StopIteration:
        raise ValueError(f"Required parameter '{name}' not found in event")

def get_cast_member(cast_id):
    try:
        table = dynamodb_resource.Table(cast_table)
        key_expression = Key(cast_pk).eq(cast_id)
        query_data = table.query(
                KeyConditionExpression=key_expression
            )
        return query_data['Items']
    except Exception:
        print(f'Error querying table: {cast_table}.')

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

def detect_key_figures(video_s3_path):
    bucket, key = extract_bucket_key(video_s3_path)

    job_id = start_celebrity_detection(bucket, key)
    print(f"Started celebrity detection job: {job_id}")

    while True:
        response = get_celebrity_detection_results(job_id)
        status = response['JobStatus']

        if status in ['SUCCEEDED', 'FAILED']:
            print("JOB COMPLETE....")
            break

        print("Job in progress...")
        time.sleep(10)

    unique_celebrities = {}  # Dictionary to store unique celebrities

    if status == 'SUCCEEDED':
        for celebrity in response['Celebrities']:
            celeb = celebrity['Celebrity']

            # Only process celebrities with 95%+ confidence
            if celeb['Confidence'] >= 95.0:
                celeb_id = celeb['Id']

                # Store or update celebrity info only if not already stored
                if celeb_id not in unique_celebrities:
                    celebrity_info = {
                        'name': celeb['Name'],
                        'confidence': celeb['Confidence'],
                        'id': celeb_id,
                        'first_appearance': celebrity['Timestamp']
                    }

                    # If you have additional celebrity info in DynamoDB
                    try:
                        query_items = get_cast_member(celeb_id)
                        if query_items:
                            celebrity_info.update(query_items[0])
                    except Exception as e:
                        print(f"Error fetching additional celebrity info: {str(e)}")

                    unique_celebrities[celeb_id] = celebrity_info
    else:
        print("Detection failed....")

    # Convert the dictionary values to a list for the final output
    final_output = list(unique_celebrities.values())
    return final_output

def populate_function_response(event, response_body):
    return {
        'response': {
            'actionGroup': event['actionGroup'],
            'function': event['function'],
            'functionResponse': {
                'responseBody': {
                    'TEXT': {
                        'body': str(response_body)
                    }
                }
            }
        }
    }

def lambda_handler(event, context):
    print(event)

    function = event.get('function', '')
    parameters = event.get('parameters', [])
    video_s3_path = get_named_parameter(event, "video_s3_path")

    if function == 'detect_key_figures':
        result = detect_key_figures(video_s3_path)
    else:
        result = f"Error, function '{function}' not recognized"

    response = populate_function_response(event, result)
    print(response)
    return response
