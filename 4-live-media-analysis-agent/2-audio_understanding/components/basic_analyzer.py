"""
Basic audio content analyzer using only transcript text
"""

import json
import re
import boto3


class BasicAudioContentAnalyzer:
    """Basic analyzer using only transcript text"""
    
    def __init__(self, model_id=None, region=None):
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        self.model_id = model_id
    
    def analyze_content(self, sentences):
        """Analyze content using only transcript text"""
        if not sentences:
            return {"chapters": [], "overall_summary": "No content to analyze", "total_duration": 0.0}
        
        transcript_text = " ".join([s['sentence'] for s in sentences])
        
        prompt = f"""Analyze the following transcript and organize it into chapters and topics.
        
        TRANSCRIPT:
        {transcript_text}
        
        Please create a structured analysis with the following JSON format:
        
        {{
          "chapters": [
            {{
              "title": "Chapter Title",
              "start_time": 0.0,
              "end_time": 30.0,
              "summary": "Brief chapter summary",
              "topics": [
                {{
                  "title": "Topic Title",
                  "start_time": 0.0,
                  "end_time": 15.0,
                  "description": "Topic description",
                  "key_points": ["Point 1", "Point 2"]
                }}
              ]
            }}
          ],
          "overall_summary": "Overall content summary",
          "total_duration": 120.0
        }}
        
        Return only the JSON structure, no additional text."""
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            
            result = json.loads(response['body'].read())
            analysis_text = result['content'][0]['text']
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from response")
                
        except Exception as e:
            print(f"❌ Error in basic analysis: {e}")
            return {
                "chapters": [],
                "overall_summary": "Analysis failed",
                "total_duration": sentences[-1]['end_time'] if sentences else 0.0
            }
