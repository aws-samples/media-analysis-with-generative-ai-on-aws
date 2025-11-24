import boto3
import json
import asyncio
import time
import copy
from datetime import datetime
from IPython.display import display, HTML, clear_output
import pandas as pd

class TextSpotlightAgent:
    """Agent for analyzing transcript text using AWS Bedrock Nova Lite"""
    
    def __init__(self, aws_region='us-east-1', min_sentences_threshold=5, clip_creator=None):
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        self.min_sentences_threshold = min_sentences_threshold
        self.is_running = False
        self.last_analyzed_count = 0
        self.insights_buffer = []
        self.prompt_buffers = []  # Array to store incremental buffers
        self.chapter_responses = []  # Array to store previous chapter responses
        self.previously_generated_last_chapter = None  # Store only the last chapter from each response
        self.finalized_chapters = []  # Store all chapters except the last one for cleanup
        self.current_non_finalized_chapter = None  # Store the current non-finalized chapter
        
        # Display handle for real-time table updates
        self.display_handle = None
        
        # Clip creator for chapter clips
        self.clip_creator = clip_creator
        self.clips_dir = "output/clips"
        
        # Metrics tracking
        self.bedrock_metrics = {
            'total_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'request_timestamps': [],
            'call_details': []
        }
    
    async def start_analysis(self):
        """Start continuous analysis of sentence buffer"""
        print("🤖 Starting TextSpotlight Agent...")
        
        # Initialize the chapter table display
        self._initialize_chapter_table()
        
        self.is_running = True
        last_table_update = 0
        
        while self.is_running:
            try:
                await self._analyze_current_buffer_pc()
                
                # Refresh table every 5 seconds to show newly created clips
                current_time = time.time()
                if current_time - last_table_update > 5:
                    if self.finalized_chapters:
                        self._update_chapter_table()
                    last_table_update = current_time
                
                await asyncio.sleep(10)  # Poll every 10 seconds
            except Exception as e:
                print(f"❌ TextSpotlight analysis error: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    def _initialize_chapter_table(self):
        """Initialize the chapter table display"""
        html_initial = """
        <div style="margin: 20px 0;">
            <h3 style="color: #2E86AB; font-family: Arial, sans-serif;">
                📚 Finalized Chapters (0 total)
            </h3>
            <div style="border: 2px solid #2E86AB; border-radius: 8px; padding: 20px; background-color: #f8f9fa; text-align: center;">
                <p style="color: #666; font-family: Arial, sans-serif; font-size: 14px;">
                    ⏳ Waiting for chapters to be detected...
                </p>
            </div>
        </div>
        """
        self.display_handle = display(HTML(html_initial), display_id=True)
    
    async def _analyze_current_buffer(self):
        """Analyze current sentence buffer if new content available"""
        # Import global buffer from main module
        from __main__ import SENTENCE_JSON_BUFFER
        
        # Clone the buffer to avoid working with the original instance
        sentence_buffer_clone = copy.deepcopy(SENTENCE_JSON_BUFFER)
        current_count = len(sentence_buffer_clone)
        
        # Only analyze if we have new sentences and at least 3 total
        if current_count > self.last_analyzed_count and current_count >= 3:
            print(f"\n🔍 TextSpotlight analyzing {current_count} sentences...")
            
            # Prepare content for analysis
            sentences_text = " ".join([item['sentence'] for item in SENTENCE_JSON_BUFFER])
            
            prompt = f"""You are analyzing a growing transcript from live audio. This transcript is continuously expanding as new sentences are added. Based on the current content, provide insights including:

1. **Headlines**: Key topics or main points discussed
2. **Topics**: Main themes and subjects covered
3. **Chapters**: Logical sections or segments (if enough content) and you are absoutely sure.
4. **Key Moments**: Important statements or insights with approximate timing
5. Do not repeat the finding every time, but if you see any change in the previous finding , please update.
6. DO not be very agressive to create headlines, topics, chapters and key moments, create content only if you are 100% confidence , else please wait for the next iterations to generate these findings.
7. For every finding, please include start time and end time.
8. Make sure once processed timline for the findings is not processed again.
9. Mention the timeline processed for each finding. ( start time and end time )
10.Avoid creating short chapters, clearly check for theme change or context change for creating chapters. consolidate as much as possible as we may use it for video clipping.

Remember: This is a growing transcript, so focus on what has been said so far while noting that more content may follow.

Please provide your analysis in JSON format.

Current transcript ({current_count} sentences):
{sentences_text}

Sentence timing data:
{json.dumps(SENTENCE_JSON_BUFFER, indent=2)}

Provide your analysis in a structured JSON format:"""

            try:
                # Call Nova Lite model
                start_time = time.time()
                # Get model ID from global config or use default
                model_id = globals().get('AUDIO_MODEL_ID', 'amazon.nova-lite-v1:0')
                response = self.bedrock_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps({
                        'messages': [{
                            'role': 'user',
                            'content': [{'text': prompt}]
                        }],
                        "inferenceConfig": {
                        "maxTokens": 1000,
                        "temperature": 0.1
                        }
                    })
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"⏱️ Nova Lite response time: {response_time:.2f} seconds")
                
                response_body = json.loads(response['body'].read())
                insights = response_body['output']['message']['content'][0]['text']
                
                # Store insights
                insight_data = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'sentence_count': current_count,
                    'insights': insights
                }
                self.insights_buffer.append(insight_data)
                
                # Display insights
                self._display_insights(insights, current_count)
                
                self.last_analyzed_count = current_count
                
            except Exception as e:
                print(f"❌ Error calling Nova Lite: {e}")
    
    def cleanup_cloned_buffer(self, sentence_buffer_clone):
        """Clean up cloned sentence buffer based on finalized chapters"""
        if not self.finalized_chapters:
            return sentence_buffer_clone
            
        try:
            # Find the latest finalized chapter's end time
            latest_end_time = 0
            for chapter in self.finalized_chapters:
                end_time = chapter.get('end_time', 0)
                if ':' in str(end_time):
                    time_parts = str(end_time).split(':')
                    end_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + float(time_parts[2])
                else:
                    end_seconds = float(end_time)
                latest_end_time = max(latest_end_time, end_seconds)
            
            if latest_end_time > 0:
                # Remove sentences before the latest finalized chapter end time
                original_count = len(sentence_buffer_clone)
                cutoff_index = 0
                
                for i, sentence in enumerate(sentence_buffer_clone):
                    sentence_start = sentence.get('start_time', 0)
                    if sentence_start > latest_end_time:
                        cutoff_index = i
                        break
                
                if cutoff_index > 0:
                    sentence_buffer_clone = sentence_buffer_clone[cutoff_index:]
                    cleaned_count = len(sentence_buffer_clone)
                    print(f"🧹 Cleaned cloned buffer: removed {original_count - cleaned_count} sentences before finalized chapters")
                    
        except Exception as e:
            print(f"⚠️ Buffer cleanup failed: {e}")
            
        return sentence_buffer_clone

    async def _analyze_current_buffer_pc(self):
        """Analyze current sentence buffer with prompt caching for chapter identification"""
        from __main__ import SENTENCE_JSON_BUFFER
        
        # Clone the buffer to avoid working with the original instance
        sentence_buffer_clone = copy.deepcopy(SENTENCE_JSON_BUFFER)
        
        # Clean up the cloned buffer based on finalized chapters
        sentence_buffer_clone = self.cleanup_cloned_buffer(sentence_buffer_clone)
        
        current_count = len(sentence_buffer_clone)
        
        if current_count >= self.min_sentences_threshold:
            print(f"\n🔍 TextSpotlight PC analyzing {current_count} sentences for chapters...")
            
            # System prompt with caching checkpoint
            system_prompt = {
                #"role": "system",
                "content": [{
                    "text": """You are an expert content analyzer specializing in identifying and creating chapters from live audio transcript ( which is array of sentences in a json format with each sentence associated with start and end time) . Your task is to:

1. Identify distinct chapters based on topic changes, context shifts, or natural breaks
2. Provide DO NOT consolidated sentences for each chapter
3. Determine accurate start and end times
4. Create meaningful chapter titles
5. Write concise synopses (max 15-20 words)

Guidelines:
- Only create chapters when you're absolutely confident about topic/context changes
- Consolidate content meaningfully - avoid creating short chapter, try and see you can consolidate as much as possible.
- Use timing data to provide accurate start/end times
- Focus on substantial content topic shifts, not minor topic variations
- Each chapter should have sufficient content to warrant separation
- if you feel the chapter is incomplete without any logical completion, then do not generate chapter for the same.""",
                    "cachePoint": { "type": "default" }
                }]
            }
        
            
            # Create content array with cloned sentence buffer
            content_array = []
            
            # Add query first
            content_array.append({
                "text": """Query: Identify chapters and provide response in JSON format with only these attributes: start_time, end_time, chapter_title, synopsis, consolidated transcript. Format: [{"start_time": number, "end_time": number, "chapter_title": "string", "synopsis": "string","consolidated_transcript": "string"}]""",
                "cachePoint": { "type": "default" }
            })
            
            # Add the entire cloned sentence buffer
            content_array.append({
                "text": f"""Complete Transcript Buffer: {sentence_buffer_clone}""",
            })

            # content_array.append({
            #     "text": f"""Previously generated Last Chapter : {self.current_non_finalized_chapter}"""
            # })
            
            messages = [{
                "role": "user", 
                "content": content_array
            }]

            #print(f" Messages used in prompt => {messages}")
            try:
                print(f"🕐 Model invoked at: {datetime.now().strftime('%H:%M:%S')}")
                start_time = time.time()
                # Get model ID from global config or use default
                model_id = globals().get('AUDIO_MODEL_ID', 'amazon.nova-lite-v1:0')
                response = self.bedrock_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps({
                        "system": system_prompt["content"],
                        'messages': messages,
                        "inferenceConfig": {
                            "maxTokens": 1500,
                            "temperature": 0.1
                        }
                    })
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"⏱️ Nova Lite response time: {response_time:.2f} seconds")
                
                response_body = json.loads(response['body'].read())
                chapter_analysis = response_body['output']['message']['content'][0]['text']
                
                # Record chapter response for future prompts
                # Clean markdown code block markers
                if chapter_analysis.startswith('```json'):
                    chapter_analysis = chapter_analysis[7:]  # Remove ```json
                if chapter_analysis.startswith('```'):
                    chapter_analysis = chapter_analysis[3:]  # Remove ```
                if chapter_analysis.endswith('```'):
                    chapter_analysis = chapter_analysis[:-3]  # Remove trailing ```
                chapter_analysis = chapter_analysis.strip()  # Remove extra whitespace
                
                # Clean up sentence buffer if multiple chapters found
                try:
                    import re
                    import json as json_lib
                    
                    # Try to parse as JSON first
                    try:
                        chapter_data = json_lib.loads(chapter_analysis)
                        # Handle different JSON structures
                        if isinstance(chapter_data, list):
                            chapters = chapter_data  # Direct array of chapters
                        elif isinstance(chapter_data, dict):
                            chapters = chapter_data.get('chapters', [])  # Object with chapters key
                        else:
                            chapters = []
                    except Exception as e:
                        print(f"JSON parsing failed: {e}")
                        # Fallback: extract chapters using regex
                        chapter_matches = re.findall(r'"start_time":\s*"?([^",}]+)"?', chapter_analysis)
                        chapters = [{'start_time': match} for match in chapter_matches]

                    print(f"Chapters length {len(chapters)}")
                    # If more than one chapter, store all except last as finalized
                    if len(chapters) > 1:
                        # Get new chapters to add
                        new_chapters = chapters[:-1]
                        current_count = len(self.finalized_chapters)
                        
                        # Store all chapters except the last one as finalized
                        self.finalized_chapters.extend(new_chapters)
                        print(f"📚 Added {len(new_chapters)} finalized chapters, total: {len(self.finalized_chapters)}")
                        
                        # Queue clip creation for new chapters
                        if self.clip_creator:
                            for idx, chapter in enumerate(new_chapters, start=current_count + 1):
                                self.clip_creator.create_chapter_clip(
                                    chapter_id=idx,
                                    start_time=chapter.get('start_time', 0),
                                    end_time=chapter.get('end_time', 0),
                                    chapter_title=chapter.get('chapter_title', 'Untitled')
                                )
                        
                        # Update the real-time table
                        self._update_chapter_table()
                    
                    # Always store the last chapter as current non-finalized
                    if len(chapters) > 0:
                        self.current_non_finalized_chapter = chapters[-1]
                                
                except Exception as e:
                    print(f"⚠️ Chapter analysis parsing failed: {e}")
                
                #print(f" Chapter Analysis => {chapter_analysis}")
                # Extract and print token usage information
                usage = response_body.get('usage', {})
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                total_tokens = input_tokens + output_tokens
                
                # Track metrics
                self.bedrock_metrics['total_requests'] += 1
                self.bedrock_metrics['total_input_tokens'] += input_tokens
                self.bedrock_metrics['total_output_tokens'] += output_tokens
                self.bedrock_metrics['total_tokens'] += total_tokens
                self.bedrock_metrics['request_timestamps'].append(datetime.now())
                self.bedrock_metrics['call_details'].append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'function': '_analyze_current_buffer_pc',
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'response_time': response_time
                })
                
                print(f"\n📊 Token Usage - {usage}")
                
                # Store chapter analysis
                chapter_data = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'sentence_count': current_count,
                    'chapter_analysis': chapter_analysis,
                    'type': 'chapter_analysis_pc',
                    'token_usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens
                    }
                }
                self.insights_buffer.append(chapter_data)
                
                # Display chapter analysis
                self._display_chapter_analysis(chapter_analysis, current_count)
                
                self.last_analyzed_count = current_count
                
            except Exception as e:
                print(f"❌ Error calling Nova Lite with prompt caching: {e}")

    async def _analyze_current_buffer_pc_claude(self):
        """Analyze current sentence buffer with prompt caching for chapter identification using Claude"""
        from __main__ import SENTENCE_JSON_BUFFER
        
        # Clone the buffer to avoid working with the original instance
        sentence_buffer_clone = copy.deepcopy(SENTENCE_JSON_BUFFER)
        current_count = len(sentence_buffer_clone)
        
        if current_count > self.last_analyzed_count and current_count >= 5:
            print(f"\n🔍 TextSpotlight PC Claude analyzing {current_count} sentences for chapters...")
            
            # System prompt with caching checkpoint
            system_prompt = {
                "content": [{
                    "text": """You are an expert content analyzer specializing in identifying chapters with short synoposis, start time and end time . Your task is to:

1. Identify distinct chapters based on topic changes, context shifts, or natural breaks
2. Provide DO NOT consolidated sentences for each chapter
3. Determine accurate start and end times
4. Create meaningful chapter titles
5. Write concise synopses with 30 - 40 words max

Guidelines:
- Only create chapters when you're absolutely confident about topic/context changes
- Consolidate content meaningfully - avoid creating too many short chapters
- Use timing data to provide accurate start/end times
- Focus on substantial content shifts, not minor topic variations
- Each chapter should have sufficient content to warrant separation
- If you feel the chapter is incomplete without any logical completion, then do not generate chapter for the same
- Review previously last generated chapter input if exist and start from that to previously last generated chapter context to generate any incremental chapter from that point.""",
                    "cache_control": {"type": "ephemeral"}
                }]
            }
        
            
            # Create incremental sentence buffers based on analysis calls
            content_array = []

             # Add query first
            content_array.append({
                "type": "text",
                "text": """Query: Identify chapters and provide start time, end time, chapter title, synopsis. please provide json response, if not chapters found return empty json like {}."""
            })
            
            # Add all existing prompt buffers
            for i, buffer in enumerate(self.prompt_buffers):
                content_array.append({
                    "type": "text",
                    "text": f"""Transcript Buffer : {buffer}"""
                })
            
            # Add new sentences since last analysis as new buffer
            if current_count > self.last_analyzed_count:
                new_sentences = sentence_buffer_clone[self.last_analyzed_count:current_count]
                self.prompt_buffers.append(new_sentences)
                content_array.append({
                    "type": "text",
                    "text": f"""Transcript Buffer : {new_sentences}""",
                    "cache_control": { "type": "ephemeral" }
                })


            content_array.append({
                "type": "text",
                "text": f"""Previously generated Last Chapter : {self.previously_generated_last_chapter}"""
            })

            
            messages = [{
                "role": "user", 
                "content": content_array
            }]

            # print(f" Messages used in prompt => {json.dumps({
            #             "system": system_prompt["content"][0]["text"],
            #             'messages': messages,
            #             "max_tokens": 1500,
            #             "temperature": 0.1
            #         })}")
            
            try:
                start_time = time.time()
                # Get model ID from global config or use default
                model_id = globals().get('AUDIOVISUAL_MODEL_ID', 'global.anthropic.claude-sonnet-4-20250514-v1:0')
                response = self.bedrock_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "system": system_prompt["content"][0]["text"],
                        'messages': messages,
                        "max_tokens": 1500,
                        "temperature": 0.1
                    })
                )
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"⏱️ Model response time: {response_time:.2f} seconds")
                
                response_body = json.loads(response['body'].read())
                print(f"Debug - Claude response structure: {list(response_body.keys())}")
                
                # Handle Claude response structure
                if 'content' in response_body:
                    chapter_analysis = response_body['content'][0]['text']
                elif 'completion' in response_body:
                    chapter_analysis = response_body['completion']
                else:
                    # Fallback - print full structure for debugging
                    print(f"Unknown response structure: {response_body}")
                    chapter_analysis = str(response_body)
                
                # Record chapter response for future prompts
                # Clean markdown code block markers
                if chapter_analysis.startswith('```json'):
                    chapter_analysis = chapter_analysis[7:]  # Remove ```json
                if chapter_analysis.startswith('```'):
                    chapter_analysis = chapter_analysis[3:]  # Remove ```
                if chapter_analysis.endswith('```'):
                    chapter_analysis = chapter_analysis[:-3]  # Remove trailing ```
                chapter_analysis = chapter_analysis.strip()  # Remove extra whitespace
                
                # Extract last chapter from response
                try:
                    import re
                    # Find all chapter patterns in the response
                    chapter_matches = re.findall(r'"chapter_title":\s*"[^"]*"[^}]*}', chapter_analysis)
                    if chapter_matches:
                        self.previously_generated_last_chapter = chapter_matches[-1]  # Get last chapter
                except:
                    # Fallback: use simple text splitting
                    lines = chapter_analysis.split('\n')
                    chapter_lines = [line for line in lines if 'chapter' in line.lower()]
                    if chapter_lines:
                        self.previously_generated_last_chapter = chapter_lines[-1]
                
                self.chapter_responses.append(chapter_analysis)
                
                # Extract and print token usage information
                usage = response_body.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                total_tokens = input_tokens + output_tokens
                
                print(f"\n📊 Token Usage - {usage}")
                
                # Store chapter analysis
                chapter_data = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'sentence_count': current_count,
                    'chapter_analysis': chapter_analysis,
                    'type': 'chapter_analysis_pc_claude',
                    'token_usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens
                    }
                }
                self.insights_buffer.append(chapter_data)
                
                # Display chapter analysis
                self._display_chapter_analysis(chapter_analysis, current_count)
                
                self.last_analyzed_count = current_count
                
            except Exception as e:
                print(f"❌ Error calling Claude with prompt caching: {e}")

    def _update_chapter_table(self):
        """Update the real-time chapter table display"""
        try:
            if not self.finalized_chapters:
                return
            
            # Prepare data for table
            import os
            table_data = []
            for i, chapter in enumerate(self.finalized_chapters, 1):
                # Format times
                start_time = chapter.get('start_time', 'N/A')
                end_time = chapter.get('end_time', 'N/A')
                
                # Convert to readable format if numeric
                if isinstance(start_time, (int, float)):
                    start_str = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
                else:
                    start_str = str(start_time)
                    
                if isinstance(end_time, (int, float)):
                    end_str = f"{int(end_time // 60):02d}:{int(end_time % 60):02d}"
                else:
                    end_str = str(end_time)
                
                # Truncate synopsis if too long
                synopsis = chapter.get('synopsis', 'N/A')
                if len(synopsis) > 60:
                    synopsis = synopsis[:57] + "..."
                
                # Check if clip exists
                safe_title = "".join(c for c in chapter.get('chapter_title', 'Untitled') if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title.replace(' ', '_')[:50]
                clip_filename = f"chapter_{i:03d}_{safe_title}.mp4"
                clip_path = os.path.join(self.clips_dir, clip_filename)
                
                if os.path.exists(clip_path):
                    # Create video player HTML
                    playback_cell = f'''<video width="200" height="150" controls style="border-radius: 4px;">
                        <source src="{clip_path}" type="video/mp4">
                    </video>'''
                else:
                    playback_cell = '<span style="color: #999; font-size: 12px;">⏳ Creating...</span>'
                
                table_data.append({
                    '#': i,
                    'Chapter Title': chapter.get('chapter_title', 'Untitled'),
                    'Start': start_str,
                    'End': end_str,
                    'Synopsis': synopsis,
                    'Playback': playback_cell
                })
            
            # Create DataFrame
            df = pd.DataFrame(table_data)
            
            # Style the table with HTML
            html_table = f"""
            <div style="margin: 20px 0;">
                <h3 style="color: #2E86AB; font-family: Arial, sans-serif;">
                    📚 Finalized Chapters ({len(self.finalized_chapters)} total)
                </h3>
                <div style="max-height: 400px; overflow-y: auto; border: 2px solid #2E86AB; border-radius: 8px; padding: 10px; background-color: #f8f9fa;">
                    {df.to_html(index=False, escape=False, classes='chapter-table', border=0)}
                </div>
            </div>
            <style>
                .chapter-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    table-layout: fixed;
                }}
                .chapter-table th {{
                    background-color: #2E86AB;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                .chapter-table th:nth-child(1) {{
                    width: 40px;  /* # column - narrow */
                }}
                .chapter-table th:nth-child(2) {{
                    width: 20%;  /* Chapter Title */
                }}
                .chapter-table th:nth-child(3) {{
                    width: 60px;  /* Start */
                }}
                .chapter-table th:nth-child(4) {{
                    width: 60px;  /* End */
                }}
                .chapter-table th:nth-child(5) {{
                    width: 30%;  /* Synopsis */
                }}
                .chapter-table th:nth-child(6) {{
                    width: 220px;  /* Playback - wider for video */
                }}
                .chapter-table td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
                .chapter-table td:nth-child(1) {{
                    text-align: center;  /* Center the # */
                    font-weight: bold;
                }}
                .chapter-table td:nth-child(6) {{
                    text-align: center;  /* Center the video player */
                }}
                .chapter-table tr:hover {{
                    background-color: #e3f2fd;
                }}
                .chapter-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
            """
            
            # Display or update the table
            if self.display_handle is None:
                self.display_handle = display(HTML(html_table), display_id=True)
            else:
                self.display_handle.update(HTML(html_table))
                
        except Exception as e:
            print(f"⚠️ Error updating chapter table: {e}")
    
    def _display_chapter_analysis(self, analysis, sentence_count):
        """Display the chapter analysis results"""
        print("\n" + "=" * 80)
        print(f"📚 CHAPTER ANALYSIS ({sentence_count} sentences analyzed)")
        print("=" * 80)
        print(analysis)
        print("=" * 80)

    def _display_insights(self, insights, sentence_count):
        """Display the generated insights"""
        print("\n" + "=" * 80)
        print(f"🧠 TEXTSPOTLIGHT INSIGHTS ({sentence_count} sentences analyzed)")
        print("=" * 80)
        print(insights)
        print("=" * 80)
    
    def stop_analysis(self):
        """Stop the analysis process"""
        self.is_running = False
        
        # Finalize the last chapter if it exists
        if self.current_non_finalized_chapter:
            print("📚 Finalizing last chapter...")
            self.finalized_chapters.append(self.current_non_finalized_chapter)
            
            # Create clip for the last chapter
            if self.clip_creator:
                chapter_id = len(self.finalized_chapters)
                self.clip_creator.create_chapter_clip(
                    chapter_id=chapter_id,
                    start_time=self.current_non_finalized_chapter.get('start_time', 0),
                    end_time=self.current_non_finalized_chapter.get('end_time', 0),
                    chapter_title=self.current_non_finalized_chapter.get('chapter_title', 'Untitled')
                )
                
                # Wait for all clips to be created
                print("⏳ Waiting for all clips to be created...")
                self.clip_creator.clip_queue.join()  # Wait for queue to be empty
                print("✅ All clips created")
            
            # Update table one last time after all clips are done
            self._update_chapter_table()
            self.current_non_finalized_chapter = None
        
        print("🛑 TextSpotlight Agent stopped")
    
    def get_final_insights(self):
        """Get all insights generated during the session"""
        return self.insights_buffer.copy()
    
    def print_bedrock_metrics(self):
        """Print comprehensive Bedrock usage metrics"""
        print("\n" + "=" * 80)
        print("📊 BEDROCK USAGE METRICS")
        print("=" * 80)
        
        metrics = self.bedrock_metrics
        
        # Overall statistics
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Total Input Tokens: {metrics['total_input_tokens']:,}")
        print(f"Total Output Tokens: {metrics['total_output_tokens']:,}")
        print(f"Total Tokens: {metrics['total_tokens']:,}")
        
        # Calculate requests per minute
        if len(metrics['request_timestamps']) > 1:
            time_span = (metrics['request_timestamps'][-1] - metrics['request_timestamps'][0]).total_seconds() / 60
            if time_span > 0:
                requests_per_min = metrics['total_requests'] / time_span
                print(f"Requests per Minute: {requests_per_min:.2f}")
        
        # Individual call details
        if metrics['call_details']:
            print(f"\n--- Call Details ---")
            for i, call in enumerate(metrics['call_details'], 1):
                print(f"Call {i} at {call['timestamp']}: {call['input_tokens']} in, {call['output_tokens']} out, {call['response_time']:.2f}s")
        
        print("=" * 80)

    def print_final_summary(self):
        """Print summary of all insights generated"""
        # if self.insights_buffer:
        #     print("\n" + "=" * 80)
        #     print("🧠 TEXTSPOTLIGHT FINAL SUMMARY")
        #     print("=" * 80)
            
        #     for i, insight in enumerate(self.insights_buffer, 1):
        #         print(f"\n--- Analysis {i} at {insight['timestamp']} ({insight['sentence_count']} sentences) ---")
        #         print(insight['insights'])
            
        #     print("=" * 80)
        # else:
        #     print("\n🧠 No insights generated during this session.")
        
        # Print finalized chapters
        if self.finalized_chapters:
            print("\n" + "=" * 80)
            print("📚 FINALIZED CHAPTERS")
            print("=" * 80)
            for i, chapter in enumerate(self.finalized_chapters, 1):
                print(f"\n--- Chapter {i} ---")
                print(json.dumps(chapter, indent=2))
            print("=" * 80)
        else:
            print("\n📚 No finalized chapters yet.")
        
        # Print current non-finalized chapter
        if self.current_non_finalized_chapter:
            print("\n" + "=" * 80)
            print("📖 CURRENT NON-FINALIZED CHAPTER")
            print("=" * 80)
            print(json.dumps(self.current_non_finalized_chapter, indent=2))
            print("=" * 80)
        else:
            print("\n📖 No current non-finalized chapter.")

print("✅ TextSpotlightAgent class defined!")
