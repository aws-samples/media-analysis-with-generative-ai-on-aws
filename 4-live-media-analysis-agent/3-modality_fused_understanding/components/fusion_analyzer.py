"""
Fusion Analyzer - Multimodal AI analysis combining visual and audio
Module-specific component for Modality Fusion Understanding
"""

import asyncio
import time
import os
import queue
import copy
import threading
import json
import subprocess
import boto3
import base64
from datetime import datetime
from pathlib import Path
import traceback

# Import shared components
try:
    from src.shared import log_component
except ImportError:
    # Fallback if shared components not available
    def log_component(component, message, level="INFO"):
        # Simple fallback that respects log levels
        if level in ["ERROR", "WARNING"] or level == "INFO":
            print(f"[{component}] {message}")


class FusionAnalyzer:
    """Combines visual and audio analysis using multimodal AI"""
    
    def __init__(self, aws_region='us-east-1', sentence_buffer=None, analysis_results=None, output_dir='output', keep_n_chapters=1, memory_client=None, memory_id=None, actor_id=None, session_id=None):
        self.aws_region = aws_region
        
        # Configure Bedrock client with standard retries
        from botocore.config import Config
        retry_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'standard'
            }
        )
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region, config=retry_config)
        self.analysis_queue = queue.Queue()
        self.is_running = False
        self.sentence_buffer = sentence_buffer if sentence_buffer is not None else []
        self.analysis_results = analysis_results if analysis_results is not None else {}
        self.output_dir = output_dir
        self.keep_n_chapters = keep_n_chapters  # Number of finalized chapters to keep in context
        
        # Conversational history for Claude (like working notebook)
        self.messages = []  # Store conversation history
        
        # For table display
        self.display_handle = None
        self.finalized_chapters = []  # Chapters excluding the last incomplete one
        
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_cache_read_tokens = 0
        self.chunk_metrics = []  # Store per-chunk metrics
        
        # Store the main thread's event loop for cross-thread display updates
        try:
            import asyncio
            self.main_loop = asyncio.get_event_loop()
        except Exception:
            self.main_loop = None
        
        # AgentCore memory integration
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        
    def initialize_display(self):
        """Initialize the chapter table display"""
        self._initialize_chapter_table()
    
    def start_analysis(self):
        """Start multimodal analysis worker"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.worker_thread.start()
        log_component("FusionAnalyzer", "✅ Started Fusion analyzer")
    
    def _analysis_worker(self):
        """Worker thread for processing analysis requests"""
        import time
        
        while self.is_running:
            try:
                analysis_request = self.analysis_queue.get(timeout=1)
                self._perform_fusion_analysis(analysis_request)
                self.analysis_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log_component("FusionAnalyzer", f"❌ Analysis worker error: {e}", "ERROR")
        
        # After shutdown signal, drain remaining queue with timeout
        log_component("FusionAnalyzer", "   🔄 Draining remaining queue items...", "DEBUG")
        drain_timeout = 60  # 60 seconds to drain remaining items
        drain_start = time.time()
        
        while (time.time() - drain_start) < drain_timeout and not getattr(self, '_shutdown_timeout', False):
            try:
                # Use timeout instead of nowait to catch items added during drain
                analysis_request = self.analysis_queue.get(timeout=2)
                log_component("FusionAnalyzer", f"   📋 Processing queued chunk {analysis_request.get('chunk_id', '?')}", "DEBUG")
                self._perform_fusion_analysis(analysis_request)
                self.analysis_queue.task_done()
            except queue.Empty:
                # Queue empty - check if really done
                if self.analysis_queue.empty():
                    log_component("FusionAnalyzer", "   ✅ Queue drained successfully", "DEBUG")
                    break
                # Otherwise continue waiting
            except Exception as e:
                log_component("FusionAnalyzer", f"❌ Analysis worker error during drain: {e}", "ERROR")
        
        if getattr(self, '_shutdown_timeout', False):
            log_component("FusionAnalyzer", "   ⏰ Worker stopped due to timeout", "WARNING")
        elif not self.analysis_queue.empty():
            log_component("FusionAnalyzer", f"   ⚠️ Drain timeout with {self.analysis_queue.qsize()} items remaining", "WARNING")
    
    def queue_analysis(self, chunk_id, filmstrip_path, start_time, end_time, shot_change_frames=None, spectrogram_data=None):
        """Queue a chunk for multimodal analysis with spectrogram data"""
        analysis_request = {
            'chunk_id': chunk_id,
            'filmstrip_path': filmstrip_path,
            'start_time': start_time,
            'end_time': end_time,
            'shot_change_frames': shot_change_frames or [],
            'spectrogram_data': spectrogram_data,
            'timestamp': time.time()
        }
        self.analysis_queue.put(analysis_request)
        if shot_change_frames:
            log_component("FusionAnalyzer", f"📋 Queued analysis for chunk {chunk_id} ({len(shot_change_frames)} shot changes)", "DEBUG")
        else:
            log_component("FusionAnalyzer", f"📋 Queued analysis for chunk {chunk_id}", "DEBUG")
    
    def _perform_fusion_analysis(self, request):
        """Perform multimodal analysis using Claude with conversational history and prompt caching"""
        try:
            chunk_id = request['chunk_id']
            filmstrip_path = request['filmstrip_path']
            start_time = request['start_time']
            end_time = request['end_time']
            shot_change_frames = request.get('shot_change_frames', [])
            spectrogram_data = request.get('spectrogram_data', None)
            
            log_component("FusionAnalyzer", f"🔍 Analyzing chunk {chunk_id}: {start_time:.1f}s - {end_time:.1f}s", "DEBUG")
            if shot_change_frames:
                log_component("FusionAnalyzer", f"   📍 Shot changes detected at frame indices: {shot_change_frames}", "DEBUG")
            
            # Get transcript for the current chunk
            transcript_data = self._get_transcript_for_timerange(start_time, end_time, self.sentence_buffer)
            transcript_text = transcript_data['text']
            transcript_sentences = transcript_data['sentences']
            
            # Encode filmstrip image
            encoded_image = base64.b64encode(self._encode_image(filmstrip_path)).decode('utf-8')
            
            # System prompt (cached with ephemeral cache control)
            system_prompt = """
# Video Topic Analysis Assistant

You analyze video content in 20-second increments to identify topics and chapters. Each chunk provides:
- **Filmstrip**: 4×5 grid (20 frames) with timestamps and grid positions [Row×Column]
- **Transcript**: Timestamped sentences with start/end times
- **Audio Spectrogram Analysis**: Acoustic features including tempo, spectral centroid, RMS energy, zero crossing rate, and delivery style insights
- **Shot Changes**: Optional timestamps indicating visual scene transitions

Your task is to incrementally build a structured analysis by deciding whether to extend existing topics or create new ones based on natural content flow.

---

## Core Principles

1. **Audio-Enhanced Topic Summaries**: MANDATORY - Include audio emotions and delivery style from spectrogram data in ALL topic summaries
2. **Content-Driven Decisions**: Let the content naturally determine topic and chapter boundaries
3. **Emotional Audio Integration**: Use spectrogram insights (energy levels, tempo, delivery style) to describe speaker emotions and tone
4. **Continuity**: Maintain natural flow between topics with contiguous coverage
5. **No Micro-Topics**: Topics should be at least 5 seconds duration
6. **Multimodal Analysis**: Consider all available data (visual + audio + spectrogram) to understand content shifts
7. **Extract Visual Text**: Include any visible text, signs, labels, or written content from filmstrips in topic summaries

## Audio Spectrogram Integration Requirements

When spectrogram analysis is provided, you MUST:
- **Include audio emotions** in topic titles (e.g., "Energetic Introduction", "Calm Technical Discussion", "Emphatic Conclusion")
- **Reference delivery style** in summaries using provided audio characteristics (animated vs conversational, dynamic vs measured)
- **Connect audio patterns** to content themes (high energy for exciting topics, calm delivery for technical explanations)
- **Describe speaker tone** based on spectral features and energy levels
6. **Emotions and tonality**: Include emotions based on the spectrogram analysis data.

---

## Timestamp Priority for Boundaries

When determining topic start_time and end_time, use this hierarchy:

**Priority 1: Transcript Timestamps** (when available)
- Use sentence_start_time_in_sec and sentence_end_time_in_sec from the JSON transcript
- Most accurate for content boundaries
- If transcript array is empty or contains "[No transcript available for this time range]", skip to Priority 2

**Priority 2: Shot Change Timestamps** (when transcript unavailable)
- Use timestamps from "SHOT CHANGES DETECTED" section (e.g., "45.20s [grid 2×1, frame 5]")
- These indicate visual scene transitions and are MORE PRECISE than chunk boundaries
- Extract the timestamp value (e.g., 45.20s) for topic boundaries

**Priority 3: Filmstrip Frame Timestamps** (fallback only)
- Use frame timestamps from grid labels (e.g., "[1×1] | 0.5s")
- Only use when NO transcript AND NO shot changes are available
- Least precise but always available

**CRITICAL**: NEVER use chunk boundaries (0s, 20s, 40s, 60s, etc.) as topic boundaries when better data is available. Always prefer: (1) transcript timestamps like "3.456s" from sentence_start_time_in_sec or sentence_end_time_in_sec (BEST), (2) shot change timestamps like "45.20s" (GOOD), or (3) frame timestamps like "12.5s" from grid labels (ACCEPTABLE). Chunk boundaries should only be used as a last resort.

**Important**: This hierarchy is ONLY for determining start/end times. For deciding whether content has shifted (topic detection), consider ALL available data holistically.

---

## Topic Detection Logic

**When to Extend Existing Topic:**
- Content continues the same theme or subject
- Natural progression of the current discussion
- Related concepts within the same domain

**When to Create New Topic:**
- Content substantially shifts to a different subject within the same domain
- Clear transition in what's being discussed
- New topic must be at least 5 seconds duration

**When to Create New Chapter:**
- Major domain or subject change (e.g., from "Serverless" to "Databases")
- Significant shift in overall theme or category

---

## Continuity Rules - CRITICAL

1. **NO GAPS ALLOWED**: Every second of video MUST belong to a topic. Topics must be strictly contiguous.
2. **Extend When Uncertain**: If unsure whether to create a new topic or extend existing one, ALWAYS extend to avoid gaps.
3. **Boundary Alignment**: When creating a new topic, its start_time MUST equal the previous topic's end_time (no gaps between topics).
4. **No Overlaps**: A topic's start_time must be >= previous topic's end_time.
5. **Avoid Micro-Topics**: Don't create topics shorter than 5 seconds unless absolutely necessary.
6. **Rapid Transitions**: In fast-paced content (sports, action), group related moments into longer topics rather than creating many micro-topics.

---

## Input Data Format

**Transcript Example:**
```json
[
  {
    "sentence_start_time_in_sec": 3.456,
    "sentence_end_time_in_sec": 7.892,
    "sentence": "Welcome to AWS re:Invent 2023."
  }
]
```

**Filmstrip**: 4×5 grid with red borders. Each frame labeled with [Row×Column] and timestamp.

**Shot Changes** (when provided):
```
SHOT CHANGES DETECTED:
3 shot changes at: 45.20s [grid 2×1, frame 5], 52.15s [grid 3×3, frame 12]
```

---

## Output Format

**Return ONLY incremental updates as JSON:**

```json
{
  "actions": [
    {
      "type": "new_chapter",
      "id": "h1",
      "chapter": "Introduction and Welcome"
    },
    {
      "type": "new_topic",
      "id": "t1",
      "chapter_id": "h1",
      "topic_summary": "Welcome message and conference overview with visible AWS re:Invent logo",
      "start_time": 3.456,
      "end_time": 25.2,
      "chunks": [0, 1]
    },
    {
      "type": "update_topic",
      "id": "t1",
      "topic_summary": "Welcome message, conference overview, and session agenda with AWS re:Invent branding",
      "end_time": 47.892,
      "chunks": [0, 1, 2]
    }
  ],
  "analysis_status": {
    "total_chunks_processed": 3,
    "processing_notes": "Extended introduction topic to include agenda discussion"
  }
}
```

**Action Types:**
- `new_chapter`: Create new chapter with unique ID (h1, h2, h3...)
- `new_topic`: Create new topic under existing chapter (t1, t2, t3...)
- `update_topic`: Modify existing topic (extend duration, refine summary, add chunks)

**Rules:**
- Only output what CHANGED or is NEW
- Reference existing chapters/topics by their IDs
- Use sequential IDs: h1, h2, h3... for chapters; t1, t2, t3... for topics
- Chunk numbers are integers in arrays: [0, 1, 2] not ["0", "1", "2"]
- Topic summaries should be 50-150 words, detailed enough for users to understand content before viewing

---

## Examples

### Example 1: Extending a Topic
```
Previous: Topic t1 "AWS Lambda Introduction" (0s-40s)
New chunk: Speaker continues with Lambda pricing details (40s-60s)
Audio: Calm, measured delivery (RMS: 0.08, Tempo: 95 BPM)
Decision: EXTEND - same subject, natural continuation

Output:
{
  "actions": [
    {
      "type": "update_topic",
      "id": "t1",
      "topic_summary": "AWS Lambda introduction covering core concepts and pricing models with calm, technical delivery style",
      "end_time": 60.0,
      "chunks": [0, 1, 2]
    }
  ]
}
```

### Example 2: Creating a New Topic
```
Previous: Topic t1 "Lambda Pricing" (0s-60s)
New chunk: Speaker shifts to "API Gateway setup" (60s-80s)
Audio: Energetic, animated delivery (RMS: 0.15, Spectral Centroid: 2200 Hz)
Decision: NEW TOPIC - different subject, still serverless domain

Output:
{
  "actions": [
    {
      "type": "new_topic",
      "id": "t2",
      "chapter_id": "h1",
      "topic_summary": "Energetic demonstration of API Gateway integration with Lambda functions",
      "start_time": 60.0,
      "end_time": 80.0,
      "chunks": [3, 4]
    }
  ]
}
```

### Example 3: Creating a New Chapter
```
Previous: Chapter h1 "Serverless Computing" with Lambda/API Gateway topics
New chunk: "Now let's discuss RDS database options" (120s-140s)
Audio: Emphatic, dynamic delivery (RMS: 0.12, Tempo: 110 BPM)
Decision: NEW CHAPTER - major domain change from serverless to databases

Output:
{
  "actions": [
    {
      "type": "new_chapter",
      "id": "h2",
      "chapter": "Database Services"
    },
    {
      "type": "new_topic",
      "id": "t3",
      "chapter_id": "h2",
      "topic_summary": "Emphatic introduction to Amazon RDS database options with dynamic presentation style",
      "start_time": 120.0,
      "end_time": 140.0,
      "chunks": [6]
    }
  ]
}
```

### Example 4: Handling Missing Transcript with Shot Changes
```
Chunk 9 (180s-200s): No transcript, but shot changes detected at: 185.5s, 192.3s
Previous topic ended at 180.0s
Decision: Use shot change timestamps for precise boundaries

Output:
{
  "actions": [
    {
      "type": "new_topic",
      "id": "t5",
      "chapter_id": "h2",
      "topic_summary": "Visual demonstration showing database configuration screens with RDS console interface",
      "start_time": 180.0,
      "end_time": 185.5,
      "chunks": [9]
    },
    {
      "type": "new_topic",
      "id": "t6",
      "chapter_id": "h2",
      "topic_summary": "Transition to live database deployment showing terminal commands and status indicators",
      "start_time": 185.5,
      "end_time": 192.3,
      "chunks": [9]
    }
  ]
}
```

### Example 5: No Transcript, No Shot Changes (Rare)
```
Chunk 10 (200s-220s): No transcript, no shot changes detected
Previous topic ended at 200.0s
Decision: Use filmstrip frame timestamps for boundaries

Output:
{
  "actions": [
    {
      "type": "update_topic",
      "id": "t6",
      "topic_summary": "Continued database deployment sequence showing terminal output and progress indicators across multiple frames",
      "end_time": 212.5,
      "chunks": [9, 10]
    }
  ]
}
```
Note: Used frame timestamp 212.5s (from grid position) instead of chunk boundary 220s.

---

## Important Notes

- **CRITICAL**: Every second of video must be covered by a topic - no gaps allowed
- Review your previous responses in the conversation history to understand existing topics/chapters
- When in doubt between extending or creating new topic: EXTEND to maintain coverage
- If you create a new topic, ensure its start_time equals the previous topic's end_time
- Create new topics/chapters only when content substantially shifts
- Always include visible text from filmstrips in your summaries
- Ensure topics are at least 5 seconds duration
- **AVOID CHUNK BOUNDARIES**: Don't use perfect 20-second intervals (0s, 20s, 40s, 60s, etc.) as topic boundaries unless absolutely necessary. Use transcript timestamps (Priority 1) or shot change timestamps (Priority 2) for precision


CRITICAL REQUIREMENTS BASED on SPECTROGRAM ANALYSIS:
- All topics MUST reference the speaker's delivery style using the spectrogram data
- Include "audio_tone" field for each chapter describing vocal characteristics
- Add "audio_delivery_analysis" field summarizing overall speaking patterns
- Use tempo to identify pacing changes between sections
- Use RMS energy to identify emphasis and key moments
- Use spectral centroid to gauge speaker engagement and excitement levels
- Descriptions should reflect whether content is delivered with high energy, calmly, urgently, etc.
- DONOT mention the audio charateristic measure numbers while you build the summary.
- MORE focus to the script delivered with inclusion of little emotions identified using audio analysis.
Now analyze the provided chunk data and return your incremental JSON response.
"""
            
            # Build new message with cache control (like working notebook)
            # Format transcript as JSON with precise timestamps
            transcript_json = json.dumps(transcript_sentences, indent=2, ensure_ascii=False)
            
            # Build shot change description
            shot_change_text = ""
            if shot_change_frames:
                # Calculate approximate timestamps for shot changes
                chunk_duration = end_time - start_time
                shot_change_times = []
                for frame_idx in shot_change_frames:
                    # Frame index 0-19 extracted at 0.5-19.5 seconds (middle of each second)
                    time_in_chunk = ((frame_idx + 0.5) / 20) * chunk_duration
                    absolute_time = start_time + time_in_chunk
                    # Calculate grid position (row, col) from frame index
                    row = (frame_idx // 5) + 1  # 5 columns per row, 1-indexed
                    col = (frame_idx % 5) + 1   # 1-indexed
                    shot_change_times.append(f"{absolute_time:.2f}s [grid {row}×{col}, frame {frame_idx}]")
                shot_change_text = f"\n\nSHOT CHANGES DETECTED:\n{len(shot_change_frames)} shot changes at: {', '.join(shot_change_times)}\n\n"
                # Log the shot change text for visibility
                log_component("FusionAnalyzer", f"📍 Shot changes for chunk {chunk_id}: {shot_change_text.strip()}", "DEBUG")

            # log_component("FusionAnalyzer", f"MESSAGES LIST BEFORE NEW MESSAGE APPEND:\n{self.messages}")
            
            # Build spectrogram analysis section
            spectrogram_text = ""
            if spectrogram_data and spectrogram_data.get('audio_features'):
                audio_features = spectrogram_data['audio_features']
                audio_description = spectrogram_data.get('audio_description', 'Audio analysis available')
                
                spectrogram_text = f"""

AUDIO SPECTROGRAM ANALYSIS:
Duration: {audio_features.get('duration', 0):.1f} seconds
Tempo: {audio_features.get('tempo', 0):.1f} BPM
Spectral Centroid: {audio_features.get('spectral_centroid_mean', 0):.1f} Hz
RMS Energy: {audio_features.get('rms_mean', 0):.4f}
Zero Crossing Rate: {audio_features.get('zero_crossing_rate_mean', 0):.4f}
Audio Characteristics: {audio_description}

AUDIO INSIGHTS FOR CONTENT ANALYSIS:
- Spectral Centroid ({audio_features.get('spectral_centroid_mean', 0):.1f} Hz): {'High energy/animated' if audio_features.get('spectral_centroid_mean', 0) > 2000 else 'Calm/conversational'} delivery
- RMS Energy ({audio_features.get('rms_mean', 0):.4f}): {'Dynamic/emphatic' if audio_features.get('rms_mean', 0) > 0.1 else 'Steady/measured'} speaking style
- Tempo ({audio_features.get('tempo', 0):.1f} BPM): {'Rapid/urgent' if audio_features.get('tempo', 0) > 120 else 'Deliberate/thoughtful'} pacing

"""

            new_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Chunk identifier: chunk_{chunk_id:04d}. Time range: {start_time:.1f}s - {end_time:.1f}s.\n\nHere is the audio transcript for this chunk with precise timestamps:\n\n{transcript_json} \n\n{spectrogram_text}",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Above is the chunk filmstrip showing 20 frames in a 4x5 grid with clear borders and labels, uniformly sampled across the 20-second chunk (1 frame per second). Each frame shows its grid position [Row×Column] and timestamp below it.{shot_change_text}"
                    }
                ]
            }
            
            # Append to conversation history
            self.messages.append(new_message)
            
            log_component("FusionAnalyzer", f"Sending {len(self.messages)} messages to Bedrock Claude", "INFO")
            
            # Prepare request for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                "messages": self.messages,
                "temperature": 0.1
            }
            
            start_llm_call = time.time()
            
            # Call Bedrock Claude
            # Get model ID from global config or use default
            model_id = globals().get('AUDIOVISUAL_MODEL_ID', 'global.anthropic.claude-sonnet-4-20250514-v1:0')
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            log_component("FusionAnalyzer", f"Total time taken by Bedrock: {time.time() - start_llm_call:.2f}s", "DEBUG")
            
            # Parse response
            response_body = json.loads(response['body'].read())
            usage = response_body.get('usage', {})
            
            # Extract token counts
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            cache_read = usage.get('cache_read_input_tokens', 0)
            cache_write = usage.get('cache_creation_input_tokens', 0)
            
            # Update totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cache_read_tokens += cache_read
            self.total_cache_write_tokens += cache_write
            
            # Calculate cache hit ratio for this call
            total_input_with_cache = input_tokens + cache_read
            cache_hit_ratio = (cache_read / total_input_with_cache * 100) if total_input_with_cache > 0 else 0
            
            # Store per-chunk metrics
            chunk_metric = {
                'chunk_id': chunk_id,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cache_read': cache_read,
                'cache_write': cache_write,
                'cache_hit_ratio': cache_hit_ratio
            }
            self.chunk_metrics.append(chunk_metric)
            
            # Print usage statistics for this chunk
            log_component("FusionAnalyzer", f"\n📊 Model Usage for chunk {chunk_id}:", "DEBUG")
            log_component("FusionAnalyzer", f"   Input tokens: {input_tokens:,}", "DEBUG")
            log_component("FusionAnalyzer", f"   Output tokens: {output_tokens:,}", "DEBUG")
            log_component("FusionAnalyzer", f"   Total tokens: {input_tokens + output_tokens + cache_read + cache_write:,}", "DEBUG")
            
            # Print cache statistics
            if cache_read > 0 or cache_write > 0:
                log_component("FusionAnalyzer", f"   💾 Cache read tokens: {cache_read:,}", "DEBUG")
                log_component("FusionAnalyzer", f"   💾 Cache write tokens: {cache_write:,}", "DEBUG")
                log_component("FusionAnalyzer", f"   📈 Cache hit ratio: {cache_hit_ratio:.1f}%", "DEBUG")
                if cache_read > 0:
                    savings_pct = (cache_read / total_input_with_cache) * 100
                    log_component("FusionAnalyzer", f"   💰 Cache savings: {savings_pct:.1f}% of total input", "DEBUG")
            
            # Print cumulative statistics
            total_all_tokens = self.total_input_tokens + self.total_output_tokens + self.total_cache_read_tokens + self.total_cache_write_tokens
            overall_cache_hit_ratio = (self.total_cache_read_tokens / (self.total_input_tokens + self.total_cache_read_tokens) * 100) if (self.total_input_tokens + self.total_cache_read_tokens) > 0 else 0
            
            log_component("FusionAnalyzer", f"\n📈 Cumulative Totals (after {len(self.chunk_metrics)} chunks):", "DEBUG")
            log_component("FusionAnalyzer", f"   Total input tokens: {self.total_input_tokens:,}", "DEBUG")
            log_component("FusionAnalyzer", f"   Total output tokens: {self.total_output_tokens:,}", "DEBUG")
            log_component("FusionAnalyzer", f"   Total cache write: {self.total_cache_write_tokens:,}", "DEBUG")
            log_component("FusionAnalyzer", f"   Total cache read: {self.total_cache_read_tokens:,}", "DEBUG")
            log_component("FusionAnalyzer", f"   Overall cache hit ratio: {overall_cache_hit_ratio:.1f}%", "DEBUG")
            log_component("FusionAnalyzer", f"   Grand total tokens: {total_all_tokens:,}", "DEBUG")
            
            content = response_body['content'][0]['text']
            
            # Clear image and cache_control from last message
            self.messages[-1] = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Chunk identifier: chunk_{chunk_id:04d}. Time range: {start_time:.1f}s - {end_time:.1f}s.\n\nHere is the audio transcript for this chunk with precise timestamps:\n\n{transcript_json}\n\n"
                    }
                ]
            }
            
            # Add assistant response to conversation
            self.messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content
                    }
                ]
            })
            
            # Create memory event for Bedrock response
            # self._create_memory_event_for_bedrock_response(chunk_id, content)

        
            # Parse and store analysis
            try:
                # Extract JSON from response (handle preamble text)
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1).strip()
                else:
                    raise json.JSONDecodeError("No JSON block found", content, 0)
                
                incremental_result = json.loads(content)
                
                # Process incremental updates and build complete analysis result
                analysis_result = self._process_incremental_updates(incremental_result, chunk_id, start_time, end_time, transcript_text, transcript_sentences)
                
                # Detect and log overlapping topics (validation)
                self._detect_overlapping_topics(analysis_result, chunk_id)
                
                # Store result
                self.analysis_results[chunk_id] = analysis_result
                
                # Print formatted analysis result
                self._print_analysis_result(chunk_id, analysis_result)
                
                # Update finalized chapters table (exclude last incomplete chapter)
                self.update_finalized_chapters()
                
            except json.JSONDecodeError as e:
                log_component("FusionAnalyzer", f"❌ Failed to parse analysis JSON for chunk {chunk_id}: {e}", "ERROR")
                log_component("FusionAnalyzer", f"   Raw content: {content}...", "ERROR")
                
        except Exception as e:
            log_component("FusionAnalyzer", f"❌ Fusion analysis error for chunk {chunk_id}: {e} {traceback.format_exc()}", "ERROR")
    
    def _process_incremental_updates(self, incremental_result, chunk_id, start_time, end_time, transcript_text, transcript_sentences):
        """Process incremental updates and build complete analysis result"""
        
        # Get previous analysis result or initialize empty structure
        if chunk_id > 0 and (chunk_id - 1) in self.analysis_results:
            # Start with previous complete structure
            analysis_result = copy.deepcopy(self.analysis_results[chunk_id - 1])
            chapters = analysis_result.get('chapters', [])
        else:
            # First chunk - initialize empty structure
            analysis_result = {'chapters': []}
            chapters = []
        
        # Create ID mappings for quick lookup
        chapter_map = {h.get('chapter'): i for i, h in enumerate(chapters)}
        topic_map = {}
        for h_idx, chapter in enumerate(chapters):
            for t_idx, topic in enumerate(chapter.get('topics', [])):
                if 'id' not in topic:
                    topic['id'] = f"t{len(topic_map) + 1}"  # Assign ID if missing
                topic_map[topic['id']] = (h_idx, t_idx)
        
        # Process each action from incremental result
        actions = incremental_result.get('actions', [])
        
        for action in actions:
            action_type = action.get('type')
            
            if action_type == 'new_chapter':
                # Add new chapter with ID
                new_chapter = {
                    '_id': action.get('id'),  # Store the ID from LLM
                    'chapter': action.get('chapter'),
                    'confidence': action.get('confidence', 0),
                    'topics': []
                }
                chapters.append(new_chapter)
                chapter_map[new_chapter['chapter']] = len(chapters) - 1
                
            elif action_type == 'new_topic':
                # Add new topic to specified chapter
                chapter_id = action.get('chapter_id')
                
                # Find chapter by ID
                chapter_idx = None
                for i, h in enumerate(chapters):
                    if '_id' not in h:  # Assign ID if missing (for backward compatibility)
                        h['_id'] = f"h{i+1}"
                    if h.get('_id') == chapter_id:
                        chapter_idx = i
                        break
                
                if chapter_idx is not None:
                    new_topic = {
                        'id': action.get('id'),
                        'topic_summary': action.get('topic_summary'),
                        'start_time': action.get('start_time'),
                        'end_time': action.get('end_time'),
                        'chunks': action.get('chunks', [])
                    }
                    chapters[chapter_idx]['topics'].append(new_topic)
                    topic_map[new_topic['id']] = (chapter_idx, len(chapters[chapter_idx]['topics']) - 1)
                    
                    # Create memory event for new topic
                    self._create_memory_event_for_topic(new_topic, 'new_topic', chunk_id)
                
            elif action_type == 'update_topic':
                # Update existing topic
                topic_id = action.get('id')
                
                if topic_id in topic_map:
                    h_idx, t_idx = topic_map[topic_id]
                    topic = chapters[h_idx]['topics'][t_idx]
                    
                    # Update fields that are provided
                    if 'topic_summary' in action:
                        topic['topic_summary'] = action['topic_summary']
                    if 'start_time' in action:
                        topic['start_time'] = action['start_time']
                    if 'end_time' in action:
                        topic['end_time'] = action['end_time']
                    if 'chunks' in action:
                        topic['chunks'] = action['chunks']
                    
                    # Create memory event for updated topic
                    self._create_memory_event_for_topic(topic, 'update_topic', chunk_id)
        
        # Update analysis result with processed chapters
        analysis_result['chapters'] = chapters
        analysis_result['chunk_id'] = chunk_id
        analysis_result['time_range'] = f"{start_time:.1f}s-{end_time:.1f}s"
        analysis_result['transcript'] = transcript_text
        analysis_result['transcript_sentences'] = transcript_sentences
        analysis_result['analysis_status'] = incremental_result.get('analysis_status', {})
        
        return analysis_result
    
    def _print_analysis_result(self, chunk_id, analysis_result):
        """Print formatted analysis result after each chunk"""
        import textwrap
        
        chapters = analysis_result.get('chapters', [])
        analysis_status = analysis_result.get('analysis_status', {})
        
        print("\n" + "="*90)
        print(f"✅ ANALYSIS COMPLETE FOR CHUNK {chunk_id}".center(90))
        print("="*90)
        
        # Print statistics
        total_topics = sum(len(chapter.get('topics', [])) for chapter in chapters)
        print(f"\n📊 Current State:")
        print(f"   • Chapters: {len(chapters)}")
        print(f"   • Topics: {total_topics}")
        print(f"   • Chunks Processed: {analysis_status.get('total_chunks_processed', chunk_id + 1)}")
        
        # Print analysis notes
        if analysis_status.get('notes'):
            print("\n💡 Analysis Notes:")
            notes = analysis_status['notes']
            wrapped_notes = textwrap.fill(notes, width=84, initial_indent='   ', subsequent_indent='   ')
            print(wrapped_notes)
        
        # Print each chapter with topics
        if chapters:
            log_component("FusionAnalyzer", f"\n{'─'*90}", "DEBUG")
            log_component("FusionAnalyzer", "📚 CHAPTERS & TOPICS:", "DEBUG")
            log_component("FusionAnalyzer", f"{'─'*90}", "DEBUG")
            
            for chapter_idx, chapter_data in enumerate(chapters, 1):
                chapter = chapter_data.get('chapter', 'Untitled')
                confidence = chapter_data.get('confidence', 0)
                topics = chapter_data.get('topics', [])
                
                log_component("FusionAnalyzer", f"\n📖 Chapter {chapter_idx}: {chapter}", "DEBUG")
                
                for topic_idx, topic in enumerate(topics, 1):
                    summary = topic.get('topic_summary', 'No summary')
                    start_time = topic.get('start_time', 0)
                    end_time = topic.get('end_time', 0)
                    chunks = topic.get('chunks', [])
                    duration = end_time - start_time
                    
                    log_component("FusionAnalyzer", f"\n   🎬 Topic {topic_idx}:", "DEBUG")
                    
                    # Wrap summary
                    wrapped_summary = textwrap.fill(summary, width=80, initial_indent='      ', subsequent_indent='      ')
                    log_component("FusionAnalyzer", wrapped_summary, "DEBUG")
                    
                    log_component("FusionAnalyzer", f"      ⏱️  {self._format_time(start_time)} → {self._format_time(end_time)} (Duration: {self._format_time(duration)})", "DEBUG")
                    log_component("FusionAnalyzer", f"      🔢 Precise: {start_time:.3f}s → {end_time:.3f}s", "DEBUG")
                    log_component("FusionAnalyzer", f"      📦 Chunks: {len(chunks)} ({chunks[0]} to {chunks[-1]})", "DEBUG")
        
        print("\n" + "="*90 + "\n")
    
    def _get_transcript_for_timerange(self, start_time, end_time, sentence_buffer):
        """Extract transcript text and structured data for specific time range"""
        relevant_sentences = []
        relevant_sentences_json = []
        
        for sentence_data in sentence_buffer:
            sentence_start = sentence_data.get('start_time', 0)
            sentence_end = sentence_data.get('end_time', 0)
            
            # Check if sentence overlaps with time range
            if (sentence_start <= end_time and sentence_end >= start_time):
                # Add to text list
                relevant_sentences.append(sentence_data['sentence'])
                
                # Add to JSON structure
                relevant_sentences_json.append({
                    "sentence_start_time_in_sec": sentence_data['start_time'],
                    "sentence_end_time_in_sec": sentence_data['end_time'],
                    "sentence": sentence_data['sentence']
                })
        
        # Return both text and JSON structure
        transcript_text = " ".join(relevant_sentences) if relevant_sentences else "[No transcript available for this time range]"
        
        return {
            'text': transcript_text,
            'sentences': relevant_sentences_json
        }
    
    def _detect_overlapping_topics(self, analysis_result, chunk_id):
        """Detect and log overlapping topics without fixing them"""
        chapters = analysis_result.get('chapters', [])
        overlap_count = 0
        
        for chapter_idx, chapter in enumerate(chapters):
            topics = chapter.get('topics', [])
            if len(topics) <= 1:
                continue
            
            # Sort topics by start time
            topics_sorted = sorted(topics, key=lambda t: t.get('start_time', 0))
            
            # Check for overlaps
            for i in range(len(topics_sorted) - 1):
                current_topic = topics_sorted[i]
                next_topic = topics_sorted[i + 1]
                
                current_end = current_topic.get('end_time', 0)
                next_start = next_topic.get('start_time', 0)
                
                # If there's an overlap (current end > next start)
                if current_end > next_start:
                    overlap_duration = current_end - next_start
                    overlap_count += 1
                    
                    log_component("FusionAnalyzer", 
                        f"⚠️ OVERLAP DETECTED in chunk {chunk_id}, chapter {chapter_idx + 1}:", 
                        "WARNING")
                    log_component("FusionAnalyzer", 
                        f"   Topic {i+1}: '{current_topic.get('summary', 'N/A')[:50]}...' ends at {current_end:.3f}s", 
                        "WARNING")
                    log_component("FusionAnalyzer", 
                        f"   Topic {i+2}: '{next_topic.get('summary', 'N/A')[:50]}...' starts at {next_start:.3f}s", 
                        "WARNING")
                    log_component("FusionAnalyzer", 
                        f"   Overlap duration: {overlap_duration:.3f}s", 
                        "WARNING")
        
        if overlap_count > 0:
            log_component("FusionAnalyzer", 
                f"📊 Total overlaps detected in chunk {chunk_id}: {overlap_count}", 
                "WARNING")
        else:
            log_component("FusionAnalyzer", 
                f"✅ No overlapping topics detected in chunk {chunk_id}", "DEBUG")
    
    def _encode_image(self, image_path):
        """Read image bytes for Bedrock"""
        with open(image_path, 'rb') as image_file:
            return image_file.read()
    
    def _create_memory_event_for_bedrock_response(self, chunk_id, response_content):
        """Create memory event for Bedrock analysis response"""
        log_component("FusionAnalyzer", f"🔍 _create_memory_event_for_bedrock_response called for chunk {chunk_id}", "DEBUG")
        
        if not self.memory_client or not self.memory_id:
            # Only log once when first chunk is analyzed
            if not hasattr(self, '_memory_warning_logged'):
                log_component("FusionAnalyzer", "⚠️ Memory client not configured - Bedrock responses will not be saved to memory", "WARNING")
                self._memory_warning_logged = True
            return
        
        try:
            log_component("FusionAnalyzer", f"💾 Creating memory event for chunk {chunk_id}...", "DEBUG")
            
            # Create event with Bedrock response
            #messages = [(response_content, "ASSISTANT")]
            messages = [
                {
                    'conversational': {
                        'content': {
                            'text': response_content
                        },
                        'role': 'ASSISTANT'
                    }
                }
            ]
            
            self.memory_client.create_event(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=self.session_id,
                eventTimestamp=datetime.now(),
                payload=messages
            )
            
            log_component("FusionAnalyzer", f"✅ Memory event created successfully for chunk {chunk_id}", "DEBUG")
            log_component("FusionAnalyzer", f"   Memory ID: {self.memory_id}", "DEBUG")
            log_component("FusionAnalyzer", f"   Session ID: {self.session_id}", "DEBUG")
            
        except Exception as e:
            log_component("FusionAnalyzer", f"❌ Failed to create memory event for chunk {chunk_id}: {e}", "ERROR")
            import traceback
            log_component("FusionAnalyzer", f"   Traceback: {traceback.format_exc()}", "ERROR")
    
    def _create_memory_event_for_topic(self, topic, action_type, chunk_id):
        """Create memory event for new or updated topics"""
        if not self.memory_client or not self.memory_id:
            return
        
        try:
            # Include complete topic information in the content
            topic_info = {
                'action': action_type,
                'topic_id': topic.get('id'),
                'topic_summary': topic.get('topic_summary'),
                'start_time': topic.get('start_time'),
                'end_time': topic.get('end_time'),
                'chunks': topic.get('chunks', []),
                'chunk_id': chunk_id
            }
            
            content_text = json.dumps(topic_info, indent=2)
            
            messages = [
                {
                    'conversational': {
                        'content': {
                            'text': content_text
                        },
                        'role': 'ASSISTANT'
                    }
                }
            ]
            
            self.memory_client.create_event(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=self.session_id,
                eventTimestamp=datetime.now(),
                payload=messages
            )
            
            log_component("FusionAnalyzer", f"✅ Memory event created for {action_type} - Topic {topic.get('id')}", "DEBUG")
            
        except Exception as e:
            log_component("FusionAnalyzer", f"❌ Failed to create memory event for {action_type}: {e}", "ERROR")
    
    def stop_analysis(self):
        """Stop analysis worker and wait for completion"""
        self.is_running = False
        
        # Wait for queue to be empty
        log_component("FusionAnalyzer", "🛑 Stopping fusion analyzer...")
        log_component("FusionAnalyzer", "   ⏳ Waiting for pending analyses to complete...", "DEBUG")
        
        # Wait for queue with timeout
        import time
        timeout = 300  # 300 seconds timeout
        start_time = time.time()
        
        # Wait for queue AND worker thread
        while (not self.analysis_queue.empty() or 
               (hasattr(self, 'worker_thread') and self.worker_thread.is_alive())) and (time.time() - start_time) < timeout:
            time.sleep(1)
            
            # Progress logging every 15 seconds
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 15 == 0:
                queue_size = self.analysis_queue.qsize()
                worker_status = 'active' if hasattr(self, 'worker_thread') and self.worker_thread.is_alive() else 'stopped'
                log_component("FusionAnalyzer", f"   ⏳ Still waiting... Queue: {queue_size}, Worker: {worker_status}", "DEBUG")
        
        if not self.analysis_queue.empty() or (hasattr(self, 'worker_thread') and self.worker_thread.is_alive()):
            queue_size = self.analysis_queue.qsize()
            log_component("FusionAnalyzer", f"   ⚠️ Timeout: {queue_size} analyses still pending", "WARNING")
            # Signal worker to stop draining
            self._shutdown_timeout = True
        else:
            log_component("FusionAnalyzer", "   ✅ All pending analyses completed", "DEBUG")
        
        # Wait for worker thread to finish
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            if self.worker_thread.is_alive():
                log_component("FusionAnalyzer", "   ⚠️ Worker thread still running (daemon will exit)", "WARNING")
            else:
                log_component("FusionAnalyzer", "   ✅ Worker thread stopped", "DEBUG")
        
        # Finalize the last chapter and update table
        log_component("FusionAnalyzer", "   📚 Finalizing last chapter...", "DEBUG")
        self.finalize_all_chapters()
        
        # Wait for clip creation to complete
        self.wait_for_clip_creation()
        
        # Ensure all threads complete before main process exits
        log_component("FusionAnalyzer", "   🔄 Ensuring all background threads complete...", "DEBUG")
        import threading
        for thread in threading.enumerate():
            if thread != threading.current_thread() and not thread.daemon:
                if thread.is_alive():
                    log_component("FusionAnalyzer", f"   ⏳ Waiting for thread: {thread.name}", "DEBUG")
                    thread.join(timeout=30)
        
        # Print final metrics summary
        self.print_token_metrics()
        
        log_component("FusionAnalyzer", "🛑 Fusion analyzer stopped")
    
    def generate_final_summary(self):
        """Generate a comprehensive summary of all analyzed content"""
        log_component("FusionAnalyzer", "📝 Generating final summary of all content...", "DEBUG")
        
        if not self.analysis_results:
            log_component("FusionAnalyzer", "⚠️ No analysis results available for summary", "WARNING")
            return
        
        # Get the final analysis result (contains all cumulative chapters)
        final_chunk_id = max(self.analysis_results.keys())
        final_result = self.analysis_results[final_chunk_id]
        chapters = final_result.get('chapters', [])
        
        if not chapters:
            log_component("FusionAnalyzer", "⚠️ No chapters found for summary", "WARNING")
            return
        
        # Calculate total duration for context
        total_duration = 0
        if chapters:
            last_chapter = chapters[-1]
            if last_chapter.get('topics'):
                total_duration = last_chapter['topics'][-1].get('end_time', 0)
        
        duration_str = f"{int(total_duration // 60):02d}:{int(total_duration % 60):02d}"
        
        # Create adaptive summary prompt
        summary_prompt = f"""Based on the complete video analysis below, first determine the content type and then generate an appropriate final summary.

VIDEO ANALYSIS RESULTS:
Duration: {duration_str} ({total_duration:.1f} seconds)
Chapters: {len(chapters)}
{json.dumps(chapters, indent=2)}

INSTRUCTIONS:
1. First, analyze the content to determine its type and provide a confidence level (High/Medium/Low)
2. Then provide a tailored summary based on the content type:

**For Conference/Keynote/Presentation Content:**
- **Event Overview**: Brief description of the event/presentation
- **Key Speakers**: List of speakers mentioned (if any)
- **Main Topics**: Bullet points of key themes and concepts covered
- **Key Takeaways**: Important insights and learnings
- **Call to Action**: Any next steps or recommendations mentioned (if applicable)

**For Movie/Film/Entertainment Content:**
- **Genre & Style**: Type of film and cinematic approach
- **Plot Summary**: Brief narrative overview of the story
- **Key Characters**: Main characters and their roles (if identifiable)
- **Themes**: Central themes and messages
- **Cinematic Elements**: Notable visual or audio elements observed

**For Documentary Content:**
- **Subject Matter**: What the documentary covers
- **Key Information**: Main facts and insights presented
- **Narrative Structure**: How the story unfolds
- **Expert Voices**: Speakers or experts featured (if any)
- **Educational Value**: Key learnings and takeaways

**For Tutorial/Educational Content:**
- **Learning Objectives**: What the content teaches
- **Step-by-Step Breakdown**: Main instructional segments
- **Tools/Technologies**: Software, tools, or concepts covered
- **Skill Level**: Beginner, intermediate, or advanced
- **Practical Applications**: How to apply the knowledge

**For Other/Mixed Content Types:**
- **Content Type**: Describe the specific type of content identified
- **Main Purpose**: What the content aims to achieve or communicate
- **Key Elements**: Most important aspects or segments
- **Target Audience**: Who this content is intended for
- **Summary**: Overall description and main points covered

**OUTPUT REQUIREMENTS:**
- Start with: "**Content Type Identified:** [Type] (Confidence: [High/Medium/Low])"
- Use the appropriate template above
- Keep each section concise but informative
- Include specific details from the analysis when available

Adapt your response format to match the identified content type. Be comprehensive but concise."""

        try:
            # Replicate existing Bedrock call pattern
            start_llm_call = time.time()
            
            # Get model ID from global config or use default
            model_id = globals().get('AUDIOVISUAL_MODEL_ID', 'global.anthropic.claude-sonnet-4-20250514-v1:0')
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4096,
                    "system": "You are an intelligent video content analyzer that provides tailored summaries based on content type. Analyze the video structure and topics to determine the most appropriate summary format.",
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "temperature": 0.3
                })
            )
            
            llm_call_duration = time.time() - start_llm_call
            response_body = json.loads(response['body'].read())
            summary = response_body['content'][0]['text']
            
            # Track token usage like existing calls
            input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
            
            log_component("FusionAnalyzer", f"✅ Final summary generated ({llm_call_duration:.1f}s, {input_tokens}→{output_tokens} tokens)", "DEBUG")
            
            print("\n" + "="*80)
            print("📋 CONTENT SUMMARY")
            print("="*80)
            print(summary)
            print("="*80 + "\n")
            
            return summary
            
        except Exception as e:
            log_component("FusionAnalyzer", f"❌ Error generating final summary: {e}", "ERROR")
            return None
    
    def wait_for_clip_creation(self, timeout=120):
        """Wait for background clip creation to complete"""
        if hasattr(self, 'clip_thread') and self.clip_thread.is_alive():
            log_component("FusionAnalyzer", "⏳ Waiting for clip creation to complete...")
            try:
                self.clip_thread.join(timeout=timeout)
                if self.clip_thread.is_alive():
                    log_component("FusionAnalyzer", f"⚠️ Clip creation timeout after {timeout}s", "WARNING")
                    log_component("FusionAnalyzer", "   Some clips may still be processing in background", "WARNING")
                else:
                    log_component("FusionAnalyzer", "✅ All clips created successfully")
            except Exception as e:
                log_component("FusionAnalyzer", f"❌ Error waiting for clips: {e}", "ERROR")
        else:
            log_component("FusionAnalyzer", "ℹ️ No clip creation in progress", "DEBUG")
    
    def print_token_metrics(self):
        """Print comprehensive token usage metrics"""
        if not self.chunk_metrics:
            print("\n⚠️ No token metrics available")
            return
        
        print("\n" + "="*100)
        print("📊 TOKEN USAGE METRICS SUMMARY".center(100))
        print("="*100)
        
        # Overall statistics
        total_all_tokens = self.total_input_tokens + self.total_output_tokens + self.total_cache_read_tokens + self.total_cache_write_tokens
        total_cache_tokens = self.total_cache_read_tokens + self.total_cache_write_tokens
        overall_cache_hit_ratio = (self.total_cache_read_tokens / (self.total_input_tokens + self.total_cache_read_tokens) * 100) if (self.total_input_tokens + self.total_cache_read_tokens) > 0 else 0
        
        print(f"\n🎯 Overall Statistics:")
        print(f"   • Total chunks analyzed: {len(self.chunk_metrics)}")
        print(f"   • Total input tokens: {self.total_input_tokens:,}")
        print(f"   • Total output tokens: {self.total_output_tokens:,}")
        print(f"   • Total cache write tokens: {self.total_cache_write_tokens:,}")
        print(f"   • Total cache read tokens: {self.total_cache_read_tokens:,}")
        print(f"   • Overall cache hit ratio: {overall_cache_hit_ratio:.1f}%")
        print(f"   • Grand total tokens: {total_all_tokens:,}")
        
        # Cost estimation (Claude Sonnet 4 pricing)
        # Input: $3 per 1M tokens, Cache write: $3.75 per 1M, Cache read: $0.30 per 1M, Output: $15 per 1M
        input_cost = (self.total_input_tokens / 1_000_000) * 3.0
        cache_write_cost = (self.total_cache_write_tokens / 1_000_000) * 3.75
        cache_read_cost = (self.total_cache_read_tokens / 1_000_000) * 0.30
        output_cost = (self.total_output_tokens / 1_000_000) * 15.0
        total_cost = input_cost + cache_write_cost + cache_read_cost + output_cost
        
        # Calculate cost without caching (for comparison)
        cost_without_cache = ((self.total_input_tokens + self.total_cache_read_tokens + self.total_cache_write_tokens) / 1_000_000) * 3.0 + output_cost
        savings = cost_without_cache - total_cost
        savings_pct = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0
        
        print(f"\n💰 Estimated Cost (Claude Sonnet 4):")
        print(f"   • Input tokens cost: ${input_cost:.4f}")
        print(f"   • Cache write cost: ${cache_write_cost:.4f}")
        print(f"   • Cache read cost: ${cache_read_cost:.4f}")
        print(f"   • Output tokens cost: ${output_cost:.4f}")
        print(f"   • Total cost: ${total_cost:.4f}")
        print(f"   • Cost without caching: ${cost_without_cache:.4f}")
        print(f"   • Savings from caching: ${savings:.4f} ({savings_pct:.1f}%)")
        
        # Per-chunk breakdown
        print(f"\n📋 Per-Chunk Breakdown:")
        print(f"   {'Chunk':<8} {'Input':<10} {'Output':<10} {'Cache R':<10} {'Cache W':<10} {'Hit %':<8}")
        print(f"   {'-'*66}")
        
        for metric in self.chunk_metrics:
            print(f"   {metric['chunk_id']:<8} "
                  f"{metric['input_tokens']:<10,} "
                  f"{metric['output_tokens']:<10,} "
                  f"{metric['cache_read']:<10,} "
                  f"{metric['cache_write']:<10,} "
                  f"{metric['cache_hit_ratio']:<8.1f}")
        
        print("\n" + "="*100 + "\n")
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def update_finalized_chapters(self):
        """Update chapter table with all chapters but only create clips for finalized ones"""
        if not self.analysis_results:
            return
        
        # Get the latest analysis result
        sorted_results = sorted(self.analysis_results.items(), key=lambda x: x[0])
        latest_result = sorted_results[-1][1]
        
        all_chapters = latest_result.get('chapters', [])
        
        log_component("FusionAnalyzer", f"📊 Chapter update: Found {len(all_chapters)} total chapters", "DEBUG")
        
        # Debug: Log chapter details
        for i, chapter in enumerate(all_chapters, 1):
            topics_count = len(chapter.get('topics', []))
            log_component("FusionAnalyzer", f"   Chapter {i}: {chapter.get('chapter', 'Untitled')} ({topics_count} topics)", "DEBUG")
        
        if len(all_chapters) >= 1:
            # Always update table with ALL chapters (including incomplete last one)
            self.all_chapters_for_display = all_chapters
            log_component("FusionAnalyzer", f"   📺 Setting all_chapters_for_display with {len(all_chapters)} chapters", "DEBUG")
            self._update_chapter_table()
            log_component("FusionAnalyzer", f"   📺 Called _update_chapter_table()", "DEBUG")
            
            # Only create clips for finalized chapters (all except the last one)
            if len(all_chapters) > 1:
                finalized_chapters = all_chapters[:-1]
                newly_finalized_count = len(finalized_chapters) - len(self.finalized_chapters)
                
                if newly_finalized_count > 0:
                    log_component("FusionAnalyzer", f"✨ {newly_finalized_count} new chapter(s) finalized - creating clips!")
                    self.finalized_chapters = finalized_chapters
                    self._create_chapter_clips()
                
                # Clean up old messages to save tokens
                self._cleanup_old_messages(all_chapters)
            else:
                log_component("FusionAnalyzer", f"⏳ Only 1 chapter so far - no clips created yet")
        else:
            log_component("FusionAnalyzer", f"⏳ No chapters found yet")
    
    def finalize_all_chapters(self):
        """Finalize ALL chapters including the last one (called at shutdown)"""
        if not self.analysis_results:
            log_component("FusionAnalyzer", "⚠️ No analysis results to finalize", "WARNING")
            return
        
        # Get the latest analysis result
        sorted_results = sorted(self.analysis_results.items(), key=lambda x: x[0])
        latest_result = sorted_results[-1][1]
        
        chapters = latest_result.get('chapters', [])
        
        if not chapters:
            log_component("FusionAnalyzer", "⚠️ No chapters found", "WARNING")
            return
        
        log_component("FusionAnalyzer", f"📚 Finalizing all {len(chapters)} chapters (including last incomplete one)")
        
        # Finalize ALL chapters (including the last one)
        self.finalized_chapters = chapters
        
        # Create clips for all chapters
        self._create_chapter_clips()
        
        # Update the table display with all chapters
        self._update_chapter_table()
        
        log_component("FusionAnalyzer", f"✅ All {len(chapters)} chapters finalized and displayed")
    
    def _cleanup_old_messages(self, all_chapters):
        """
        Clean up conversation history to keep only recent chapters
        Keeps messages for: last N finalized chapters + current chapter
        """
        # Handle None case (treat as no limit)
        if self.keep_n_chapters is None:
            return
        
        if len(all_chapters) <= self.keep_n_chapters + 1:
            # Not enough chapters to clean up yet
            return
        
        # Determine which chapters to keep
        # Keep: last N finalized chapters + current (non-finalized) chapter
        chapters_to_keep = all_chapters[-(self.keep_n_chapters + 1):]
        
        # Get chunk IDs to keep
        chunks_to_keep = set()
        for chapter in chapters_to_keep:
            for topic in chapter.get('topics', []):
                chunks_to_keep.update(topic.get('chunks', []))
        
        log_component("FusionAnalyzer", f"🧹 Cleaning up conversation history...", "DEBUG")
        log_component("FusionAnalyzer", f"   Keeping last {self.keep_n_chapters} finalized chapter(s) + current chapter", "DEBUG")
        log_component("FusionAnalyzer", f"   Chunks to keep: {sorted(chunks_to_keep)}", "DEBUG")
        
        # Filter messages - keep only those for chunks we want to keep
        # Messages are in pairs: user message (with chunk_id in text) + assistant response
        original_count = len(self.messages)
        filtered_messages = []
        
        i = 0
        while i < len(self.messages):
            message = self.messages[i]
            
            # Check if this is a user message with chunk identifier
            if message.get('role') == 'user':
                content = message.get('content', [])
                if content and isinstance(content, list):
                    # Look for chunk identifier in text
                    text_content = next((c.get('text', '') for c in content if c.get('type') == 'text'), '')
                    
                    # Extract chunk_id from text like "Chunk identifier: chunk_0001"
                    keep_this_pair = False
                    for chunk_id in chunks_to_keep:
                        chunk_identifier = f"chunk_{chunk_id:04d}"
                        if chunk_identifier in text_content:
                            keep_this_pair = True
                            break
                    
                    if keep_this_pair:
                        # Keep this user message
                        filtered_messages.append(message)
                        
                        # Keep the corresponding assistant response (next message)
                        if i + 1 < len(self.messages) and self.messages[i + 1].get('role') == 'assistant':
                            filtered_messages.append(self.messages[i + 1])
                            i += 2
                        else:
                            i += 1
                    else:
                        # Skip this user message and its response
                        if i + 1 < len(self.messages) and self.messages[i + 1].get('role') == 'assistant':
                            i += 2  # Skip both user and assistant
                        else:
                            i += 1  # Skip just user
                else:
                    # User message without proper content structure - skip
                    i += 1
            else:
                # Non-user message (orphaned assistant message) - skip
                i += 1
        
        self.messages = filtered_messages
        removed_count = original_count - len(self.messages)
        
        log_component("FusionAnalyzer", f"   ✅ Removed {removed_count} messages (kept {len(self.messages)} messages)", "DEBUG")
        log_component("FusionAnalyzer", f"   💾 This will reduce token usage in future requests", "DEBUG")
    
    def _create_chapter_clips(self):
        """Create video clips for finalized chapters AND topics in background thread"""
        import threading
        
        def create_clips_background():
            """Background thread function for creating clips"""
            import subprocess
            
            # Try multiple possible recording locations and formats
            recording_paths = [
                f"{self.output_dir}/recording",
                f"{self.output_dir}",
                "../sample_videos"
            ]
            
            recording_path = None
            
            # Look for video files in possible locations
            for rec_dir in recording_paths:
                if os.path.exists(rec_dir):
                    # Look for various video formats
                    video_files = []
                    for ext in ['.mxf', '.mp4', '.avi', '.mov', '.mkv']:
                        video_files.extend([f for f in os.listdir(rec_dir) if f.lower().endswith(ext)])
                    
                    if video_files:
                        recording_path = os.path.join(rec_dir, video_files[0])
                        log_component("FusionAnalyzer", f"📹 Found video file: {recording_path}", "DEBUG")
                        break
            
            if not recording_path:
                log_component("FusionAnalyzer", "⚠️ No video recording file found for clip creation", "WARNING")
                log_component("FusionAnalyzer", f"   Searched in: {recording_paths}", "WARNING")
                return
            clips_dir = f"{self.output_dir}/clips"
            os.makedirs(clips_dir, exist_ok=True)
            
            for i, chapter in enumerate(self.finalized_chapters, 1):
                topics = chapter.get('topics', [])
                if not topics:
                    continue
                
                # Get time range from first and last topic
                start_time = topics[0].get('start_time', 0)
                end_time = topics[-1].get('end_time', 0)
                duration = end_time - start_time
                
                # Create safe filename for chapter
                chapter_title = chapter.get('chapter', 'Untitled')
                safe_title = "".join(c for c in chapter_title if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title.replace(' ', '_')[:50]
                clip_filename = f"chapter_{i:03d}_{safe_title}.mp4"
                clip_path = os.path.join(clips_dir, clip_filename)
                
                # Create chapter clip if it doesn't exist
                if not os.path.exists(clip_path):
                    log_component("FusionAnalyzer", f"✂️  Creating clip for chapter {i}: {chapter_title} ({start_time:.1f}s-{end_time:.1f}s)", "DEBUG")
                    
                    cmd = [
                        'ffmpeg', '-i', recording_path,
                        '-ss', str(start_time), '-t', str(duration),
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-c:a', 'aac',
                        '-avoid_negative_ts', 'make_zero',
                        '-y', clip_path
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0 and os.path.exists(clip_path):
                            clip_size = os.path.getsize(clip_path)
                            log_component("FusionAnalyzer", f"✅ Created chapter clip: {clip_filename} ({clip_size} bytes)", "DEBUG")
                        else:
                            log_component("FusionAnalyzer", f"❌ Failed to create clip for chapter {i}", "ERROR")
                            if result.stderr:
                                log_component("FusionAnalyzer", f"   FFmpeg error: {result.stderr[:200]}", "ERROR")
                    except subprocess.TimeoutExpired:
                        log_component("FusionAnalyzer", f"⏰ Timeout creating clip for chapter {i} - skipping", "WARNING")
                    except Exception as e:
                        log_component("FusionAnalyzer", f"❌ Error creating clip for chapter {i}: {e}", "ERROR")
                
                # Create clips for each topic in this chapter
                for j, topic in enumerate(topics, 1):
                    topic_start = topic.get('start_time', 0)
                    topic_end = topic.get('end_time', 0)
                    topic_duration = topic_end - topic_start
                    topic_summary = topic.get('topic_summary', 'No summary')
                    
                    # Validate topic timestamps
                    if topic_duration <= 0:
                        log_component("FusionAnalyzer", f"⚠️ Skipping chapter {i}, topic {j}: Invalid duration ({topic_duration:.2f}s)", "WARNING")
                        continue
                    
                    if topic_start < 0:
                        log_component("FusionAnalyzer", f"⚠️ Skipping chapter {i}, topic {j}: Negative start time ({topic_start:.2f}s)", "WARNING")
                        continue
                    
                    # Create safe filename for topic (use first 30 chars of summary)
                    safe_summary = "".join(c for c in topic_summary if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_summary = safe_summary.replace(' ', '_')[:30]
                    topic_clip_filename = f"chapter_{i:03d}_topic_{j:02d}_{safe_summary}.mp4"
                    topic_clip_path = os.path.join(clips_dir, topic_clip_filename)
                    
                    # Skip if topic clip already exists
                    if os.path.exists(topic_clip_path):
                        continue
                    
                    log_component("FusionAnalyzer", f"✂️  Creating clip for chapter {i}, topic {j}: {topic_summary[:50]}... ({topic_start:.1f}s-{topic_end:.1f}s, duration: {topic_duration:.1f}s)", "DEBUG")
                    
                    cmd = [
                        'ffmpeg', '-i', recording_path,
                        '-ss', str(topic_start), '-t', str(topic_duration),
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-c:a', 'aac',
                        '-avoid_negative_ts', 'make_zero',
                        '-y', topic_clip_path
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                        if result.returncode == 0 and os.path.exists(topic_clip_path):
                            clip_size = os.path.getsize(topic_clip_path)
                            log_component("FusionAnalyzer", f"✅ Created topic clip: {topic_clip_filename} ({clip_size} bytes)", "DEBUG")
                        else:
                            log_component("FusionAnalyzer", f"❌ Failed to create clip for chapter {i}, topic {j}", "ERROR")
                            log_component("FusionAnalyzer", f"   Topic: {topic_summary[:80]}", "ERROR")
                            log_component("FusionAnalyzer", f"   Timestamps: start={topic_start:.3f}s, end={topic_end:.3f}s, duration={topic_duration:.3f}s", "ERROR")
                            if result.stderr:
                                # Print more of the error for debugging
                                stderr_lines = result.stderr.split('\n')
                                # Get last 10 lines which usually contain the actual error
                                error_msg = '\n   '.join(stderr_lines[-10:])
                                log_component("FusionAnalyzer", f"   FFmpeg error:\n   {error_msg}", "ERROR")
                    except subprocess.TimeoutExpired:
                        log_component("FusionAnalyzer", f"⏰ Timeout creating clip for chapter {i}, topic {j} - skipping", "WARNING")
                    except Exception as e:
                        log_component("FusionAnalyzer", f"❌ Error creating clip for chapter {i}, topic {j}: {e}", "ERROR")
        
        # Run clip creation in background thread to avoid blocking analysis
        self.clip_thread = threading.Thread(target=create_clips_background, daemon=False)
        self.clip_thread.start()
        log_component("FusionAnalyzer", "🎬 Started background clip creation (chapters + topics)", "DEBUG")
    
    def _initialize_chapter_table(self):
        """Initialize the chapter table display - just set flag, actual display happens in notebook"""
        self._display_needs_update = False
        log_component("FusionAnalyzer", "✅ Chapter table initialized (display handled by notebook)")
    
    def get_chapter_table_html(self):
        """Generate HTML for chapter table (called from notebook)"""
        if not hasattr(self, 'all_chapters_for_display') or not self.all_chapters_for_display:
            return """
            <div style="margin: 20px 0;">
                <h3 style="color: #2E86AB; font-family: Arial, sans-serif;">
                    📚 Chapters (0 total)
                </h3>
                <div style="max-height: 400px; overflow-y: auto; border: 2px solid #2E86AB; border-radius: 8px; padding: 10px; background-color: #f8f9fa;">
                    <p style="text-align: center; color: #999; padding: 20px;">
                        ⏳ Waiting for chapters to be detected...
                    </p>
                </div>
            </div>
            """
        
        return self._build_chapter_table_html()
    
    def _update_chapter_table(self):
        """Mark that chapter table needs update (actual display happens in notebook)"""
        # Just set the flag - the notebook will handle the actual display
        self._display_needs_update = True
        log_component("FusionAnalyzer", f"📊 Chapter table data updated ({len(self.all_chapters_for_display) if hasattr(self, 'all_chapters_for_display') else 0} chapters)", "DEBUG")
    
    def _build_chapter_table_html(self):
        """Build HTML for chapter table (internal method)"""
        try:
            if not hasattr(self, 'all_chapters_for_display') or not self.all_chapters_for_display:
                return ""
            
            clips_dir = f"{self.output_dir}/clips"
            
            # Build HTML for collapsible chapter table
            chapters_html = ""
            
            for i, chapter in enumerate(self.all_chapters_for_display, 1):
                chapter_title = chapter.get('chapter', 'Untitled')
                topics = chapter.get('topics', [])
                
                if not topics:
                    continue
                
                # Get chapter time range from topics
                chapter_start = topics[0].get('start_time', 0)
                chapter_end = topics[-1].get('end_time', 0)
                chapter_duration = chapter_end - chapter_start
                
                # Format chapter times
                start_str = f"{int(chapter_start // 60):02d}:{int(chapter_start % 60):02d}"
                end_str = f"{int(chapter_end // 60):02d}:{int(chapter_end % 60):02d}"
                duration_str = f"{int(chapter_duration // 60):02d}:{int(chapter_duration % 60):02d}"
                
                # Check if this chapter is finalized (has clips)
                is_finalized = hasattr(self, 'finalized_chapters') and i <= len(self.finalized_chapters)
                
                if is_finalized:
                    # Check if clip exists
                    safe_title = "".join(c for c in chapter_title if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_title = safe_title.replace(' ', '_')[:50]
                    clip_filename = f"chapter_{i:03d}_{safe_title}.mp4"
                    clip_path = os.path.join(clips_dir, clip_filename)
                    
                    if os.path.exists(clip_path):
                        playback_html = f'''<video width="200" height="120" controls style="border-radius: 4px;">
                            <source src="{clip_path}" type="video/mp4">
                        </video>'''
                    else:
                        playback_html = '<span style="color: #999; font-size: 12px;">⏳ Creating clip...</span>'
                else:
                    # Chapter not finalized yet
                    playback_html = '<span style="color: #ffc107; font-size: 12px;">⏳ In progress...</span>'
                
                # Build topics HTML
                topics_html = ""
                for j, topic in enumerate(topics, 1):
                    topic_start = topic.get('start_time', 0)
                    topic_end = topic.get('end_time', 0)
                    topic_duration = topic_end - topic_start
                    topic_summary = topic.get('topic_summary', 'No summary')
                    
                    # Format topic times
                    t_start_str = f"{int(topic_start // 60):02d}:{int(topic_start % 60):02d}"
                    t_end_str = f"{int(topic_end // 60):02d}:{int(topic_end % 60):02d}"
                    t_duration_str = f"{int(topic_duration // 60):02d}:{int(topic_duration % 60):02d}"
                    
                    # Check if topic clip exists
                    safe_summary = "".join(c for c in topic_summary if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_summary = safe_summary.replace(' ', '_')[:30]
                    topic_clip_filename = f"chapter_{i:03d}_topic_{j:02d}_{safe_summary}.mp4"
                    topic_clip_path = os.path.join(clips_dir, topic_clip_filename)
                    
                    if os.path.exists(topic_clip_path):
                        topic_playback_html = f'''<video width="180" height="100" controls style="border-radius: 4px; margin-top: 8px;">
                            <source src="{topic_clip_path}" type="video/mp4">
                        </video>'''
                    elif is_finalized:
                        topic_playback_html = '<div style="margin-top: 8px;"><span style="color: #999; font-size: 11px;">⏳ Creating clip...</span></div>'
                    else:
                        topic_playback_html = '<div style="margin-top: 8px;"><span style="color: #ffc107; font-size: 11px;">⏳ In progress...</span></div>'
                    
                    topics_html += f"""
                    <div style="margin: 8px 0; padding: 8px; background: #f8f9fa; border-left: 3px solid #28a745; border-radius: 4px;">
                        <div style="font-weight: bold; color: #495057; margin-bottom: 4px;">Topic {j}</div>
                        <div style="font-size: 12px; color: #6c757d; margin-bottom: 4px;">
                            <strong>Time:</strong> {t_start_str} → {t_end_str} (Duration: {t_duration_str})
                        </div>
                        <div style="font-size: 13px; color: #495057; margin-bottom: 4px;">{topic_summary}</div>
                        {topic_playback_html}
                    </div>
                    """
                
                # Build chapter row HTML with audio understanding styling
                chapters_html += f"""
                <div class="chapter-container">
                    <div class="chapter-header" onclick="toggleChapter({i})">
                        📚 Chapter {i}: {chapter_title} ({start_str} - {end_str}, {duration_str}) - {len(topics)} topics
                    </div>
                    <div id="chapter_topics_{i}" class="chapter-content">
                        <div style="margin-bottom: 15px;">
                            {playback_html}
                        </div>
                        <h4 style="margin: 0 0 12px 0; color: #495057;">Topics in this Chapter:</h4>
                        {topics_html}
                    </div>
                </div>
                """
            
            # Complete HTML with styling and JavaScript
            html_table = f"""
            <style>
            .chapter-container {{
                border: 2px solid #ddd;
                border-radius: 8px;
                margin: 10px 0;
                background: #f9f9f9;
            }}
            .chapter-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                cursor: pointer;
                border-radius: 6px 6px 0 0;
                font-weight: bold;
            }}
            .chapter-content {{
                padding: 20px;
                display: block;
            }}
            </style>
            <div style="margin: 20px 0;">
                <h3 style="color: #2E86AB; font-family: Arial, sans-serif;">
                    📚 Chapters ({len(self.all_chapters_for_display)} total)
                </h3>
                <div style="max-height: 600px; overflow-y: auto; border: 2px solid #2E86AB; border-radius: 8px; padding: 16px; background-color: #f8f9fa;">
                    {chapters_html}
                </div>
            </div>
            <script>
                function toggleChapter(chapterId) {{
                    var topics = document.getElementById('chapter_topics_' + chapterId);
                    if (topics.style.display === 'none' || topics.style.display === '') {{
                        topics.style.display = 'block';
                    }} else {{
                        topics.style.display = 'none';
                    }}
                }}
            </script>
            """
            
            # Return the HTML table
            return html_table
                
        except Exception as e:
            # Log errors for debugging
            log_component("FusionAnalyzer", f"❌ Error building chapter table HTML: {e}", "ERROR")
            return f"<div style='color: red;'>Error building table: {e}</div>"


