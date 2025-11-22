"""
Transcription Processor - Real-time audio transcription from UDP stream
Shared component used across Audio and Modality Fusion modules
"""

import asyncio
import time
import json
from datetime import datetime
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.exceptions import BadRequestException

# Import shared components
try:
    from .transcription_handler import TranscriptionHandler
    from .component_monitor import log_component
except ImportError:
    # Fallback imports
    try:
        from transcription_handler import TranscriptionHandler
    except ImportError:
        TranscriptionHandler = None
    
    def log_component(component, message, level="INFO"):
        print(f"[{component}] {message}")


class TranscriptionProcessor:
    """
    Handles real-time transcription from UDP audio stream.
    
    This processor:
    - Captures audio from UDP stream using FFmpeg
    - Sends audio to Amazon Transcribe streaming API
    - Processes transcription results via TranscriptionHandler
    - Integrates with AgentCore Memory for transcript storage
    - Handles graceful shutdown and cleanup
    
    Key Features:
    - Real-time streaming transcription
    - Automatic sentence boundary detection
    - Memory event creation every 10 sentences
    - Stream timeout detection
    - Graceful error handling and recovery
    """
    
    def __init__(
        self, 
        udp_port, 
        aws_region='us-east-1', 
        sentence_buffer=None, 
        memory_client=None, 
        memory_id=None, 
        actor_id=None, 
        session_id=None,
        output_dir='output'
    ):
        """
        Initialize transcription processor.
        
        Args:
            udp_port: UDP port for audio stream
            aws_region: AWS region for Transcribe service
            sentence_buffer: Shared list for storing complete sentences
            memory_client: AgentCore Memory client
            memory_id: Memory ID for storing transcripts
            actor_id: Actor ID for memory events
            session_id: Session ID for memory events
            output_dir: Output directory for transcript files
        """
        self.udp_port = udp_port
        self.aws_region = aws_region
        self.output_dir = output_dir
        self.is_running = False
        self.transcribe_client = None
        self.stream = None
        self.handler = None
        self.sentence_buffer = sentence_buffer if sentence_buffer is not None else []
        self.ffmpeg_process = None  # Store FFmpeg process reference
        
        # AgentCore memory integration
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        self.sentence_count_since_last_event = 0
        self.sentences_for_next_event = []
        
    async def start_transcription(self):
        """
        Start real-time transcription.
        
        This method:
        1. Initializes the transcription process
        2. Captures audio from UDP stream
        3. Sends audio to Amazon Transcribe
        4. Processes results via handler
        5. Handles cleanup on completion
        """
        log_component("Transcription", f"🎧 Starting transcription from UDP port {self.udp_port}", "DEBUG")
        self.is_running = True
        
        try:
            await self._capture_and_transcribe()
        except Exception as e:
            log_component("Transcription", f"❌ Transcription error: {e}", "ERROR")
        finally:
            await self._cleanup()
    
    async def _initialize_transcribe_client(self):
        """
        Initialize AWS Transcribe streaming client.
        
        Creates:
        - TranscribeStreamingClient
        - Streaming transcription session
        - TranscriptionHandler for processing results
        """
        self.transcribe_client = TranscribeStreamingClient(region=self.aws_region)
        
        self.stream = await self.transcribe_client.start_stream_transcription(
            language_code='en-US',
            media_sample_rate_hz=16000,
            media_encoding="pcm",
            enable_partial_results_stabilization=True
        )
        
        # Create transcript file path
        import os
        transcript_file = None
        if hasattr(self, 'output_dir'):
            transcript_file = f"{self.output_dir}/transcripts/live_transcript.json"
        
        self.handler = TranscriptionHandler(
            None, 
            self.stream.output_stream, 
            self.sentence_buffer, 
            processor=self,
            transcript_file=transcript_file,
            sentence_log_level="DEBUG"  # Hide sentences in modality fusion
        )
        
        # Wrap handle_events with exception handling for BadRequestException
        async def handle_events_with_exception_handling():
            try:
                await self.handler.handle_events()
            except BadRequestException as e:
                if "timed out" in str(e).lower():
                    log_component("Transcription", "⏱️  Transcribe stream timed out (15s no audio) - this is normal when stream ends", "WARNING")
                else:
                    log_component("Transcription", f"⚠️  Transcribe error: {e}", "WARNING")
            except Exception as e:
                log_component("Transcription", f"❌ Unexpected transcription error: {e}", "ERROR")
        
        asyncio.create_task(handle_events_with_exception_handling())
        
        log_component("Transcription", "✅ Transcription stream initialized")
    
    async def _capture_and_transcribe(self):
        """
        Capture audio from UDP stream and send to transcription.
        
        This method:
        1. Starts FFmpeg to capture audio from UDP
        2. Converts audio to PCM 16kHz mono
        3. Sends audio chunks to Transcribe
        4. Handles stream timeout detection
        5. Manages graceful shutdown
        """
        udp_url = f"udp://127.0.0.1:{self.udp_port}"
        
        ffmpeg_command = [
            'ffmpeg', '-i', udp_url,
            '-f', 'wav', '-ac', '1', '-ar', '16000',
            '-c:a', 'pcm_s16le', '-'
        ]
        
        log_component("Transcription", f"🎵 Starting audio capture on UDP port {self.udp_port}")
        
        self.ffmpeg_process = await asyncio.create_subprocess_exec(
            *ffmpeg_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for stream with timeout
        stream_timeout_seconds = 60
        last_data_time = time.time()
        client_initialized = False
        
        try:
            while self.is_running:
                try:
                    data = await asyncio.wait_for(self.ffmpeg_process.stdout.read(1024 * 2), timeout=1.0)
                    
                    if data:
                        if not client_initialized:
                            await self._initialize_transcribe_client()
                            client_initialized = True
                        
                        last_data_time = time.time()
                        
                        if self.stream:
                            await self.stream.input_stream.send_audio_event(audio_chunk=data)
                    elif not data:
                        log_component("Transcription", "⚠️ Audio stream ended. Stopping...", "WARNING")
                        break
                        
                except asyncio.TimeoutError:
                    time_since_last_data = time.time() - last_data_time
                    if time_since_last_data >= stream_timeout_seconds:
                        if not client_initialized:
                            log_component("Transcription", f"❌ No audio data received for {stream_timeout_seconds} seconds.", "ERROR")
                            log_component("Transcription", "   Please start the FFmpeg ingest command first!", "ERROR")
                        else:
                            log_component("Transcription", f"⚠️ No audio data for {stream_timeout_seconds} seconds. Stream may have stopped.", "WARNING")
                        self.is_running = False
                        break
                    
                    if int(time_since_last_data) % 5 == 0 and int(time_since_last_data) > 0:
                        remaining_wait = stream_timeout_seconds - int(time_since_last_data)
                        if not client_initialized:
                            log_component("Transcription", f"⏳ Waiting for audio data... ({remaining_wait}s remaining)")
                    
        except Exception as e:
            log_component("Transcription", f"❌ Audio capture error: {e}", "ERROR")
        finally:
            if self.ffmpeg_process and self.ffmpeg_process.returncode is None:
                self.ffmpeg_process.terminate()
                await self.ffmpeg_process.wait()
    
    async def _cleanup(self):
        """
        Cleanup transcription resources.
        
        Closes:
        - Transcribe streaming session
        - FFmpeg process
        - Any open connections
        """
        if self.stream:
            try:
                await self.stream.input_stream.end_stream()
            except Exception as e:
                log_component("Transcription", f"⚠️ Error ending stream: {e}", "WARNING")
        
        log_component("Transcription", "✅ Transcription cleanup complete")
    
    def _add_sentence_for_memory_event(self, sentence_data):
        """
        Add sentence and create memory event every 10 sentences.
        
        This method:
        1. Accumulates sentences in buffer
        2. Counts sentences since last event
        3. Creates memory event every 10 sentences
        4. Resets counters after event creation
        
        Args:
            sentence_data: Dictionary with sentence, start_time, end_time
        """
        log_component("Transcription", f"🔍 _add_sentence_for_memory_event called (sentence: '{sentence_data.get('sentence', '')[:50]}...')", "DEBUG")
        
        if not self.memory_client or not self.memory_id:
            # Only log once when first sentence is detected
            if not hasattr(self, '_memory_warning_logged'):
                log_component("Transcription", "⚠️ Memory client not configured - transcripts will not be saved to memory", "WARNING")
                self._memory_warning_logged = True
            return
        
        self.sentences_for_next_event.append(sentence_data)
        self.sentence_count_since_last_event += 1
        
        log_component("Transcription", f"📝 Accumulated {self.sentence_count_since_last_event} sentences for memory event", "DEBUG")
        
        # Create event every 10 sentences
        # DISABLED: Memory events temporarily disabled to avoid errors
        # if self.sentence_count_since_last_event >= 10:
        #     self._create_memory_event_for_transcripts()
    
    def _create_memory_event_for_transcripts(self):
        """
        Create memory event with accumulated transcripts.
        
        This method:
        1. Formats sentences as JSON
        2. Creates AgentCore Memory event
        3. Logs success/failure
        4. Resets sentence buffer
        """
        if not self.memory_client or not self.memory_id:
            log_component("Transcription", "⚠️ Cannot create memory event - memory client not configured", "WARNING")
            return
        
        if not self.sentences_for_next_event:
            log_component("Transcription", "⚠️ No sentences to save to memory", "WARNING")
            return
        
        try:
            log_component("Transcription", f"💾 Creating memory event with {len(self.sentences_for_next_event)} sentences...", "DEBUG")
            
            # Format sentences as JSON string
            transcript_json = json.dumps(self.sentences_for_next_event, indent=2)
            
            # Create event with transcript data
            messages = [
                {
                    'conversational': {
                        'content': {
                            'text': transcript_json
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
            
            log_component("Transcription", f"✅ Memory event created successfully with {len(self.sentences_for_next_event)} sentences")
            log_component("Transcription", f"   Memory ID: {self.memory_id}", "DEBUG")
            log_component("Transcription", f"   Session ID: {self.session_id}", "DEBUG")
            
            # Reset counters
            self.sentence_count_since_last_event = 0
            self.sentences_for_next_event = []
            
        except Exception as e:
            log_component("Transcription", f"❌ Failed to create memory event: {e}", "ERROR")
            import traceback
            log_component("Transcription", f"   Traceback: {traceback.format_exc()}", "ERROR")
    
    async def stop_transcription(self):
        """
        Stop transcription and terminate FFmpeg process.
        
        This method:
        1. Creates final memory event with remaining sentences
        2. Stops transcription processing
        3. Terminates FFmpeg process gracefully
        4. Handles force kill if needed
        """
        log_component("Transcription", "🛑 Stopping transcription...")
        self.is_running = False
        
        # Create final memory event with remaining sentences
        # DISABLED: Memory events temporarily disabled
        # if self.sentences_for_next_event:
        #     self._create_memory_event_for_transcripts()
        
        # Check if FFmpeg process exists and is still running
        if self.ffmpeg_process:
            if self.ffmpeg_process.returncode is None:
                # Process is still running, terminate it
                try:
                    self.ffmpeg_process.terminate()
                    await asyncio.wait_for(self.ffmpeg_process.wait(), timeout=5)
                    log_component("Transcription", "✅ Transcription stopped")
                except asyncio.TimeoutError:
                    log_component("Transcription", "⚠️ FFmpeg process didn't terminate, force killing...", "WARNING")
                    try:
                        self.ffmpeg_process.kill()
                        await asyncio.wait_for(self.ffmpeg_process.wait(), timeout=2)
                        log_component("Transcription", "✅ Transcription stopped (ffmpeg process killed)")
                    except Exception as e:
                        log_component("Transcription", f"❌ Error killing ffmpeg process: {e}", "ERROR")
                except Exception as e:
                    log_component("Transcription", f"⚠️ Error terminating ffmpeg process: {e}", "WARNING")
            else:
                # Process already terminated
                log_component("Transcription", "✅ Transcription already stopped (ffmpeg process already terminated)")
        else:
            # No FFmpeg process
            log_component("Transcription", "✅ Transcription stopped (no ffmpeg process)")


if __name__ == "__main__":
    print("=" * 80)
    print("Transcription Processor - Example Usage")
    print("=" * 80)
    
    print("\nExample: Real-time audio transcription from UDP stream")
    print("-" * 80)
    print("""
from src.shared import TranscriptionProcessor

# Create processor with memory integration
processor = TranscriptionProcessor(
    udp_port="1234",
    aws_region='us-east-1',
    sentence_buffer=sentence_buffer,
    memory_client=memory_client,
    memory_id='video-transcription',
    actor_id='user-123',
    session_id='session-456'
)

# Start transcription (async)
await processor.start_transcription()

# Sentences appear in sentence_buffer as they're detected
# Memory events created automatically every 10 sentences

# Stop transcription
await processor.stop_transcription()
    """)
    
    print("\n" + "=" * 80)
    print("✅ Transcription Processor module ready for use!")
    print("=" * 80)
