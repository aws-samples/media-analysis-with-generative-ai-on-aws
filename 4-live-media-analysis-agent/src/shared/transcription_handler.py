"""
Transcription Handler - Processes Amazon Transcribe streaming results
Shared component used across Audio and Modality Fusion modules
"""

from datetime import datetime
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# Import component monitor
try:
    from .component_monitor import log_component
except ImportError:
    # Fallback if component_monitor not available
    def log_component(component, message, level="INFO"):
        print(f"[{component}] {message}")


class TranscriptionHandler(TranscriptResultStreamHandler):
    """
    Enhanced transcription handler for processing streaming results.
    
    This handler processes Amazon Transcribe streaming results and:
    - Buffers partial results with stable items
    - Detects sentence boundaries using punctuation
    - Creates complete sentences with precise timestamps
    - Integrates with TranscriptionProcessor for memory events
    
    Key Features:
    - Handles both partial and final results
    - Tracks stable items to avoid duplicates
    - Creates sentences on punctuation detection
    - Supports memory event creation via processor callback
    """
    
    def __init__(self, transcript_queue, transcript_result_stream, sentence_buffer=None, processor=None, transcript_file=None, sentence_log_level="INFO"):
        """
        Initialize transcription handler.
        
        Args:
            transcript_queue: Queue for transcript events (legacy, can be None)
            transcript_result_stream: Amazon Transcribe result stream
            sentence_buffer: Shared list for storing complete sentences
            processor: Reference to TranscriptionProcessor for memory events
            transcript_file: Optional file path to write transcripts to
            sentence_log_level: Log level for sentence logging ("INFO" or "DEBUG")
                               Use "INFO" for audio_understanding (show sentences)
                               Use "DEBUG" for modality_fusion (hide sentences at INFO level)
        """
        super().__init__(transcript_result_stream)
        self.transcript_queue = transcript_queue
        self.partial_buffer = {}
        self.last_processed_stable_key = None
        self.sentence_processed_keys = set()
        self.sentence_buffer = sentence_buffer if sentence_buffer is not None else []
        self.processor = processor  # Reference to TranscriptionProcessor for memory events
        self.transcript_file = transcript_file
        self.sentence_log_level = sentence_log_level  # Control sentence logging verbosity
        
        # Initialize transcript file if provided (JSON format)
        if self.transcript_file:
            import os
            import json
            os.makedirs(os.path.dirname(self.transcript_file), exist_ok=True)
            # Initialize with empty array
            with open(self.transcript_file, 'w') as f:
                json.dump([], f)
    
    def _create_sentence_from_buffer(self):
        """
        Create a complete sentence from buffered items.
        
        This method:
        1. Sorts buffered items by timestamp
        2. Combines pronunciation items into words
        3. Adds punctuation at the end
        4. Creates sentence with start/end timestamps
        5. Triggers memory event creation if processor available
        """
        if not self.partial_buffer:
            return
        
        sorted_items = sorted(self.partial_buffer.values(), key=lambda x: x.start_time)
        sentence_words = []
        sentence_start = None
        sentence_end = None
        punctuation = ""
        items_to_remove = []
        
        for item in sorted_items:
            item_key = f"{item.start_time}_{item.end_time}_{item.content}"
            
            if item.item_type == "pronunciation":
                sentence_words.append(item.content)
                if sentence_start is None:
                    sentence_start = item.start_time
                sentence_end = item.end_time
                items_to_remove.append(item_key)
            elif item.item_type == "punctuation":
                punctuation = item.content.strip()
                items_to_remove.append(item_key)
                break
        
        if sentence_words:
            sentence = " ".join(sentence_words) + punctuation
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            sentence_data = {
                'sentence': sentence,
                'start_time': sentence_start,
                'end_time': sentence_end,
                'timestamp': timestamp
            }
            
            self.sentence_buffer.append(sentence_data)
            
            log_component("Transcription", f"📝 Sentence: {sentence} ({sentence_start:.1f}s-{sentence_end:.1f}s)", self.sentence_log_level)
            
            # Write to transcript file if configured (JSON format)
            if self.transcript_file:
                try:
                    import json
                    # Read existing data
                    with open(self.transcript_file, 'r') as f:
                        sentences = json.load(f)
                    
                    # Append new sentence in the format expected by _get_transcript_for_timerange
                    sentences.append({
                        "start_time": sentence_start,
                        "end_time": sentence_end,
                        "sentence": sentence,
                        "timestamp": timestamp
                    })
                    
                    # Write back
                    with open(self.transcript_file, 'w') as f:
                        json.dump(sentences, f, indent=2)
                except Exception as e:
                    log_component("Transcription", f"⚠️ Failed to write to transcript file: {e}", "WARNING")
            
            # Trigger memory event creation if processor is available
            if self.processor:
                log_component("Transcription", f"🔍 _add_sentence_for_memory_event called (sentence: '{sentence_data.get('sentence', '')[:50]}...')", "DEBUG")
                self.processor._add_sentence_for_memory_event(sentence_data)
            else:
                # Debug: log if processor is not available
                if not hasattr(self, '_processor_warning_logged'):
                    log_component("Transcription", "⚠️ Processor reference not available in handler - memory events will not be created", "WARNING")
                    self._processor_warning_logged = True
            
            for key in items_to_remove:
                if key in self.partial_buffer:
                    del self.partial_buffer[key]
                    self.sentence_processed_keys.add(key)
    
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """
        Handle incoming transcript events from Amazon Transcribe.
        
        This processes both partial and final results:
        - Partial results: Process stable items incrementally
        - Final results: Process all remaining items
        
        Args:
            transcript_event: Transcript event from Amazon Transcribe
        """
        try:
            results = transcript_event.transcript.results
            
            for result in results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    
                    if result.is_partial:
                        if hasattr(alt, 'items') and alt.items:
                            found_last_processed = self.last_processed_stable_key is None
                            
                            for item in alt.items:
                                if hasattr(item, 'item_type'):
                                    item_key = f"{item.start_time}_{item.end_time}_{item.content}"
                                    
                                    if not found_last_processed:
                                        if item_key == self.last_processed_stable_key:
                                            found_last_processed = True
                                        continue
                                    
                                    if hasattr(item, 'stable') and item.stable:
                                        if item_key not in self.partial_buffer:
                                            self.partial_buffer[item_key] = item
                                            self.last_processed_stable_key = item_key
                                            
                                            if item.item_type == "punctuation":
                                                self._create_sentence_from_buffer()
                    else:
                        if hasattr(alt, 'items') and alt.items:
                            for item in alt.items:
                                if hasattr(item, 'item_type'):
                                    item_key = f"{item.start_time}_{item.end_time}_{item.content}"
                                    
                                    if item_key in self.sentence_processed_keys:
                                        continue
                                    
                                    if item_key not in self.partial_buffer:
                                        self.partial_buffer[item_key] = item
                                        
                                        if item.item_type == "punctuation":
                                            self._create_sentence_from_buffer()
                        
                        remaining_items = {k: v for k, v in self.partial_buffer.items() if k not in self.sentence_processed_keys}
                        self.partial_buffer = remaining_items
                        self.sentence_processed_keys.clear()
                        
                        if not remaining_items:
                            self.last_processed_stable_key = None
        except Exception as e:
            log_component("Transcription", f"❌ Transcript event error: {e}", "ERROR")


if __name__ == "__main__":
    print("=" * 80)
    print("Transcription Handler - Example Usage")
    print("=" * 80)
    
    print("\nExample: Processing Amazon Transcribe streaming results")
    print("-" * 80)
    print("""
from src.shared import TranscriptionHandler

# Create handler with sentence buffer
sentence_buffer = []
handler = TranscriptionHandler(
    transcript_queue=None,
    transcript_result_stream=stream.output_stream,
    sentence_buffer=sentence_buffer,
    processor=transcription_processor
)

# Handler processes events automatically
asyncio.create_task(handler.handle_events())

# Sentences appear in sentence_buffer as they're detected
for sentence_data in sentence_buffer:
    print(f"{sentence_data['sentence']} ({sentence_data['start_time']:.1f}s)")
    """)
    
    print("\n" + "=" * 80)
    print("✅ Transcription Handler module ready for use!")
    print("=" * 80)
