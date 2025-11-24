"""
Transcript Processor for Audio Understanding

This module provides helper classes for processing Amazon Transcribe streaming results,
handling complex buffering logic and item processing operations.

Author: Audio Understanding Team
"""

from datetime import datetime


class TranscriptItemProcessor:
    """Handles complex transcript item processing and buffering operations"""
    
    def __init__(self, sentence_builder):
        self.sentence_builder = sentence_builder
        self.partial_buffer = {}
        self.last_processed_stable_key = None
        self.sentence_processed_keys = set()
    
    def process_partial_result_items(self, items):
        """Process partial result items with stability tracking"""
        found_last_processed = self.last_processed_stable_key is None
        
        for item in items:
            if not self._is_valid_item(item):
                continue
                
            item_key = self._create_item_key(item)
            
            # Skip items until we find our last processed stable item
            if not found_last_processed:
                if item_key == self.last_processed_stable_key:
                    found_last_processed = True
                continue
            
            # Process stable items
            if hasattr(item, 'stable') and item.stable:
                if item_key not in self.partial_buffer:
                    self.partial_buffer[item_key] = item
                    self.last_processed_stable_key = item_key
                    
                    # Trigger sentence creation on punctuation
                    if item.item_type == "punctuation":
                        return self._create_sentence_from_buffer()
        
        return None
    
    def process_final_result_items(self, items):
        """Process final result items"""
        sentences_created = []
        
        for item in items:
            if not self._is_valid_item(item):
                continue
                
            item_key = self._create_item_key(item)
            
            # Skip already processed items
            if item_key in self.sentence_processed_keys:
                continue
            
            # Add new items to buffer
            if item_key not in self.partial_buffer:
                self.partial_buffer[item_key] = item
                
                # Create sentence on punctuation
                if item.item_type == "punctuation":
                    sentence = self._create_sentence_from_buffer()
                    if sentence:
                        sentences_created.append(sentence)
        
        # Clean up processed items
        self._cleanup_processed_items()
        
        return sentences_created
    
    def _create_sentence_from_buffer(self):
        """Build complete sentences from buffered word items with comma handling"""
        if not self.partial_buffer:
            return None
        
        sorted_items = sorted(self.partial_buffer.values(), key=lambda x: x.start_time)
        sentence_words = []
        sentence_start = None
        sentence_end = None
        punctuation = ""
        items_to_remove = []
        
        for item in sorted_items:
            item_key = self._create_item_key(item)
            
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
            # Use SentenceBuilder for comma handling
            sentence_data = self.sentence_builder.add_sentence_fragment(
                sentence_words, sentence_start, sentence_end, punctuation
            )
            
            # Clean up processed items
            for key in items_to_remove:
                if key in self.partial_buffer:
                    del self.partial_buffer[key]
                    self.sentence_processed_keys.add(key)
            
            return sentence_data
        
        return None
    
    def _cleanup_processed_items(self):
        """Clean up processed items from buffers"""
        remaining_items = {
            k: v for k, v in self.partial_buffer.items() 
            if k not in self.sentence_processed_keys
        }
        self.partial_buffer = remaining_items
        self.sentence_processed_keys.clear()
        
        if not remaining_items:
            self.last_processed_stable_key = None
    
    def _is_valid_item(self, item):
        """Check if transcript item is valid for processing"""
        if not item or not hasattr(item, 'item_type'):
            return False
        
        required_attrs = ['start_time', 'end_time', 'content']
        return all(hasattr(item, attr) for attr in required_attrs)
    
    def _create_item_key(self, item):
        """Create unique key for transcript item"""
        return f"{item.start_time}_{item.end_time}_{item.content}"
    
    def finalize_pending_sentences(self):
        """Force completion of any pending sentences"""
        return self.sentence_builder.finalize_pending()


class SentenceFormatter:
    """Handles sentence formatting and output operations"""
    
    @staticmethod
    def format_sentence_output(sentence_data):
        """Format sentence data for display"""
        if not sentence_data:
            return None
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"📝 SENTENCE: {sentence_data['text']}")
        print(f"⏱️ Time: {sentence_data['start_time']:.3f}s-{sentence_data['end_time']:.3f}s")
        
        return {
            'text': sentence_data['text'],
            'start_time': sentence_data['start_time'],
            'end_time': sentence_data['end_time'],
            'timestamp': timestamp
        }
    
    @staticmethod
    def format_final_sentence_output(sentence_data):
        """Format final sentence data for display"""
        if not sentence_data:
            return None
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"📝 FINAL SENTENCE: {sentence_data['text']}")
        print(f"⏱️ Time: {sentence_data['start_time']:.3f}s")
        
        return {
            'text': sentence_data['text'],
            'start_time': sentence_data['start_time'],
            'end_time': sentence_data['end_time'],
            'timestamp': timestamp
        }


class TranscriptEventValidator:
    """Handles transcript event validation operations"""
    
    @staticmethod
    def validate_transcript_event(transcript_event):
        """Validate transcript event structure"""
        if not transcript_event or not hasattr(transcript_event, 'transcript'):
            return False
        
        if not transcript_event.transcript or not hasattr(transcript_event.transcript, 'results'):
            return False
        
        return bool(transcript_event.transcript.results)
    
    @staticmethod
    def validate_result(result):
        """Validate individual result structure"""
        if not result or not hasattr(result, 'alternatives') or not result.alternatives:
            return False
        
        alt = result.alternatives[0]
        return alt is not None
    
    @staticmethod
    def get_result_items(result):
        """Safely extract items from result"""
        if not result or not hasattr(result, 'alternatives') or not result.alternatives:
            return []
        
        alt = result.alternatives[0]
        if not alt or not hasattr(alt, 'items'):
            return []
        
        return alt.items or []