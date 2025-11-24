"""
Sentence Builder for Audio Understanding

This module provides the SentenceBuilder class for handling sentence construction
from Amazon Transcribe streaming results with proper comma handling and sentence
fragment combination.

Author: Audio Understanding Team
"""

class SentenceBuilder:
    """Handles sentence building and comma combination logic"""
    
    def __init__(self):
        self.pending_sentence = None
        
    def add_sentence_fragment(self, words, start_time, end_time, punctuation):
        """Add a sentence fragment and return complete sentence if ready"""
        if punctuation == ",":
            # Store comma-separated fragment for later combination
            if self.pending_sentence is None:
                self.pending_sentence = {
                    'words': words,
                    'start_time': start_time,
                    'punctuation': punctuation
                }
            else:
                # Extend existing pending sentence
                self.pending_sentence['words'].extend(words)
                self.pending_sentence['punctuation'] = punctuation
            return None  # Not ready yet
        else:
            # Complete sentence - combine with pending if exists
            if self.pending_sentence:
                # Combine with pending comma-separated part
                pending_comma = self.pending_sentence.get('punctuation', '')
                combined_text = (" ".join(self.pending_sentence['words']) + pending_comma + 
                               " " + " ".join(words) + punctuation)
                result = {
                    'text': combined_text,
                    'start_time': self.pending_sentence['start_time'],
                    'end_time': end_time
                }
                self.pending_sentence = None  # Clear pending
                return result
            else:
                # Regular complete sentence
                return {
                    'text': " ".join(words) + punctuation,
                    'start_time': start_time,
                    'end_time': end_time
                }
    
    def finalize_pending(self):
        """Force completion of any pending sentence"""
        if self.pending_sentence:
            result = {
                'text': " ".join(self.pending_sentence['words']) + self.pending_sentence.get('punctuation', ''),
                'start_time': self.pending_sentence['start_time'],
                'end_time': self.pending_sentence['start_time'] + 1.0
            }
            self.pending_sentence = None
            return result
        return None