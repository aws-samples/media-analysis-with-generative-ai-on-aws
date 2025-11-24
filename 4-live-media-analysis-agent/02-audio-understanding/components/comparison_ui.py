"""
Comparison UI utilities for audio understanding analysis
"""

import tempfile
import subprocess
import base64
from IPython.display import HTML, display


class ComparisonUIBuilder:
    """Builds collapsible comparison UI for basic vs enhanced analysis"""
    
    def __init__(self, video_path):
        self.video_path = video_path
    
    def create_audio_clip_data(self, start_time, end_time, clip_id):
        """Create audio clip and return base64 data for embedding"""
        try:
            clip_duration = end_time - start_time
            temp_clip = tempfile.NamedTemporaryFile(suffix=f'_clip_{clip_id}.wav', delete=False)
            clip_path = temp_clip.name
            temp_clip.close()
            
            # Extract clip using FFmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-ss', str(start_time), '-t', str(clip_duration),
                '-i', self.video_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', clip_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Read audio file and encode as base64
                with open(clip_path, 'rb') as f:
                    audio_data = f.read()
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                return audio_b64
            return None
        except:
            return None
    
    def get_css_styles(self):
        """Return CSS styles for the collapsible interface"""
        return """
        <style>
        .chapter-container {
            border: 2px solid #ddd;
            border-radius: 8px;
            margin: 10px 0;
            background: #f9f9f9;
        }
        .chapter-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            cursor: pointer;
            border-radius: 6px 6px 0 0;
            font-weight: bold;
        }
        .chapter-content {
            padding: 20px;
            display: block;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
        }
        .basic-analysis {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #999;
        }
        .enhanced-analysis {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }
        .audio-section {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #ff9800;
        }
        .embedded-audio {
            margin: 10px 0;
        }
        </style>
        """
    
    def get_javascript(self):
        """Return JavaScript for toggle functionality"""
        return """
        <script>
        function toggleChapter(chapterId) {
            var content = document.getElementById('content-' + chapterId);
            if (content.style.display === 'none' || content.style.display === '') {
                content.style.display = 'block';
            } else {
                content.style.display = 'none';
            }
        }
        </script>
        """
    
    def create_audio_html(self, audio_b64):
        """Create HTML for embedded audio player"""
        if audio_b64:
            return f"""
            <div class="embedded-audio">
                <audio controls style="width: 100%; max-width: 400px;">
                    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
        else:
            return "<p style='color: red;'>❌ Audio clip could not be generated</p>"
    
    def create_chapter_html(self, chapter_num, basic_ch, enhanced_ch, start_time, end_time, audio_html):
        """Create HTML for a single chapter comparison"""
        return f"""
        <div class="chapter-container">
            <div class="chapter-header" onclick="toggleChapter({chapter_num})">
                📖 Chapter {chapter_num} ({start_time:.1f}s - {end_time:.1f}s) - Click to expand comparison
            </div>
            <div id="content-{chapter_num}" class="chapter-content">
                <div class="comparison-grid">
                    <div class="basic-analysis">
                        <h4>📝 Basic Analysis (Transcript Only)</h4>
                        <p><strong>Title:</strong> {basic_ch.get('title', 'N/A')}</p>
                        <p><strong>Summary:</strong> {basic_ch.get('summary', 'No summary available')}</p>
                        <p><strong>Topics:</strong> {len(basic_ch.get('topics', []))} topics detected</p>
                    </div>
                    <div class="enhanced-analysis">
                        <h4>🎵 Enhanced Analysis (Transcript + Spectrogram)</h4>
                        <p><strong>Title:</strong> {enhanced_ch.get('title', 'N/A')}</p>
                        <p><strong>Summary:</strong> {enhanced_ch.get('summary', 'No summary available')}</p>
                        <p><strong>Audio Tone:</strong> {enhanced_ch.get('audio_tone', 'Not detected')}</p>
                        <p><strong>Topics:</strong> {len(enhanced_ch.get('topics', []))} topics detected</p>
                    </div>
                </div>
                <div class="audio-section">
                    <h4>🎧 Audio Clip for This Chapter</h4>
                    <p><strong>Duration:</strong> {end_time - start_time:.1f} seconds</p>
                    {audio_html}
                </div>
            </div>
        </div>
        """
    
    def build_comparison_ui(self, basic_analysis, enhanced_analysis):
        """Build complete collapsible comparison UI"""
        if not (basic_analysis and enhanced_analysis):
            return None
        
        basic_chapters = basic_analysis.get('chapters', [])
        enhanced_chapters = enhanced_analysis.get('chapters', [])
        max_chapters = max(len(basic_chapters), len(enhanced_chapters))
        
        # Pre-generate all audio clips
        audio_data = {}
        for i in range(max_chapters):
            basic_ch = basic_chapters[i] if i < len(basic_chapters) else {}
            enhanced_ch = enhanced_chapters[i] if i < len(enhanced_chapters) else {}
            
            start_time = enhanced_ch.get('start_time', basic_ch.get('start_time', 0))
            end_time = enhanced_ch.get('end_time', basic_ch.get('end_time', 30))
            
            audio_b64 = self.create_audio_clip_data(start_time, end_time, i+1)
            if audio_b64:
                audio_data[i+1] = audio_b64
        
        # Build HTML content
        html_content = self.get_css_styles() + self.get_javascript()
        html_content += "<h2>📊 Chapter-by-Chapter Comparison: Basic vs Enhanced Analysis</h2>"
        
        for i in range(max_chapters):
            basic_ch = basic_chapters[i] if i < len(basic_chapters) else {}
            enhanced_ch = enhanced_chapters[i] if i < len(enhanced_chapters) else {}
            
            start_time = enhanced_ch.get('start_time', basic_ch.get('start_time', 0))
            end_time = enhanced_ch.get('end_time', basic_ch.get('end_time', 30))
            
            audio_html = self.create_audio_html(audio_data.get(i+1))
            chapter_html = self.create_chapter_html(i+1, basic_ch, enhanced_ch, start_time, end_time, audio_html)
            html_content += chapter_html
        
        return html_content
    
    def display_comparison(self, basic_analysis, enhanced_analysis):
        """Display the complete comparison UI"""
        html_content = self.build_comparison_ui(basic_analysis, enhanced_analysis)
        if html_content:
            display(HTML(html_content))
        else:
            print("❌ No comparison data available")
