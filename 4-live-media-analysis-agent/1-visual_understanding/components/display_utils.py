"""
Display utilities for visual understanding analysis results
"""

import json
import tempfile
import subprocess
import base64
import os
from IPython.display import HTML, display


def create_video_clip(video_path, start_time, end_time, clip_id):
    """Create video clip and return base64 data for embedding"""
    try:
        duration = float(end_time) - float(start_time)
        temp_clip = tempfile.NamedTemporaryFile(suffix=f'_clip_{clip_id}.mp4', delete=False)
        clip_path = temp_clip.name
        temp_clip.close()
        
        # Extract clip using FFmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration),
            '-i', video_path, '-c:v', 'libx264', '-c:a', 'aac', 
            '-preset', 'fast', '-crf', '23', clip_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Read video file and encode as base64
            with open(clip_path, 'rb') as f:
                video_data = f.read()
            video_b64 = base64.b64encode(video_data).decode('utf-8')
            # Clean up temp file
            os.unlink(clip_path)
            return video_b64
        return None
    except:
        return None


def display_analysis_results(analysis, video_path=None):
    """
    Display video analysis results in a user-friendly collapsible format
    
    Args:
        analysis (str): JSON string containing video analysis results
        video_path (str): Path to the source video file for creating clips
    """
    try:
        analysis_data = json.loads(analysis)
        video_data = analysis_data['video_analysis']
        
        html = """
        <style>
        .section-container { border: 2px solid #ddd; border-radius: 8px; margin: 10px 0; background: #f9f9f9; }
        .section-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; cursor: pointer; border-radius: 6px 6px 0 0; font-weight: bold; }
        .section-content { padding: 20px; display: none; }
        .overview-section { background: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; }
        .text-section { background: #fff3e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800; }
        .visual-section { background: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50; }
        .chapter-section { background: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 4px solid #9c27b0; margin: 10px 0; }
        .safety-section { background: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336; }
        .movement-section { background: #f1f8e9; padding: 15px; border-radius: 8px; border-left: 4px solid #8bc34a; }
        .spatial-section { background: #fce4ec; padding: 15px; border-radius: 8px; border-left: 4px solid #e91e63; }
        .color-section { background: #fff8e1; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; }
        .video-clip { background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
        
        <script>
        function toggleSection(sectionId) {
            var content = document.getElementById('content-' + sectionId);
            content.style.display = content.style.display === 'none' ? 'block' : 'none';
        }
        </script>
        
        <h2>📹 Video Analysis Results</h2>
        """
        
        # Overview
        overview = video_data['overview']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('overview')">
                🎬 Overview - {overview['title']} - Click to expand
            </div>
            <div id="content-overview" class="section-content">
                <div class="overview-section">
                    <p><strong>Title:</strong> {overview['title']}</p>
                    <p><strong>Genre:</strong> {overview['genre']}</p>
                    <p><strong>Duration:</strong> {overview['duration_analyzed']}</p>
                    <p><strong>Frames:</strong> {overview['total_frames_analyzed']}</p>
                    <p><strong>Summary:</strong> {overview['summary']}</p>
                </div>
            </div>
        </div>
        """
        
        # Text Recognition
        text_data = video_data['text_recognition']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('text')">
                📝 Text Recognition ({len(text_data['details'])} items) - Click to expand
            </div>
            <div id="content-text" class="section-content">
                <div class="text-section">
                    <ul>
        """
        for detail in text_data['details']:
            html += f"<li>{detail}</li>"
        html += "</ul></div></div></div>"
        
        # Movement Dynamics
        movement = video_data['movement_dynamics']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('movement')">
                🏃 Movement & Dynamics ({len(movement['details'])} items) - Click to expand
            </div>
            <div id="content-movement" class="section-content">
                <div class="movement-section">
                    <ul>
        """
        for detail in movement['details']:
            html += f"<li>{detail}</li>"
        html += "</ul></div></div></div>"
        
        # Spatial Compositions
        spatial = video_data['spatial_compositions']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('spatial')">
                📐 Spatial Compositions ({len(spatial['details'])} items) - Click to expand
            </div>
            <div id="content-spatial" class="section-content">
                <div class="spatial-section">
                    <ul>
        """
        for detail in spatial['details']:
            html += f"<li>{detail}</li>"
        html += "</ul></div></div></div>"
        
        # Color & Visual Properties
        color = video_data['color_visual_properties']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('color')">
                🎨 Color & Visual Properties ({len(color['details'])} items) - Click to expand
            </div>
            <div id="content-color" class="section-content">
                <div class="color-section">
                    <ul>
        """
        for detail in color['details']:
            html += f"<li>{detail}</li>"
        html += "</ul></div></div></div>"
        
        # Visual Elements
        visual = video_data['visual_elements']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('visual')">
                👁️ Visual Elements - Click to expand
            </div>
            <div id="content-visual" class="section-content">
                <div class="visual-section">
                    <h4>👥 People Details:</h4>
                    <ul>
        """
        for detail in visual['people_details']:
            html += f"<li>{detail}</li>"
        
        html += "</ul><h4>🎯 Object Details:</h4><ul>"
        for detail in visual['object_details']:
            html += f"<li>{detail}</li>"
        
        html += "</ul><h4>🌍 Environment Details:</h4><ul>"
        for detail in visual['environment_details']:
            html += f"<li>{detail}</li>"
        html += "</ul></div></div></div>"
        
        # Content Moderation
        moderation = video_data['content_moderation']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('safety')">
                🛡️ Content Moderation ({len(moderation['details'])} items) - Click to expand
            </div>
            <div id="content-safety" class="section-content">
                <div class="safety-section">
                    <ul>
        """
        for detail in moderation['details']:
            html += f"<li>{detail}</li>"
        html += "</ul></div></div></div>"
        
        # Narrative Analysis
        narrative = video_data['narrative_analysis']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('narrative')">
                📖 Narrative Analysis ({len(narrative['details'])} items) - Click to expand
            </div>
            <div id="content-narrative" class="section-content">
                <div class="visual-section">
                    <ul>
        """
        for detail in narrative['details']:
            html += f"<li>{detail}</li>"
        html += "</ul></div></div></div>"
        
        # Chapters with video clips
        chapters = video_data['chapters']
        html += f"""
        <div class="section-container">
            <div class="section-header" onclick="toggleSection('chapters')">
                📚 Chapters ({len(chapters)} segments) - Click to expand
            </div>
            <div id="content-chapters" class="section-content">
        """
        
        for i, chapter in enumerate(chapters):
            duration = float(chapter['end_time']) - float(chapter['start_time'])
            html += f"""
            <div class="chapter-section">
                <h4>Chapter {chapter['chapter_number']}: {chapter['title']}</h4>
                <p><strong>Time:</strong> {chapter['start_time']}s - {chapter['end_time']}s ({duration:.1f}s)</p>
                <p><strong>Setting:</strong> {chapter['setting']}</p>
                <p><strong>Mood:</strong> {chapter['mood']}</p>
                <p><strong>Description:</strong> {chapter['description']}</p>
            """
            
            # Add video clip if video_path is provided
            if video_path and os.path.exists(video_path):
                video_b64 = create_video_clip(video_path, chapter['start_time'], chapter['end_time'], i+1)
                if video_b64:
                    html += f"""
                    <div class="video-clip">
                        <p><strong>🎬 Chapter Video Clip:</strong></p>
                        <video controls style="width: 100%; max-width: 600px;">
                            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                            Your browser does not support the video element.
                        </video>
                    </div>
                    """
                else:
                    html += "<p style='color: orange;'>⚠️ Could not generate video clip for this chapter</p>"
            
            if chapter.get('key_events'):
                html += "<p><strong>Key Events:</strong></p><ul>"
                for event in chapter['key_events']:
                    html += f"<li>{event}</li>"
                html += "</ul>"
            if chapter.get('characters_present'):
                html += f"<p><strong>Characters:</strong> {', '.join(chapter['characters_present'])}</p>"
            html += "</div>"
        
        html += "</div></div>"
        
        display(HTML(html))
        
    except json.JSONDecodeError:
        print('❌ Invalid JSON response')
        print(analysis)
    except Exception as e:
        print(f'❌ Error: {e}')
        print(analysis)
