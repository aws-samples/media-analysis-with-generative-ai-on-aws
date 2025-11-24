import glob
import json
import os
import time
from IPython.display import HTML, display, clear_output

def create_scene_results_table(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS):
    """Create comprehensive table with scene videos, filmstrips, and analysis"""
    
    # Get all scene files
    scene_clips = sorted(glob.glob(f"{OUTPUT_DIR}/clips/scene_*.mp4"))
    scene_filmstrips = sorted(glob.glob(f"{OUTPUT_DIR}/scenes/scene_*.jpg"))
    
    if not scene_clips and not scene_filmstrips:
        print("❌ No scene files found for table display")
        return
    
    print(f"📊 Creating scene results table with {len(scene_clips)} clips and {len(scene_filmstrips)} filmstrips")
    
    # Create HTML table (header will be added after we know scene count)
    html_content = """
    <script>
    function showPopup(popupId) {
        document.getElementById(popupId).style.display = 'block';
    }
    function hidePopup(popupId) {
        document.getElementById(popupId).style.display = 'none';
    }
    function toggleAnalysis(analysisId) {
        var compact = document.getElementById('compact_' + analysisId);
        var full = document.getElementById('full_' + analysisId);
        
        if (compact.style.display === 'none') {
            compact.style.display = 'block';
            full.style.display = 'none';
        } else {
            compact.style.display = 'none';
            full.style.display = 'block';
        }
    }
    function showFilmstripZoom(imgElement) {
        var filmstripPath = imgElement.src;
        var sceneInfo = imgElement.nextElementSibling.textContent.trim();
        
        var overlay = document.createElement('div');
        overlay.id = 'filmstrip-zoom-overlay';
        overlay.className = 'filmstrip-zoom-overlay';
        overlay.onclick = function() { hideFilmstripZoom(); };
        
        overlay.innerHTML = '<span class="filmstrip-zoom-close" onclick="hideFilmstripZoom()">&times;</span><div class="filmstrip-zoom-content" onclick="event.stopPropagation()"><img src="' + filmstripPath + '" class="filmstrip-zoom-img"><div class="filmstrip-info">' + sceneInfo + ' - Click outside to close</div></div>';
        
        document.body.appendChild(overlay);
        overlay.style.display = 'block';
    }
    function hideFilmstripZoom() {
        var overlay = document.getElementById('filmstrip-zoom-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
    </script>
        <style>
    .scene-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: Arial, sans-serif;
        font-size: 14px;
        table-layout: fixed;
    }
    .scene-table th {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
        text-align: left;
        padding: 12px;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    .scene-table th, .scene-table td {
        border-bottom: 1px solid #ddd;
        padding: 10px;
        vertical-align: top;
        text-align: left;
    }
    .scene-table tr:hover {
        background-color: #e3f2fd;
    }
    .scene-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .scene-number {
        text-align: center;
        font-weight: bold;
        font-size: 14px;
        width: 40px;
        padding: 10px;
        vertical-align: middle;
    }
    .scene-table th:nth-child(1) {
        width: 40px;
        text-align: center;
    }
    .scene-table td:nth-child(1) {
        text-align: center;
    }
    .scene-video {
        text-align: center;
        width: 360px;
        padding: 8px;
    }
    .scene-video video {
        width: 100%;
        max-width: 340px;
        height: auto;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .scene-filmstrip {
        text-align: center;
        width: 230px;
        padding: 5px;
    }
    .scene-summary {
        width: 560px;
        font-size: 12px;
        position: relative;
        padding: 8px;
    }
    .expand-btn {
        background: #28a745;
        color: white;
        border: none;
        padding: 3px 6px;
        border-radius: 3px;
        cursor: pointer;
        font-size: 10px;
        margin-top: 5px;
        width: 100%;
    }
    .expand-btn:hover {
        background: #218838;
    }
    .filmstrip-zoom-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.9);
        z-index: 2000;
        cursor: pointer;
    }
    .filmstrip-zoom-content {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        max-width: 95vw;
        max-height: 95vh;
        cursor: default;
    }
    .filmstrip-zoom-img {
        max-width: 100%;
        max-height: 95vh;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .filmstrip-zoom-close {
        position: absolute;
        top: 20px;
        right: 30px;
        color: white;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        z-index: 2001;
    }
    .filmstrip-zoom-close:hover {
        color: #ccc;
    }
    .filmstrip-info {
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        background: rgba(0,0,0,0.7);
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 14px;
    }
    </style>
    
    <table class="scene-table">
        <thead>
            <tr>
                <th>Scene #</th>
                <th>Video Clip</th>
                <th>Scene Filmstrip</th>
                <th>Bedrock Analysis Summary</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Get scene numbers from files
    scene_numbers = set()
    for clip in scene_clips:
        scene_num = int(os.path.basename(clip).split('_')[1])
        scene_numbers.add(scene_num)
    for filmstrip in scene_filmstrips:
        scene_num = int(os.path.basename(filmstrip).split('_')[1])
        scene_numbers.add(scene_num)
    
    # Add header with scene count
    header_html = f"""
    <div style="margin: 20px 0;">
        <h3 style="color: #2E86AB; font-family: Arial, sans-serif;">
            🎬 Scene Analysis Results ({len(scene_numbers)} scenes)
        </h3>
        <div style="max-height: 600px; overflow-y: auto; border: 2px solid #2E86AB; border-radius: 8px; padding: 10px; background-color: #f8f9fa;">
    """
    html_content = header_html + html_content
    
    # Create table rows for each scene
    for scene_num in sorted(scene_numbers):
        # Find corresponding files
        clip_file = None
        filmstrip_file = None
        
        for clip in scene_clips:
            if f"scene_{scene_num:04d}_" in clip:
                clip_file = clip
                break
        
        for filmstrip in scene_filmstrips:
            if f"scene_{scene_num:04d}_" in filmstrip:
                filmstrip_file = filmstrip
                break
        
        # Create table row
        html_content += f"""
            <tr>
                <td class="scene-number">{scene_num + 1}</td>
                <td class="scene-video">
        """
        
        # Video clip cell
        if clip_file and os.path.exists(clip_file):
            file_size = os.path.getsize(clip_file) / (1024*1024)
            # Extract duration from filename
            import re
            duration_match = re.search(r'_(\d+\.?\d*)s-(\d+\.?\d*)s\.mp4', clip_file)
            if duration_match:
                start_time = float(duration_match.group(1))
                end_time = float(duration_match.group(2))
                clip_duration = end_time - start_time
            else:
                clip_duration = 0.0
            html_content += f"""
                    <video width="280" height="200" controls preload="metadata">
                        <source src="{clip_file}" type="video/mp4">
                        Your browser does not support video.
                    </video>
                    <div style="font-size: 10px; margin-top: 5px;">
                        {os.path.basename(clip_file)}<br>
                        Duration: {clip_duration:.1f}s<br>Size: {file_size:.1f} MB
                    </div>
            """
        else:
            html_content += "<div style='color: #999;'>No video clip</div>"
        
        html_content += "</td><td class='scene-filmstrip'>"
        
        # Filmstrip cell
        if filmstrip_file and os.path.exists(filmstrip_file):
            html_content += f"""
                    <img src="{filmstrip_file}" width="180" style="border: 1px solid #ccc; cursor: pointer;" onclick="showFilmstripZoom(this)" title="Click to zoom">
                    <div style="font-size: 10px; margin-top: 5px;">
                        {os.path.basename(filmstrip_file)}
                    </div>
            """
        else:
            html_content += "<div style='color: #999;'>No filmstrip</div>"
        
        html_content += "</td><td class='scene-summary'>"
        
        # Analysis summary cell
        summary_content = get_scene_analysis_summary_fixed(scene_num, SCENE_ANALYSIS_RESULTS)
        html_content += summary_content
        
        html_content += "</td></tr>"
    
    html_content += f"""
        </tbody>
    </table>
    
        </div>
    </div>
    """
    
    return html_content

def get_scene_analysis_summary_fixed(scene_num, SCENE_ANALYSIS_RESULTS):
    """Get analysis summary from global storage"""
    
    if scene_num in SCENE_ANALYSIS_RESULTS:
        data = SCENE_ANALYSIS_RESULTS[scene_num]
        analysis = data['analysis']
        analysis_time = data['analysis_time']
        
        formatted = format_analysis_summary(analysis)
        formatted += f"""
            <div class="summary-item" style="margin-top: 8px; padding-top: 5px; border-top: 1px solid #eee;">
                <span class="summary-label">Analysis Time:</span> {analysis_time:.2f}s
            </div>
            <div class="summary-item">
                <span class="summary-label">Frames:</span> {data['start_frame']}-{data['end_frame']}
            </div>
        """
        return formatted
    
    return """
        <div class="summary-item">
            <span class="summary-label">Status:</span> Analysis pending
        </div>
        <div class="summary-item">
            <span class="summary-label">Note:</span> Run live processing with Bedrock analysis
        </div>
    """

def format_analysis_summary(analysis):
    """Format Bedrock analysis into compact HTML summary with expand option"""
    if 'error' in analysis:
        return f"""
            <div class="summary-item">
                <span class="summary-label">Error:</span> {analysis['error']}
            </div>
        """
    
    # Create compact summary (first 3 items)
    compact_html = ""
    full_html = ""
    
    items = [
        ('Scene Type', analysis.get('scene_type', '')),
        ('Setting', analysis.get('setting', '')),
        ('Objects', ', '.join(analysis.get('objects', [])[:3])),
        ('People', ', '.join(analysis.get('people', [])[:2])),
        ('Mood', analysis.get('mood', '')),
        ('Summary', analysis.get('summary', ''))
    ]
    
    # Compact version - first 3 non-empty items
    compact_count = 0
    for label, value in items:
        if value and compact_count < 3:
            compact_html += f"""
                <div class="summary-item">
                    <span class="summary-label">{label}:</span> {value[:50]}{'...' if len(str(value)) > 50 else ''}
                </div>
            """
            compact_count += 1
    
    # Full version - all items
    for label, value in items:
        if value:
            full_html += f"""
                <div class="summary-item">
                    <span class="summary-label">{label}:</span> {value}
                </div>
            """
    
    # Create unique ID for this analysis
    analysis_id = f"analysis_{hash(str(analysis)) % 10000}"
    
    # Return compact view with expand/collapse functionality
    return f"""
        <div id="compact_{analysis_id}" class="compact-analysis">
            {compact_html}
            <button class="expand-btn" onclick="toggleAnalysis('{analysis_id}')">▼ Show Full Analysis</button>
        </div>
        <div id="full_{analysis_id}" class="full-analysis" style="display: none;">
            {full_html}
            <button class="expand-btn" onclick="toggleAnalysis('{analysis_id}')">▲ Show Less</button>
        </div>
    """

def refresh_table_display(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS, table_output=None):
    """Refresh the table display with latest data using dedicated output widget"""
    try:
        if table_output:
            with table_output:
                clear_output(wait=True)
                table_html = create_scene_results_table(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS)
                if table_html:
                    display(HTML(table_html))
                    print(f"📊 Table updated at {time.strftime('%H:%M:%S')}")
        else:
            clear_output(wait=True)
            table_html = create_scene_results_table(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS)
            if table_html:
                display(HTML(table_html))
                print(f"📊 Table updated at {time.strftime('%H:%M:%S')}")
        return True
    except Exception as e:
        print(f"❌ Table update failed: {e}")
        return False

def display_scene_results_table(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS, table_output=None):
    """Display the comprehensive scene results table"""
    print("📊 Generating comprehensive scene results table...")
    
    table_html = create_scene_results_table(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS)
    if table_html:
        if table_output:
            with table_output:
                clear_output(wait=True)
                display(HTML(table_html))
        else:
            display(HTML(table_html))
        print("✅ Scene results table displayed!")
    else:        
        print("❌ Failed to create scene results table")

def update_results_table_realtime(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS, table_output=None):
    """Update results table in real-time as scenes are processed"""
    # Display initial empty table
    table_html = create_scene_results_table(OUTPUT_DIR, SCENE_ANALYSIS_RESULTS)
    if table_html:
        if table_output:
            with table_output:
                clear_output(wait=True)
                display(HTML(table_html))
        else:
            clear_output(wait=True)
            display(HTML(table_html))
    
    return True
