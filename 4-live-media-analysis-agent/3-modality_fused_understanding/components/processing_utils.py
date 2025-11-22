"""
Processing Utilities for Modality Fused Understanding

This module provides the main processing orchestration functions
for multi-modal fusion processing.
"""

import asyncio
import time
from typing import Callable, Optional


class ProcessingUtils:
    """Utility class for processing orchestration"""
    
    @staticmethod
    async def start_fusion_processing(
        duration_minutes: int,
        recording_manager,
        chunk_processor,
        chunk_monitor,
        fusion_analyzer,
        transcription_processor,
        stream_monitor_class,
        log_component: Callable,
        refresh_chapter_table: Callable
    ):
        """
        Start all three processing streams in parallel
        
        Args:
            duration_minutes: Maximum processing duration
            recording_manager: Recording manager instance
            chunk_processor: Chunk processor instance
            chunk_monitor: Chunk monitor instance
            fusion_analyzer: Fusion analyzer instance
            transcription_processor: Transcription processor instance
            stream_monitor_class: StreamMonitor class
            log_component: Logging function
            refresh_chapter_table: Table refresh function
        """
        log_component("Main", "🚀 Starting Modality Fusion Processing...")
        log_component("Main", f"⏰ Processing duration: {duration_minutes} minutes")

        try:
            # Start all processors
            log_component("Main", "📹 Starting recording manager...")
            recording_manager.start_recording()

            log_component("Main", "🎬 Starting chunk processor...")
            chunk_processor.start_processing()

            log_component("Main", "🔍 Starting chunk monitor...")
            chunk_monitor.start_monitoring()

            log_component("Main", "🧠 Starting fusion analyzer...")
            fusion_analyzer.start_analysis()

            log_component("Main", "🎧 Starting transcription...")
            transcription_task = asyncio.create_task(
                transcription_processor.start_transcription()
            )

            # Initialize stream monitor for centralized stream-end detection
            stream_monitor = stream_monitor_class(chunk_monitor, transcription_processor)
            
            # Monitor for duration timeout OR stream end
            log_component("Main", f"⏳ Processing for {duration_minutes} minutes (or until stream ends)...", "DEBUG")
            start_time = time.time()
            last_refresh_time = time.time()
            
            while True:
                # Check duration timeout
                if time.time() - start_time >= duration_minutes * 60:
                    log_component("Main", f"🛑 Duration elapsed ({duration_minutes} minutes), stopping processors...")
                    break
                    
                # Check stream end
                if stream_monitor.stream_appears_ended():
                    log_component("Main", "📡 Stream appears to have ended (60s timeout), stopping processors...")
                    break
                
                # Refresh display every 3 seconds
                if time.time() - last_refresh_time >= 3:
                    refresh_chapter_table()
                    last_refresh_time = time.time()
                    
                await asyncio.sleep(1)

            # Shutdown sequence
            await ProcessingUtils._shutdown_processors(
                recording_manager,
                transcription_processor,
                chunk_monitor,
                chunk_processor,
                fusion_analyzer,
                transcription_task,
                log_component,
                refresh_chapter_table
            )

        except KeyboardInterrupt:
            log_component("Main", "⏹️ Processing interrupted by user", "WARNING")
        except Exception as e:
            log_component("Main", f"❌ Processing error: {e}", "ERROR")
        finally:
            log_component("Main", "✅ Processing complete")

    @staticmethod
    async def _shutdown_processors(
        recording_manager,
        transcription_processor,
        chunk_monitor,
        chunk_processor,
        fusion_analyzer,
        transcription_task,
        log_component: Callable,
        refresh_chapter_table: Callable
    ):
        """Handle graceful shutdown of all processors"""
        
        # 1. Stop data sources (no new data flowing in)
        log_component("Main", "🛑 Stopping recording manager...")
        recording_manager.stop_recording()

        log_component("Main", "🛑 Stopping transcription...")
        await transcription_processor.stop_transcription()

        # 2. Stop chunk monitor (no new filmstrips/analyses queued)
        log_component("Main", "🛑 Stopping chunk monitor...")
        chunk_monitor.stop_monitoring()

        # 3. Stop chunk processor (FFmpeg)
        log_component("Main", "🛑 Stopping chunk processor...")
        chunk_processor.stop_processing()

        # 4. Brief pause for any in-flight operations
        log_component("Main", "⏳ Waiting for in-flight operations...", "DEBUG")
        time.sleep(2)

        # 5. Stop fusion analyzer (process remaining queue)
        log_component("Main", "🛑 Stopping fusion analyzer...")
        fusion_analyzer.stop_analysis()

        # 6. Final chapter table refresh (show all chapters including last one)
        log_component("Main", "📊 Final chapter table refresh...")
        refresh_chapter_table()

        # 7. Generate final summary of all content
        log_component("Main", "📝 Generating final content summary...")
        fusion_analyzer.generate_final_summary()
        
        # 8. Wait for all clip creation to complete
        fusion_analyzer.wait_for_clip_creation()
        
        # 9. Final table refresh after all clips are created
        log_component("Main", "📊 Final table refresh - all processing complete")
        refresh_chapter_table()
        
        # Wait for transcription task to complete
        if transcription_task:
            try:
                await asyncio.wait_for(transcription_task, timeout=10)
            except asyncio.TimeoutError:
                transcription_task.cancel()
                try:
                    await transcription_task
                except asyncio.CancelledError:
                    pass


# Convenience function for direct import
async def start_fusion_processing(
    duration_minutes: int,
    recording_manager,
    chunk_processor,
    chunk_monitor,
    fusion_analyzer,
    transcription_processor,
    stream_monitor_class,
    log_component: Callable,
    refresh_chapter_table: Callable
):
    """Convenience function for fusion processing"""
    return await ProcessingUtils.start_fusion_processing(
        duration_minutes,
        recording_manager,
        chunk_processor,
        chunk_monitor,
        fusion_analyzer,
        transcription_processor,
        stream_monitor_class,
        log_component,
        refresh_chapter_table
    )
