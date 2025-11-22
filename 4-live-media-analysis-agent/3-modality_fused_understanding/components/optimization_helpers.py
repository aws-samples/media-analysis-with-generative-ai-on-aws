"""
Helper functions for optimization demonstrations (prompt caching and smart context windowing)
"""

import json
import time
import glob
import base64
from pathlib import Path
from .demo_utils import (
    load_system_prompt,
    print_payload_structure, 
    print_token_metrics,
    print_summary_table,
    print_first_call_explanation,
    print_windowing_comparison_table,
    print_windowing_key_learnings
)
from .fusion_analyzer import FusionAnalyzer


def run_prompt_caching_demo(sample_dir, output_dir, aws_region, chunk_duration, sentence_buffer, model_id=None):
    """
    Run prompt caching demonstration with minimal code in notebook
    Returns: metrics_list for comparison
    """
    # Check for demo data
    if Path(sample_dir).exists():
        print(f"✅ Using demo data from {sample_dir}/")
        filmstrips = sorted(glob.glob(f"{sample_dir}/filmstrips/filmstrip_*.jpg"))
        transcript_file = f"{sample_dir}/transcripts/live_transcript.json"
    else:
        print(f"ℹ️  Using {output_dir}/")
        filmstrips = sorted(glob.glob(f"{output_dir}/filmstrips/filmstrip_*.jpg"))
        transcript_file = f"{output_dir}/transcripts/live_transcript.json"
    
    if not filmstrips:
        print("❌ No filmstrips found.")
        return []
    
    print(f"✅ Found {len(filmstrips)} filmstrips")
    
    # Load transcript
    sample_transcript = []
    if Path(transcript_file).exists():
        with open(transcript_file, 'r') as f:
            sample_transcript = json.load(f)
        print(f"✅ Loaded {len(sample_transcript)} transcript sentences\n")
    
    # Create analyzer
    class CachingDemoAnalyzer(FusionAnalyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.call_number = 0
            self.metrics_list = []
            self.expected_calls = 0
            self.completed_calls = 0
        
        def _perform_fusion_analysis(self, request):
            try:
                self.call_number += 1
                chunk_id = request['chunk_id']
                filmstrip_path = request['filmstrip_path']
                start_time = request['start_time']
                end_time = request['end_time']
                
                print(f"\n{'#'*50}")
                print(f"🔍 CHUNK {chunk_id} ({start_time}s-{end_time}s)")
                print(f"{'#'*50}\n")
                
                # Prepare data
                transcript_data = self._get_transcript_for_timerange(start_time, end_time, self.sentence_buffer)
                with open(filmstrip_path, 'rb') as f:
                    image_data = base64.standard_b64encode(f.read()).decode('utf-8')
                
                system_prompt = load_system_prompt()
                
                # Build request with cache breakpoints
                transcript_json = json.dumps(transcript_data['sentences'], indent=2, ensure_ascii=False)
                current_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Chunk {chunk_id} ({start_time}s-{end_time}s)\\n\\nTranscript:\\n{transcript_json}",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]
                }
                
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
                    "messages": self.messages + [current_message],
                    "temperature": 0.1
                }
                
                print_payload_structure(request_body, self.call_number)
                
                print(f"🚀 Calling Amazon Bedrock API...\n")
                start_time_call = time.time()
                
                model_id_to_use = getattr(self, 'model_id_override', None) or model_id or "global.anthropic.claude-sonnet-4-20250514-v1:0"
                response = self.bedrock_client.invoke_model(
                    modelId=model_id_to_use,
                    body=json.dumps(request_body)
                )
                
                call_duration = time.time() - start_time_call
                
                response_body = json.loads(response['body'].read())
                usage = response_body.get('usage', {})
                
                metrics = print_token_metrics(usage, self.call_number, call_duration)
                self.metrics_list.append(metrics)
                self.completed_calls += 1
                
                if self.completed_calls == 1:
                    print_first_call_explanation()
                
                print(f"✅ Completed {self.completed_calls}/{self.expected_calls} calls\n")
                
                # Update conversation history
                response_text = response_body['content'][0]['text']
                self.messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"Chunk {chunk_id} ({start_time}s-{end_time}s)\\n\\nTranscript:\\n{transcript_json}"}]
                })
                self.messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_text}]
                })
                
                return response
                
            except Exception as e:
                print(f"❌ Error: {e}")
                raise
    
    # Run demo
    analyzer = CachingDemoAnalyzer(
        aws_region=aws_region,
        sentence_buffer=sample_transcript,
        analysis_results={},
        output_dir=output_dir,
        memory_client=None,
        memory_id=None,
        actor_id=None,
        session_id=None
    )
    
    # Store model_id for the analyzer to use
    analyzer.model_id_override = model_id
    
    analyzer.start_analysis()
    
    # Process 4 chunks
    num_chunks = min(4, len(filmstrips))
    analyzer.expected_calls = num_chunks
    
    for i in range(num_chunks):
        analyzer.queue_analysis(
            i, filmstrips[i],
            i * chunk_duration,
            (i + 1) * chunk_duration,
            []
        )
    
    # Wait for completion
    print(f"\n⏳ Waiting for all {num_chunks} Bedrock API calls...")
    max_wait = 180
    start_wait = time.time()
    
    while analyzer.completed_calls < num_chunks and (time.time() - start_wait) < max_wait:
        time.sleep(2)
    
    print(f"✅ All {num_chunks} calls completed!\n")
    time.sleep(1)
    
    # Print summary
    print_summary_table(analyzer.metrics_list)
    
    analyzer.stop_analysis()
    return analyzer.metrics_list


def run_windowing_demo(sample_dir, output_dir, aws_region, chunk_duration, sentence_buffer, n_chapters=0, model_id=None):
    """
    Run smart context windowing demonstration
    Returns: (no_window_metrics, with_window_metrics, savings_pct)
    """
    # Check for demo data
    if Path(sample_dir).exists():
        print(f"✅ Using demo data from {sample_dir}/")
        filmstrips = sorted(glob.glob(f"{sample_dir}/filmstrips/filmstrip_*.jpg"))
        transcript_file = f"{sample_dir}/transcripts/live_transcript.json"
    else:
        print(f"ℹ️  Using {output_dir}/")
        filmstrips = sorted(glob.glob(f"{output_dir}/filmstrips/filmstrip_*.jpg"))
        transcript_file = f"{output_dir}/transcripts/live_transcript.json"
    
    if not filmstrips:
        print("❌ No filmstrips found.")
        return [], [], 0
    
    print(f"✅ Found {len(filmstrips)} filmstrips")
    
    # Load transcript
    sample_transcript = []
    if Path(transcript_file).exists():
        with open(transcript_file, 'r') as f:
            sample_transcript = json.load(f)
        print(f"✅ Loaded {len(sample_transcript)} transcript sentences")
    
    class WindowingDemoAnalyzer(FusionAnalyzer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.context_metrics = []
            self.demo_mode = True
            self.expected_chunks = 0
            self.completed_chunks = 0
        
        def _perform_fusion_analysis(self, request):
            chunk_id = request['chunk_id']
            
            try:
                result = super()._perform_fusion_analysis(request)
                
                # Measure context size
                num_messages = len(self.messages)
                context_tokens = 0
                for msg in self.messages:
                    content = msg.get('content', [])
                    if isinstance(content, list):
                        for item in content:
                            if item.get('type') == 'text':
                                text = item.get('text', '')
                                if text:
                                    context_tokens += len(text) // 4
                
                # Estimate chapters in context
                if self.analysis_results:
                    latest_result = self.analysis_results.get(max(self.analysis_results.keys()), {})
                    total_chapters = len(latest_result.get('chapters', []))
                    
                    if self.keep_n_chapters is not None and total_chapters > self.keep_n_chapters + 1:
                        chapters_in_context = min(total_chapters, self.keep_n_chapters + 1)
                    else:
                        chapters_in_context = total_chapters
                else:
                    chapters_in_context = 0
                
                self.context_metrics.append({
                    'chunk_id': chunk_id,
                    'num_messages': num_messages,
                    'context_tokens': context_tokens,
                    'chapters_in_context': chapters_in_context
                })
                
                print(f"   📊 Context after chunk {chunk_id}: {num_messages} messages, {context_tokens:,} tokens, {chapters_in_context} chapters")
                
                self.completed_chunks += 1
                return result
                
            except TypeError as e:
                print(f"   ⚠️  Skipping chunk {chunk_id} due to error: {e}")
                self.completed_chunks += 1
                return None
    
    # Test without windowing
    print(f"\n{'#'*50}")
    print("WITHOUT Windowing (baseline)")
    print(f"{'#'*50}")
    
    analyzer_no_window = WindowingDemoAnalyzer(
        aws_region=aws_region,
        sentence_buffer=sample_transcript,
        analysis_results={},
        output_dir=output_dir,
        keep_n_chapters=None,
        memory_client=None,
        memory_id=None,
        actor_id=None,
        session_id=None
    )
    
    # Store model_id for the analyzer to use
    analyzer_no_window.model_id_override = model_id
    
    analyzer_no_window.start_analysis()
    
    num_chunks = min(3, len(filmstrips))
    analyzer_no_window.expected_chunks = num_chunks
    
    for i in range(num_chunks):
        analyzer_no_window.queue_analysis(
            i, filmstrips[i],
            i * chunk_duration,
            (i + 1) * chunk_duration,
            []
        )
    
    # Wait for completion
    max_wait = 180
    start_wait = time.time()
    
    while analyzer_no_window.completed_chunks < num_chunks and (time.time() - start_wait) < max_wait:
        time.sleep(2)
    
    print(f"✅ Completed without windowing")
    analyzer_no_window.is_running = False
    
    # Test with windowing
    print(f"\n{'#'*50}")
    print(f"WITH Windowing (keep_n_chapters = {n_chapters})")
    print(f"{'#'*50}")
    
    analyzer_with_window = WindowingDemoAnalyzer(
        aws_region=aws_region,
        sentence_buffer=sample_transcript,
        analysis_results={},
        output_dir=output_dir,
        keep_n_chapters=n_chapters,
        memory_client=None,
        memory_id=None,
        actor_id=None,
        session_id=None
    )
    
    # Store model_id for the analyzer to use
    analyzer_with_window.model_id_override = model_id
    
    analyzer_with_window.start_analysis()
    analyzer_with_window.expected_chunks = num_chunks
    
    for i in range(num_chunks):
        analyzer_with_window.queue_analysis(
            i, filmstrips[i],
            i * chunk_duration,
            (i + 1) * chunk_duration,
            []
        )
    
    start_wait = time.time()
    while analyzer_with_window.completed_chunks < num_chunks and (time.time() - start_wait) < max_wait:
        time.sleep(2)
    
    print(f"✅ Completed with windowing")
    analyzer_with_window.is_running = False
    
    # Calculate savings
    total_no_win = sum(m['context_tokens'] for m in analyzer_no_window.context_metrics)
    total_with_win = sum(m['context_tokens'] for m in analyzer_with_window.context_metrics)
    savings_pct = ((total_no_win - total_with_win) / total_no_win * 100) if total_no_win > 0 else 0
    
    # Print comparison
    print_windowing_comparison_table(
        analyzer_no_window.context_metrics,
        analyzer_with_window.context_metrics,
        n_chapters
    )
    
    print_windowing_key_learnings(savings_pct)
    
    return analyzer_no_window.context_metrics, analyzer_with_window.context_metrics, savings_pct


def print_optimization_summary(caching_metrics=None, windowing_savings=None):
    """Print a concise summary of optimization results"""
    print("\n" + "="*80)
    print("📊 OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    if caching_metrics:
        # Calculate caching savings
        total_regular = sum(m.get('regular_input', 0) for m in caching_metrics)
        total_cache_read = sum(m.get('cache_read', 0) for m in caching_metrics)
        total_input = total_regular + total_cache_read
        cache_hit_ratio = (total_cache_read / total_input * 100) if total_input > 0 else 0
        
        print(f"\n🎯 PROMPT CACHING RESULTS:")
        print(f"   • Cache Hit Ratio: {cache_hit_ratio:.1f}%")
        print(f"   • Total Input Tokens: {total_input:,}")
        print(f"   • Cached Tokens: {total_cache_read:,}")
        print(f"   • Cost Savings: ~{cache_hit_ratio * 0.9:.0f}% on cached content")
    
    if windowing_savings is not None:
        print(f"\n🪟 SMART CONTEXT WINDOWING RESULTS:")
        print(f"   • Token Reduction: {windowing_savings:.1f}%")
        print(f"   • Context Size: Bounded vs Unbounded")
        print(f"   • Memory Efficiency: Maintained with accuracy")
    
    if caching_metrics and windowing_savings is not None:
        combined_savings = min(85, cache_hit_ratio * 0.9 + windowing_savings * 0.4)
        print(f"\n💰 COMBINED OPTIMIZATION:")
        print(f"   • Total Estimated Savings: ~{combined_savings:.0f}%")
        print(f"   • Prompt Caching + Smart Windowing")
        print(f"   • Scalable to unlimited video length")
    
    print("="*80)
