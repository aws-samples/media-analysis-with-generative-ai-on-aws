# Prompt Caching Demo Data

This directory contains sample data for demonstrating prompt caching in the notebook's Step 4.5 demo cell.

## Contents

### Filmstrips (5 samples)
- `filmstrip_0000_4x5.jpg` - Chunk 0 (0-20s)
- `filmstrip_0001_4x5.jpg` - Chunk 1 (20-40s)
- `filmstrip_0002_4x5.jpg` - Chunk 2 (40-60s)
- `filmstrip_0003_4x5.jpg` - Chunk 3 (60-80s)
- `filmstrip_0004_4x5.jpg` - Chunk 4 (80-100s)

Each filmstrip is a 4×5 grid showing 20 frames sampled uniformly across a 20-second chunk.

### Transcripts
- `live_transcript.json` - Complete transcript with timestamped sentences

## Usage

The Step 4.5 demo cell in the notebook will automatically use this directory if available, allowing you to run the prompt caching demonstration without needing to process a full video first.

## Demo Purpose

This demo shows:
1. **Complete JSON payload** for each API call with cache breakpoints highlighted
2. **Token usage metrics** for each individual call
3. **Cache performance evolution** across calls (0% → 60-70% → 70-80%)
4. **Cumulative summary table** showing all calls and total savings

## Cache Strategy

The demo implements a **dual cache breakpoint strategy**:
- **Breakpoint #1**: System prompt (static, always cached)
- **Breakpoint #2**: Growing conversation (incremental caching)

This results in 60-80% cost savings on cached content!
