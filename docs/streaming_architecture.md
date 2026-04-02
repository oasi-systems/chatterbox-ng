---
name: Streaming architecture reference
description: Complete map of all streaming-related files, classes, and methods — read this to understand or modify the streaming pipeline
type: reference
---

## File map (in pipeline order)

### T3 — Token generation
- `src/chatterbox/models/t3/t3.py` → `inference_streaming()` — yields speech tokens one at a time
- `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py` — token repetition threshold (set to 4)

### S3Gen — Token→Mel→Audio
- `src/chatterbox/models/s3gen/transformer/attention.py` → `rel_shift()` — fixed for asymmetric Q/K (KV-cache)
- `src/chatterbox/models/s3gen/transformer/embedding.py` → `position_encoding_cached(query_len, key_len)` — relative pos encoding for cached attention
- `src/chatterbox/models/s3gen/transformer/upsample_encoder.py` → `EncoderCaches` dataclass + `forward_cached()` — encoder KV-cache path
- `src/chatterbox/models/s3gen/flow.py` → `inference_cached()` (encoder KV-cache) + `inference_windowed()` (CFM context window)
- `src/chatterbox/models/s3gen/flow_matching.py` → `solve_euler()` / `basic_euler()` with `freeze_len` param
- `src/chatterbox/models/s3gen/s3gen.py` → `StreamState` (7 fields) + `streaming_step_cached()` + `streaming_step()` (fallback)

### Orchestrator
- `src/chatterbox/streaming.py` → `ChatterboxStreamingTTS` — main streaming class
  - `generate_stream()` — public API, yields audio chunks
  - `_generate_stream_pipelined()` — sentence-level pipelining (default)
  - `_emit_chunk()` — dispatches to cached or fallback path
  - `_split_sentences()` — sentence boundary detection

### CUDA optimizations
- `src/chatterbox/cuda_optimizations.py` → `optimize_for_cuda()` — BF16, torch.compile, SDPA
- `examples/l4_production_streaming.py` — production example with quality presets

### Exports
- `src/chatterbox/__init__.py` → exports `ChatterboxStreamingTTS`, `optimize_for_cuda`

## Key constants
- `S3GEN_SR = 24000` (output audio sample rate)
- `S3_SR = 16000` (s3 tokenizer sample rate)
- `token_mel_ratio = 2` (1 token → 2 mel frames)
- `pre_lookahead_len = 3` (trimmed when finalize=False → 6 mel frames)
- Token rate: 25Hz, Mel rate: 50Hz

## StreamState fields
```
hifi_cache_source     — HiFiGAN waveform continuity cache
prev_stable_mel_len   — mel frames already vocoded
is_first_chunk        — first chunk flag
encoder_caches        — EncoderCaches (KV-cache for 6+4 conformer layers)
cached_encoder_output — accumulated projected encoder output (B, mel_frames, 80)
cached_mel            — accumulated generated mel (B, 80, mel_frames)
spk_embedding_proj    — projected speaker embedding (cached once)
```

## Quality tuning knobs
| Parameter | Default | Effect |
|-----------|---------|--------|
| `streaming_cfm_steps` | 4 | ODE steps for intermediate chunks. 4=fast/metallic, 8=balanced, 12=quality |
| `chunk_tokens` | 25 | Tokens buffered per chunk. Lower=more chunks, higher=better quality |
| `min_initial_tokens` | 15 | Minimum tokens before first emission |
| `context_frames` | 20 | CFM context window size (in `streaming_step_cached`) |
| `exaggeration` | 0.5 | Voice expressiveness |
| `cfg_weight` | 0.5 | Classifier-free guidance weight |

## Known issues
- Speech rate is too fast vs reference — T3 limitation, not streaming
- Slightly metallic with 4 CFM steps — use 8+ for production
- Encoder KV-cache is approximate (bidirectional encoder, old positions frozen)
