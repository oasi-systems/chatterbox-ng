"""
ChatterBox NG — Production streaming on NVIDIA L4 GPU.

Optimized for best quality/latency balance on L4 (24GB VRAM, Ada Lovelace).

Key tuning parameters for quality:
- streaming_cfm_steps: ODE steps for intermediate chunks (more = better quality, slower)
  - 4: fastest, slightly metallic
  - 8: good balance (recommended for production)
  - 12-16: near-standard quality, slower per chunk
  Final chunk always uses model default (10 steps).

- chunk_tokens: speech tokens buffered before emitting audio (~40ms per token)
  - 15: low latency, more chunks, slightly lower quality
  - 25: default, good balance
  - 40: fewer chunks, better quality per chunk

- context_frames: mel frames of context for CFM window (more = smoother transitions)
  - 10: minimum viable
  - 20: default
  - 40: best continuity, slightly slower

Usage:
    python l4_production_streaming.py --ref speaker.wav --text "Your text here"
"""
import argparse
import time
import logging
import numpy as np
import torch
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model(device="cuda"):
    """Load and optimize model for L4 production."""
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.cuda_optimizations import optimize_for_cuda

    logger.info("Loading model...")
    model = ChatterboxTTS.from_pretrained(device)

    logger.info("Applying CUDA optimizations...")
    optimize_for_cuda(
        model,
        compile_mode="reduce-overhead",  # best for repeated inference
        use_bf16=True,                   # L4 supports BF16 natively
        compile_models=True,             # compile encoder, CFM, HiFiGAN, T3
    )

    # Warmup — first inference triggers torch.compile tracing
    logger.info("Warmup inference (torch.compile tracing)...")
    _ = model.generate("Warmup.", audio_prompt_path=None)
    logger.info("Model ready.")

    return model


def stream_tts(
    model,
    text: str,
    ref_audio: str,
    output_path: str = "output.wav",
    # Quality/latency knobs
    streaming_cfm_steps: int = 8,
    chunk_tokens: int = 25,
    min_initial_tokens: int = 15,
    # Generation params
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
):
    """Stream TTS with optimized settings for L4.

    Returns:
        tuple: (full_watermarked_audio, sample_rate, metrics_dict)
    """
    from chatterbox.streaming import ChatterboxStreamingTTS

    streamer = ChatterboxStreamingTTS(
        model,
        chunk_tokens=chunk_tokens,
        min_initial_tokens=min_initial_tokens,
        streaming_cfm_steps=streaming_cfm_steps,
    )

    chunks = []
    chunk_latencies = []
    t_start = time.time()
    first_chunk_time = None

    for i, chunk in enumerate(streamer.generate_stream(
        text,
        audio_prompt_path=ref_audio,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        sentence_pipelining=True,  # always recommended
    )):
        now = time.time()
        latency = now - t_start
        if first_chunk_time is None:
            first_chunk_time = latency

        chunk_dur_ms = len(chunk) / streamer.sample_rate * 1000
        chunk_latencies.append(latency * 1000)
        chunks.append(chunk)

        logger.info(
            f"Chunk {i:2d}: {chunk_dur_ms:6.0f}ms audio, "
            f"latency {latency*1000:.0f}ms, "
            f"RTF {latency / (chunk_dur_ms/1000):.2f}x"
        )
        t_start = now

    # Watermark and save
    full_audio = streamer.get_full_watermarked()
    sf.write(output_path, full_audio, streamer.sample_rate)

    total_audio_dur = len(full_audio) / streamer.sample_rate
    total_wall = sum(chunk_latencies) / 1000

    metrics = {
        "total_audio_s": total_audio_dur,
        "total_wall_s": total_wall,
        "overall_rtf": total_wall / total_audio_dur,
        "first_chunk_ms": first_chunk_time * 1000 if first_chunk_time else 0,
        "mean_chunk_ms": np.mean(chunk_latencies),
        "n_chunks": len(chunks),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Audio duration:    {metrics['total_audio_s']:.2f}s")
    logger.info(f"Wall time:         {metrics['total_wall_s']:.2f}s")
    logger.info(f"Overall RTF:       {metrics['overall_rtf']:.2f}x")
    logger.info(f"First chunk:       {metrics['first_chunk_ms']:.0f}ms")
    logger.info(f"Mean chunk:        {metrics['mean_chunk_ms']:.0f}ms")
    logger.info(f"Chunks:            {metrics['n_chunks']}")
    logger.info(f"Saved: {output_path}")

    return full_audio, streamer.sample_rate, metrics


# --- Quality presets ---

PRESETS = {
    "fast": {
        "streaming_cfm_steps": 4,
        "chunk_tokens": 15,
        "min_initial_tokens": 10,
    },
    "balanced": {
        "streaming_cfm_steps": 8,
        "chunk_tokens": 25,
        "min_initial_tokens": 15,
    },
    "quality": {
        "streaming_cfm_steps": 12,
        "chunk_tokens": 40,
        "min_initial_tokens": 25,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatterBox NG — L4 Production Streaming")
    parser.add_argument("--ref", required=True, help="Reference audio path for voice cloning")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output audio path")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="balanced",
                        help="Quality preset: fast, balanced, quality")
    parser.add_argument("--cfm-steps", type=int, help="Override CFM ODE steps")
    parser.add_argument("--chunk-tokens", type=int, help="Override chunk token count")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = setup_model(args.device)

    preset = PRESETS[args.preset].copy()
    if args.cfm_steps:
        preset["streaming_cfm_steps"] = args.cfm_steps
    if args.chunk_tokens:
        preset["chunk_tokens"] = args.chunk_tokens

    stream_tts(
        model,
        text=args.text,
        ref_audio=args.ref,
        output_path=args.output,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        **preset,
    )
