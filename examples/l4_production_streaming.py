"""
ChatterBox NG — Production streaming on NVIDIA L4 GPU.

Optimized for real-time telephony (16kHz) with meanflow (2 ODE steps).

Key tuning parameters:
- chunk_tokens: speech tokens buffered before emitting audio (~40ms per token)
  - 25: default, good TTFA/quality balance
- exaggeration: 0.0–1.0, expressiveness (0.5 = natural for call center)
- cfg_weight: 0.0–1.0, adherence to reference voice (0.5 = balanced)

Performance on L4:
- TTFA: ~173ms (adaptive schedule 5→10→20→25)
- RTF: ~0.67x (comfortably real-time)
- T3: ~17ms/tok (no torch.compile — SDPA handles kernel fusion)

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


def setup_model(device="cuda", meanflow=True):
    """Load and optimize model for L4 production."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.cuda_optimizations import optimize_for_cuda, warmup_model

    logger.info("Loading model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device, meanflow=meanflow)

    logger.info("Applying CUDA optimizations (BF16, SDPA, TF32)...")
    optimize_for_cuda(model, use_bf16=True)

    return model


def stream_tts(
    model,
    text: str,
    ref_audio: str,
    output_path: str = "output.wav",
    output_sr: int = 16000,
    chunk_tokens: int = 25,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    language_id: str = "it",
):
    """Stream TTS with optimized settings for L4."""
    from chatterbox.streaming import ChatterboxStreamingTTS
    from chatterbox.cuda_optimizations import warmup_model

    model.prepare_conditionals(ref_audio, exaggeration=exaggeration)
    warmup_model(model, device=model.device)

    streamer = ChatterboxStreamingTTS(
        model,
        chunk_tokens=chunk_tokens,
        output_sample_rate=output_sr,
    )

    chunks = []
    chunk_latencies = []
    t_start = time.time()
    first_chunk_time = None

    for i, chunk in enumerate(streamer.generate_stream(
        text,
        language_id=language_id,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
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

    # Save
    full_audio = np.concatenate(chunks)
    sf.write(output_path, full_audio, streamer.sample_rate)

    total_audio_dur = len(full_audio) / streamer.sample_rate
    total_wall = sum(chunk_latencies) / 1000

    logger.info(f"\n{'='*50}")
    logger.info(f"Audio duration:    {total_audio_dur:.2f}s")
    logger.info(f"Wall time:         {total_wall:.2f}s")
    logger.info(f"Overall RTF:       {total_wall / total_audio_dur:.2f}x")
    logger.info(f"First chunk:       {first_chunk_time * 1000:.0f}ms")
    logger.info(f"Mean chunk:        {np.mean(chunk_latencies):.0f}ms")
    logger.info(f"Chunks:            {len(chunks)}")
    logger.info(f"Saved: {output_path}")

    return full_audio, streamer.sample_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatterBox NG — L4 Production Streaming")
    parser.add_argument("--ref", required=True, help="Reference audio path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output audio path")
    parser.add_argument("--output-sr", type=int, default=16000, help="Output sample rate (16000 for telephony)")
    parser.add_argument("--chunk-tokens", type=int, default=25, help="Tokens per chunk")
    parser.add_argument("--language", default="it", help="Language code")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-meanflow", action="store_true", help="Disable meanflow (10 ODE steps, slower)")
    args = parser.parse_args()

    model = setup_model(args.device, meanflow=not args.no_meanflow)

    stream_tts(
        model,
        text=args.text,
        ref_audio=args.ref,
        output_path=args.output,
        output_sr=args.output_sr,
        chunk_tokens=args.chunk_tokens,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        language_id=args.language,
    )
