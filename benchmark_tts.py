"""
Benchmarking tool for ChatterBox TTS.

Measures:
- First-chunk latency (time to first audio in streaming mode)
- Total generation time
- Real-Time Factor (RTF = generation_time / audio_duration)
- Audio quality metrics (if reference available)

Usage:
    python benchmark_tts.py --model multilingual --text "Ciao mondo!" --ref_audio ref.wav
    python benchmark_tts.py --model turbo --text "Hello world!" --iterations 5
    python benchmark_tts.py --suite  # Run full benchmark suite
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_type: str
    text: str
    text_length: int
    language_id: Optional[str]

    # Timing
    first_chunk_latency_ms: float = 0.0
    total_time_ms: float = 0.0
    num_chunks: int = 0

    # Audio
    audio_duration_s: float = 0.0
    total_samples: int = 0

    # Derived
    rtf: float = 0.0  # Real-Time Factor (< 1.0 = faster than real-time)
    chars_per_second: float = 0.0

    # Streaming
    sentence_pipelining: bool = False
    chunk_tokens: int = 25

    def compute_derived(self):
        if self.audio_duration_s > 0:
            self.rtf = (self.total_time_ms / 1000.0) / self.audio_duration_s
        if self.total_time_ms > 0:
            self.chars_per_second = self.text_length / (self.total_time_ms / 1000.0)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    device: str = ""
    model_type: str = ""

    def summary(self) -> dict:
        if not self.results:
            return {}
        fcls = [r.first_chunk_latency_ms for r in self.results]
        rtfs = [r.rtf for r in self.results]
        total_times = [r.total_time_ms for r in self.results]
        return {
            "device": self.device,
            "model_type": self.model_type,
            "num_runs": len(self.results),
            "first_chunk_latency_ms": {
                "mean": np.mean(fcls),
                "std": np.std(fcls),
                "min": np.min(fcls),
                "max": np.max(fcls),
                "p50": np.median(fcls),
                "p95": np.percentile(fcls, 95),
            },
            "rtf": {
                "mean": np.mean(rtfs),
                "std": np.std(rtfs),
                "min": np.min(rtfs),
                "max": np.max(rtfs),
            },
            "total_time_ms": {
                "mean": np.mean(total_times),
                "std": np.std(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
            },
        }


def benchmark_streaming(
    model,
    text: str,
    ref_audio_path: Optional[str] = None,
    language_id: Optional[str] = None,
    chunk_tokens: int = 25,
    sentence_pipelining: bool = False,
    **gen_kwargs,
) -> BenchmarkResult:
    """Run a single streaming benchmark."""
    from chatterbox.streaming import ChatterboxStreamingTTS

    result = BenchmarkResult(
        model_type=type(model).__name__,
        text=text,
        text_length=len(text),
        language_id=language_id,
        chunk_tokens=chunk_tokens,
        sentence_pipelining=sentence_pipelining,
    )

    streamer = ChatterboxStreamingTTS(model, chunk_tokens=chunk_tokens)

    kwargs = dict(
        text=text,
        sentence_pipelining=sentence_pipelining,
        **gen_kwargs,
    )
    if ref_audio_path:
        kwargs["audio_prompt_path"] = ref_audio_path
    if language_id:
        kwargs["language_id"] = language_id

    all_audio = []
    start_time = time.perf_counter()
    first_chunk_time = None

    for chunk in streamer.generate_stream(**kwargs):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter()
        all_audio.append(chunk)

    end_time = time.perf_counter()

    if first_chunk_time:
        result.first_chunk_latency_ms = (first_chunk_time - start_time) * 1000
    result.total_time_ms = (end_time - start_time) * 1000
    result.num_chunks = len(all_audio)

    if all_audio:
        total = np.concatenate(all_audio)
        result.total_samples = len(total)
        result.audio_duration_s = len(total) / SAMPLE_RATE

    result.compute_derived()
    return result


def benchmark_sync(
    model,
    text: str,
    ref_audio_path: Optional[str] = None,
    language_id: Optional[str] = None,
    **gen_kwargs,
) -> BenchmarkResult:
    """Run a single synchronous (non-streaming) benchmark."""
    result = BenchmarkResult(
        model_type=type(model).__name__,
        text=text,
        text_length=len(text),
        language_id=language_id,
    )

    kwargs = dict(text=text, **gen_kwargs)
    if ref_audio_path:
        kwargs["audio_prompt_path"] = ref_audio_path

    # Use the model's generate method
    start_time = time.perf_counter()

    if hasattr(model, 'generate'):
        wav = model.generate(**kwargs)
    else:
        raise ValueError(f"Model {type(model)} has no generate method")

    end_time = time.perf_counter()

    result.first_chunk_latency_ms = (end_time - start_time) * 1000  # same as total for sync
    result.total_time_ms = (end_time - start_time) * 1000
    result.num_chunks = 1

    if hasattr(wav, 'numpy'):
        wav_np = wav.squeeze().cpu().numpy()
    else:
        wav_np = np.asarray(wav).squeeze()

    result.total_samples = len(wav_np)
    result.audio_duration_s = len(wav_np) / SAMPLE_RATE
    result.compute_derived()

    return result


# --- Default benchmark texts ---

BENCHMARK_TEXTS = {
    "short_en": "Hello, how are you today?",
    "medium_en": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
    "long_en": (
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends "
        "of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or "
        "to eat: it was a hobbit-hole, and that means comfort."
    ),
    "short_it": "Buongiorno! Come stai oggi?",
    "medium_it": "Il dott. Rossi ha comprato 42 libri per 100 euro alla libreria del centro.",
    "long_it": (
        "Buongiorno! Oggi il dott. Rossi ha comprato 42 libri per 100 euro alla libreria del centro. "
        "La NATO ha organizzato un incontro il 15 marzo 2024 alle 14:30. "
        "Il PIL è cresciuto del 3% secondo l'ISTAT. "
        "Per informazioni, chiamare il numero 06 1234 5678."
    ),
}


def run_suite(
    model,
    ref_audio_path: Optional[str] = None,
    iterations: int = 3,
    streaming: bool = True,
    texts: Optional[dict] = None,
) -> dict:
    """Run a full benchmark suite.

    Returns:
        Dictionary with per-text results and summary statistics.
    """
    import torch

    texts = texts or BENCHMARK_TEXTS
    device = str(model.device) if hasattr(model, 'device') else "unknown"
    model_type = type(model).__name__

    all_results = {}

    for text_name, text in texts.items():
        lang = "it" if "_it" in text_name else None
        suite = BenchmarkSuite(device=device, model_type=model_type)

        # Warmup run
        if streaming:
            _ = benchmark_streaming(model, text, ref_audio_path, language_id=lang)
        else:
            _ = benchmark_sync(model, text, ref_audio_path)

        # Timed runs
        for i in range(iterations):
            if streaming:
                result = benchmark_streaming(model, text, ref_audio_path, language_id=lang)
            else:
                result = benchmark_sync(model, text, ref_audio_path)
            suite.results.append(result)
            logger.info(
                f"  [{text_name}] run {i+1}/{iterations}: "
                f"FCL={result.first_chunk_latency_ms:.0f}ms, "
                f"total={result.total_time_ms:.0f}ms, "
                f"RTF={result.rtf:.3f}, "
                f"duration={result.audio_duration_s:.2f}s"
            )

        all_results[text_name] = suite.summary()

    return {
        "device": device,
        "model_type": model_type,
        "streaming": streaming,
        "iterations": iterations,
        "results": all_results,
    }


def print_results(suite_results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'='*70}")
    print(f"  ChatterBox TTS Benchmark")
    print(f"  Device: {suite_results['device']}")
    print(f"  Model: {suite_results['model_type']}")
    print(f"  Mode: {'Streaming' if suite_results['streaming'] else 'Synchronous'}")
    print(f"  Iterations: {suite_results['iterations']}")
    print(f"{'='*70}\n")

    for text_name, stats in suite_results["results"].items():
        print(f"  {text_name}:")
        if "first_chunk_latency_ms" in stats:
            fcl = stats["first_chunk_latency_ms"]
            print(f"    First chunk:  {fcl['mean']:.0f}ms (p50={fcl['p50']:.0f}ms, p95={fcl['p95']:.0f}ms)")
        if "total_time_ms" in stats:
            tt = stats["total_time_ms"]
            print(f"    Total time:   {tt['mean']:.0f}ms (+/- {tt['std']:.0f}ms)")
        if "rtf" in stats:
            rtf = stats["rtf"]
            faster = "FASTER" if rtf['mean'] < 1.0 else "SLOWER"
            print(f"    RTF:          {rtf['mean']:.3f}x ({faster} than real-time)")
        print()

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="ChatterBox TTS Benchmark")
    parser.add_argument("--model", default="multilingual",
                        choices=["standard", "multilingual", "turbo"])
    parser.add_argument("--text", default=None, help="Text to synthesize")
    parser.add_argument("--ref_audio", default=None, help="Reference audio path")
    parser.add_argument("--language_id", default=None, help="Language ID (e.g., 'it')")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--suite", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--sync", action="store_true", help="Benchmark synchronous mode")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    logger.info(f"Loading {args.model} model on {DEVICE}...")
    if args.model == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    elif args.model == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model = ChatterboxTurboTTS.from_pretrained(DEVICE)
    else:
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if args.suite:
        results = run_suite(
            model,
            ref_audio_path=args.ref_audio,
            iterations=args.iterations,
            streaming=not args.sync,
        )
        print_results(results)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")
    elif args.text:
        if not args.sync:
            result = benchmark_streaming(
                model, args.text, args.ref_audio, args.language_id,
            )
            print(f"First chunk: {result.first_chunk_latency_ms:.0f}ms")
            print(f"Total: {result.total_time_ms:.0f}ms")
            print(f"RTF: {result.rtf:.3f}")
            print(f"Audio: {result.audio_duration_s:.2f}s ({result.num_chunks} chunks)")
        else:
            result = benchmark_sync(model, args.text, args.ref_audio)
            print(f"Total: {result.total_time_ms:.0f}ms")
            print(f"RTF: {result.rtf:.3f}")
            print(f"Audio: {result.audio_duration_s:.2f}s")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
