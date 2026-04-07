#!/usr/bin/env python3
"""
Benchmark: streaming performance comparison.

Compares efficient (windowed CFM) vs full reprocess streaming.
Measures FCL, per-chunk latency, RTF, total wall time.

Usage:
    python benchmarks/bench_streaming.py --ref speaker.wav
    python benchmarks/bench_streaming.py --ref speaker.wav --languages it en fr de es pt
"""
import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# Test sentences per language (short / medium / long)
TEST_SENTENCES = {
    "it": [
        "Buongiorno.",
        "Il suo saldo è di milleduecentocinquanta euro e zero centesimi.",
        "La informo che la sua pratica numero dodici-tre-quattro-cinque è stata "
        "approvata. Il pagamento è previsto entro il quindici marzo. Per ulteriori "
        "informazioni, prema uno. Per parlare con un operatore, prema due.",
    ],
    "en": [
        "Good morning.",
        "Your account balance is one thousand two hundred fifty euros.",
        "I would like to inform you that your application number twelve-three-four-five "
        "has been approved. Payment is expected by March fifteenth. For more "
        "information, press one. To speak with an operator, press two.",
    ],
    "fr": [
        "Bonjour.",
        "Votre solde est de mille deux cent cinquante euros.",
        "Je vous informe que votre dossier numéro douze-trois-quatre-cinq a été "
        "approuvé. Le paiement est prévu avant le quinze mars. Pour plus "
        "d'informations, appuyez sur un. Pour parler à un opérateur, appuyez sur deux.",
    ],
    "de": [
        "Guten Morgen.",
        "Ihr Kontostand beträgt eintausendzweihundertfünfzig Euro.",
        "Ich möchte Sie darüber informieren, dass Ihr Antrag Nummer zwölf-drei-vier-fünf "
        "genehmigt wurde. Die Zahlung wird bis zum fünfzehnten März erwartet. Für weitere "
        "Informationen drücken Sie die Eins. Um mit einem Mitarbeiter zu sprechen, drücken Sie die Zwei.",
    ],
    "es": [
        "Buenos días.",
        "Su saldo es de mil doscientos cincuenta euros.",
        "Le informo que su solicitud número doce-tres-cuatro-cinco ha sido aprobada. "
        "El pago está previsto antes del quince de marzo. Para más información, "
        "pulse uno. Para hablar con un operador, pulse dos.",
    ],
    "pt": [
        "Bom dia.",
        "O seu saldo é de mil duzentos e cinquenta euros.",
        "Informamos que o seu pedido número doze-três-quatro-cinco foi aprovado. "
        "O pagamento está previsto até quinze de março. Para mais informações, "
        "pressione um. Para falar com um operador, pressione dois.",
    ],
}

SENTENCE_LABELS = ["short", "medium", "long"]


@dataclass
class ChunkMetric:
    index: int
    audio_duration_ms: float
    latency_ms: float
    cumulative_audio_ms: float
    cumulative_wall_ms: float


@dataclass
class RunResult:
    mode: str  # "efficient" or "full"
    language: str
    sentence_label: str
    text: str
    fcl_ms: float = 0.0
    total_audio_s: float = 0.0
    total_wall_s: float = 0.0
    rtf: float = 0.0
    n_chunks: int = 0
    mean_chunk_ms: float = 0.0
    median_chunk_ms: float = 0.0
    p95_chunk_ms: float = 0.0
    chunk_latencies: List[float] = field(default_factory=list)


def run_benchmark(
    model,
    text: str,
    language: str,
    ref_audio: str,
    efficient: bool,
    warmup: bool = False,
) -> RunResult:
    """Run a single streaming benchmark."""
    from chatterbox.streaming import ChatterboxStreamingTTS

    streamer = ChatterboxStreamingTTS(
        model,
        chunk_tokens=25,
        min_initial_tokens=15,
        output_sample_rate=16000,
        adaptive_chunking=True,
        efficient_streaming=efficient,
        cfm_context_frames=30,
    )

    chunks = []
    chunk_latencies = []
    t_wall_start = time.perf_counter()
    first_chunk_time = None

    for i, chunk in enumerate(streamer.generate_stream(
        text,
        audio_prompt_path=ref_audio,
        language_id=language,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
    )):
        now = time.perf_counter()
        latency = now - t_wall_start
        if first_chunk_time is None:
            first_chunk_time = latency
        chunk_latencies.append((now - t_wall_start) * 1000)
        chunks.append(chunk)
        t_wall_start = now

    if warmup:
        return RunResult(mode="warmup", language=language, sentence_label="warmup", text=text)

    total_audio = np.concatenate(chunks) if chunks else np.array([])
    sr = streamer.sample_rate
    total_audio_s = len(total_audio) / sr
    total_wall_s = sum(chunk_latencies) / 1000

    return RunResult(
        mode="efficient" if efficient else "full",
        language=language,
        sentence_label="",
        text=text[:60],
        fcl_ms=first_chunk_time * 1000 if first_chunk_time else 0,
        total_audio_s=total_audio_s,
        total_wall_s=total_wall_s,
        rtf=total_wall_s / total_audio_s if total_audio_s > 0 else 0,
        n_chunks=len(chunks),
        mean_chunk_ms=float(np.mean(chunk_latencies)) if chunk_latencies else 0,
        median_chunk_ms=float(np.median(chunk_latencies)) if chunk_latencies else 0,
        p95_chunk_ms=float(np.percentile(chunk_latencies, 95)) if chunk_latencies else 0,
        chunk_latencies=chunk_latencies,
    )


def print_comparison(results: List[RunResult]):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 100)
    print(f"{'STREAMING BENCHMARK RESULTS':^100}")
    print("=" * 100)

    # Group by language + sentence
    groups = {}
    for r in results:
        key = (r.language, r.sentence_label)
        groups.setdefault(key, {})[r.mode] = r

    header = (
        f"{'Lang':>4} {'Size':>6} │ "
        f"{'FCL(eff)':>8} {'FCL(full)':>9} {'Δ':>6} │ "
        f"{'RTF(eff)':>8} {'RTF(full)':>9} {'Δ':>6} │ "
        f"{'Mean(eff)':>9} {'Mean(full)':>10} {'Speedup':>8}"
    )
    print(header)
    print("─" * 100)

    for (lang, label), modes in sorted(groups.items()):
        eff = modes.get("efficient")
        full = modes.get("full")
        if not eff or not full:
            continue

        fcl_delta = ((eff.fcl_ms - full.fcl_ms) / full.fcl_ms * 100) if full.fcl_ms > 0 else 0
        rtf_delta = ((eff.rtf - full.rtf) / full.rtf * 100) if full.rtf > 0 else 0
        mean_speedup = full.mean_chunk_ms / eff.mean_chunk_ms if eff.mean_chunk_ms > 0 else 0

        print(
            f"{lang:>4} {label:>6} │ "
            f"{eff.fcl_ms:>7.0f}ms {full.fcl_ms:>8.0f}ms {fcl_delta:>+5.0f}% │ "
            f"{eff.rtf:>8.2f}x {full.rtf:>9.2f}x {rtf_delta:>+5.0f}% │ "
            f"{eff.mean_chunk_ms:>8.0f}ms {full.mean_chunk_ms:>9.0f}ms {mean_speedup:>7.1f}x"
        )

    print("=" * 100)

    # Summary
    eff_results = [r for r in results if r.mode == "efficient"]
    full_results = [r for r in results if r.mode == "full"]
    if eff_results and full_results:
        print(f"\nOverall averages:")
        print(f"  Efficient: FCL={np.mean([r.fcl_ms for r in eff_results]):.0f}ms, "
              f"RTF={np.mean([r.rtf for r in eff_results]):.2f}x, "
              f"Mean chunk={np.mean([r.mean_chunk_ms for r in eff_results]):.0f}ms")
        print(f"  Full:      FCL={np.mean([r.fcl_ms for r in full_results]):.0f}ms, "
              f"RTF={np.mean([r.rtf for r in full_results]):.2f}x, "
              f"Mean chunk={np.mean([r.mean_chunk_ms for r in full_results]):.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Streaming performance benchmark")
    parser.add_argument("--ref", required=True, help="Reference audio for voice cloning")
    parser.add_argument("--languages", nargs="+", default=["it", "en"],
                        help="Languages to benchmark (default: it en)")
    parser.add_argument("--sentences", nargs="+", default=["short", "medium", "long"],
                        choices=SENTENCE_LABELS, help="Sentence lengths to test")
    parser.add_argument("--runs", type=int, default=1, help="Runs per config (for averaging)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", help="Save results to JSON file")
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip full reprocess benchmark (only test efficient)")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    from chatterbox import ChatterboxMultilingualTTS
    from chatterbox.cuda_optimizations import optimize_for_cuda

    model = ChatterboxMultilingualTTS.from_pretrained(args.device)
    optimize_for_cuda(model, compile_mode="default", use_bf16=True)

    # Warmup
    print("Warmup (torch.compile tracing)...")
    run_benchmark(model, "Warmup.", "en", args.ref, efficient=True, warmup=True)
    run_benchmark(model, "Warmup.", "en", args.ref, efficient=False, warmup=True)
    torch.cuda.synchronize()

    # Benchmark
    results = []
    modes = [True] if args.skip_full else [True, False]
    mode_names = {True: "efficient", False: "full"}

    total_runs = len(args.languages) * len(args.sentences) * len(modes) * args.runs
    run_idx = 0

    for lang in args.languages:
        if lang not in TEST_SENTENCES:
            print(f"Skipping {lang} — no test sentences")
            continue

        for sent_idx, label in enumerate(SENTENCE_LABELS):
            if label not in args.sentences:
                continue

            text = TEST_SENTENCES[lang][sent_idx]

            for efficient in modes:
                for run in range(args.runs):
                    run_idx += 1
                    mode = mode_names[efficient]
                    print(f"[{run_idx}/{total_runs}] {lang}/{label}/{mode} (run {run+1})")

                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    result = run_benchmark(model, text, lang, args.ref, efficient=efficient)
                    result.sentence_label = label
                    results.append(result)

                    print(f"  FCL={result.fcl_ms:.0f}ms RTF={result.rtf:.2f}x "
                          f"mean_chunk={result.mean_chunk_ms:.0f}ms chunks={result.n_chunks}")

    # Print comparison
    print_comparison(results)

    # Save JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
