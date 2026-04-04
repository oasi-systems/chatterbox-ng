"""
ChatterBox NG — Real-time TTS Streaming Server

WebSocket server che trasmette audio in tempo reale mentre il modello genera.
Il client riceve chunk audio PCM appena disponibili — zero attesa per la frase completa.

Architettura:
    Client WebSocket → {"text": "...", "voice": "sara"}
    Server → chunk PCM 24kHz int16 in tempo reale
    Server → {"done": true, "duration": 4.2} al termine

Produzione:
    - Preload voci al boot (prepare_conditionals una sola volta)
    - Streaming chunk-by-chunk via WebSocket (latenza primo chunk ~1-2s su L4)
    - Watermark applicato in background dopo lo stream
    - Health check + metriche Prometheus-ready

Usage:
    # Server
    python realtime_tts_server.py --voices voices/ --device cuda --port 8765

    # Client (qualsiasi linguaggio)
    wscat -c ws://localhost:8765/tts
    > {"text": "Buonanotte, sono Sara.", "voice": "sara", "language": "it"}
    < [binary PCM chunks...]
    < {"done": true, "duration": 3.2, "chunks": 5}

    # voices/ directory:
    #   sara.wav
    #   marco.wav
    #   lella.wav
"""
import asyncio
import json
import logging
import struct
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("tts-server")


class VoiceRegistry:
    """Pre-loaded voice profiles for zero-latency voice switching."""

    def __init__(self, model, voices_dir: str, exaggeration: float = 0.5):
        self.model = model
        self.voices: Dict[str, object] = {}
        self.exaggeration = exaggeration

        voices_path = Path(voices_dir)
        if not voices_path.exists():
            logger.warning(f"Voices directory {voices_dir} not found")
            return

        for wav_file in sorted(voices_path.glob("*.wav")):
            name = wav_file.stem.lower()
            logger.info(f"Loading voice: {name} ({wav_file})")
            model.prepare_conditionals(str(wav_file), exaggeration=exaggeration)
            self.voices[name] = model.conds
            logger.info(f"  → {name} ready")

        logger.info(f"Loaded {len(self.voices)} voices: {list(self.voices.keys())}")

    def activate(self, voice_name: str) -> bool:
        """Switch active voice. Returns False if voice not found."""
        conds = self.voices.get(voice_name.lower())
        if conds is None:
            return False
        self.model.conds = conds
        return True


class TTSEngine:
    """Wraps model + streaming for concurrent request handling."""

    def __init__(self, device: str = "cuda", voices_dir: str = "voices/",
                 meanflow: bool = False, output_sr: int = 24000,
                 trt_engine_dir: str = None):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from chatterbox.streaming import ChatterboxStreamingTTS

        logger.info(f"Loading model on {device} (meanflow={meanflow}, output_sr={output_sr})...")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device, meanflow=meanflow)
        self.output_sr = output_sr
        self.device = device

        # CUDA optimizations (with optional TensorRT)
        if "cuda" in str(device):
            from chatterbox.cuda_optimizations import optimize_for_cuda, warmup_model
            optimize_for_cuda(
                self.model,
                use_tensorrt=trt_engine_dir is not None,
                trt_engine_dir=trt_engine_dir,
            )
            logger.info("CUDA optimizations applied")
            self._warmup_fn = lambda: warmup_model(self.model, device=device)
        else:
            self._warmup_fn = None

        self.StreamingTTS = ChatterboxStreamingTTS
        self.voice_registry = VoiceRegistry(self.model, voices_dir)
        self.sample_rate = output_sr

        # Warmup compiled kernels after voices are loaded
        if self._warmup_fn and self.voice_registry.voices:
            # Activate first voice for warmup
            first_voice = next(iter(self.voice_registry.voices))
            self.voice_registry.activate(first_voice)
            self._warmup_fn()
            self._warmup_fn = None

        # Serialization lock — model is not thread-safe
        self._lock = asyncio.Lock()

        # Metrics
        self.total_requests = 0
        self.total_audio_seconds = 0.0
        self.total_latency_seconds = 0.0

    async def stream_tts(self, text: str, voice: str = None,
                         language: str = "it", on_chunk=None):
        """Generate TTS and call on_chunk(pcm_bytes) for each audio chunk.

        Args:
            text: input text
            voice: voice name (must be pre-loaded)
            language: language code
            on_chunk: async callback receiving raw PCM int16 bytes

        Returns:
            dict with generation metrics
        """
        async with self._lock:
            self.total_requests += 1

            # Activate voice
            if voice:
                if not self.voice_registry.activate(voice):
                    raise ValueError(f"Voice '{voice}' not found. Available: {list(self.voice_registry.voices.keys())}")
            elif not self.model.conds:
                raise ValueError("No voice active. Provide 'voice' or call with audio_prompt_path.")

            streamer = self.StreamingTTS(
                self.model,
                chunk_tokens=25,
                min_initial_tokens=15,
                output_sample_rate=self.sample_rate if self.sample_rate != 24000 else None,
            )

            t_start = time.time()
            first_chunk_time = None
            n_chunks = 0
            total_samples = 0

            # Run generation in thread pool (blocking torch ops)
            gen = streamer.generate_stream(
                text=text,
                language_id=language,
            )

            # Iterate chunks — yield to event loop between chunks
            loop = asyncio.get_event_loop()
            chunks_iter = iter(gen)

            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, chunks_iter)
                except StopIteration:
                    break

                if first_chunk_time is None:
                    first_chunk_time = time.time() - t_start

                # Convert float32 → int16 PCM
                pcm = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
                n_chunks += 1
                total_samples += len(pcm)

                if on_chunk:
                    await on_chunk(pcm.tobytes())

            duration = total_samples / self.sample_rate
            wall_time = time.time() - t_start
            self.total_audio_seconds += duration
            self.total_latency_seconds += wall_time

            return {
                "done": True,
                "duration": round(duration, 2),
                "wall_time": round(wall_time, 2),
                "first_chunk_ms": round(first_chunk_time * 1000) if first_chunk_time else 0,
                "chunks": n_chunks,
                "rtf": round(wall_time / duration, 2) if duration > 0 else 0,
                "voice": voice,
                "language": language,
            }


async def websocket_handler(websocket, engine: TTSEngine):
    """Handle a single WebSocket connection."""
    remote = websocket.remote_address
    logger.info(f"Client connected: {remote}")

    try:
        async for message in websocket:
            try:
                req = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
                continue

            text = req.get("text", "").strip()
            if not text:
                await websocket.send(json.dumps({"error": "Missing 'text'"}))
                continue

            voice = req.get("voice")
            language = req.get("language", "it")

            logger.info(f"[{remote}] Generating: voice={voice}, lang={language}, text={text[:60]}...")

            try:
                async def send_chunk(pcm_bytes):
                    await websocket.send(pcm_bytes)

                metrics = await engine.stream_tts(
                    text=text,
                    voice=voice,
                    language=language,
                    on_chunk=send_chunk,
                )

                # Send completion message
                await websocket.send(json.dumps(metrics))
                logger.info(f"[{remote}] Done: {metrics['duration']}s audio, "
                           f"first_chunk={metrics['first_chunk_ms']}ms, "
                           f"RTF={metrics['rtf']}x")

            except ValueError as e:
                await websocket.send(json.dumps({"error": str(e)}))
            except Exception as e:
                logger.exception(f"[{remote}] Generation error")
                await websocket.send(json.dumps({"error": f"Generation failed: {e}"}))

    except Exception:
        pass
    finally:
        logger.info(f"Client disconnected: {remote}")


async def health_handler(path, request_headers):
    """HTTP health check on /health."""
    if path == "/health":
        return (200, [], b'{"status": "ok"}\n')
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ChatterBox NG Real-time TTS Server")
    parser.add_argument("--voices", default="voices/", help="Directory with voice .wav files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--meanflow", action="store_true",
                        help="Use meanflow S3Gen (2 ODE steps, ~5x faster CFM)")
    parser.add_argument("--output-sr", type=int, default=24000,
                        help="Output sample rate (default: 24000, use 16000 for Asterisk)")
    parser.add_argument("--tensorrt", default=None, metavar="DIR",
                        help="Directory with TRT/ONNX engines (from python -m chatterbox.trt_export)")
    args = parser.parse_args()

    engine = TTSEngine(device=args.device, voices_dir=args.voices,
                       meanflow=args.meanflow, output_sr=args.output_sr,
                       trt_engine_dir=args.tensorrt)

    async def run():
        try:
            import websockets
        except ImportError:
            print("pip install websockets")
            return

        async with websockets.serve(
            lambda ws: websocket_handler(ws, engine),
            args.host,
            args.port,
            process_request=health_handler,
            max_size=2**20,  # 1MB max message
            ping_interval=30,
            ping_timeout=10,
        ):
            logger.info(f"TTS Server ready on ws://{args.host}:{args.port}/")
            logger.info(f"Health check: http://{args.host}:{args.port}/health")
            logger.info(f"Voices: {list(engine.voice_registry.voices.keys())}")
            await asyncio.Future()  # run forever

    asyncio.run(run())


if __name__ == "__main__":
    main()
