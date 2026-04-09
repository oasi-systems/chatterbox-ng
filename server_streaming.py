"""
WebSocket and SSE streaming server for ChatterBox TTS.

Provides endpoints for real-time audio streaming and dictionary management:
- WebSocket: /ws/tts — bidirectional, sends raw audio chunks as binary frames
- SSE: /sse/tts — server-sent events with base64-encoded audio chunks
- REST: /api/dictionary — CRUD for custom pronunciation dictionaries

Concurrency model:
- asyncio.Lock serializes GPU inference (one request at a time)
- Sync generators run in thread pool (don't block event loop)
- Per-request conditionals snapshot prevents voice identity corruption
- Queue tracks active/waiting requests for observability

Usage:
    python server_streaming.py [--host 0.0.0.0] [--port 8765] [--model multilingual]

Client examples:

    WebSocket (JavaScript):
        const ws = new WebSocket('ws://localhost:8765/ws/tts');
        ws.onopen = () => ws.send(JSON.stringify({
            text: "Ciao mondo!",
            language_id: "it",
            audio_prompt_b64: "<base64 wav>",
        }));
        ws.onmessage = (e) => {
            if (e.data instanceof Blob) {
                // Raw PCM float32 audio chunk at 24kHz
                playAudioChunk(e.data);
            } else {
                // JSON status message
                console.log(JSON.parse(e.data));
            }
        };

    SSE (JavaScript):
        const params = new URLSearchParams({text: "Ciao!", language_id: "it"});
        const es = new EventSource(`/sse/tts?${params}`);
        es.addEventListener('audio', (e) => {
            const chunk = base64ToFloat32Array(e.data);
            playAudioChunk(chunk);
        });
        es.addEventListener('done', () => es.close());

    Dictionary (curl):
        # Add entry
        curl -X POST http://localhost:8765/api/dictionary \
             -H 'Content-Type: application/json' \
             -d '{"word": "IBAN", "respelling": "i ban", "language_id": "it"}'

        # List entries
        curl http://localhost:8765/api/dictionary?language_id=it

        # Remove entry
        curl -X DELETE http://localhost:8765/api/dictionary \
             -H 'Content-Type: application/json' \
             -d '{"word": "IBAN", "language_id": "it"}'
"""
import argparse
import asyncio
import base64
import json
import logging
import tempfile
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_app(model_type: str = "multilingual"):
    """Create the ASGI application with all endpoints.

    Returns:
        (app, model_loader) tuple. Call model_loader() to preload the model.
    """
    try:
        from starlette.applications import Starlette
        from starlette.routing import Route, WebSocketRoute
        from starlette.requests import Request
        from starlette.responses import StreamingResponse, JSONResponse
        from starlette.websockets import WebSocket
    except ImportError:
        raise ImportError(
            "starlette is required for the streaming server. "
            "Install with: pip install starlette uvicorn"
        )

    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    SAMPLE_RATE = 24000

    # --- Model holder with thread-safe initialization ---
    _model = {"instance": None}
    _model_lock = threading.Lock()

    def _get_model():
        if _model["instance"] is None:
            with _model_lock:
                if _model["instance"] is None:
                    _model["instance"] = _load_model(model_type, DEVICE)
        return _model["instance"]

    def _load_model(mtype, device):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from chatterbox.cuda_optimizations import optimize_for_cuda
        model = ChatterboxMultilingualTTS.from_pretrained(device)
        if device == "cuda" or (isinstance(device, str) and "cuda" in device):
            optimize_for_cuda(model, compile_mode="default", use_bf16=True)
        return model

    # --- Custom dictionary (shared via global G2P singleton) ---
    from chatterbox.g2p import CustomDictionary, configure_default_pipeline, get_default_pipeline
    _dictionary = CustomDictionary()
    configure_default_pipeline(custom_dict=_dictionary, auto_respell=True)
    _dict_lock = threading.Lock()

    # --- Inference serialization ---
    # Only one request can use the GPU model at a time.
    # asyncio.Lock ensures fair FIFO ordering in the event loop.
    _inference_lock = asyncio.Lock()
    _request_stats = {"active": 0, "queued": 0, "total": 0}

    def _prepare_prompt(audio_b64: Optional[str]) -> Optional[str]:
        """Decode base64 audio prompt to temp file, return path."""
        if not audio_b64:
            return None
        audio_bytes = base64.b64decode(audio_b64)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name

    def _generate_chunks_sync(params: dict):
        """Sync generator that yields audio chunks. Runs in thread pool.

        Supports SSML: if text contains <speak> or SSML tags, auto-detection
        in generate_stream() handles per-segment prosody, emphasis, breaks.
        """
        from chatterbox.streaming import ChatterboxStreamingTTS

        model = _get_model()
        prompt_path = _prepare_prompt(params.get("audio_prompt_b64"))

        streamer = ChatterboxStreamingTTS(
            model,
            chunk_tokens=params.get("chunk_tokens", 25),
            output_sample_rate=params.get("output_sample_rate", 16000),
            efficient_streaming=params.get("efficient_streaming", True),
            cfm_context_frames=params.get("cfm_context_frames", 30),
        )

        gen_kwargs = {
            "text": params["text"],
            "temperature": params.get("temperature", 0.8),
            "repetition_penalty": params.get("repetition_penalty", 1.2),
            "exaggeration": params.get("exaggeration", 0.5),
            "cfg_weight": params.get("cfg_weight", 0.5),
            "sentence_pipelining": params.get("sentence_pipelining", True),
        }
        if prompt_path:
            gen_kwargs["audio_prompt_path"] = prompt_path
        if params.get("language_id"):
            gen_kwargs["language_id"] = params["language_id"]

        for chunk in streamer.generate_stream(**gen_kwargs):
            yield chunk

    async def _generate_chunks_async(params: dict):
        """Async generator: serializes GPU access, streams chunks via queue.

        Flow:
        1. Acquire inference lock (queued requests wait here)
        2. Spawn sync generator in thread — pushes chunks to queue
        3. Async loop reads queue and yields chunks immediately (true streaming)
        4. Release lock when done

        This ensures TTFA = time to first chunk, NOT time to generate all audio.
        """
        import queue as queue_mod

        _request_stats["queued"] += 1
        async with _inference_lock:
            _request_stats["queued"] -= 1
            _request_stats["active"] += 1
            _request_stats["total"] += 1
            try:
                chunk_queue = queue_mod.Queue()
                _SENTINEL = object()

                def _producer():
                    try:
                        for chunk in _generate_chunks_sync(params):
                            chunk_queue.put(chunk)
                    except Exception as e:
                        chunk_queue.put(e)
                    finally:
                        chunk_queue.put(_SENTINEL)

                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, _producer)

                while True:
                    # Poll queue without blocking the event loop
                    while True:
                        try:
                            item = chunk_queue.get_nowait()
                            break
                        except queue_mod.Empty:
                            await asyncio.sleep(0.01)

                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                _request_stats["active"] -= 1

    # --- WebSocket endpoint ---
    async def ws_tts(websocket: WebSocket):
        await websocket.accept()
        try:
            data = await websocket.receive_text()
            params = json.loads(data)

            if "text" not in params:
                await websocket.send_text(json.dumps({"error": "Missing 'text' field"}))
                await websocket.close()
                return

            await websocket.send_text(json.dumps({
                "status": "generating",
                "sample_rate": SAMPLE_RATE,
                "queue_position": _request_stats["queued"],
            }))

            async for chunk in _generate_chunks_async(params):
                audio_bytes = chunk.astype(np.float32).tobytes()
                await websocket.send_bytes(audio_bytes)

            await websocket.send_text(json.dumps({"status": "done"}))

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    # --- SSE endpoint ---
    async def sse_tts(request: Request):
        params = dict(request.query_params)
        if "text" not in params:
            return JSONResponse({"error": "Missing 'text' parameter"}, status_code=400)

        # Parse numeric params
        for key in ("temperature", "repetition_penalty", "exaggeration", "cfg_weight"):
            if key in params:
                params[key] = float(params[key])
        for key in ("chunk_tokens",):
            if key in params:
                params[key] = int(params[key])
        if "sentence_pipelining" in params:
            params["sentence_pipelining"] = params["sentence_pipelining"].lower() in ("true", "1", "yes")

        async def event_generator():
            meta = json.dumps({"sample_rate": SAMPLE_RATE})
            yield f"event: meta\ndata: {meta}\n\n"

            async for chunk in _generate_chunks_async(params):
                audio_b64 = base64.b64encode(chunk.astype(np.float32).tobytes()).decode()
                yield f"event: audio\ndata: {audio_b64}\n\n"

            yield f"event: done\ndata: {{}}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # =====================================================================
    # Custom Dictionary REST API
    # =====================================================================

    async def dict_get(request: Request):
        """GET /api/dictionary — list entries.

        Query params:
            language_id: optional, filter by language
            word: optional, look up a specific word
        """
        language_id = request.query_params.get("language_id")
        word = request.query_params.get("word")

        if word and language_id:
            result = _dictionary.lookup(word, language_id)
            if result is None:
                return JSONResponse({"word": word, "found": False}, status_code=404)
            return JSONResponse({"word": word, "respelling": result, "language_id": language_id})

        entries = _dictionary.list_entries(language_id)
        return JSONResponse({"entries": entries})

    async def dict_post(request: Request):
        """POST /api/dictionary — add entry or batch entries.

        Body (single):
            {"word": "IBAN", "respelling": "i ban", "language_id": "it"}

        Body (batch):
            {"entries": [{"word": "...", "respelling": "...", "language_id": "..."}]}

        Body (load YAML):
            {"yaml_path": "/path/to/dict.yaml", "language_id": "it"}
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        # Load YAML file
        if "yaml_path" in body:
            yaml_path = body["yaml_path"]
            lang = body.get("language_id")
            with _dict_lock:
                _dictionary.load_yaml(yaml_path, language_id=lang)
            entries = _dictionary.list_entries(lang)
            return JSONResponse({"status": "loaded", "path": yaml_path, "entries": entries})

        # Batch add
        if "entries" in body:
            added = 0
            with _dict_lock:
                for entry in body["entries"]:
                    word = entry.get("word")
                    respelling = entry.get("respelling")
                    if not word or not respelling:
                        continue
                    _dictionary.add(word, respelling, language_id=entry.get("language_id"))
                    added += 1
            return JSONResponse({"status": "ok", "added": added})

        # Single add
        word = body.get("word")
        respelling = body.get("respelling")
        if not word or not respelling:
            return JSONResponse(
                {"error": "Missing 'word' and/or 'respelling' field"}, status_code=400
            )
        lang = body.get("language_id")
        with _dict_lock:
            _dictionary.add(word, respelling, language_id=lang)
        return JSONResponse({"status": "ok", "word": word, "respelling": respelling, "language_id": lang})

    async def dict_delete(request: Request):
        """DELETE /api/dictionary — remove entry.

        Body: {"word": "IBAN", "language_id": "it"}
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        word = body.get("word")
        if not word:
            return JSONResponse({"error": "Missing 'word' field"}, status_code=400)
        lang = body.get("language_id")
        with _dict_lock:
            removed = _dictionary.remove(word, language_id=lang)
        if not removed:
            return JSONResponse({"error": "Entry not found", "word": word}, status_code=404)
        return JSONResponse({"status": "ok", "removed": word, "language_id": lang})

    async def dict_handler(request: Request):
        """Route /api/dictionary to GET/POST/DELETE based on HTTP method."""
        if request.method == "GET":
            return await dict_get(request)
        elif request.method == "POST":
            return await dict_post(request)
        elif request.method == "DELETE":
            return await dict_delete(request)
        return JSONResponse({"error": "Method not allowed"}, status_code=405)

    # --- Health check ---
    async def health(request: Request):
        dict_entries = _dictionary.list_entries()
        total_entries = sum(len(v) for v in dict_entries.values())
        return JSONResponse({
            "status": "ok",
            "model_type": model_type,
            "model_loaded": _model["instance"] is not None,
            "sample_rate": SAMPLE_RATE,
            "features": {
                "ssml": True,
                "efficient_streaming": True,
                "custom_dictionary": True,
                "concurrent_requests": True,
                "languages": ["it", "en", "fr", "de", "es", "pt"],
            },
            "requests": dict(_request_stats),
            "dictionary_entries": total_entries,
        })

    app = Starlette(
        routes=[
            WebSocketRoute("/ws/tts", ws_tts),
            Route("/sse/tts", sse_tts),
            Route("/api/dictionary", dict_handler, methods=["GET", "POST", "DELETE"]),
            Route("/health", health),
        ],
    )

    return app, lambda: _get_model()


def main():
    parser = argparse.ArgumentParser(description="ChatterBox Streaming TTS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--model", default="multilingual",
                        choices=["standard", "multilingual", "turbo"],
                        help="Model type to load")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")
    parser.add_argument("--dict", nargs="*", metavar="YAML_PATH",
                        help="Load dictionary YAML files on startup")
    parser.add_argument("--dict-lang", default=None,
                        help="Language ID for --dict files (default: global)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting ChatterBox Streaming Server on {args.host}:{args.port}")
    logger.info(f"Model type: {args.model}")

    app, model_loader = create_app(args.model)

    # Load startup dictionaries into the global G2P singleton
    if args.dict:
        from chatterbox.g2p import get_default_pipeline
        pipeline = get_default_pipeline()
        for yaml_path in args.dict:
            pipeline.dictionary.load_yaml(yaml_path, language_id=args.dict_lang)
            logger.info(f"Loaded dictionary: {yaml_path}")

    if args.preload:
        logger.info("Preloading model...")
        model = model_loader()
        logger.info("Model loaded.")

    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install uvicorn"
        )


if __name__ == "__main__":
    main()
