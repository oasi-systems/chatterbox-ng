"""
WebSocket and SSE streaming server for ChatterBox TTS.

Provides two endpoints for real-time audio streaming:
- WebSocket: /ws/tts — bidirectional, sends raw audio chunks as binary frames
- SSE: /sse/tts — server-sent events with base64-encoded audio chunks

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
"""
import argparse
import asyncio
import base64
import json
import logging
import struct
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_app(model_type: str = "multilingual"):
    """Create the ASGI application with WebSocket and SSE endpoints.

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

    # Model holder
    _model = {"instance": None}

    def _get_model():
        if _model["instance"] is None:
            _model["instance"] = _load_model(model_type, DEVICE)
        return _model["instance"]

    def _load_model(mtype, device):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS.from_pretrained(device)

    def _prepare_prompt(audio_b64: Optional[str], model) -> Optional[str]:
        """Decode base64 audio prompt to temp file, return path."""
        if not audio_b64:
            return None
        audio_bytes = base64.b64decode(audio_b64)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name

    def _generate_chunks(params: dict):
        """Generator that yields audio chunks as bytes."""
        from chatterbox.streaming import ChatterboxStreamingTTS

        model = _get_model()
        prompt_path = _prepare_prompt(params.get("audio_prompt_b64"), model)

        streamer = ChatterboxStreamingTTS(
            model,
            chunk_tokens=params.get("chunk_tokens", 25),
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

    # --- WebSocket endpoint ---
    async def ws_tts(websocket: WebSocket):
        await websocket.accept()
        try:
            # Receive request as JSON
            data = await websocket.receive_text()
            params = json.loads(data)

            if "text" not in params:
                await websocket.send_text(json.dumps({"error": "Missing 'text' field"}))
                await websocket.close()
                return

            await websocket.send_text(json.dumps({
                "status": "generating",
                "sample_rate": SAMPLE_RATE,
            }))

            # Stream audio chunks as binary frames
            loop = asyncio.get_event_loop()
            chunk_gen = _generate_chunks(params)

            for chunk in chunk_gen:
                # Send raw PCM float32 bytes
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
            # Send metadata
            meta = json.dumps({"sample_rate": SAMPLE_RATE})
            yield f"event: meta\ndata: {meta}\n\n"

            for chunk in _generate_chunks(params):
                # Base64-encode the float32 audio
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

    # --- Health check ---
    async def health(request: Request):
        return JSONResponse({
            "status": "ok",
            "model_type": model_type,
            "model_loaded": _model["instance"] is not None,
            "sample_rate": SAMPLE_RATE,
        })

    app = Starlette(
        routes=[
            WebSocketRoute("/ws/tts", ws_tts),
            Route("/sse/tts", sse_tts),
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting ChatterBox Streaming Server on {args.host}:{args.port}")
    logger.info(f"Model type: {args.model}")

    app, model_loader = create_app(args.model)
    if args.preload:
        logger.info("Preloading model...")
        model_loader()
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
