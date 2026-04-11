# ChatterBox NG — Multilingual Streaming TTS

**ChatterBox NG** is a multilingual, real-time streaming text-to-speech system built on [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox).

Optimized for production telephony (Asterisk, WebSocket) with voice cloning in 23+ languages.

## Features

- **23 languages** — Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese
- **Real-time streaming** — adaptive chunking with ~173ms first-chunk latency on L4 (RTF 0.67x)
- **Voice cloning** — zero-shot from 5-10s reference audio
- **SSML support** — `<break>`, `<emphasis>`, `<prosody>`, `<say-as>`, `<phoneme>` with full normalization
- **Text normalization** — numbers, dates, currency, abbreviations for IT/FR/DE/ES/PT/EN
- **G2P pipeline** — foreign word respelling via espeak-ng + custom dictionaries
- **Custom dictionary API** — runtime pronunciation CRUD via REST
- **Concurrent requests** — asyncio.Lock serialized GPU access, thread pool offload
- **Voice humanizer** — natural breathing sounds inserted in real-time
- **16kHz output** — telephony-ready (Asterisk/G.711), with native 24kHz option
- **CUDA optimized** — BF16, Flash Attention/SDPA, TF32, CFM decoder warmup

## Installation

```shell
git clone https://github.com/scalcerano/chatterbox-ng.git
cd chatterbox-ng
pip install -e .
```

## Quick Start

### Streaming (Real-Time)

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.streaming import ChatterboxStreamingTTS
from chatterbox.cuda_optimizations import optimize_for_cuda, warmup_model

# Load model
model = ChatterboxMultilingualTTS.from_pretrained("cuda", meanflow=True)
optimize_for_cuda(model)
model.prepare_conditionals("agent_voice.wav")
warmup_model(model, device="cuda")

# Stream at 16kHz (default) — ready for Asterisk
streamer = ChatterboxStreamingTTS(model)

for chunk in streamer.generate_stream(
    text="Buongiorno, la informo che la sua pratica è stata approvata.",
    language_id="it",
):
    send_to_asterisk(chunk)  # numpy float32 array at 16kHz
```

### SSML (Telephony IVR)

SSML is auto-detected — just send markup in the `text` field:

```python
ssml = """
<speak>
    <prosody rate="95%">
        Buongiorno, la informo che la sua pratica
        <say-as interpret-as="characters">ABC</say-as>
        <say-as interpret-as="number">12345</say-as>
        è stata approvata.
    </prosody>
    <break time="500ms"/>
    <emphasis level="strong">
        Il pagamento è previsto entro il
        <say-as interpret-as="date" format="dmy">15/03/2024</say-as>.
    </emphasis>
</speak>
"""

for chunk in streamer.generate_stream(text=ssml, language_id="it"):
    send_to_asterisk(chunk)
```

**Supported SSML tags:**

| Tag | Effect | Example |
|-----|--------|---------|
| `<break time="500ms"/>` | Insert silence | `<break time="1.5s"/>` |
| `<emphasis level="strong">` | Expressiveness (0.2-0.8) | `strong`, `moderate`, `reduced` |
| `<prosody rate="90%">` | Speaking rate | `slow`, `fast`, `x-slow`, `95%` |
| `<say-as interpret-as="date" format="dmy">` | Date normalization | `15/03/2024` → "quindici marzo..." |
| `<say-as interpret-as="currency">` | Currency | `€1250` → "milleduecentocinquanta euro" |
| `<say-as interpret-as="number">` | Number | `12345` → "dodicimilatrecentoquarantacinque" |
| `<say-as interpret-as="characters">` | Spell out | `ABC` → "A B C" |
| `<say-as interpret-as="telephone">` | Phone digits | `+39 02 1234` → "3 9 0 2 1 2 3 4" |
| `<say-as interpret-as="time">` | Time | `14:30` → "quattordici e trenta" |
| `<say-as interpret-as="ordinal">` | Ordinal | `5` → "quinto" |
| `<phoneme ph="ʃmɪt">Schmidt</phoneme>` | IPA pronunciation | Converts to orthographic respelling |
| `<sub alias="...">` | Text substitution | `<sub alias="OMS">WHO</sub>` |
| `<p>`, `<s>` | Paragraph/sentence | Auto-breaks (600ms / 300ms) |

### Custom Dictionary

```python
from chatterbox.g2p import CustomDictionary, G2PPipeline

# Create dictionary
dictionary = CustomDictionary()
dictionary.load_yaml("dictionaries/italian_telephony.yaml", language_id="it")
dictionary.add("Schmidt", "shmit", language_id="it")

# Use in G2P pipeline
g2p = G2PPipeline(custom_dict=dictionary)
text = g2p.process("Il sig. Schmidt ha chiamato", lang="it")
# → "Il sig. shmit ha chiamato"
```

### Streaming Server

```bash
# Start server with model preloading and custom dictionaries
python server_streaming.py --preload --dict dictionaries/italian_telephony.yaml --dict-lang it
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws/tts` | WebSocket | Bidirectional streaming (binary PCM chunks) |
| `/sse/tts` | GET | Server-Sent Events (base64 chunks) |
| `/api/dictionary` | GET/POST/DELETE | Custom dictionary CRUD |
| `/health` | GET | Server status, request queue, features |

**WebSocket protocol (`/ws/tts`):**

1. Client sends JSON with text + parameters
2. Server replies JSON `{"status": "generating", "sample_rate": 16000}`
3. Server streams binary frames (raw float32 PCM at `output_sample_rate`)
4. Server sends JSON `{"status": "done"}`

```python
import asyncio, json, struct
import websockets
import numpy as np

async def synthesize(text, language="it", ref_audio_b64=None):
    async with websockets.connect("ws://localhost:8765/ws/tts") as ws:
        # 1. Send request
        await ws.send(json.dumps({
            "text": text,
            "language_id": language,
            "audio_prompt_b64": ref_audio_b64,  # base64-encoded WAV (optional)
            # Optional parameters (with defaults):
            # "temperature": 0.8,
            # "repetition_penalty": 1.2,
            # "exaggeration": 0.5,
            # "cfg_weight": 0.5,
            # "output_sample_rate": 16000,
            # "chunk_tokens": 25,
            # "sentence_pipelining": false,
        }))

        # 2. Receive chunks
        audio_chunks = []
        async for message in ws:
            if isinstance(message, bytes):
                # Binary frame = float32 PCM audio chunk
                chunk = np.frombuffer(message, dtype=np.float32)
                audio_chunks.append(chunk)
            else:
                status = json.loads(message)
                if status.get("status") == "done":
                    break

        return np.concatenate(audio_chunks)  # full audio at output_sample_rate
```

**Dictionary API examples:**

```bash
# Add entry
curl -X POST http://localhost:8765/api/dictionary \
     -H 'Content-Type: application/json' \
     -d '{"word": "IBAN", "respelling": "i ban", "language_id": "it"}'

# List entries
curl http://localhost:8765/api/dictionary?language_id=it

# Batch add
curl -X POST http://localhost:8765/api/dictionary \
     -H 'Content-Type: application/json' \
     -d '{"entries": [{"word": "SEPA", "respelling": "sepa"}, {"word": "CVV", "respelling": "ci vu vu", "language_id": "it"}]}'

# Load YAML
curl -X POST http://localhost:8765/api/dictionary \
     -H 'Content-Type: application/json' \
     -d '{"yaml_path": "/path/to/dict.yaml", "language_id": "it"}'

# Remove entry
curl -X DELETE http://localhost:8765/api/dictionary \
     -H 'Content-Type: application/json' \
     -d '{"word": "IBAN", "language_id": "it"}'
```

### Monolithic Generation (no streaming)

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained("cuda", meanflow=True)

wav = model.generate(
    "The total invoice amount is $425,000, payment due by March 15th.",
    language_id="en",
    audio_prompt_path="voice.wav",
)
# wav is a tensor at 24kHz
```

### Voice Humanizer

Adds natural breathing sounds between sentences in real-time during streaming.

```python
from chatterbox.humanizer import VoiceHumanizer

humanizer = VoiceHumanizer.from_reference("agent_voice.wav")
streamer = ChatterboxStreamingTTS(model, humanizer=humanizer)

for chunk in streamer.generate_stream(text="...", language_id="it"):
    send_to_asterisk(chunk)  # breaths already inserted
```

## Configuration

### Quality Tuning

| Parameter | Default | Notes |
|-----------|---------|-------|
| `output_sample_rate` | 16000 | 16kHz for telephony. Use 24000 for native quality |
| `exaggeration` | 0.5 | Voice expressiveness. Call center: 0.3-0.5 |
| `cfg_weight` | 0.5 | Voice timbre fidelity. For faithful cloning: 0.5-0.7 |
| `chunk_tokens` | 25 | Max tokens per chunk (after adaptive ramp-up) |
| `adaptive_chunking` | True | Progressive chunk sizes for low FCL |
| `adaptive_schedule` | (5,10,20,25) | Token counts per chunk |
| Reference audio | — | 5-10s of clean speech. Affects timbre, NOT speed |

### Performance (L4 GPU, meanflow=True)

| Metric | Value |
|--------|-------|
| TTFA (first audio) | ~173ms |
| RTF (real-time factor) | 0.67x |
| T3 token speed | ~17ms/tok (60+ tok/s) |
| Latency spikes | Zero (CFM decoder pre-warmed) |

### Performance Presets

| Scenario | Setup |
|----------|-------|
| Max quality | `meanflow=False`, `cfg_weight=0.7` |
| Balanced (default) | `meanflow=True`, `cfg_weight=0.5` |
| Lowest latency | `meanflow=True`, `adaptive_schedule=(3, 8, 15, 25)` |

## Text Normalization

Built-in normalizers for 6 European languages convert raw text to spoken form:

| Input | Language | Output |
|-------|----------|--------|
| `425.000 euro` | IT | `quattrocentoventicinquemila euro` |
| `$425,000` | EN | `four hundred and twenty-five thousand dollars` |
| `15/03/2024` | FR | `quinze mars deux mille vingt-quatre` |
| `14:30 Uhr` | DE | `vierzehn Uhr dreißig` |

Numbers, dates, times, currency, ordinals, abbreviations are all handled automatically.

## Supported Languages

Arabic (ar) · Danish (da) · German (de) · Greek (el) · English (en) · Spanish (es) · Finnish (fi) · French (fr) · Hebrew (he) · Hindi (hi) · Italian (it) · Japanese (ja) · Korean (ko) · Malay (ms) · Dutch (nl) · Norwegian (no) · Polish (pl) · Portuguese (pt) · Russian (ru) · Swedish (sv) · Swahili (sw) · Turkish (tr) · Chinese (zh)

## Built-in PerTh Watermarking

Every audio file includes [Resemble AI's Perth](https://github.com/resemble-ai/perth) imperceptible neural watermarks.

```python
import perth, librosa

audio, sr = librosa.load("output.wav", sr=None)
watermarker = perth.PerthImplicitWatermarker()
watermark = watermarker.get_watermark(audio, sample_rate=sr)
print(f"Watermark: {watermark}")  # 0.0 (none) or 1.0 (present)
```

## Acknowledgements

Based on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by [Resemble AI](https://resemble.ai).

- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

## Disclaimer

Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
