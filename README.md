# ChatterBox NG — Multilingual Streaming TTS

**ChatterBox NG** is a multilingual, real-time streaming text-to-speech system built on [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox).

Optimized for production telephony (Asterisk, WebSocket) with voice cloning in 23+ languages.

## Features

- **23 languages** — Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese
- **Real-time streaming** — adaptive chunking with ~200ms first-chunk latency on L4
- **Voice cloning** — zero-shot from 5-10s reference audio
- **Text normalization** — numbers, dates, currency, abbreviations → spoken words (IT/FR/DE/ES/PT/EN)
- **Voice humanizer** — natural breathing sounds inserted in real-time
- **16kHz output** — telephony-ready (Asterisk/G.711), with native 24kHz option
- **CUDA optimized** — BF16, torch.compile, Flash Attention, TF32

## Installation

```shell
git clone https://github.com/user/chatterbox-ng.git
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

### WebSocket Server

```bash
python examples/realtime_tts_server.py --device cuda --meanflow --output-sr 16000
```

## Configuration

### Adaptive Chunking

```python
# Default: adaptive ON, schedule (5, 10, 20, 25) — best for real-time
streamer = ChatterboxStreamingTTS(model)

# More aggressive first chunk
streamer = ChatterboxStreamingTTS(model, adaptive_schedule=(3, 8, 15, 25))

# Fixed chunk sizes (legacy)
streamer = ChatterboxStreamingTTS(model, adaptive_chunking=False, min_initial_tokens=15)
```

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

### Performance Presets

| Scenario | Setup |
|----------|-------|
| Max quality | `meanflow=False`, `cfg_weight=0.7` |
| Balanced (default) | `meanflow=True`, `adaptive_chunking=True` |
| Max speed | `meanflow=True` + TensorRT |
| Lowest latency | `adaptive_schedule=(3, 8, 15, 25)` |

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

Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)

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
