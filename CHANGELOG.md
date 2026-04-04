# ChatterBox NG — Changelog

## v0.3.0 — Production Real-Time Streaming on L4 (2026-04-04)

> Ottimizzazione completa per telefonia italiana real-time su NVIDIA L4.
> Streaming audio ora **identico** al monolitico `generate()`.

### Meanflow S3Gen (5x CFM speedup)
- Pesi turbo S3Gen (`s3gen_meanflow.safetensors`) caricabili nel modello multilingue
- 2 ODE steps (vs 10), niente CFG batch doubling — **~5x speedup sulla parte CFM**
- `ChatterboxMultilingualTTS.from_pretrained("cuda", meanflow=True)`

### CUDA L4 Optimizations
- `optimize_for_cuda()` — setup one-call: BF16, torch.compile, SDPA, TF32, cuDNN benchmark
- `warmup_model()` — pre-trigger JIT compilation prima della prima request
- Flash/MemEfficient SDPA per encoder attention

### Streaming Quality Fix (CRITICO)
- **Full reprocess pipeline** — ogni chunk esegue encoder + CFM completi sull'intera sequenza accumulata
- Audio streaming ora **identico** al monolitico `generate()`
- HiFiGAN cache per continuità audio senza click/pop

### Streaming Resampler 24kHz → 16kHz
- `StreamingResampler` con `scipy.resample_poly` — bit-exact con offline
- Zero artefatti ai bordi dei chunk
- `output_sample_rate=16000` per integrazione Asterisk

### TensorRT Export (opt-in)
- `trt_export.py` — ONNX export per HiFiGAN e CFM estimator
- `trt_runtime.py` — wrappers drop-in TRT/ORT con fallback automatico
- `pip install chatterbox-ng[tensorrt]`

### WebSocket Server Aggiornato
- `examples/realtime_tts_server.py` con flag `--meanflow`, `--output-sr`, `--tensorrt`
- Client HTML integrato

### Bug Fix
- **fix: prompt_feat conditioning** — il decoder CFM riceveva `cond=zeros` invece del mel del reference audio, perdendo completamente l'identità vocale
- **fix: ODE steps streaming** — step ridotti (4 vs 10) causavano distorsione grave; streaming ora usa sempre gli step completi del monolitico

### Deprecati e Rimossi
I seguenti parametri sono **ignorati** — causano tutti degradazione audio:
- `streaming_cfm_steps` — usa sempre step completi
- `use_cfm_windowing` — freeze corrompe consistenza ODE
- `use_kv_cache` — encoder bidirezionale produce K/V stale

### Come Usare

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.streaming import ChatterboxStreamingTTS
from chatterbox.cuda_optimizations import optimize_for_cuda, warmup_model

# Carica con meanflow
model = ChatterboxMultilingualTTS.from_pretrained("cuda", meanflow=True)
optimize_for_cuda(model)
model.prepare_conditionals("voce_agente.wav")
warmup_model(model, device="cuda")

# Streaming a 16kHz per Asterisk
streamer = ChatterboxStreamingTTS(model, chunk_tokens=25, output_sample_rate=16000)
for chunk in streamer.generate_stream(
    text="Buongiorno, la sua pratica è stata approvata.",
    language_id="it", exaggeration=0.5, cfg_weight=0.5,
):
    asterisk_channel.write(chunk)
```

### Configurazioni Performance

| Scenario | Setup |
|----------|-------|
| Massima qualità | `meanflow=False`, `cfg_weight=0.7` |
| Bilanciato | `meanflow=True`, `cfg_weight=0.5` |
| Massima velocità | `meanflow=True` + TensorRT |
| Telefonia 16kHz | `output_sample_rate=16000` |

---

## v0.2.0 — Italian Support, True Streaming & Audio Post-Processing (2026-04-02)

> First release as **ChatterBox NG** (Next Generation), fork of ChatterBox by Resemble AI.
> Distribution name: `chatterbox-ng` | Python module: `import chatterbox` (unchanged)

### Italian Language Support

- **Text normalization pipeline** (`italian_text_normalize()`) with 11-step processing:
  - 26 abbreviazioni italiane: `dott.` → "dottore", `sig.ra` → "signora", `prof.ssa` → "professoressa", etc.
  - Numeri cardinali e ordinali: `42` → "quarantadue", `1°` → "primo", `3ª` → "terza"
  - Decimali con virgola: `3,14` → "tre virgola quattordici"
  - Date numeriche e scritte: `15/03/2024` → "quindici marzo duemilaventiquattro"
  - Orari: `14:30` → "le quattordici e trenta"
  - Numeri di telefono: `+39 02 1234567` → gruppi naturali separati
  - Valute: `100€` → "cento euro", `1$` → "un dollaro"
  - Sigle intelligenti: NATO → letta come parola, PIL → "pi i elle" (spelling)
  - Simboli: `€`, `%`, `&`, `@`, `«»` → testo parlato
- **Prosody normalization** per intonazione naturale italiana:
  - Tag questions: `, vero.` → `, vero?` (intonazione interrogativa automatica)
  - Ellissi, em-dash, punteggiatura ripetuta normalizzate
  - Spaziatura post-virgola per ritmo naturale
- **Punteggiatura italiana** in `punc_norm()`: guillemets `«»`, testo vuoto in italiano

### True Streaming TTS

- **`ChatterboxStreamingTTS`**: orchestratore streaming che genera audio in tempo reale
  - Compatibile con tutti i modelli: Standard, Multilingual, Turbo
  - `generate_stream()` → yield di chunk audio numpy a 24kHz
  - Chunk configurabili: `chunk_tokens` (default 25, ~1s audio) e `min_initial_tokens` (default 15)
  - `get_full_watermarked()` per watermark Perth post-streaming
- **T3 streaming**: `inference_streaming()` e `inference_turbo_streaming()` — yield token per token con KV-cache
- **S3Gen streaming**: `StreamState` dataclass + `streaming_step()` con:
  - Growing-sequence mel con `finalize=False` (6 mel frame stabili trimmate)
  - HiFiGAN `cache_source` per continuita waveform tra chunk
  - Trim fade solo sul primo chunk per ridurre spillover reference
- **Sentence pipelining**: `sentence_pipelining=True` divide il testo in frasi, ognuna processata indipendentemente da T3, audio continuo via HiFiGAN cache condivisa

### Audio Post-Processing

- **`audio_processing.py`** — modulo post-processing broadcast-quality:
  - `lufs_normalize()`: normalizzazione loudness a -16 LUFS (standard broadcast)
  - `de_ess()`: riduzione sibilanti con analisi FFT per-frame, threshold e riduzione configurabili
  - `match_room_tone()`: shaping spettrale per match acustico con audio reference
  - `post_process()`: pipeline combinata (de-ess → room tone → LUFS)

### Gradio Streaming App

- **`gradio_streaming_app.py`**: interfaccia Gradio con streaming real-time
  - Selezione modello (standard / multilingual / turbo) e lingua (23 lingue)
  - Controlli: exaggeration, CFG, temperature, seed, min_p, top_p, repetition penalty
  - Opzioni streaming: chunk size, sentence pipelining
  - Post-processing: toggle on/off, target LUFS configurabile

### WebSocket / SSE Server

- **`server_streaming.py`**: server ASGI per integrazione web
  - `ws://host/ws/tts` — WebSocket: chunk audio PCM float32 binari
  - `http://host/sse/tts` — SSE: chunk audio base64 con eventi `meta`, `audio`, `done`
  - `/health` — health check con stato modello
  - Supporto preload modello, model type selezionabile

### Test Suite

- **65 test** in `tests/`:
  - `test_italian_normalization.py`: 34 test per abbreviazioni, numeri, date, orari, telefoni, valute, sigle, simboli, prosodia, edge cases
  - `test_audio_processing.py`: 15 test per de-ess, room tone matching, LUFS normalization, pipeline
  - `test_streaming.py`: 16 test per sentence splitting, import verification

### Benchmarking

- **`benchmark_tts.py`**: strumento di benchmarking completo
  - First-chunk latency, RTF (Real-Time Factor), total time
  - Suite con testi EN/IT (short, medium, long)
  - Warmup + statistiche: mean, std, min, max, p50, p95
  - Output JSON per tracking nel tempo
  - Modalita streaming e sincrona

### Dependencies

- Aggiunto `num2words>=0.5.13` per conversione numeri italiani
- `pyloudnorm` gia presente per LUFS normalization
- `starlette` + `uvicorn` opzionali per il server streaming
