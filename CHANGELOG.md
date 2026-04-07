# ChatterBox NG — Changelog

## v0.4.0 — SSML, Dictionary API, Concurrent Requests, O(1) Streaming (2026-04-07)

> SSML completo per telefonia, API custom dictionary, request isolation,
> windowed CFM O(1), INT8 quantization. 107 test.

### SSML Fully Functional

- **`<say-as>`** ora normalizza direttamente via num2words:
  - `interpret-as="date"` con attributo `format` (dmy/mdy/ymd): `15/03/2024` → "quindici marzo duemilaventiquattro"
  - `interpret-as="currency"`: `€1250` → "milleduecentocinquanta euro"
  - `interpret-as="number"`: `12345` → "dodicimilatrecentoquarantacinque"
  - `interpret-as="ordinal"`: `5` → "quinto"
  - `interpret-as="time"`: `14:30` → "quattordici e trenta"
  - Tutte le 6 lingue EU (IT/EN/FR/DE/ES/PT) con nomi mesi e forme specifiche
- **`<phoneme>`** ora funziona: IPA → respelling ortografico via tabelle G2P
  - `<phoneme ph="ʃmɪt">Schmidt</phoneme>` → "scimit" (italiano)
  - Fallback al testo originale se conversione fallisce
- **`<emphasis>`**: `strong`=0.8, `moderate`=0.5, `reduced`=0.3 → exaggeration
- **`<prosody rate>`**: `slow`/`fast`/percentuale → cfg_weight
- **`<break>`**: silenzio in ms/s, `strength` attribute
- **`<p>`, `<s>`**: auto-break 600ms/300ms
- **Auto-detection**: nessun flag necessario, SSML rilevato automaticamente

### Custom Dictionary REST API

- `GET /api/dictionary` — lista entries (filtro per lingua)
- `POST /api/dictionary` — add singolo, batch, o load YAML
- `DELETE /api/dictionary` — rimuovi entry
- `CustomDictionary.remove()` e `list_entries()` aggiunti
- Flag CLI `--dict` per caricare YAML all'avvio

### Concurrent Requests (Request Isolation)

- **`asyncio.Lock`** serializza accesso GPU — FIFO, nessuna corruzione identità vocale
- **Thread pool offload** — generatori sync in `run_in_executor()`, event loop mai bloccato
- **Thread-safe model loading** — double-check locking su `_get_model()`
- **Request stats** — active/queued/total requests nel `/health` endpoint

### O(1) Windowed CFM Streaming

- `streaming_step_efficient()` — primo chunk: full CFM (identità vocale), successivi: CFM solo su [context + new] frames
- `decode_cfm_windowed()` — CFM UNet vede solo la finestra, non tutta la sequenza
- `efficient_streaming=True` (default), `cfm_context_frames=30`
- Costo per chunk costante indipendentemente dalla lunghezza totale

### INT8 Weight-Only Quantization

- `quantize_t3_int8(model)` — 3 backend (torchao → torch.ao → manual)
- `Int8WeightLinear` — per-channel symmetric INT8 con dequant a inference
- ~2x memory reduction su T3
- Flag CLI `--int8` nel server

### G2P Pipeline

- `G2PPipeline` con espeak-ng per 6 lingue EU
- `CustomDictionary` con priorità: dizionario > auto-respelling
- `ipa_to_respelling()` standalone function per SSML phoneme
- Tabelle IPA→ortografia per IT/EN/FR/DE/ES/PT
- Dizionari YAML per telefonia inclusi (`dictionaries/`)

### Phoneme Embeddings

- Token phonemici per 6 lingue EU
- Training LoRA T3 in corso (v4, MLS dataset)

### Text Normalization (6 EU Languages)

- IT/EN/FR/DE/ES/PT: numeri, date, orari, valute, ordinali, abbreviazioni, telefoni
- `normalize_text_for_language(text, lang)` dispatcher

### Benchmark Script

- `benchmarks/bench_streaming.py` — A/B efficient vs full, 6 lingue, FCL/RTF/p95
- Test sentences short/medium/long per lingua

### Test Suite: 107 test

- `test_ssml.py`: 42 test (SSML parsing, say-as normalization, phoneme IPA)
- `test_g2p.py`: 34 test (dictionary CRUD, foreign detection, tokenizer)
- `test_server_api.py`: 13 test (dictionary API, concurrency model, server structure)
- `test_phoneme_embeddings.py`: 18 test

---

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

### Adaptive Chunking (50-60% lower first-chunk latency)
- Progressive chunk sizes: 5→10→20→25 tokens (default, `adaptive_chunking=True`)
- First audio chunk arrives **~50-60% faster** than fixed chunking
- Benchmarked: 899ms vs 2204ms FCL on MPS, with equal or better perceived quality
- Customizable schedule via `adaptive_schedule` parameter

### Voice Humanizer (breathing)
- **`VoiceHumanizer`** — post-processor che aggiunge respiri naturali tra le frasi
- Template di respiri reali adattati al profilo spettrale del parlante (spectral transfer)
- Funziona con qualsiasi voce senza riestrarre campioni
- Inserisce respiri **solo** nei gap di silenzio reali (mai taglia speech)
- Salta gap dove il T3 ha già generato suoni naturali (RMS detection)
- Durata respiro proporzionale al parlato precedente
- 8 template verificati inclusi nel pacchetto (`breath_templates/`)
- `VoiceHumanizer.from_reference("voice.wav")` → pronto all'uso

### Streaming Quality Fix (CRITICO)
- **Full reprocess pipeline** — ogni chunk esegue encoder + CFM completi sull'intera sequenza accumulata
- Audio streaming ora **identico** al monolitico `generate()`
- HiFiGAN cache per continuità audio senza click/pop

### Streaming Resampler 24kHz → 16kHz (default)
- `StreamingResampler` con `scipy.resample_poly` — bit-exact con offline
- Zero artefatti ai bordi dei chunk
- **16kHz è il default** — pronto per telefonia/Asterisk senza configurazione
- `output_sample_rate=24000` per qualità nativa senza resampling

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
streamer = ChatterboxStreamingTTS(model)  # 16kHz default, adaptive chunking ON
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
| Qualità nativa 24kHz | `output_sample_rate=24000` |

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
