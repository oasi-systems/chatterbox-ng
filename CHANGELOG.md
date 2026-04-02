# ChatterBox NG — Changelog

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
