"""
Gradio Streaming TTS App for ChatterBox.

Supports real-time streaming audio output, multilingual (including Italian),
audio post-processing, and sentence pipelining.
"""
import random
import logging
from typing import Optional

import numpy as np
import torch
import gradio as gr

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SAMPLE_RATE = 24000


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model(model_type: str = "multilingual"):
    """Load TTS model."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    return ChatterboxMultilingualTTS.from_pretrained(DEVICE)


def generate_streaming(
    model_state,
    model_type,
    text,
    audio_prompt_path,
    language_id,
    exaggeration,
    temperature,
    seed_num,
    cfg_weight,
    min_p,
    top_p,
    repetition_penalty,
    chunk_tokens,
    sentence_pipelining,
    post_process_enabled,
    target_lufs,
):
    """Generator function for streaming audio output."""
    from chatterbox.streaming import ChatterboxStreamingTTS

    # Load model if needed or if model type changed
    if model_state is None or model_state.get("type") != model_type:
        model = load_model(model_type)
        model_state = {"model": model, "type": model_type}
    else:
        model = model_state["model"]

    if seed_num != 0:
        set_seed(int(seed_num))

    streamer = ChatterboxStreamingTTS(
        model,
        chunk_tokens=int(chunk_tokens),
        min_initial_tokens=15,
    )

    # Determine language_id
    lang = language_id if language_id and language_id != "auto" else None
    if model_type != "multilingual":
        lang = None

    # Build generation kwargs
    gen_kwargs = dict(
        text=text,
        audio_prompt_path=audio_prompt_path,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        sentence_pipelining=sentence_pipelining,
    )
    if lang:
        gen_kwargs["language_id"] = lang

    # Stream audio chunks
    all_audio = []
    for chunk in streamer.generate_stream(**gen_kwargs):
        all_audio.append(chunk)
        # Yield cumulative audio for Gradio (it replaces the component each time)
        cumulative = np.concatenate(all_audio, axis=0)
        yield model_state, (SAMPLE_RATE, cumulative)

    # Post-processing on final audio
    if post_process_enabled and all_audio:
        from chatterbox.audio_processing import post_process
        full_audio = np.concatenate(all_audio, axis=0)

        # Load reference audio for room tone matching if available
        ref_audio = None
        if audio_prompt_path:
            try:
                import librosa
                ref_audio, _ = librosa.load(audio_prompt_path, sr=SAMPLE_RATE)
            except Exception:
                pass

        processed = post_process(
            full_audio,
            SAMPLE_RATE,
            reference=ref_audio,
            target_lufs=target_lufs,
        )
        yield model_state, (SAMPLE_RATE, processed)


# --- Language options ---
LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "it": "Italian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "he": "Hebrew",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "el": "Greek",
    "ms": "Malay",
    "sw": "Swahili",
}


with gr.Blocks(title="ChatterBox Streaming TTS", theme=gr.themes.Soft()) as demo:
    model_state = gr.State(None)

    gr.Markdown("# ChatterBox Streaming TTS")
    gr.Markdown("Real-time streaming text-to-speech with Italian support, post-processing, and sentence pipelining.")

    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                value="Buongiorno! Oggi il dott. Rossi ha comprato 42 libri per 100 euro. La NATO ha organizzato un incontro il 15 marzo 2024 alle 14:30.",
                label="Text to synthesize",
                max_lines=5,
                lines=3,
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio (voice to clone)",
            )

            with gr.Row():
                model_type = gr.Dropdown(
                    choices=["standard", "multilingual", "turbo"],
                    value="multilingual",
                    label="Model",
                )
                language_id = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="it",
                    label="Language",
                )

            with gr.Row():
                exaggeration = gr.Slider(0.25, 2.0, step=0.05, value=0.5, label="Exaggeration")
                cfg_weight = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="CFG/Pace")

            with gr.Accordion("Advanced Options", open=False):
                with gr.Row():
                    temperature = gr.Slider(0.05, 5.0, step=0.05, value=0.8, label="Temperature")
                    seed_num = gr.Number(value=0, label="Seed (0 = random)")
                with gr.Row():
                    min_p = gr.Slider(0.0, 1.0, step=0.01, value=0.05, label="min_p")
                    top_p = gr.Slider(0.0, 1.0, step=0.01, value=1.0, label="top_p")
                repetition_penalty = gr.Slider(1.0, 2.0, step=0.1, value=1.2, label="Repetition Penalty")

            with gr.Accordion("Streaming Options", open=False):
                chunk_tokens = gr.Slider(10, 50, step=5, value=25, label="Chunk size (tokens, ~40ms each)")
                sentence_pipelining = gr.Checkbox(value=True, label="Sentence pipelining (split text into sentences)")

            with gr.Accordion("Post-Processing", open=False):
                post_process_enabled = gr.Checkbox(value=True, label="Enable post-processing")
                target_lufs = gr.Slider(-30.0, -6.0, step=1.0, value=-16.0, label="Target Loudness (LUFS)")

            run_btn = gr.Button("Generate (Streaming)", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Output Audio", streaming=True)

    run_btn.click(
        fn=generate_streaming,
        inputs=[
            model_state, model_type, text, ref_wav, language_id,
            exaggeration, temperature, seed_num, cfg_weight,
            min_p, top_p, repetition_penalty,
            chunk_tokens, sentence_pipelining,
            post_process_enabled, target_lufs,
        ],
        outputs=[model_state, audio_output],
    )


if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=True)
