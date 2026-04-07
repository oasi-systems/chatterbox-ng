from ..llama_configs import LLAMA_CONFIGS


# Original BPE vocab size (pretrained multilingual)
BPE_VOCAB_SIZE = 2454


class T3Config:
    def __init__(self, text_tokens_dict_size=704, phoneme_mode=False):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

        # Phoneme embedding support
        self.phoneme_mode = phoneme_mode
        self.bpe_vocab_size = BPE_VOCAB_SIZE

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size >= BPE_VOCAB_SIZE

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls):
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=BPE_VOCAB_SIZE)

    @classmethod
    def multilingual_phoneme(cls):
        """Create configuration for multilingual TTS with phoneme tokens.

        Extends the BPE vocab (2454) with IPA phoneme tokens.
        Original pretrained weights map to IDs 0-2453.
        Phoneme tokens start at ID 2454.
        """
        from chatterbox.phoneme_tokens import get_all_new_tokens
        n_new = len(get_all_new_tokens())
        return cls(
            text_tokens_dict_size=BPE_VOCAB_SIZE + n_new,
            phoneme_mode=True,
        )
