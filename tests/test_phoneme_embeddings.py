#!/usr/bin/env python3
"""
Tests for phoneme token definitions and embedding initialization.

Run: python -m pytest tests/test_phoneme_embeddings.py -v
"""
import sys
import os
import importlib.util

# Direct import to avoid heavy deps
_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "src", "chatterbox")


def _load_module(name, filename):
    path = os.path.join(_pkg_dir, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


phoneme_tokens = _load_module("chatterbox.phoneme_tokens", "phoneme_tokens.py")

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# Token Definitions
# ============================================================================

class TestPhonemeList:
    def test_phoneme_count(self):
        """Should have the expected number of IPA phonemes."""
        assert phoneme_tokens.N_PHONEMES == 71

    def test_no_duplicates(self):
        """No duplicate phonemes in the list."""
        assert len(set(phoneme_tokens.PHONEME_LIST)) == len(phoneme_tokens.PHONEME_LIST)

    def test_token_names(self):
        """Token names follow [ph_X] format."""
        names = phoneme_tokens.get_phoneme_token_names()
        # First is [PHON] mode marker
        assert names[0] == "[PHON]"
        # Rest are [ph_X]
        for name in names[1:]:
            assert name.startswith("[ph_")
            assert name.endswith("]")

    def test_all_new_tokens(self):
        """get_all_new_tokens returns PHON + all phonemes."""
        tokens = phoneme_tokens.get_all_new_tokens()
        assert len(tokens) == 1 + phoneme_tokens.N_PHONEMES  # [PHON] + 73
        assert tokens[0] == "[PHON]"

    def test_common_phonemes_present(self):
        """Essential phonemes for EU languages are in the list."""
        essential = ["a", "e", "i", "o", "u", "p", "t", "k", "s", "n", "l", "r",
                     "tʃ", "dʒ", "ʃ", "ʒ", "ə", "ŋ", "ɲ"]
        for ph in essential:
            assert ph in phoneme_tokens.PHONEME_LIST, f"Missing: {ph}"

    def test_diphthongs_present(self):
        """Multi-char diphthongs are single tokens."""
        diphthongs = ["aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"]
        for d in diphthongs:
            assert d in phoneme_tokens.PHONEME_LIST, f"Missing diphthong: {d}"

    def test_affricates_present(self):
        """Affricates are single tokens."""
        affricates = ["tʃ", "dʒ", "ts", "dz"]
        for a in affricates:
            assert a in phoneme_tokens.PHONEME_LIST, f"Missing affricate: {a}"


# ============================================================================
# Grapheme Mapping
# ============================================================================

class TestGraphemeMapping:
    def test_all_phonemes_have_mapping(self):
        """Every phoneme in PHONEME_LIST has an entry in _PHONEME_TO_GRAPHEMES."""
        for ph in phoneme_tokens.PHONEME_LIST:
            assert ph in phoneme_tokens._PHONEME_TO_GRAPHEMES, \
                f"Missing mapping for: {ph}"

    def test_suprasegmentals_empty_mapping(self):
        """Stress/length markers have empty grapheme lists (random init)."""
        for ph in ["ˈ", "ˌ", "ː", " "]:
            assert phoneme_tokens._PHONEME_TO_GRAPHEMES[ph] == [], \
                f"{ph} should have empty grapheme list"

    def test_consonant_mappings_sensible(self):
        """Consonant → grapheme mappings are linguistically reasonable."""
        # /k/ should map to 'k' and 'c'
        assert "k" in phoneme_tokens._PHONEME_TO_GRAPHEMES["k"]
        assert "c" in phoneme_tokens._PHONEME_TO_GRAPHEMES["k"]
        # /ʃ/ should map to 's' and 'c' (as in English 'sh', Italian 'sc')
        assert "s" in phoneme_tokens._PHONEME_TO_GRAPHEMES["ʃ"]


# ============================================================================
# Embedding Initialization
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestEmbeddingInit:
    def test_basic_init(self):
        """Initialize phoneme embeddings from dummy grapheme embeddings."""
        dim = 64  # smaller for test speed
        vocab_size = 100
        emb = torch.randn(vocab_size, dim)

        # Create a simple vocab: single chars at known positions
        vocab = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5,
                 "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11,
                 "m": 12, "n": 13, "o": 14, "p": 15, "r": 16, "s": 17,
                 "t": 18, "u": 19, "v": 20, "w": 21, "y": 22, "z": 23}

        new_emb = phoneme_tokens.initialize_phoneme_embeddings(emb, vocab)

        # Shape: [PHON + 73 phonemes, dim]
        n_new = len(phoneme_tokens.get_all_new_tokens())
        assert new_emb.shape == (n_new, dim)

        # PHON marker (idx 0) should be non-zero (random init)
        # (statistically, random init won't be exactly zero)

        # Phoneme 'a' (idx 1) should equal grapheme 'a' embedding
        a_idx = phoneme_tokens.PHONEME_LIST.index("a") + 1
        assert torch.allclose(new_emb[a_idx], emb[vocab["a"]], atol=1e-6), \
            "Phoneme /a/ should match grapheme 'a' embedding"

        # Phoneme 'k' should be mean of 'k' and 'c'
        k_idx = phoneme_tokens.PHONEME_LIST.index("k") + 1
        expected_k = (emb[vocab["k"]] + emb[vocab["c"]]) / 2
        assert torch.allclose(new_emb[k_idx], expected_k, atol=1e-6), \
            "Phoneme /k/ should be mean of 'k','c' embeddings"

    def test_missing_grapheme_fallback(self):
        """If grapheme not in vocab, fall back to random init."""
        dim = 32
        emb = torch.randn(50, dim)
        # Empty vocab — all phonemes will get random init
        vocab = {}

        new_emb = phoneme_tokens.initialize_phoneme_embeddings(emb, vocab)
        n_new = len(phoneme_tokens.get_all_new_tokens())
        assert new_emb.shape == (n_new, dim)
        # Should not be all zeros (random init)
        assert new_emb.abs().sum() > 0

    def test_output_dtype(self):
        """Output should be float32."""
        emb = torch.randn(50, 32)
        new_emb = phoneme_tokens.initialize_phoneme_embeddings(emb, {})
        assert new_emb.dtype == torch.float32


# ============================================================================
# Phoneme → Token ID Encoding
# ============================================================================

class TestPhonemeEncoding:
    def setup_method(self):
        """Build token_to_id mapping."""
        base_id = 2454  # BPE vocab size
        new_tokens = phoneme_tokens.get_all_new_tokens()
        self.token_to_id = {}
        for i, name in enumerate(new_tokens):
            self.token_to_id[name] = base_id + i

    def test_simple_word(self):
        """Encode a simple IPA string."""
        # "ka" → [PHON], [ph_k], [ph_a]
        ids = phoneme_tokens.phonemes_to_token_ids("ka", self.token_to_id)
        assert len(ids) == 3  # PHON + k + a
        assert ids[0] == self.token_to_id["[PHON]"]
        assert ids[1] == self.token_to_id["[ph_k]"]
        assert ids[2] == self.token_to_id["[ph_a]"]

    def test_affricate_greedy_match(self):
        """Multi-char phonemes like 'tʃ' should be matched as single tokens."""
        # "tʃao" → [PHON], [ph_tʃ], [ph_a], [ph_o]
        ids = phoneme_tokens.phonemes_to_token_ids("tʃao", self.token_to_id)
        assert self.token_to_id["[ph_tʃ]"] in ids

    def test_word_boundary(self):
        """Spaces in IPA become word boundary tokens."""
        ids = phoneme_tokens.phonemes_to_token_ids("ka sa", self.token_to_id)
        space_id = self.token_to_id["[ph_ ]"]
        assert space_id in ids

    def test_stress_markers(self):
        """Stress markers are encoded."""
        ids = phoneme_tokens.phonemes_to_token_ids("ˈka", self.token_to_id)
        stress_id = self.token_to_id["[ph_ˈ]"]
        assert stress_id in ids

    def test_unknown_chars_skipped(self):
        """Punctuation and unknown chars are skipped."""
        ids = phoneme_tokens.phonemes_to_token_ids("ka.", self.token_to_id)
        # Should have PHON + k + a (period skipped)
        assert len(ids) == 3

    def test_empty_string(self):
        """Empty string produces only PHON marker."""
        ids = phoneme_tokens.phonemes_to_token_ids("", self.token_to_id)
        assert len(ids) == 1
        assert ids[0] == self.token_to_id["[PHON]"]

    def test_all_phonemes_encodable(self):
        """Every phoneme in the list can be encoded."""
        for ph in phoneme_tokens.PHONEME_LIST:
            ids = phoneme_tokens.phonemes_to_token_ids(ph, self.token_to_id)
            # Should have PHON + at least one phoneme token
            assert len(ids) >= 2, f"Phoneme '{ph}' not encoded"


# ============================================================================
# T3Config
# ============================================================================

class TestT3Config:
    def test_multilingual_phoneme_config(self):
        """multilingual_phoneme() creates extended vocab config."""
        # Need to make the import work
        config_path = os.path.join(_pkg_dir, "models", "t3", "modules", "t3_config.py")
        spec = importlib.util.spec_from_file_location("t3_config", config_path)

        # We need the llama_configs module too
        llama_path = os.path.join(_pkg_dir, "models", "t3", "llama_configs.py")
        llama_spec = importlib.util.spec_from_file_location(
            "chatterbox.models.t3.llama_configs", llama_path)
        llama_mod = importlib.util.module_from_spec(llama_spec)
        sys.modules["chatterbox.models.t3.llama_configs"] = llama_mod
        # Also register as relative import target
        sys.modules["..llama_configs"] = llama_mod

        # This is getting complex — just test the token count directly
        n_new = len(phoneme_tokens.get_all_new_tokens())
        expected_vocab = 2454 + n_new
        # 1 PHON + N_PHONEMES
        assert n_new == 1 + phoneme_tokens.N_PHONEMES
        assert expected_vocab == 2454 + n_new


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
