"""
Test suite for G2P pipeline.

Tests:
1. Custom dictionary lookups
2. Foreign word detection
3. IPA → respelling conversion (when espeak-ng available)
4. Pipeline integration (normalize → G2P → output)
5. Passthrough of native words (no unnecessary respelling)
6. Edge cases: punctuation, numbers, mixed text

Run: python -m pytest tests/test_g2p.py -v
"""
import sys
import os
import importlib.util

# Direct module import to avoid loading heavy deps from chatterbox.__init__
_g2p_path = os.path.join(os.path.dirname(__file__), "..", "src", "chatterbox", "g2p.py")
_spec = importlib.util.spec_from_file_location("chatterbox.g2p", _g2p_path)
_g2p_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g2p_mod)

import pytest

G2PPipeline = _g2p_mod.G2PPipeline
CustomDictionary = _g2p_mod.CustomDictionary
_tokenize_for_g2p = _g2p_mod._tokenize_for_g2p
_FOREIGN_PATTERNS = _g2p_mod._FOREIGN_PATTERNS
_HAS_PHONEMIZER = _g2p_mod._HAS_PHONEMIZER


# ============================================================================
# Tokenizer tests
# ============================================================================

class TestTokenizer:
    def test_simple_sentence(self):
        tokens = _tokenize_for_g2p("Il sig. Schmidt ha chiamato")
        assert "".join(tokens) == "Il sig. Schmidt ha chiamato"
        assert "Schmidt" in tokens

    def test_punctuation_preserved(self):
        tokens = _tokenize_for_g2p("Ciao, mondo!")
        assert "".join(tokens) == "Ciao, mondo!"

    def test_numbers_preserved(self):
        tokens = _tokenize_for_g2p("Ho 42 anni")
        assert "".join(tokens) == "Ho 42 anni"

    def test_empty_string(self):
        tokens = _tokenize_for_g2p("")
        assert tokens == [] or "".join(tokens) == ""

    def test_accented_chars(self):
        tokens = _tokenize_for_g2p("perché così è")
        # Accented words should be kept as single tokens
        word_tokens = [t for t in tokens if t.strip() and t[0].isalpha()]
        assert "perch" in word_tokens or "perché" in word_tokens  # depends on unicode normalization


# ============================================================================
# Custom Dictionary tests
# ============================================================================

class TestCustomDictionary:
    def test_add_and_lookup(self):
        d = CustomDictionary()
        d.add("Schmidt", "shmit", language_id="it")
        assert d.lookup("Schmidt", "it") == "shmit"
        assert d.lookup("Schmidt", "fr") is None  # Not in French dict

    def test_global_entry(self):
        d = CustomDictionary()
        d.add("IBAN", "i ban")  # No language_id = global
        assert d.lookup("IBAN", "it") == "i ban"
        assert d.lookup("IBAN", "fr") == "i ban"
        assert d.lookup("IBAN", "de") == "i ban"

    def test_language_specific_overrides_global(self):
        d = CustomDictionary()
        d.add("ok", "o kei")  # global
        d.add("ok", "occhei", language_id="it")  # Italian-specific
        assert d.lookup("ok", "it") == "occhei"  # Italian wins
        assert d.lookup("ok", "fr") == "o kei"   # Falls back to global

    def test_case_insensitive(self):
        d = CustomDictionary()
        d.add("SEPA", "sepa", language_id="it")
        assert d.lookup("sepa", "it") == "sepa"
        assert d.lookup("Sepa", "it") == "sepa"
        assert d.lookup("SEPA", "it") == "sepa"

    def test_lookup_miss(self):
        d = CustomDictionary()
        assert d.lookup("nonexistent", "it") is None

    def test_remove_language_specific(self):
        d = CustomDictionary()
        d.add("test", "tèst", language_id="it")
        assert d.remove("test", language_id="it") is True
        assert d.lookup("test", "it") is None

    def test_remove_global(self):
        d = CustomDictionary()
        d.add("IBAN", "i ban")
        assert d.remove("IBAN") is True
        assert d.lookup("IBAN", "it") is None

    def test_remove_nonexistent(self):
        d = CustomDictionary()
        assert d.remove("ghost", language_id="it") is False
        assert d.remove("ghost") is False

    def test_list_entries_all(self):
        d = CustomDictionary()
        d.add("IBAN", "i ban")
        d.add("Schmidt", "shmit", language_id="it")
        d.add("Müller", "miuller", language_id="de")
        result = d.list_entries()
        assert "global" in result
        assert result["global"]["iban"] == "i ban"
        assert result["it"]["schmidt"] == "shmit"
        assert result["de"]["müller"] == "miuller"

    def test_list_entries_by_language(self):
        d = CustomDictionary()
        d.add("IBAN", "i ban")
        d.add("Schmidt", "shmit", language_id="it")
        result = d.list_entries(language_id="it")
        assert "it" in result
        assert "global" in result
        assert result["it"]["schmidt"] == "shmit"
        assert result["global"]["iban"] == "i ban"
        # German shouldn't be present
        assert "de" not in result


# ============================================================================
# Foreign Word Detection tests
# ============================================================================

class TestForeignWordDetection:
    def setup_method(self):
        self.g2p = G2PPipeline()

    def test_italian_detects_german_names(self):
        assert self.g2p._is_foreign_word("Schmidt", "it")
        assert self.g2p._is_foreign_word("Schwartz", "it")

    def test_italian_detects_english_patterns(self):
        assert self.g2p._is_foreign_word("through", "it")
        assert self.g2p._is_foreign_word("thought", "it")

    def test_italian_passthrough_common_loanwords(self):
        assert not self.g2p._is_foreign_word("computer", "it")
        assert not self.g2p._is_foreign_word("software", "it")
        assert not self.g2p._is_foreign_word("internet", "it")

    def test_italian_passthrough_native_words(self):
        assert not self.g2p._is_foreign_word("chiamato", "it")
        assert not self.g2p._is_foreign_word("buongiorno", "it")
        assert not self.g2p._is_foreign_word("pratica", "it")

    def test_short_words_passthrough(self):
        assert not self.g2p._is_foreign_word("il", "it")
        assert not self.g2p._is_foreign_word("le", "fr")
        assert not self.g2p._is_foreign_word("ok", "it")

    def test_french_detects_english(self):
        assert self.g2p._is_foreign_word("through", "fr")

    def test_german_detects_english(self):
        assert self.g2p._is_foreign_word("thought", "de")

    def test_english_detects_german(self):
        assert self.g2p._is_foreign_word("Schäfer", "en")


# ============================================================================
# Pipeline Integration tests (dictionary-only, no espeak needed)
# ============================================================================

class TestPipelineDictionaryOnly:
    def setup_method(self):
        self.dict = CustomDictionary()
        self.dict.add("Schmidt", "shmit", language_id="it")
        self.dict.add("IBAN", "i ban")
        self.dict.add("CVV", "ci vu vu", language_id="it")
        self.dict.add("McDonald", "mecdonald", language_id="it")
        self.dict.add("Schmidt", "chmitt", language_id="de")
        self.g2p = G2PPipeline(custom_dict=self.dict)

    def test_dictionary_replacement_in_sentence(self):
        result = self.g2p.process("Il signor Schmidt ha chiamato", lang="it")
        assert "shmit" in result
        assert "signor" in result  # native word untouched
        assert "chiamato" in result  # native word untouched

    def test_dictionary_language_specific(self):
        result_it = self.g2p.process("Schmidt", lang="it")
        result_de = self.g2p.process("Schmidt", lang="de")
        assert "shmit" in result_it
        assert "chmitt" in result_de

    def test_global_dictionary_entry(self):
        result = self.g2p.process("Il codice IBAN", lang="it")
        assert "i ban" in result

    def test_punctuation_preserved(self):
        result = self.g2p.process("Ciao, Schmidt!", lang="it")
        assert "shmit" in result
        assert "," in result
        assert "!" in result

    def test_native_text_untouched(self):
        text = "Buongiorno, la sua pratica è stata approvata."
        result = self.g2p.process(text, lang="it")
        assert result == text  # Nothing should change

    def test_unsupported_language_passthrough(self):
        text = "Hello world"
        result = self.g2p.process(text, lang="ja")
        assert result == text

    def test_mixed_native_and_foreign(self):
        result = self.g2p.process("Il signor McDonald ha il CVV", lang="it")
        assert "mecdonald" in result
        assert "ci vu vu" in result
        assert "signor" in result


# ============================================================================
# espeak-ng Respelling tests (SKIPPED if espeak-ng not installed)
# ============================================================================

@pytest.mark.skipif(not _HAS_PHONEMIZER, reason="phonemizer/espeak-ng not installed")
class TestEspeakRespelling:
    def setup_method(self):
        self.g2p = G2PPipeline()

    def test_respell_german_name_to_italian(self):
        result = self.g2p.respell("Schmidt", source_lang="de", target_lang="it")
        assert result is not None
        assert len(result) > 0
        print(f"  Schmidt (DE→IT): {result}")

    def test_respell_english_to_italian(self):
        result = self.g2p.respell("thought", source_lang="en", target_lang="it")
        assert result is not None
        print(f"  thought (EN→IT): {result}")

    def test_respell_french_to_italian(self):
        result = self.g2p.respell("Beaujolais", source_lang="fr", target_lang="it")
        assert result is not None
        print(f"  Beaujolais (FR→IT): {result}")

    def test_pipeline_auto_respell(self):
        """Foreign words should be auto-detected and respelled."""
        result = self.g2p.process("Il signor Schwartz ha chiamato", lang="it")
        # Schwartz should be respelled (detected as foreign)
        assert "Schwartz" not in result or result != "Il signor Schwartz ha chiamato"
        print(f"  Schwartz in IT: {result}")

    def test_pipeline_preserves_native(self):
        """Native Italian words should NOT be respelled."""
        text = "La pratica è stata approvata dal direttore."
        result = self.g2p.process(text, lang="it")
        assert result == text  # Nothing should change


# ============================================================================
# Full Pipeline tests (normalize + G2P)
# ============================================================================

class TestFullPipeline:
    """Test the complete flow: text → normalize → G2P → output."""

    def setup_method(self):
        self.dict = CustomDictionary()
        self.dict.add("IBAN", "i ban")
        self.dict.add("SEPA", "sepa")
        self.dict.add("Unicredit", "unikrèdit", language_id="it")
        self.g2p = G2PPipeline(custom_dict=self.dict)

    def test_italian_banking_sentence(self):
        # After euro_text_normalizers, numbers are already words
        text = "il codice IBAN per il bonifico SEPA a Unicredit"
        result = self.g2p.process(text, lang="it")
        assert "i ban" in result
        assert "sepa" in result
        assert "unikrèdit" in result

    def test_passthrough_already_normalized(self):
        # Numbers should already be normalized by euro_text_normalizers
        text = "quattrocentoventicinquemila euro"
        result = self.g2p.process(text, lang="it")
        assert result == text  # No change needed

    def test_french_sentence(self):
        text = "le compte IBAN de monsieur Dupont"
        result = self.g2p.process(text, lang="fr")
        assert "i ban" in result  # Global dict entry
        assert "monsieur" in result  # Native word preserved

    def test_german_sentence(self):
        text = "die IBAN nummer"
        result = self.g2p.process(text, lang="de")
        assert "i ban" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
