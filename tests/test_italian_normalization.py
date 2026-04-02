"""Tests for Italian text normalization in ChatterBox TTS."""
import re
import logging
import pytest

logger = logging.getLogger(__name__)

# Import the normalization code by reading the source file directly,
# to avoid triggering the full chatterbox package import chain.
# In CI with the full package installed, you can use:
#   from chatterbox.models.tokenizers.tokenizer import italian_text_normalize
import importlib.util
import os

_tokenizer_path = os.path.join(
    os.path.dirname(__file__), "..", "src", "chatterbox", "models", "tokenizers", "tokenizer.py"
)


def _load_normalize_func():
    """Load italian_text_normalize without importing the full package."""
    with open(os.path.abspath(_tokenizer_path)) as f:
        source = f.read()

    lines = source.split('\n')
    start = end = None
    for i, line in enumerate(lines):
        if '_ITALIAN_ABBREVIATIONS' in line and start is None:
            start = i
        if 'def add_russian_stress' in line:
            end = i
            break

    code = '\n'.join(lines[start:end])
    ns = {"re": re, "logger": logger}
    exec(code, ns)
    return ns['italian_text_normalize']


italian_text_normalize = _load_normalize_func()


# --- Abbreviations ---

class TestAbbreviations:
    def test_dottore(self):
        assert "dottore" in italian_text_normalize("Il dott. Rossi")

    def test_dottoressa(self):
        assert "dottoressa" in italian_text_normalize("La dott.ssa Bianchi")

    def test_professore(self):
        assert "professore" in italian_text_normalize("Il prof. Verdi")

    def test_signore(self):
        assert "signore" in italian_text_normalize("Il sig. Neri")

    def test_signora(self):
        assert "signora" in italian_text_normalize("La sig.ra Rossi")

    def test_avvocato(self):
        assert "avvocato" in italian_text_normalize("L'avv. Conti")

    def test_ingegnere(self):
        assert "ingegnere" in italian_text_normalize("L'ing. Ferri")

    def test_eccetera(self):
        assert "eccetera" in italian_text_normalize("pane, latte, ecc.")


# --- Numbers ---

class TestNumbers:
    def test_cardinal(self):
        result = italian_text_normalize("Ho 42 gatti")
        assert "quarantadue" in result

    def test_zero(self):
        result = italian_text_normalize("Valore 0")
        assert "zero" in result

    def test_large_number(self):
        result = italian_text_normalize("1000000 abitanti")
        assert "milione" in result.lower()

    def test_decimal_comma(self):
        result = italian_text_normalize("3,14 gradi")
        assert "tre virgola quattordici" in result

    def test_ordinal_masculine(self):
        result = italian_text_normalize("Il 1° piano")
        assert "primo" in result

    def test_ordinal_feminine(self):
        result = italian_text_normalize("La 3ª edizione")
        assert "terza" in result


# --- Dates ---

class TestDates:
    def test_date_slash(self):
        result = italian_text_normalize("15/03/2024")
        assert "quindici" in result
        assert "marzo" in result
        assert "duemilaventiquattro" in result

    def test_date_written(self):
        result = italian_text_normalize("il 15 marzo 2024")
        assert "quindici" in result
        assert "marzo" in result

    def test_first_day(self):
        result = italian_text_normalize("il 1 gennaio")
        assert "primo" in result
        assert "gennaio" in result

    def test_date_month_abbreviation(self):
        result = italian_text_normalize("il 25 dic 1990")
        assert "venticinque" in result
        assert "dicembre" in result


# --- Times ---

class TestTimes:
    def test_time_hhmm(self):
        result = italian_text_normalize("14:30")
        assert "quattordici" in result
        assert "trenta" in result

    def test_time_exact_hour(self):
        result = italian_text_normalize("9:00")
        assert "nove" in result
        assert "zero" not in result  # should say "le nove" not "le nove e zero"


# --- Phone Numbers ---

class TestPhoneNumbers:
    def test_italian_phone(self):
        result = italian_text_normalize("06 1234 5678")
        # Should be split into groups
        assert "sei" in result

    def test_international_phone(self):
        result = italian_text_normalize("+39 02 1234567")
        assert "trentanove" in result


# --- Currency ---

class TestCurrency:
    def test_euro_after(self):
        result = italian_text_normalize("100€")
        assert "cento euro" in result

    def test_euro_before(self):
        result = italian_text_normalize("€50")
        assert "cinquanta euro" in result

    def test_dollar_singular(self):
        result = italian_text_normalize("1$")
        assert "un dollaro" in result

    def test_dollar_plural(self):
        result = italian_text_normalize("5$")
        assert "cinque dollari" in result


# --- Acronyms ---

class TestAcronyms:
    def test_nato_word(self):
        # NATO should be read as a word
        result = italian_text_normalize("La NATO decide")
        assert "nato" in result.lower()

    def test_pil_spelled(self):
        # PIL should be spelled out
        result = italian_text_normalize("Il PIL cresce")
        assert "pi" in result.lower()  # P = pi
        assert "i" in result.lower()   # I = i
        assert "elle" in result.lower()  # L = elle


# --- Symbols ---

class TestSymbols:
    def test_percent(self):
        result = italian_text_normalize("3%")
        assert "percento" in result

    def test_ampersand(self):
        result = italian_text_normalize("pane & burro")
        assert " e " in result


# --- Prosody ---

class TestProsody:
    def test_tag_question_vero(self):
        result = italian_text_normalize("Sei contento, vero.")
        assert result.endswith("vero?")

    def test_tag_question_no(self):
        result = italian_text_normalize("Lo sai, no!")
        assert "no?" in result

    def test_ellipsis(self):
        result = italian_text_normalize("Aspetta... penso")
        assert "\u2026" in result  # Unicode ellipsis

    def test_em_dash(self):
        result = italian_text_normalize("Il tema \u2014 importante \u2014 resta")
        assert "\u2014" not in result  # em dash removed
        assert "importante" in result

    def test_repeated_exclamation(self):
        result = italian_text_normalize("Incredibile!!!")
        assert result.count("!") == 1

    def test_comma_spacing(self):
        result = italian_text_normalize("ciao,mondo")
        assert ", " in result


# --- Edge Cases ---

class TestEdgeCases:
    def test_empty_string(self):
        result = italian_text_normalize("")
        assert result == ""

    def test_no_special_content(self):
        result = italian_text_normalize("ciao mondo")
        assert result == "ciao mondo"

    def test_mixed_content(self):
        result = italian_text_normalize("Il dott. Rossi ha 42 gatti e 100€")
        assert "dottore" in result
        assert "quarantadue" in result
        assert "cento euro" in result

    def test_sentence_with_all_features(self):
        """Integration test: text with multiple features combined."""
        text = "Il dott. Rossi, nato il 15/03/1980, ha pagato 100€ alle 14:30 per la NATO."
        result = italian_text_normalize(text)
        assert "dottore" in result
        assert "quindici" in result
        assert "marzo" in result
        assert "cento euro" in result
        assert "quattordici" in result
        assert "trenta" in result
