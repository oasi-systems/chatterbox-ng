#!/usr/bin/env python3
"""Tests for SSML parser."""
import sys, os, importlib.util

_pkg_dir = os.path.join(os.path.dirname(__file__), "..", "src", "chatterbox")
spec = importlib.util.spec_from_file_location("ssml", os.path.join(_pkg_dir, "ssml.py"))
ssml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssml)

SSMLParser = ssml.SSMLParser
SSMLSegment = ssml.SSMLSegment
parse_ssml = ssml.parse_ssml
is_ssml = ssml.is_ssml


class TestIsSSML:
    def test_plain_text(self):
        assert not is_ssml("Buongiorno, come stai?")

    def test_speak_tag(self):
        assert is_ssml("<speak>Ciao</speak>")

    def test_break_tag(self):
        assert is_ssml('Ciao <break time="500ms"/> mondo')

    def test_emphasis(self):
        assert is_ssml("<emphasis level=\"strong\">Attenzione!</emphasis>")

    def test_prosody(self):
        assert is_ssml('<prosody rate="slow">Lentamente</prosody>')

    def test_html_not_ssml(self):
        assert not is_ssml("<div>Not SSML</div>")


class TestPlainText:
    def test_returns_single_segment(self):
        segs = parse_ssml("Buongiorno, come stai?")
        assert len(segs) == 1
        assert segs[0].text == "Buongiorno, come stai?"
        assert not segs[0].is_break

    def test_with_language(self):
        segs = parse_ssml("Hello world", default_language="en")
        assert segs[0].language_id == "en"


class TestBreak:
    def test_milliseconds(self):
        segs = parse_ssml('<speak>Prima <break time="500ms"/> dopo</speak>')
        breaks = [s for s in segs if s.is_break]
        assert len(breaks) == 1
        assert breaks[0].break_duration_ms == 500.0

    def test_seconds(self):
        segs = parse_ssml('<speak>Prima <break time="1.5s"/> dopo</speak>')
        breaks = [s for s in segs if s.is_break]
        assert breaks[0].break_duration_ms == 1500.0

    def test_strength(self):
        segs = parse_ssml('<speak>A <break strength="strong"/> B</speak>')
        breaks = [s for s in segs if s.is_break]
        assert breaks[0].break_duration_ms == 600.0

    def test_text_around_break(self):
        segs = parse_ssml('<speak>Prima <break time="300ms"/> dopo</speak>')
        texts = [s.text for s in segs if not s.is_break]
        assert "Prima" in texts
        assert "dopo" in texts


class TestEmphasis:
    def test_strong(self):
        segs = parse_ssml('<speak><emphasis level="strong">Attenzione!</emphasis></speak>')
        text_segs = [s for s in segs if not s.is_break]
        assert len(text_segs) == 1
        assert text_segs[0].emphasis == "strong"
        assert text_segs[0].exaggeration == 0.8

    def test_reduced(self):
        segs = parse_ssml('<speak><emphasis level="reduced">piano</emphasis></speak>')
        text_segs = [s for s in segs if not s.is_break]
        assert text_segs[0].exaggeration == 0.3

    def test_default_moderate(self):
        segs = parse_ssml('<speak>Testo normale</speak>')
        assert segs[0].emphasis == "moderate"
        assert segs[0].exaggeration == 0.5


class TestProsody:
    def test_rate_slow(self):
        segs = parse_ssml('<speak><prosody rate="slow">Lentamente</prosody></speak>')
        assert segs[0].rate == 0.75

    def test_rate_fast(self):
        segs = parse_ssml('<speak><prosody rate="fast">Veloce</prosody></speak>')
        assert segs[0].rate == 1.25

    def test_rate_percentage(self):
        segs = parse_ssml('<speak><prosody rate="90%">Novanta</prosody></speak>')
        assert segs[0].rate == 0.9

    def test_cfg_weight_slow(self):
        segs = parse_ssml('<speak><prosody rate="x-slow">Molto lento</prosody></speak>')
        assert segs[0].cfg_weight == 0.7

    def test_cfg_weight_fast(self):
        segs = parse_ssml('<speak><prosody rate="x-fast">Molto veloce</prosody></speak>')
        assert segs[0].cfg_weight == 0.3


class TestSayAs:
    def test_characters(self):
        segs = parse_ssml('<speak><say-as interpret-as="characters">ABC</say-as></speak>')
        assert segs[0].text == "A B C"

    def test_telephone(self):
        segs = parse_ssml('<speak><say-as interpret-as="telephone">+39 02 1234</say-as></speak>')
        text = segs[0].text
        # Digits should be separated
        assert "39" in text
        assert "02" in text

    def test_currency_normalized(self):
        segs = parse_ssml('<speak><say-as interpret-as="currency">€1250</say-as></speak>',
                          default_language="it")
        text = segs[0].text.lower()
        # Should contain "euro" after normalization
        assert "euro" in text

    def test_currency_no_num2words(self):
        # Even without language, should not crash
        segs = parse_ssml('<speak><say-as interpret-as="currency">€100</say-as></speak>')
        assert len(segs) >= 1

    def test_date_dmy(self):
        segs = parse_ssml(
            '<speak><say-as interpret-as="date" format="dmy">15/03/2024</say-as></speak>',
            default_language="it"
        )
        text = segs[0].text.lower()
        # Should contain month name "marzo"
        assert "marzo" in text

    def test_date_mdy(self):
        segs = parse_ssml(
            '<speak><say-as interpret-as="date" format="mdy">03/15/2024</say-as></speak>',
            default_language="en"
        )
        text = segs[0].text.lower()
        assert "march" in text

    def test_number_normalized(self):
        segs = parse_ssml(
            '<speak><say-as interpret-as="number">12345</say-as></speak>',
            default_language="it"
        )
        text = segs[0].text.lower()
        # Should be expanded to Italian words
        assert "dodici" in text or "mila" in text or "cento" in text

    def test_ordinal(self):
        segs = parse_ssml(
            '<speak><say-as interpret-as="ordinal">5</say-as></speak>',
            default_language="it"
        )
        text = segs[0].text.lower()
        assert "quint" in text  # "quinto"

    def test_time(self):
        segs = parse_ssml(
            '<speak><say-as interpret-as="time">14:30</say-as></speak>',
            default_language="it"
        )
        text = segs[0].text.lower()
        assert "quattordici" in text or "trenta" in text


class TestPhoneme:
    def test_ipa_stored(self):
        segs = parse_ssml('<speak><phoneme alphabet="ipa" ph="ʃmɪt">Schmidt</phoneme></speak>',
                          default_language="it")
        assert segs[0].phoneme_ipa == "ʃmɪt"
        # Text should be respelled (not original "Schmidt")
        # IPA "ʃmɪt" → Italian respelling via g2p
        assert segs[0].text != ""

    def test_ipa_respelling_italian(self):
        segs = parse_ssml(
            '<speak><phoneme alphabet="ipa" ph="ʃmɪt">Schmidt</phoneme></speak>',
            default_language="it"
        )
        # IPA ʃ → "sci" in Italian → "scimit"
        text = segs[0].text.lower()
        assert text != "schmidt"  # Must be respelled
        assert "sci" in text  # ʃ → "sci"

    def test_ipa_no_language_fallback(self):
        # Without language, should fall back to original text
        segs = parse_ssml('<speak><phoneme alphabet="ipa" ph="test">Word</phoneme></speak>')
        assert segs[0].text != ""  # Should not crash


class TestSub:
    def test_alias_replacement(self):
        segs = parse_ssml('<speak><sub alias="Organizzazione Mondiale della Sanità">OMS</sub></speak>')
        assert segs[0].text == "Organizzazione Mondiale della Sanità"


class TestParagraphSentence:
    def test_paragraph_adds_break(self):
        segs = parse_ssml('<speak><p>Primo paragrafo.</p><p>Secondo paragrafo.</p></speak>')
        breaks = [s for s in segs if s.is_break]
        assert len(breaks) >= 2  # paragraph breaks

    def test_sentence_adds_break(self):
        segs = parse_ssml('<speak><s>Prima frase.</s><s>Seconda frase.</s></speak>')
        breaks = [s for s in segs if s.is_break]
        assert len(breaks) >= 2


class TestMerging:
    def test_adjacent_same_props_merged(self):
        segs = parse_ssml('<speak>Parte uno parte due</speak>')
        text_segs = [s for s in segs if not s.is_break]
        assert len(text_segs) == 1

    def test_different_emphasis_not_merged(self):
        segs = parse_ssml(
            '<speak>Normale <emphasis level="strong">forte</emphasis></speak>'
        )
        text_segs = [s for s in segs if not s.is_break]
        assert len(text_segs) == 2


class TestGracefulDegradation:
    def test_invalid_xml(self):
        segs = parse_ssml('<speak>Unclosed <break</speak>')
        assert len(segs) >= 1
        # Should not crash, returns plain text

    def test_empty_string(self):
        segs = parse_ssml("")
        assert len(segs) == 1

    def test_unknown_tags_ignored(self):
        segs = parse_ssml('<speak><custom>Testo</custom></speak>')
        text_segs = [s for s in segs if not s.is_break]
        assert any("Testo" in s.text for s in text_segs)


class TestTelephonyScenario:
    """Full integration test: realistic telephony SSML."""

    def test_ivr_greeting(self):
        ssml = '''
        <speak>
            <prosody rate="95%">
                Buongiorno, benvenuto nel servizio clienti.
            </prosody>
            <break time="500ms"/>
            <emphasis level="moderate">
                Per informazioni sul suo conto, prema uno.
            </emphasis>
            <break time="300ms"/>
            Per parlare con un operatore, prema due.
        </speak>
        '''
        segs = parse_ssml(ssml, default_language="it")
        text_segs = [s for s in segs if not s.is_break]
        break_segs = [s for s in segs if s.is_break]

        assert len(text_segs) >= 2
        assert len(break_segs) >= 2

        # First segment should have slow rate
        assert text_segs[0].rate == 0.95
        assert all(s.language_id == "it" for s in segs)

    def test_amount_with_break(self):
        ssml = '''
        <speak>
            Il suo saldo è di
            <say-as interpret-as="currency">€1250</say-as>.
            <break time="800ms"/>
            <emphasis level="strong">Desidera effettuare un'operazione?</emphasis>
        </speak>
        '''
        segs = parse_ssml(ssml, default_language="it")
        assert any(s.is_break and s.break_duration_ms == 800.0 for s in segs)
        strong_segs = [s for s in segs if not s.is_break and s.emphasis == "strong"]
        assert len(strong_segs) == 1
        # Currency should be normalized
        text_segs = [s for s in segs if not s.is_break]
        all_text = " ".join(s.text for s in text_segs).lower()
        assert "euro" in all_text


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
