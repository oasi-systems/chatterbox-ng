"""
SSML Parser for ChatterBox NG.

Parses a subset of SSML (Speech Synthesis Markup Language) relevant for
telephony applications (IVR, call center, Asterisk).

Supported tags:
    <speak>           — Root element (optional)
    <break time="Xs"/> — Insert silence (seconds or milliseconds)
    <emphasis level="strong|moderate|reduced"> — Control expressiveness
    <prosody rate="slow|medium|fast|X%"> — Speaking rate
    <prosody pitch="+X%|-X%"> — Pitch adjustment
    <say-as interpret-as="date|number|currency|characters|telephone">
    <phoneme alphabet="ipa" ph="..."> — IPA pronunciation override
    <sub alias="..."> — Substitution (e.g., abbreviations)
    <p>, <s> — Paragraph/sentence boundaries (insert natural pauses)

Usage:
    from chatterbox.ssml import SSMLParser, SSMLSegment

    parser = SSMLParser()
    segments = parser.parse('''
        <speak>
            <prosody rate="95%">
                Buongiorno, la informo che il pagamento di
                <say-as interpret-as="currency">€1.250</say-as>
                è stato ricevuto.
            </prosody>
            <break time="500ms"/>
            <emphasis level="strong">Posso aiutarla?</emphasis>
        </speak>
    ''')

    for seg in segments:
        if seg.is_break:
            insert_silence(seg.break_duration_ms)
        else:
            audio = model.generate(seg.text, language_id="it",
                                    exaggeration=seg.exaggeration,
                                    cfg_weight=seg.cfg_weight)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


def _ipa_to_respelling_safe(ipa: str, lang: str) -> str:
    """Convert IPA to respelling, with robust import fallback.

    Tries multiple import paths so this works both as a package import
    and when ssml.py is loaded standalone for testing.
    """
    try:
        try:
            from .g2p import ipa_to_respelling
        except (ImportError, SystemError):
            # Standalone loading or package not fully available
            import importlib.util
            import os
            g2p_path = os.path.join(os.path.dirname(__file__), "g2p.py")
            if os.path.exists(g2p_path):
                spec = importlib.util.spec_from_file_location("_g2p_standalone", g2p_path)
                g2p_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(g2p_mod)
                ipa_to_respelling = g2p_mod.ipa_to_respelling
            else:
                return ""
        result = ipa_to_respelling(ipa, lang)
        return result if result and result.strip() else ""
    except Exception:
        return ""


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SSMLSegment:
    """A segment of text with associated SSML properties.

    Either a text segment (with prosody/emphasis) or a break (silence).
    """
    text: str = ""
    language_id: Optional[str] = None

    # Break (silence)
    is_break: bool = False
    break_duration_ms: float = 0.0

    # Prosody
    rate: float = 1.0           # 1.0 = normal, 0.5 = half speed, 2.0 = double
    pitch_shift: float = 0.0    # semitones or percentage

    # Emphasis → maps to exaggeration parameter
    emphasis: str = "moderate"  # "strong", "moderate", "reduced", "none"

    # Phoneme override (IPA)
    phoneme_ipa: Optional[str] = None

    @property
    def exaggeration(self) -> float:
        """Map emphasis level to ChatterBox exaggeration parameter."""
        return {
            "strong": 0.8,
            "moderate": 0.5,
            "reduced": 0.3,
            "none": 0.2,
        }.get(self.emphasis, 0.5)

    @property
    def cfg_weight(self) -> float:
        """Map prosody rate to cfg_weight (affects pacing)."""
        # Slower speech → higher cfg_weight (more faithful to reference timing)
        if self.rate < 0.8:
            return 0.7
        elif self.rate > 1.2:
            return 0.3
        return 0.5


# ============================================================================
# Parser
# ============================================================================

class SSMLParser:
    """Parse SSML markup into a list of SSMLSegments.

    Handles both valid SSML (with <speak> root) and plain text.
    Invalid XML is treated as plain text (graceful degradation).
    """

    # Default break durations for structural elements
    SENTENCE_BREAK_MS = 300.0
    PARAGRAPH_BREAK_MS = 600.0

    def parse(self, text: str, default_language: str = None) -> list[SSMLSegment]:
        """Parse SSML text into segments.

        Args:
            text: SSML markup or plain text
            default_language: default language_id for segments

        Returns:
            List of SSMLSegment objects ready for generation.
        """
        text = text.strip()

        # Plain text (no XML tags) → single segment
        if not self._looks_like_ssml(text):
            return [SSMLSegment(text=text, language_id=default_language)]

        # Wrap in <speak> if not already
        if not text.startswith("<speak"):
            text = f"<speak>{text}</speak>"

        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            logger.warning(f"SSML parse error: {e}. Treating as plain text.")
            # Strip tags and return as plain text
            clean = re.sub(r'<[^>]+>', '', text)
            return [SSMLSegment(text=clean.strip(), language_id=default_language)]

        segments = []
        self._walk(root, segments, default_language, rate=1.0, emphasis="moderate")

        # Merge adjacent text segments with same properties
        merged = self._merge_segments(segments)

        # Remove empty text segments
        result = [s for s in merged if s.is_break or s.text.strip()]

        return result if result else [SSMLSegment(text="", language_id=default_language)]

    def _looks_like_ssml(self, text: str) -> bool:
        """Quick check if text contains SSML tags."""
        return bool(re.search(r'<\s*(speak|break|emphasis|prosody|say-as|phoneme|sub|p|s)\b', text))

    def _walk(self, element, segments: list, lang: str,
              rate: float, emphasis: str):
        """Recursively walk the XML tree, building segments."""

        tag = self._local_tag(element.tag)

        # --- <break> ---
        if tag == "break":
            duration = self._parse_break_time(element.get("time", "500ms"))
            # Also support strength attribute
            if "strength" in element.attrib:
                duration = self._parse_break_strength(element.get("strength"))
            segments.append(SSMLSegment(
                is_break=True,
                break_duration_ms=duration,
                language_id=lang,
            ))
            # Process tail text after break
            if element.tail and element.tail.strip():
                segments.append(SSMLSegment(
                    text=element.tail.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            return

        # --- <emphasis> ---
        if tag == "emphasis":
            emphasis = element.get("level", "moderate")

        # --- <prosody> ---
        if tag == "prosody":
            rate_attr = element.get("rate", None)
            if rate_attr:
                rate = self._parse_rate(rate_attr)

        # --- <say-as> ---
        if tag == "say-as":
            interpret_as = element.get("interpret-as", "")
            format_attr = element.get("format", None)
            inner_text = self._get_all_text(element)
            converted = self._apply_say_as(inner_text, interpret_as, lang, format_attr)
            segments.append(SSMLSegment(
                text=converted,
                language_id=lang,
                rate=rate,
                emphasis=emphasis,
            ))
            # Process tail
            if element.tail and element.tail.strip():
                segments.append(SSMLSegment(
                    text=element.tail.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            return  # Don't recurse into children

        # --- <phoneme> ---
        if tag == "phoneme":
            ph = element.get("ph", "")
            inner_text = self._get_all_text(element)
            # Convert IPA to orthographic respelling for the BPE tokenizer
            respelled = inner_text
            if ph and lang:
                respelled = _ipa_to_respelling_safe(ph, lang) or inner_text
            segments.append(SSMLSegment(
                text=respelled,
                language_id=lang,
                rate=rate,
                emphasis=emphasis,
                phoneme_ipa=ph,
            ))
            if element.tail and element.tail.strip():
                segments.append(SSMLSegment(
                    text=element.tail.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            return

        # --- <sub> ---
        if tag == "sub":
            alias = element.get("alias", "")
            if alias:
                segments.append(SSMLSegment(
                    text=alias,
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            if element.tail and element.tail.strip():
                segments.append(SSMLSegment(
                    text=element.tail.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            return

        # --- <p> (paragraph) ---
        if tag == "p":
            # Process children, then add paragraph break
            if element.text and element.text.strip():
                segments.append(SSMLSegment(
                    text=element.text.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            for child in element:
                self._walk(child, segments, lang, rate, emphasis)
            segments.append(SSMLSegment(
                is_break=True,
                break_duration_ms=self.PARAGRAPH_BREAK_MS,
                language_id=lang,
            ))
            if element.tail and element.tail.strip():
                segments.append(SSMLSegment(
                    text=element.tail.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            return

        # --- <s> (sentence) ---
        if tag == "s":
            if element.text and element.text.strip():
                segments.append(SSMLSegment(
                    text=element.text.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            for child in element:
                self._walk(child, segments, lang, rate, emphasis)
            segments.append(SSMLSegment(
                is_break=True,
                break_duration_ms=self.SENTENCE_BREAK_MS,
                language_id=lang,
            ))
            if element.tail and element.tail.strip():
                segments.append(SSMLSegment(
                    text=element.tail.strip(),
                    language_id=lang,
                    rate=rate,
                    emphasis=emphasis,
                ))
            return

        # --- Default: <speak> or unknown tags ---
        # Process text content
        if element.text and element.text.strip():
            segments.append(SSMLSegment(
                text=element.text.strip(),
                language_id=lang,
                rate=rate,
                emphasis=emphasis,
            ))

        # Recurse into children
        for child in element:
            self._walk(child, segments, lang, rate, emphasis)

        # Process tail text (text after closing tag, before next sibling)
        if element.tail and element.tail.strip():
            segments.append(SSMLSegment(
                text=element.tail.strip(),
                language_id=lang,
                rate=rate,
                emphasis=emphasis,
            ))

    # --- Helpers ---

    @staticmethod
    def _local_tag(tag: str) -> str:
        """Strip namespace from tag."""
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @staticmethod
    def _get_all_text(element) -> str:
        """Get all text content from element and children."""
        return "".join(element.itertext()).strip()

    @staticmethod
    def _parse_break_time(time_str: str) -> float:
        """Parse break time string to milliseconds."""
        time_str = time_str.strip().lower()
        if time_str.endswith("ms"):
            return float(time_str[:-2])
        elif time_str.endswith("s"):
            return float(time_str[:-1]) * 1000.0
        else:
            try:
                return float(time_str) * 1000.0  # Assume seconds
            except ValueError:
                return 500.0  # Default

    @staticmethod
    def _parse_break_strength(strength: str) -> float:
        """Parse break strength to milliseconds."""
        return {
            "none": 0.0,
            "x-weak": 100.0,
            "weak": 200.0,
            "medium": 400.0,
            "strong": 600.0,
            "x-strong": 1000.0,
        }.get(strength.lower(), 400.0)

    @staticmethod
    def _parse_rate(rate_str: str) -> float:
        """Parse prosody rate to multiplier."""
        rate_str = rate_str.strip().lower()
        # Named rates
        named = {
            "x-slow": 0.5,
            "slow": 0.75,
            "medium": 1.0,
            "fast": 1.25,
            "x-fast": 1.5,
        }
        if rate_str in named:
            return named[rate_str]
        # Percentage (e.g., "90%", "120%")
        if rate_str.endswith("%"):
            try:
                return float(rate_str[:-1]) / 100.0
            except ValueError:
                return 1.0
        # Raw number
        try:
            return float(rate_str)
        except ValueError:
            return 1.0

    @staticmethod
    def _apply_say_as(text: str, interpret_as: str, lang: str,
                      format_attr: str = None) -> str:
        """Apply say-as interpretation with explicit normalization.

        Converts structured data (dates, numbers, currency) to spoken form
        using the euro_text_normalizers, rather than hoping the general
        normalizer catches the pattern downstream.
        """
        if interpret_as in ("characters", "spell-out"):
            return " ".join(text)

        if interpret_as == "telephone":
            digits = re.sub(r'[^\d+]', ' ', text)
            return " ".join(digits.split())

        if interpret_as == "date":
            return SSMLParser._normalize_date(text, lang, format_attr)

        if interpret_as == "number":
            return SSMLParser._normalize_number(text, lang)

        if interpret_as == "currency":
            return SSMLParser._normalize_currency(text, lang)

        if interpret_as == "ordinal":
            return SSMLParser._normalize_ordinal(text, lang)

        if interpret_as == "time":
            return SSMLParser._normalize_time(text, lang)

        # Unknown interpret-as → pass through to general normalizer
        return text

    @staticmethod
    def _normalize_date(text: str, lang: str, format_attr: str = None) -> str:
        """Convert date string to spoken form.

        Supports format attribute: "dmy" (default EU), "mdy" (US), "ymd" (ISO).
        """
        try:
            from num2words import num2words
        except ImportError:
            return text

        # Extract digits from date
        parts = re.split(r'[/\-.]', text.strip())
        if len(parts) < 2:
            return text

        fmt = (format_attr or "dmy").lower()
        try:
            if fmt == "mdy":
                month, day = int(parts[0]), int(parts[1])
                year = int(parts[2]) if len(parts) > 2 else None
            elif fmt == "ymd":
                year = int(parts[0])
                month, day = int(parts[1]), int(parts[2]) if len(parts) > 2 else (int(parts[1]), 1)
            else:  # dmy (default for EU)
                day, month = int(parts[0]), int(parts[1])
                year = int(parts[2]) if len(parts) > 2 else None
        except (ValueError, IndexError):
            return text

        # Language-specific month names
        _MONTHS = {
            "it": {1: "gennaio", 2: "febbraio", 3: "marzo", 4: "aprile", 5: "maggio",
                   6: "giugno", 7: "luglio", 8: "agosto", 9: "settembre", 10: "ottobre",
                   11: "novembre", 12: "dicembre"},
            "fr": {1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai",
                   6: "juin", 7: "juillet", 8: "août", 9: "septembre", 10: "octobre",
                   11: "novembre", 12: "décembre"},
            "de": {1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai",
                   6: "Juni", 7: "Juli", 8: "August", 9: "September", 10: "Oktober",
                   11: "November", 12: "Dezember"},
            "es": {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo",
                   6: "junio", 7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre",
                   11: "noviembre", 12: "diciembre"},
            "pt": {1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio",
                   6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro",
                   11: "novembro", 12: "dezembro"},
            "en": {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
                   6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
                   11: "November", 12: "December"},
        }

        num2words_lang = lang if lang in ("it", "fr", "de", "es", "pt", "en") else "en"
        month_name = _MONTHS.get(lang, _MONTHS["en"]).get(month, str(month))

        try:
            if lang == "it":
                day_w = "primo" if day == 1 else num2words(day, lang="it")
            elif lang == "fr":
                day_w = "premier" if day == 1 else num2words(day, lang="fr")
            elif lang == "de":
                day_w = num2words(day, to="ordinal", lang="de")
            elif lang == "en":
                day_w = num2words(day, to="ordinal", lang="en")
            else:
                day_w = num2words(day, lang=num2words_lang)

            if year:
                year_w = num2words(year, lang=num2words_lang)
                return f"{day_w} {month_name} {year_w}"
            return f"{day_w} {month_name}"
        except Exception:
            return text

    @staticmethod
    def _normalize_number(text: str, lang: str) -> str:
        """Convert number to spoken form."""
        try:
            from num2words import num2words
        except ImportError:
            return text

        num2words_lang = lang if lang in ("it", "fr", "de", "es", "pt", "en") else "en"
        clean = text.strip().replace(" ", "")

        # Handle decimal separator (comma for EU, dot for EN)
        if "," in clean and lang != "en":
            parts = clean.split(",", 1)
            try:
                int_part = num2words(int(parts[0].replace(".", "")), lang=num2words_lang)
                dec_part = num2words(int(parts[1]), lang=num2words_lang)
                sep_word = {"it": "virgola", "fr": "virgule", "de": "Komma",
                            "es": "coma", "pt": "vírgula"}.get(lang, "point")
                return f"{int_part} {sep_word} {dec_part}"
            except (ValueError, Exception):
                pass

        # Remove thousands separators
        clean = clean.replace(".", "").replace(",", "")
        try:
            return num2words(int(clean), lang=num2words_lang)
        except (ValueError, Exception):
            return text

    @staticmethod
    def _normalize_currency(text: str, lang: str) -> str:
        """Convert currency to spoken form (e.g., '€1.250' → 'milleduecentocinquanta euro')."""
        try:
            from num2words import num2words
        except ImportError:
            return text

        num2words_lang = lang if lang in ("it", "fr", "de", "es", "pt", "en") else "en"

        # Extract symbol and amount
        m = re.match(r'([€$£])\s*([\d.,]+)', text.strip())
        if not m:
            m = re.match(r'([\d.,]+)\s*([€$£])', text.strip())
            if not m:
                return text
            amount_str, symbol = m.group(1), m.group(2)
        else:
            symbol, amount_str = m.group(1), m.group(2)

        # Clean amount: remove thousands separator
        if lang == "en":
            amount_str = amount_str.replace(",", "")
        else:
            amount_str = amount_str.replace(".", "").replace(",", ".")

        _CURRENCY_NAMES = {
            "€": {"it": ("euro", "euro"), "fr": ("euro", "euros"), "de": ("Euro", "Euro"),
                   "es": ("euro", "euros"), "pt": ("euro", "euros"), "en": ("euro", "euros")},
            "$": {"it": ("dollaro", "dollari"), "fr": ("dollar", "dollars"), "de": ("Dollar", "Dollar"),
                   "es": ("dólar", "dólares"), "pt": ("dólar", "dólares"), "en": ("dollar", "dollars")},
            "£": {"it": ("sterlina", "sterline"), "fr": ("livre", "livres"), "de": ("Pfund", "Pfund"),
                   "es": ("libra", "libras"), "pt": ("libra", "libras"), "en": ("pound", "pounds")},
        }

        try:
            amount = int(float(amount_str))
            words = num2words(amount, lang=num2words_lang)
            if amount == 1 and lang == "it":
                words = "un"
            names = _CURRENCY_NAMES.get(symbol, {}).get(lang, (symbol, symbol))
            name = names[0] if amount == 1 else names[1]
            return f"{words} {name}"
        except (ValueError, Exception):
            return text

    @staticmethod
    def _normalize_ordinal(text: str, lang: str) -> str:
        """Convert ordinal number to spoken form."""
        try:
            from num2words import num2words
        except ImportError:
            return text

        num2words_lang = lang if lang in ("it", "fr", "de", "es", "pt", "en") else "en"
        clean = re.sub(r'[°ºªa]', '', text.strip())
        try:
            n = int(clean)
            return num2words(n, to="ordinal", lang=num2words_lang)
        except (ValueError, Exception):
            return text

    @staticmethod
    def _normalize_time(text: str, lang: str) -> str:
        """Convert time to spoken form (e.g., '14:30' → 'quattordici e trenta')."""
        try:
            from num2words import num2words
        except ImportError:
            return text

        num2words_lang = lang if lang in ("it", "fr", "de", "es", "pt", "en") else "en"
        m = re.match(r'(\d{1,2})[:.hH](\d{2})', text.strip())
        if not m:
            return text

        h, mins = int(m.group(1)), int(m.group(2))
        try:
            h_w = num2words(h, lang=num2words_lang)
            if mins == 0:
                return h_w
            m_w = num2words(mins, lang=num2words_lang)
            sep = {"it": "e", "fr": "heures", "de": "Uhr", "es": "y",
                   "pt": "e", "en": ""}.get(lang, "")
            if sep:
                return f"{h_w} {sep} {m_w}"
            return f"{h_w} {m_w}"
        except Exception:
            return text

    @staticmethod
    def _merge_segments(segments: list[SSMLSegment]) -> list[SSMLSegment]:
        """Merge adjacent text segments with identical properties."""
        if not segments:
            return segments

        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            # Merge if both are text (not break) and have same properties
            if (not seg.is_break and not prev.is_break
                    and seg.rate == prev.rate
                    and seg.emphasis == prev.emphasis
                    and seg.language_id == prev.language_id
                    and seg.phoneme_ipa is None
                    and prev.phoneme_ipa is None):
                prev.text = (prev.text + " " + seg.text).strip()
            else:
                merged.append(seg)
        return merged


# ============================================================================
# Convenience
# ============================================================================

_default_parser: Optional[SSMLParser] = None


def get_default_parser() -> SSMLParser:
    """Get or create the default SSML parser singleton."""
    global _default_parser
    if _default_parser is None:
        _default_parser = SSMLParser()
    return _default_parser


def parse_ssml(text: str, default_language: str = None) -> list[SSMLSegment]:
    """Parse SSML text (convenience function)."""
    return get_default_parser().parse(text, default_language)


def is_ssml(text: str) -> bool:
    """Check if text contains SSML markup."""
    return get_default_parser()._looks_like_ssml(text)
