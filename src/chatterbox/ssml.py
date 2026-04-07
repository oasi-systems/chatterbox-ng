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
            inner_text = self._get_all_text(element)
            converted = self._apply_say_as(inner_text, interpret_as, lang)
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
            segments.append(SSMLSegment(
                text=inner_text,
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
    def _apply_say_as(text: str, interpret_as: str, lang: str) -> str:
        """Apply say-as interpretation.

        Note: most conversions are handled by euro_text_normalizers.
        This handles only character-by-character spelling and explicit formats.
        """
        if interpret_as == "characters" or interpret_as == "spell-out":
            # Spell out each character: "ABC" → "a, bi, ci" (in Italian)
            return " ".join(text)

        if interpret_as == "telephone":
            # Read digits individually with pauses
            digits = re.sub(r'[^\d+]', ' ', text)
            return " ".join(digits.split())

        # date, number, currency, ordinal → handled by euro_text_normalizers
        # Just return the text and let the normalizer do its job
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
