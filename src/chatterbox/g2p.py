"""
Grapheme-to-Phoneme (G2P) pipeline for ChatterBox NG.

Since the model uses BPE text tokens (not phonemes), this module does NOT pass
IPA to the model. Instead, it uses espeak-ng to get IPA pronunciation and then
RESPELLS difficult words in the target language's orthography, so the BPE
tokenizer produces the correct pronunciation.

Strategy:
1. Custom dictionary lookup (highest priority — per-client overrides)
2. Foreign word detection + espeak-ng respelling
3. Passthrough for native words (the model already handles these well)

Example:
    "Schmidt" in Italian context → "shmit"
    "McDonald" in Italian context → "mecdonald"
    "Müller" in French context → "muleur"

Usage:
    from chatterbox.g2p import G2PPipeline

    g2p = G2PPipeline()
    text = g2p.process("Il sig. Schmidt ha chiamato", lang="it")
    # → "Il sig. shmit ha chiamato"

Dependencies:
    - espeak-ng (system package): apt install espeak-ng
    - phonemizer (Python): pip install phonemizer
    Both are optional — graceful fallback to passthrough if unavailable.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# --- Check for espeak-ng / phonemizer availability ---
_HAS_PHONEMIZER = False
_phonemizer_backend = None

try:
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    from phonemizer.phonemize import phonemize
    _HAS_PHONEMIZER = True
except ImportError:
    logger.info("phonemizer not installed — G2P respelling disabled. "
                "Install with: pip install phonemizer (requires espeak-ng)")


# ============================================================================
# IPA → Native Orthography Respelling Tables
# ============================================================================
# These tables convert IPA phonemes to the closest spelling in each target
# language. The goal is NOT linguistic perfection — it's to produce text that
# the BPE model will pronounce correctly.
#
# Order matters: longer patterns must come before shorter ones to avoid
# partial matches (e.g., "tʃ" before "t").
# ============================================================================

_IPA_TO_ITALIAN = [
    # Affricates & digraphs (before single consonants)
    ("tʃ", "ci"),    # "ch" sound → ci (chiesa)
    ("dʒ", "gi"),    # "j" sound → gi (giorno)
    ("ts", "z"),     # "ts" → z (pizza)
    ("dz", "z"),     # "dz" → z (zero)
    ("ʃ", "sci"),    # "sh" → sci (scienza)
    ("ʒ", "gi"),     # "zh" → gi (approx)
    ("ɲ", "gn"),     # "ny" → gn (gnomo)
    ("ʎ", "gli"),    # "ly" → gli (famiglia)
    ("kw", "qu"),    # "kw" → qu (quando)
    # Vowels
    ("ə", "e"),      # schwa → e
    ("ɛ", "e"),      # open e → e
    ("ɔ", "o"),      # open o → o
    ("æ", "e"),      # near-open front → e
    ("ɑ", "a"),      # open back → a
    ("ʌ", "a"),      # open-mid back → a
    ("ɪ", "i"),      # near-close front → i
    ("ʊ", "u"),      # near-close back → u
    ("iː", "i"),     # long i → i
    ("uː", "u"),     # long u → u
    ("eː", "e"),     # long e → e
    ("oː", "o"),     # long o → o
    ("aː", "a"),     # long a → a
    ("aʊ", "au"),    # diphthong
    ("aɪ", "ai"),    # diphthong
    ("ɔɪ", "oi"),    # diphthong
    ("eɪ", "ei"),    # diphthong
    ("oʊ", "ou"),    # diphthong
    # Consonants
    ("θ", "t"),      # "th" voiceless → t
    ("ð", "d"),      # "th" voiced → d
    ("ŋ", "ng"),     # velar nasal
    ("x", "h"),      # voiceless velar fricative
    ("ɣ", "g"),      # voiced velar fricative
    ("ç", "h"),      # voiceless palatal fricative
    ("h", ""),       # h is silent in Italian
    ("ɹ", "r"),      # English r → r
    ("ɾ", "r"),      # tap → r
    ("w", "u"),      # semivowel w → u
    ("j", "i"),      # semivowel j → i
    ("β", "b"),      # voiced bilabial fricative → b
    ("ɸ", "f"),      # voiceless bilabial fricative → f
    # Keep simple consonants as-is
    ("k", "c"),      # k → c (default, before a/o/u — good enough for respelling)
    ("p", "p"), ("b", "b"), ("t", "t"), ("d", "d"),
    ("f", "f"), ("v", "v"), ("s", "s"), ("z", "z"),
    ("m", "m"), ("n", "n"), ("l", "l"), ("r", "r"),
    ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"),
]

_IPA_TO_FRENCH = [
    ("tʃ", "tch"),
    ("dʒ", "dj"),
    ("ʃ", "ch"),
    ("ʒ", "j"),
    ("ɲ", "gn"),
    ("ŋ", "ng"),
    ("ɑ̃", "an"),
    ("ɛ̃", "in"),
    ("ɔ̃", "on"),
    ("œ̃", "un"),
    ("ə", "e"),
    ("ɛ", "è"),
    ("ɔ", "o"),
    ("œ", "eu"),
    ("ø", "eu"),
    ("y", "u"),
    ("ɑ", "a"),
    ("æ", "è"),
    ("ʌ", "a"),
    ("ɪ", "i"),
    ("ʊ", "ou"),
    ("aʊ", "aou"),
    ("aɪ", "aï"),
    ("eɪ", "eï"),
    ("oʊ", "o"),
    ("θ", "s"),
    ("ð", "z"),
    ("x", "r"),
    ("ɣ", "r"),
    ("h", ""),
    ("ɹ", "r"),
    ("ɾ", "r"),
    ("w", "ou"),
    ("j", "y"),
    ("k", "k"), ("p", "p"), ("b", "b"), ("t", "t"), ("d", "d"),
    ("f", "f"), ("v", "v"), ("s", "s"), ("z", "z"),
    ("m", "m"), ("n", "n"), ("l", "l"), ("r", "r"),
    ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "ou"),
]

_IPA_TO_GERMAN = [
    ("tʃ", "tsch"),
    ("dʒ", "dsch"),
    ("ʃ", "sch"),
    ("ʒ", "sch"),
    ("ç", "ch"),
    ("x", "ch"),
    ("ɲ", "nj"),
    ("ŋ", "ng"),
    ("pf", "pf"),
    ("ts", "z"),
    ("ə", "e"),
    ("ɛ", "e"),
    ("ɔ", "o"),
    ("œ", "ö"),
    ("ø", "ö"),
    ("y", "ü"),
    ("ʏ", "ü"),
    ("ɑ", "a"),
    ("æ", "ä"),
    ("ʌ", "a"),
    ("ɪ", "i"),
    ("ʊ", "u"),
    ("aʊ", "au"),
    ("aɪ", "ei"),
    ("ɔɪ", "eu"),
    ("θ", "s"),
    ("ð", "s"),
    ("h", "h"),
    ("ɹ", "r"),
    ("ɾ", "r"),
    ("w", "w"),
    ("j", "j"),
    ("v", "w"),
    ("k", "k"), ("p", "p"), ("b", "b"), ("t", "t"), ("d", "d"),
    ("f", "f"), ("s", "s"), ("z", "s"),
    ("m", "m"), ("n", "n"), ("l", "l"), ("r", "r"),
    ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"),
]

_IPA_TO_SPANISH = [
    ("tʃ", "ch"),
    ("dʒ", "y"),
    ("ʃ", "sh"),
    ("ʒ", "y"),
    ("ɲ", "ñ"),
    ("ʎ", "ll"),
    ("ŋ", "ng"),
    ("rr", "rr"),
    ("ɾ", "r"),
    ("θ", "z"),
    ("ð", "d"),
    ("β", "b"),
    ("ɣ", "g"),
    ("x", "j"),
    ("ə", "e"),
    ("ɛ", "e"),
    ("ɔ", "o"),
    ("æ", "e"),
    ("ɑ", "a"),
    ("ʌ", "a"),
    ("ɪ", "i"),
    ("ʊ", "u"),
    ("aʊ", "au"),
    ("aɪ", "ai"),
    ("eɪ", "ei"),
    ("h", "j"),
    ("ɹ", "r"),
    ("w", "u"),
    ("j", "y"),
    ("k", "c"), ("p", "p"), ("b", "b"), ("t", "t"), ("d", "d"),
    ("f", "f"), ("v", "b"), ("s", "s"), ("z", "s"),
    ("m", "m"), ("n", "n"), ("l", "l"), ("r", "r"),
    ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"),
]

_IPA_TO_PORTUGUESE = [
    ("tʃ", "tch"),
    ("dʒ", "dj"),
    ("ʃ", "ch"),
    ("ʒ", "j"),
    ("ɲ", "nh"),
    ("ʎ", "lh"),
    ("ŋ", "ng"),
    ("ɾ", "r"),
    ("ʁ", "rr"),
    ("ə", "e"),
    ("ɛ", "é"),
    ("ɔ", "ó"),
    ("ã", "an"),
    ("ẽ", "en"),
    ("õ", "on"),
    ("æ", "é"),
    ("ɑ", "a"),
    ("ʌ", "a"),
    ("ɪ", "i"),
    ("ʊ", "u"),
    ("aʊ", "au"),
    ("aɪ", "ai"),
    ("eɪ", "ei"),
    ("θ", "s"),
    ("ð", "d"),
    ("h", ""),
    ("ɹ", "r"),
    ("w", "u"),
    ("j", "i"),
    ("x", "rr"),
    ("k", "c"), ("p", "p"), ("b", "b"), ("t", "t"), ("d", "d"),
    ("f", "f"), ("v", "v"), ("s", "s"), ("z", "z"),
    ("m", "m"), ("n", "n"), ("l", "l"), ("r", "r"),
    ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"),
]

_IPA_TO_ENGLISH = [
    # English doesn't need respelling — the model handles English text natively.
    # This table is here for completeness (e.g., respelling foreign names in English).
    ("tʃ", "ch"),
    ("dʒ", "j"),
    ("ʃ", "sh"),
    ("ʒ", "zh"),
    ("ɲ", "ny"),
    ("ŋ", "ng"),
    ("θ", "th"),
    ("ð", "th"),
    ("ə", "uh"),
    ("ɛ", "eh"),
    ("ɔ", "aw"),
    ("æ", "a"),
    ("ɑ", "ah"),
    ("ʌ", "uh"),
    ("ɪ", "ih"),
    ("ʊ", "oo"),
    ("aʊ", "ow"),
    ("aɪ", "eye"),
    ("ɔɪ", "oy"),
    ("eɪ", "ay"),
    ("oʊ", "oh"),
    ("ɹ", "r"),
    ("ɾ", "r"),
    ("x", "kh"),
    ("ç", "h"),
    ("h", "h"),
    ("w", "w"),
    ("j", "y"),
    ("k", "k"), ("p", "p"), ("b", "b"), ("t", "t"), ("d", "d"),
    ("f", "f"), ("v", "v"), ("s", "s"), ("z", "z"),
    ("m", "m"), ("n", "n"), ("l", "l"), ("r", "r"),
    ("a", "a"), ("e", "e"), ("i", "i"), ("o", "o"), ("u", "u"),
]

# Map language codes to respelling tables
_IPA_TABLES = {
    "it": _IPA_TO_ITALIAN,
    "fr": _IPA_TO_FRENCH,
    "de": _IPA_TO_GERMAN,
    "es": _IPA_TO_SPANISH,
    "pt": _IPA_TO_PORTUGUESE,
    "en": _IPA_TO_ENGLISH,
}

# Map our language codes to espeak-ng language codes
_ESPEAK_LANG_MAP = {
    "it": "it",
    "fr": "fr-fr",
    "de": "de",
    "es": "es",
    "pt": "pt",
    "en": "en-us",
}


# ============================================================================
# Foreign Word Detection
# ============================================================================

# Common character patterns that suggest a word is foreign to the target language
_FOREIGN_PATTERNS = {
    "it": re.compile(
        r"(?:"
        r"[wWxXyYkK]{2,}"        # double w/x/y/k (rare in Italian)
        r"|th[aeiourw]"           # th + vowel/r/w (English/German: through, three)
        r"|sch"                   # sch anywhere (German: Schmidt, Schwartz, Schubert)
        r"|(?<![cg])h(?=[aeiou])" # h + vowel NOT after c/g (foreign; chi/ghi are Italian)
        r"|oo|ee|ou[^r]"          # English vowel combinations
        r"|tion\b"                # English suffix
        r"|ght\b"                 # English suffix
        r"|ph[aeiou]"             # ph (Greek/English)
        r"|(?:^|\b)wh"            # wh at word start (English)
        r")", re.IGNORECASE
    ),
    "fr": re.compile(
        r"(?:"
        r"th[aeiourw]"            # th (English)
        r"|sch"                   # sch (German)
        r"|ght\b"                 # English suffix
        r"|oo|ee"                 # English vowel combinations
        r"|[wW]{2,}"              # double w
        r")", re.IGNORECASE
    ),
    "de": re.compile(
        r"(?:"
        r"th[aeiourw]"            # th (English)
        r"|oo|ee"                 # English vowel combinations
        r"|tion\b"                # English suffix
        r"|ght\b"                 # English suffix
        r"|ou[aeiou]"
        r")", re.IGNORECASE
    ),
    "es": re.compile(
        r"(?:"
        r"th[aeiourw]"            # th (English)
        r"|sch"                   # sch (German)
        r"|oo|ee"                 # English vowel combinations
        r"|ght\b"                 # English suffix
        r"|[wW]{2,}"              # double w
        r"|ph[aeiou]"             # ph (Greek/English)
        r")", re.IGNORECASE
    ),
    "pt": re.compile(
        r"(?:"
        r"th[aeiourw]"            # th (English)
        r"|sch"                   # sch (German)
        r"|oo|ee"                 # English vowel combinations
        r"|ght\b"                 # English suffix
        r"|[wW]{2,}"              # double w
        r")", re.IGNORECASE
    ),
    "en": re.compile(
        r"(?:"
        r"sch[aeiou]"             # German
        r"|[àáâãäèéêëìíîïòóôõöùúûü]"  # accented vowels
        r"|ß"                     # German eszett
        r")", re.IGNORECASE
    ),
}

# Words that should NEVER be respelled (common loanwords that models handle well)
_LOANWORD_PASSTHROUGH = {
    "it": {"computer", "software", "hardware", "internet", "email", "web", "online",
           "marketing", "manager", "shopping", "weekend", "ok", "hotel", "bar",
           "sport", "film", "club", "stress", "business", "partner", "team",
           "design", "brand", "trend", "start", "stop", "smart", "zoom",
           "meeting", "briefing", "training", "coaching", "feedback"},
    "fr": {"computer", "software", "internet", "email", "web", "online",
           "marketing", "manager", "shopping", "weekend", "ok", "design",
           "smartphone", "startup", "feedback", "business", "meeting"},
    "de": {"computer", "software", "internet", "email", "web", "online",
           "marketing", "manager", "shopping", "weekend", "ok", "design",
           "smartphone", "meeting", "feedback", "team", "business"},
    "es": {"computer", "software", "internet", "email", "web", "online",
           "marketing", "manager", "shopping", "weekend", "ok", "design",
           "smartphone", "feedback", "business", "meeting"},
    "pt": {"computer", "software", "internet", "email", "web", "online",
           "marketing", "manager", "shopping", "weekend", "ok", "design",
           "smartphone", "feedback", "business", "meeting"},
    "en": set(),  # English rarely needs respelling
}


# ============================================================================
# Custom Dictionary
# ============================================================================

class CustomDictionary:
    """Per-client pronunciation dictionary.

    Supports YAML format:
        # dizionario_banca_x.yaml
        IBAN: "i ban"
        SEPA: "sepa"
        Unicredit: "unikrèdit"
        CVV: "ci vu vu"
        Schmidt: "shmit"

    Higher priority than automatic G2P — always checked first.
    """

    def __init__(self):
        self._entries: dict[str, dict[str, str]] = {}  # lang → {word → respelling}
        self._global: dict[str, str] = {}  # language-independent overrides

    def load_yaml(self, path: str, language_id: str = None):
        """Load dictionary from YAML file.

        Args:
            path: path to YAML file
            language_id: if set, entries apply only to this language.
                         If None, entries apply to all languages.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Dictionary file not found: {path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            # Fallback: try loading without yaml (simple key: value format)
            data = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" in line:
                        key, val = line.split(":", 1)
                        data[key.strip().strip('"').strip("'")] = val.strip().strip('"').strip("'")

        if not isinstance(data, dict):
            logger.warning(f"Invalid dictionary format in {path}")
            return

        if language_id:
            if language_id not in self._entries:
                self._entries[language_id] = {}
            for word, respelling in data.items():
                self._entries[language_id][word.lower()] = str(respelling)
            logger.info(f"Loaded {len(data)} entries for [{language_id}] from {path}")
        else:
            for word, respelling in data.items():
                self._global[word.lower()] = str(respelling)
            logger.info(f"Loaded {len(data)} global entries from {path}")

    def add(self, word: str, respelling: str, language_id: str = None):
        """Add a single dictionary entry programmatically."""
        if language_id:
            if language_id not in self._entries:
                self._entries[language_id] = {}
            self._entries[language_id][word.lower()] = respelling
        else:
            self._global[word.lower()] = respelling

    def remove(self, word: str, language_id: str = None) -> bool:
        """Remove a dictionary entry. Returns True if entry existed."""
        w = word.lower()
        if language_id:
            if language_id in self._entries and w in self._entries[language_id]:
                del self._entries[language_id][w]
                return True
        else:
            if w in self._global:
                del self._global[w]
                return True
        return False

    def list_entries(self, language_id: str = None) -> dict:
        """List dictionary entries.

        Args:
            language_id: if set, return entries for that language + globals.
                         If None, return all entries grouped by language.

        Returns:
            dict with "global" and/or language keys mapping to {word: respelling}.
        """
        if language_id:
            result = {}
            if language_id in self._entries:
                result[language_id] = dict(self._entries[language_id])
            result["global"] = dict(self._global)
            return result
        # All entries
        result = {"global": dict(self._global)}
        for lang, entries in self._entries.items():
            result[lang] = dict(entries)
        return result

    def lookup(self, word: str, language_id: str) -> Optional[str]:
        """Look up a word. Returns respelling or None."""
        w = word.lower()
        # Language-specific first
        if language_id in self._entries and w in self._entries[language_id]:
            return self._entries[language_id][w]
        # Then global
        if w in self._global:
            return self._global[w]
        return None


# ============================================================================
# Core G2P Pipeline
# ============================================================================

class G2PPipeline:
    """Grapheme-to-Phoneme pipeline with respelling for BPE models.

    Usage:
        g2p = G2PPipeline()

        # Optional: load custom dictionary
        g2p.dictionary.load_yaml("dizionario_banca_x.yaml", language_id="it")

        # Process text
        text = g2p.process("Il sig. Schmidt ha chiamato alle 14:30", lang="it")
    """

    def __init__(self, custom_dict: CustomDictionary = None, auto_respell: bool = False):
        """
        Args:
            custom_dict: custom pronunciation dictionary (highest priority)
            auto_respell: if True, automatically respell detected foreign words
                via espeak-ng. DISABLED by default because it creates non-words
                that can confuse the BPE model. Only enable after validation.
        """
        self.dictionary = custom_dict or CustomDictionary()
        self.auto_respell = auto_respell
        self._espeak_available = _HAS_PHONEMIZER
        self._separator = None

        if self._espeak_available:
            self._separator = Separator(phone=" ", word="  ", syllable="")
            logger.info("G2P pipeline initialized with espeak-ng backend")
        else:
            logger.info("G2P pipeline initialized (dictionary-only mode)")

    def process(self, text: str, lang: str) -> str:
        """Process text through the G2P pipeline.

        Steps:
        1. Split text into words (preserving punctuation)
        2. For each word:
           a. Custom dictionary lookup (highest priority)
           b. If foreign word detected → espeak-ng → respell in target language
           c. Otherwise → passthrough (model handles native words well)
        3. Reconstruct text

        Args:
            text: input text (already normalized by euro_text_normalizers)
            lang: language code (it, fr, de, es, pt, en)

        Returns:
            Text with foreign/difficult words respelled in target language orthography.
        """
        if lang not in _IPA_TABLES:
            return text  # Unsupported language, passthrough

        # Split into tokens preserving whitespace and punctuation
        tokens = _tokenize_for_g2p(text)
        result = []

        for token in tokens:
            if not token.strip() or not token[0].isalpha():
                # Whitespace or punctuation — keep as-is
                result.append(token)
                continue

            # 1. Custom dictionary (highest priority)
            dict_entry = self.dictionary.lookup(token, lang)
            if dict_entry is not None:
                result.append(dict_entry)
                logger.debug(f"G2P dict: '{token}' → '{dict_entry}' [{lang}]")
                continue

            # 2. Auto-respelling via espeak-ng (DISABLED by default)
            # The BPE model handles most foreign words acceptably.
            # Automatic respelling creates non-words that confuse the model.
            # Only enable if you have validated espeak-ng output for your use case.
            if self.auto_respell and self._is_foreign_word(token, lang):
                respelled = self._respell_word(token, lang)
                if respelled and respelled.lower() != token.lower():
                    result.append(respelled)
                    logger.debug(f"G2P respell: '{token}' → '{respelled}' [{lang}]")
                    continue

            # 3. Passthrough — model handles native words
            result.append(token)

        return "".join(result)

    def respell(self, word: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Respell a single word from source language in target language orthography.

        Useful for explicit respelling of known foreign words.

        Args:
            word: the word to respell
            source_lang: language the word is from (for espeak-ng pronunciation)
            target_lang: language to respell into

        Returns:
            Respelled word or None if espeak-ng unavailable.
        """
        if not self._espeak_available:
            return None

        ipa = self._get_ipa(word, source_lang)
        if not ipa:
            return None

        return self._ipa_to_respelling(ipa, target_lang)

    def _is_foreign_word(self, word: str, lang: str) -> bool:
        """Detect if a word is likely foreign to the target language."""
        # Skip short words (1-2 chars) — usually native
        if len(word) <= 2:
            return False

        # Check passthrough list (common loanwords the model handles)
        passthrough = _LOANWORD_PASSTHROUGH.get(lang, set())
        if word.lower() in passthrough:
            return False

        # Check foreign character patterns
        pattern = _FOREIGN_PATTERNS.get(lang)
        if pattern and pattern.search(word):
            return True

        # Check for characters unusual in the target language
        if lang == "it" and re.search(r'[wxyjkWXYJK]', word) and len(word) > 3:
            return True
        if lang == "fr" and re.search(r'[ß]', word):
            return True

        return False

    def _respell_word(self, word: str, target_lang: str) -> Optional[str]:
        """Get pronunciation via espeak-ng and respell in target language."""
        if not self._espeak_available:
            return None

        # Use espeak-ng with the TARGET language to get how it should sound
        # in that language's phonology
        ipa = self._get_ipa(word, target_lang)
        if not ipa:
            return None

        return self._ipa_to_respelling(ipa, target_lang)

    def _get_ipa(self, word: str, lang: str) -> Optional[str]:
        """Get IPA transcription from espeak-ng."""
        if not self._espeak_available:
            return None

        espeak_lang = _ESPEAK_LANG_MAP.get(lang, lang)

        try:
            result = phonemize(
                word,
                language=espeak_lang,
                backend="espeak",
                strip=True,
                preserve_punctuation=False,
                with_stress=False,  # Skip stress marks for cleaner respelling
            )
            return result.strip()
        except Exception as e:
            logger.debug(f"espeak-ng failed for '{word}' [{lang}]: {e}")
            return None

    def _ipa_to_respelling(self, ipa: str, target_lang: str) -> str:
        """Convert IPA string to target language orthography."""
        table = _IPA_TABLES.get(target_lang, _IPA_TO_ENGLISH)

        # Remove IPA stress markers, ties, and length marks
        ipa = ipa.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
        ipa = ipa.replace("̩", "").replace("͡", "").replace("̃", "")

        result = []
        i = 0
        while i < len(ipa):
            matched = False
            # Try longest match first (up to 3 chars)
            for length in (3, 2, 1):
                if i + length <= len(ipa):
                    chunk = ipa[i:i + length]
                    for ipa_pattern, replacement in table:
                        if chunk == ipa_pattern:
                            result.append(replacement)
                            i += length
                            matched = True
                            break
                if matched:
                    break

            if not matched:
                # Skip unknown IPA characters (ties, diacritics, etc.)
                char = ipa[i]
                if char.isalpha():
                    result.append(char)  # Keep unknown alphabetic chars
                elif char == " ":
                    result.append(" ")
                # Skip non-alphabetic IPA symbols
                i += 1

        return "".join(result)


# ============================================================================
# Helper Functions
# ============================================================================

def _tokenize_for_g2p(text: str) -> list[str]:
    """Split text into words and non-word tokens (whitespace, punctuation).

    Preserves the original structure so we can reconstruct the text.
    Returns list where joining all elements reproduces the original text.

    Examples:
        "Il sig. Schmidt, ok?" → ["Il", " ", "sig", ".", " ", "Schmidt", ",", " ", "ok", "?"]
    """
    tokens = re.findall(r"[a-zA-ZàáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿßœšžŸ]+|[^a-zA-ZàáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿßœšžŸ]+", text, re.UNICODE)
    return tokens


# ============================================================================
# Convenience Functions
# ============================================================================

# Module-level singleton for simple usage
_default_pipeline: Optional[G2PPipeline] = None


def get_default_pipeline() -> G2PPipeline:
    """Get or create the default G2P pipeline singleton."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = G2PPipeline()
    return _default_pipeline


def process_text(text: str, lang: str, dictionary: CustomDictionary = None) -> str:
    """Process text through G2P pipeline (convenience function).

    Args:
        text: input text
        lang: language code (it, fr, de, es, pt, en)
        dictionary: optional custom dictionary

    Returns:
        Text with foreign words respelled.
    """
    if dictionary:
        pipeline = G2PPipeline(custom_dict=dictionary)
        return pipeline.process(text, lang)
    return get_default_pipeline().process(text, lang)


def ipa_to_respelling(ipa: str, target_lang: str) -> str:
    """Convert IPA string to target language orthography (standalone function).

    Used by SSML <phoneme> tag to convert IPA pronunciation overrides
    to orthographic respellings the BPE tokenizer can handle.

    Args:
        ipa: IPA transcription (e.g., "ˈʃmɪt" for "Schmidt")
        target_lang: target language code (it, fr, de, es, pt, en)

    Returns:
        Orthographic respelling (e.g., "shmit" for Italian)
    """
    return get_default_pipeline()._ipa_to_respelling(ipa, target_lang)
