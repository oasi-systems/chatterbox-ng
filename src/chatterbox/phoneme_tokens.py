"""
Phoneme token definitions and embedding initialization for ChatterBox NG.

Defines the IPA phoneme set for 6 European languages (IT, EN, FR, DE, ES, PT),
and provides logic to:
1. Extend the BPE vocabulary with phoneme tokens
2. Initialize phoneme embeddings from related grapheme embeddings
3. Phonemize text via espeak-ng and encode as token IDs

Phoneme tokens are added AFTER the existing BPE vocab (IDs 2454+).
The original model weights are preserved exactly — only new rows are added.

Token format: [ph_X] where X is the IPA symbol.
Example: [ph_k], [ph_tʃ], [ph_ə]
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Phoneme Set — IPA symbols for IT/EN/FR/DE/ES/PT
# ============================================================================
# Ordered list. Token ID = VOCAB_OFFSET + index.

PHONEME_LIST = [
    # --- Vowels ---
    "a",     # open front (IT: casa, ES: casa)
    "e",     # close-mid front (IT: sera, FR: été)
    "i",     # close front (all languages)
    "o",     # close-mid back (IT: come, ES: como)
    "u",     # close back (all languages)
    "ɛ",     # open-mid front (IT: bello, FR: fête)
    "ɔ",     # open-mid back (IT: cosa, FR: bonne)
    "ə",     # schwa (EN: about, DE: bitte, FR: le)
    "æ",     # near-open front (EN: cat)
    "ɑ",     # open back (EN: father, FR: pâte)
    "ʌ",     # open-mid back (EN: cup)
    "ɪ",     # near-close front (EN: sit)
    "ʊ",     # near-close back (EN: put)
    "y",     # close front rounded (FR: tu, DE: über)
    "ø",     # close-mid front rounded (FR: deux, DE: schön)
    "œ",     # open-mid front rounded (FR: peur, DE: Hölle)
    "ʏ",     # near-close front rounded (DE: hübsch)

    # --- Nasal vowels (French) ---
    "ɑ̃",     # FR: an, en
    "ɛ̃",     # FR: in, ain
    "ɔ̃",     # FR: on
    "œ̃",     # FR: un

    # --- Diphthongs (as single tokens) ---
    "aɪ",    # EN: my, DE: mein
    "aʊ",    # EN: now, DE: Haus
    "ɔɪ",    # EN: boy, DE: neu
    "eɪ",    # EN: day
    "oʊ",    # EN: go
    "ɪə",    # EN: ear
    "eə",    # EN: air
    "ʊə",    # EN: cure

    # --- Plosives ---
    "p",     # all languages
    "b",     # all languages
    "t",     # all languages
    "d",     # all languages
    "k",     # all languages
    "ɡ",     # all languages (note: IPA ɡ, not ASCII g)

    # --- Fricatives ---
    "f",     # all languages
    "v",     # all languages
    "s",     # all languages
    "z",     # all languages
    "ʃ",     # EN: she, FR: chat, DE: schön, IT: scena
    "ʒ",     # EN: vision, FR: je, PT: gente
    "θ",     # EN: think, ES: caza (Castilian)
    "ð",     # EN: this, ES: cada
    "x",     # ES: jota, DE: Bach
    "ç",     # DE: ich
    "h",     # EN: hat, DE: Haus
    "ɣ",     # ES: haga, PT: amigo
    "β",     # ES: haber
    "ɸ",     # rare, bilabial fricative
    "ʁ",     # FR: rouge, DE: rot (uvular)

    # --- Affricates ---
    "tʃ",    # EN: church, IT: ciao, ES: mucho
    "dʒ",    # EN: judge, IT: giorno
    "ts",    # DE: Zeit, IT: pizza
    "dz",    # IT: zero

    # --- Nasals ---
    "m",     # all languages
    "n",     # all languages
    "ɲ",     # IT: gnocchi, FR: agneau, ES: año
    "ŋ",     # EN: sing, DE: lang

    # --- Liquids ---
    "l",     # all languages
    "ʎ",     # IT: famiglia, ES: llama, PT: filho
    "r",     # IT/ES/PT: trilled r
    "ɾ",     # ES/PT/IT: tap r
    "ɹ",     # EN: red (approximant)
    "ʀ",     # DE/FR: uvular trill (rare)

    # --- Semivowels ---
    "j",     # all languages (yes, you)
    "w",     # all languages (we, oui)
    "ɥ",     # FR: huit (labio-palatal)

    # --- Suprasegmentals ---
    "ˈ",     # primary stress
    "ˌ",     # secondary stress
    "ː",     # vowel length

    # --- Boundary markers ---
    " ",     # word boundary (within phoneme sequence)
]

# Number of phoneme tokens
N_PHONEMES = len(PHONEME_LIST)

# Token prefix
PHONEME_PREFIX = "[ph_"
PHONEME_SUFFIX = "]"

# Special token to mark "this is phoneme mode"
PHONEME_MODE_TOKEN = "[PHON]"


def phoneme_token_name(phoneme: str) -> str:
    """Get the token name for a phoneme. E.g., 'tʃ' → '[ph_tʃ]'"""
    return f"{PHONEME_PREFIX}{phoneme}{PHONEME_SUFFIX}"


def get_phoneme_token_names() -> list[str]:
    """Get all phoneme token names in order."""
    return [PHONEME_MODE_TOKEN] + [phoneme_token_name(p) for p in PHONEME_LIST]


def get_all_new_tokens() -> list[str]:
    """Get all new tokens to add (mode marker + phonemes)."""
    return get_phoneme_token_names()


# ============================================================================
# Grapheme-to-Phoneme Embedding Initialization
# ============================================================================
# For each IPA phoneme, we map it to the BPE grapheme tokens that commonly
# produce that sound. The phoneme embedding is initialized as the MEAN
# of those grapheme embeddings. This gives the transformer a reasonable
# starting point — the phoneme /k/ starts near the grapheme "c"/"k".

# Map: IPA phoneme → list of BPE grapheme tokens (by character)
# These are individual characters that exist in the BPE vocab.
_PHONEME_TO_GRAPHEMES = {
    # Vowels
    "a": ["a"],
    "e": ["e"],
    "i": ["i"],
    "o": ["o"],
    "u": ["u"],
    "ɛ": ["e", "è"],
    "ɔ": ["o", "ò"],
    "ə": ["e", "a"],
    "æ": ["a", "e"],
    "ɑ": ["a"],
    "ʌ": ["a", "u"],
    "ɪ": ["i", "e"],
    "ʊ": ["u", "o"],
    "y": ["u", "ü"],
    "ø": ["ö", "o", "e"],
    "œ": ["ö", "o", "e"],
    "ʏ": ["ü", "u"],
    # Nasal vowels
    "ɑ̃": ["a", "n"],
    "ɛ̃": ["i", "n"],
    "ɔ̃": ["o", "n"],
    "œ̃": ["u", "n"],
    # Diphthongs
    "aɪ": ["a", "i"],
    "aʊ": ["a", "u"],
    "ɔɪ": ["o", "i"],
    "eɪ": ["e", "i"],
    "oʊ": ["o", "u"],
    "ɪə": ["i", "a"],
    "eə": ["e", "a"],
    "ʊə": ["u", "a"],
    # Plosives
    "p": ["p"],
    "b": ["b"],
    "t": ["t"],
    "d": ["d"],
    "k": ["k", "c"],
    "ɡ": ["g"],
    # Fricatives
    "f": ["f"],
    "v": ["v"],
    "s": ["s"],
    "z": ["z"],
    "ʃ": ["s", "c"],
    "ʒ": ["g", "j"],
    "θ": ["t", "h"],
    "ð": ["d"],
    "x": ["h", "c"],
    "ç": ["h", "c"],
    "h": ["h"],
    "ɣ": ["g"],
    "β": ["b", "v"],
    "ɸ": ["f"],
    "ʁ": ["r"],
    # Affricates
    "tʃ": ["c", "t"],
    "dʒ": ["g", "d"],
    "ts": ["z", "t"],
    "dz": ["z", "d"],
    # Nasals
    "m": ["m"],
    "n": ["n"],
    "ɲ": ["n", "g"],
    "ŋ": ["n", "g"],
    # Liquids
    "l": ["l"],
    "ʎ": ["l", "g"],
    "r": ["r"],
    "ɾ": ["r"],
    "ɹ": ["r"],
    "ʀ": ["r"],
    # Semivowels
    "j": ["i", "y"],
    "w": ["u", "w"],
    "ɥ": ["u", "i"],
    # Suprasegmentals — init from neutral
    "ˈ": [],  # stress — random init
    "ˌ": [],  # secondary stress — random init
    "ː": [],  # length — random init
    # Boundary
    " ": [],   # word boundary — random init
}


def initialize_phoneme_embeddings(existing_emb_weight, vocab: dict, emb_std: float = 0.0152):
    """Create initialized embeddings for phoneme tokens.

    Args:
        existing_emb_weight: tensor [2454, 1024] — current text_emb weights
        vocab: dict mapping token string → token ID
        emb_std: std for random init (for tokens without grapheme mapping)

    Returns:
        tensor [N_new_tokens, 1024] — embeddings for new tokens
    """
    import torch

    dim = existing_emb_weight.shape[1]
    n_new = len(get_all_new_tokens())
    new_emb = torch.zeros(n_new, dim)

    # First token is [PHON] mode marker — random init
    new_emb[0].normal_(0, emb_std)

    # Phoneme tokens
    for i, phoneme in enumerate(PHONEME_LIST):
        idx = i + 1  # offset by 1 for [PHON] token
        graphemes = _PHONEME_TO_GRAPHEMES.get(phoneme, [])

        if graphemes:
            # Average of corresponding grapheme embeddings
            grapheme_ids = []
            for g in graphemes:
                if g in vocab:
                    grapheme_ids.append(vocab[g])

            if grapheme_ids:
                selected = existing_emb_weight[grapheme_ids]
                new_emb[idx] = selected.mean(dim=0)
                continue

        # Fallback: random init with same distribution as existing
        new_emb[idx].normal_(0, emb_std)

    return new_emb


# ============================================================================
# Phonemizer (espeak-ng)
# ============================================================================

_HAS_PHONEMIZER = False
try:
    from phonemizer.backend import EspeakBackend
    from phonemizer.phonemize import phonemize as _phonemize
    _HAS_PHONEMIZER = True
except ImportError:
    pass

_ESPEAK_LANG_MAP = {
    "it": "it",
    "fr": "fr-fr",
    "de": "de",
    "es": "es",
    "pt": "pt",
    "en": "en-us",
}


def text_to_phonemes(text: str, lang: str) -> Optional[str]:
    """Convert text to IPA phonemes using espeak-ng.

    Args:
        text: input text
        lang: language code (it, en, fr, de, es, pt)

    Returns:
        IPA phoneme string, or None if espeak-ng unavailable.
    """
    if not _HAS_PHONEMIZER:
        logger.warning("phonemizer not installed — cannot convert to phonemes")
        return None

    espeak_lang = _ESPEAK_LANG_MAP.get(lang, lang)

    try:
        result = _phonemize(
            text,
            language=espeak_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
        return result.strip()
    except Exception as e:
        logger.error(f"Phonemization failed for [{lang}]: {e}")
        return None


def phonemes_to_token_ids(phoneme_string: str, token_to_id: dict) -> list[int]:
    """Convert an IPA phoneme string to a list of token IDs.

    Handles multi-character phonemes (tʃ, dʒ, etc.) by greedy matching.

    Args:
        phoneme_string: IPA string from espeak-ng
        token_to_id: mapping from token name → token ID

    Returns:
        List of token IDs.
    """
    ids = []

    # Add phoneme mode marker
    mode_id = token_to_id.get(PHONEME_MODE_TOKEN)
    if mode_id is not None:
        ids.append(mode_id)

    i = 0
    while i < len(phoneme_string):
        matched = False

        # Try longest phoneme match first (3 chars, then 2, then 1)
        for length in (3, 2, 1):
            if i + length <= len(phoneme_string):
                chunk = phoneme_string[i:i + length]
                token_name = phoneme_token_name(chunk)
                if token_name in token_to_id:
                    ids.append(token_to_id[token_name])
                    i += length
                    matched = True
                    break

        if not matched:
            # Skip unknown characters (punctuation passes through)
            char = phoneme_string[i]
            # Check if it's a space (word boundary)
            if char == " ":
                space_token = phoneme_token_name(" ")
                if space_token in token_to_id:
                    ids.append(token_to_id[space_token])
            i += 1

    return ids
