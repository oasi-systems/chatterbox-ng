import logging
import json
import re

import torch
from pathlib import Path
from unicodedata import category, normalize
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt


# Model repository
REPO_ID = "ResembleAI/chatterbox"

# Global instances for optional dependencies
_kakasi = None
_dicta = None
_russian_stresser = None


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese text normalization: converts kanji to hiragana; katakana remains the same."""
    global _kakasi
    
    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        
        result = _kakasi.convert(text)
        out = []
        
        for r in result:
            inp = r['orig']
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif all([is_katakana(c) for c in inp]) if inp else False:  # Safety check for empty inp
                out.append(r['orig'])

            else:
                out.append(inp)
        
        normalized_text = "".join(out)
        
        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', normalized_text)
        
        return normalized_text
        
    except ImportError:
        logger.warning("pykakasi not available - Japanese text processing skipped")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta
    
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()
        
        return _dicta.add_diacritics(text)
        
    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        logger.warning(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""
    
    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        
        return initial + medial + final
    
    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)    
    return result.strip()


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""
    
    def __init__(self, model_dir=None):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping(model_dir)
        self._init_segmenter()
    
    def _load_cangjie_mapping(self, model_dir=None):
        """Load Cangjie mapping from HuggingFace model repository."""        
        try:
            cangjie_file = hf_hub_download(
                repo_id=REPO_ID,
                filename="Cangjie5_TC.json",
                cache_dir=model_dir
            )
            
            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)
                    
        except Exception as e:
            logger.warning(f"Could not load Cangjie mapping: {e}")
    
    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None
    
    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)
    

    
    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text
        
        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


# Italian abbreviation dictionary (lowercase, applied after preprocess_text)
_ITALIAN_ABBREVIATIONS = {
    r'\bsig\.ra\b': 'signora',
    r'\bsig\.na\b': 'signorina',
    r'\bsig\b\.': 'signore',
    r'\bdott\.ssa\b': 'dottoressa',
    r'\bdott\b\.': 'dottore',
    r'\bprof\.ssa\b': 'professoressa',
    r'\bprof\b\.': 'professore',
    r'\bavv\b\.': 'avvocato',
    r'\bing\b\.': 'ingegnere',
    r'\barch\b\.': 'architetto',
    r'\bgeom\b\.': 'geometra',
    r'\brag\b\.': 'ragioniere',
    r'\bcomm\b\.': 'commendatore',
    r'\bon\b\.': 'onorevole',
    r'\bsen\b\.': 'senatore',
    r'\bgen\b\.': 'generale',
    r'\bcol\b\.': 'colonnello',
    r'\bcap\b\.': 'capitano',
    r'\bpag\b\.': 'pagina',
    r'\bvol\b\.': 'volume',
    r'\bcapit\b\.': 'capitolo',
    r'\bfig\b\.': 'figura',
    r'\btab\b\.': 'tabella',
    r'\becc\b\.': 'eccetera',
    r'\bes\b\.': 'esempio',
    r'\bn\b\.': 'numero',
}

# Italian symbol replacements
_ITALIAN_SYMBOLS = {
    '€': ' euro',
    '$': ' dollari',
    '£': ' sterline',
    '%': ' percento',
    '&': ' e ',
    '+': ' più ',
    '=': ' uguale ',
    '@': ' chiocciola ',
    '«': '"',
    '»': '"',
}

# Italian month names for date normalization
_ITALIAN_MONTHS = {
    '01': 'gennaio', '1': 'gennaio',
    '02': 'febbraio', '2': 'febbraio',
    '03': 'marzo', '3': 'marzo',
    '04': 'aprile', '4': 'aprile',
    '05': 'maggio', '5': 'maggio',
    '06': 'giugno', '6': 'giugno',
    '07': 'luglio', '7': 'luglio',
    '08': 'agosto', '8': 'agosto',
    '09': 'settembre', '9': 'settembre',
    '10': 'ottobre',
    '11': 'novembre',
    '12': 'dicembre',
    'gennaio': 'gennaio', 'febbraio': 'febbraio', 'marzo': 'marzo',
    'aprile': 'aprile', 'maggio': 'maggio', 'giugno': 'giugno',
    'luglio': 'luglio', 'agosto': 'agosto', 'settembre': 'settembre',
    'ottobre': 'ottobre', 'novembre': 'novembre', 'dicembre': 'dicembre',
    'gen': 'gennaio', 'feb': 'febbraio', 'mar': 'marzo',
    'apr': 'aprile', 'mag': 'maggio', 'giu': 'giugno',
    'lug': 'luglio', 'ago': 'agosto', 'set': 'settembre',
    'ott': 'ottobre', 'nov': 'novembre', 'dic': 'dicembre',
}

# Acronyms that should be spelled out letter-by-letter
_ITALIAN_SPELL_OUT_ACRONYMS = {
    'onu', 'usa', 'fbi', 'cia', 'bbc', 'cnn', 'rai', 'iva', 'inps', 'asl',
    'pdf', 'url', 'html', 'css', 'xml', 'api', 'gpu', 'cpu', 'ram', 'rom',
    'sms', 'gps', 'usb', 'led', 'lcd', 'dvd', 'vip', 'dna', 'rna',
    'pm', 'am', 'pc', 'tv', 'cd', 'dj', 'ok',
}

# Acronyms that should be read as words (not spelled out)
_ITALIAN_WORD_ACRONYMS = {
    'nato', 'unesco', 'unicef', 'fiat', 'istat', 'enea', 'ansa',
    'laser', 'radar', 'sim', 'pin', 'ban', 'cap',
}

# Italian letter pronunciation for spelling out acronyms
_ITALIAN_LETTER_NAMES = {
    'a': 'a', 'b': 'bi', 'c': 'ci', 'd': 'di', 'e': 'e', 'f': 'effe',
    'g': 'gi', 'h': 'acca', 'i': 'i', 'j': 'i lunga', 'k': 'cappa',
    'l': 'elle', 'm': 'emme', 'n': 'enne', 'o': 'o', 'p': 'pi',
    'q': 'cu', 'r': 'erre', 's': 'esse', 't': 'ti', 'u': 'u',
    'v': 'vu', 'w': 'doppia vu', 'x': 'ics', 'y': 'ipsilon', 'z': 'zeta',
}


def _italian_spell_acronym(acronym: str) -> str:
    """Spell out an acronym letter by letter in Italian."""
    return ' '.join(_ITALIAN_LETTER_NAMES.get(c, c) for c in acronym.lower())


def _italian_normalize_acronym(match: re.Match) -> str:
    """Decide whether to spell out or read as word an uppercase acronym."""
    word = match.group(0)
    lower = word.lower()
    # Known word-acronyms: keep as-is (the model reads them as words)
    if lower in _ITALIAN_WORD_ACRONYMS:
        return lower
    # Known spell-out acronyms or any 2-5 letter all-caps word
    if lower in _ITALIAN_SPELL_OUT_ACRONYMS or (2 <= len(word) <= 5):
        return _italian_spell_acronym(word)
    return word


def italian_text_normalize(text: str) -> str:
    """Italian text normalization: numbers to words, abbreviation expansion, symbol replacement,
    dates, times, phone numbers, acronyms, and currency expressions."""
    try:
        from num2words import num2words
    except ImportError:
        logger.warning("num2words not available - Italian number normalization skipped")
        return text

    try:
        # 1. Expand abbreviations (longer patterns first to avoid partial matches)
        for pattern, replacement in _ITALIAN_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 2. Phone numbers BEFORE symbol replacement (to preserve '+' in phone numbers)
        def replace_phone(match):
            full = match.group(0)
            parts = re.split(r'[\s\-]+', full.replace('+', ''))
            try:
                spoken_parts = []
                for part in parts:
                    if part:
                        spoken_parts.append(num2words(int(part), lang='it'))
                return ', '.join(spoken_parts)
            except Exception:
                return full

        text = re.sub(r'\+?\d{2,4}[\s\-]\d{2,4}(?:[\s\-]\d{2,7}){1,3}', replace_phone, text)

        # 3. Currency BEFORE symbol replacement (to handle €/$/ with amounts)
        def _currency_to_words(amount_str, symbol):
            currency_map = {'€': 'euro', '$': 'dollari', '£': 'sterline'}
            singular_map = {'€': 'euro', '$': 'dollaro', '£': 'sterlina'}
            amount = int(amount_str)
            amount_words = num2words(amount, lang='it')
            # "uno" → "un" before currency nouns (Italian grammar)
            if amount == 1:
                amount_words = 'un'
            name = singular_map.get(symbol, symbol) if amount == 1 else currency_map.get(symbol, symbol)
            return f"{amount_words} {name}"

        def replace_currency_after(match):
            try:
                return _currency_to_words(match.group(1), match.group(2))
            except Exception:
                return match.group(0)

        def replace_currency_before(match):
            try:
                return _currency_to_words(match.group(2), match.group(1))
            except Exception:
                return match.group(0)

        text = re.sub(r'(\d+)\s*([€$£])', replace_currency_after, text)
        text = re.sub(r'([€$£])\s*(\d+)', replace_currency_before, text)

        # 4. Replace remaining symbols
        for symbol, replacement in _ITALIAN_SYMBOLS.items():
            text = text.replace(symbol, replacement)

        # 5. Ordinals: 1°, 2°, 3ª etc. — BEFORE dates/times to avoid conflicts
        def replace_ordinal(match):
            n = int(match.group(1))
            suffix = match.group(2)
            try:
                if suffix in ('\u00aa', 'a'):  # ª or a
                    return num2words(n, to='ordinal', lang='it').rstrip('o') + 'a'
                return num2words(n, to='ordinal', lang='it')
            except Exception:
                return match.group(0)

        text = re.sub(r'(\d+)([°ºªa])(?=\s|$|[,.])', replace_ordinal, text)

        # 6. Times: HH:MM — BEFORE dates to avoid HH:MM being eaten by date regex
        def replace_time(match):
            hours, minutes = int(match.group(1)), int(match.group(2))
            try:
                h_words = num2words(hours, lang='it')
                if minutes == 0:
                    return f"le {h_words}"
                m_words = num2words(minutes, lang='it')
                return f"le {h_words} e {m_words}"
            except Exception:
                return match.group(0)

        text = re.sub(r'\b(\d{1,2}):(\d{2})\b', replace_time, text)

        # 7. Dates: DD/MM/YYYY, DD-MM-YYYY (not dot — conflicts with abbreviations)
        def replace_date_numeric(match):
            day, month, year = match.group(1), match.group(2), match.group(3)
            try:
                month_name = _ITALIAN_MONTHS.get(month.lstrip('0') or '0', month)
                day_words = 'primo' if int(day) == 1 else num2words(int(day), lang='it')
                if year:
                    year_words = num2words(int(year), lang='it')
                    return f"{day_words} {month_name} {year_words}"
                return f"{day_words} {month_name}"
            except Exception:
                return match.group(0)

        text = re.sub(
            r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b',
            replace_date_numeric, text
        )

        # "15 marzo 2024" or "15 marzo" (day + month name + optional year)
        # Only match actual month names, not month numbers
        written_months = [m for m in _ITALIAN_MONTHS.keys() if not m.isdigit()]
        month_pattern = '|'.join(sorted(written_months, key=len, reverse=True))

        def replace_date_written(match):
            day, month_str, year = match.group(1), match.group(2).lower(), match.group(3)
            try:
                month_name = _ITALIAN_MONTHS.get(month_str, month_str)
                day_words = 'primo' if int(day) == 1 else num2words(int(day), lang='it')
                if year:
                    year_words = num2words(int(year), lang='it')
                    return f"{day_words} {month_name} {year_words}"
                return f"{day_words} {month_name}"
            except Exception:
                return match.group(0)

        text = re.sub(
            rf'\b(\d{{1,2}})\s+({month_pattern})(?:\s+(\d{{4}}))?\b',
            replace_date_written, text, flags=re.IGNORECASE
        )

        # 8. Acronyms: uppercase sequences (2-5 letters, not part of a word)
        text = re.sub(r'\b[A-Z]{2,5}\b', _italian_normalize_acronym, text)

        # 9. Numbers with decimal comma: 3,14 → tre virgola quattordici
        def replace_decimal(match):
            integer_part = match.group(1)
            decimal_part = match.group(2)
            try:
                int_words = num2words(int(integer_part), lang='it')
                dec_words = num2words(int(decimal_part), lang='it')
                return f"{int_words} virgola {dec_words}"
            except Exception:
                return match.group(0)

        text = re.sub(r'(\d+),(\d+)', replace_decimal, text)

        # 10. Cardinal numbers (standalone or within text)
        def replace_number(match):
            num_str = match.group(0)
            try:
                n = int(num_str)
                return num2words(n, lang='it')
            except Exception:
                return num_str

        text = re.sub(r'\b\d+\b', replace_number, text)

        # 11. Italian prosody normalization
        # Ensure trailing question marks for rhetorical Italian patterns
        # "vero" / "no" / "eh" at end of sentence without ? → add ?
        text = re.sub(r',\s*(vero|no|eh|già|giusto)\s*([.!]|$)', r', \1?', text)

        # Normalize ellipsis for natural pause: "..." → "…" (single char, cleaner for model)
        text = re.sub(r'\.{3,}', '…', text)

        # Italian em-dash parenthetical: "— testo —" → ", testo," for natural pause
        text = re.sub(r'\s*[—–]\s*', ', ', text)

        # Repeated punctuation (emphasis): "!!" or "???" → single with implicit emphasis
        text = re.sub(r'([!?])\1+', r'\1', text)

        # Ensure space after comma for natural rhythm (common Italian writing omits it)
        text = re.sub(r',(?=\S)', ', ', text)

        # 12. Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    except Exception as e:
        logger.warning(f"Italian text normalization failed: {e}")
        return text


def add_russian_stress(text: str) -> str:
    """Russian text normalization: adds stress marks to Russian text."""
    global _russian_stresser
    
    try:
        if _russian_stresser is None:
            from russian_text_stresser.text_stresser import RussianTextStresser
            _russian_stresser = RussianTextStresser()
        
        return _russian_stresser.stress_text(text)
        
    except ImportError:
        logger.warning("russian_text_stresser not available - Russian stress labeling skipped")
        return text
    except Exception as e:
        logger.warning(f"Russian stress labeling failed: {e}")
        return text


class MTLTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        model_dir = Path(vocab_file_path).parent
        self.cangjie_converter = ChineseCangjieConverter(model_dir)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def preprocess_text(self, raw_text: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
        """
        Text preprocessor that handles lowercase conversion and NFKD normalization.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if nfkd_normalize:
            preprocessed_text = normalize("NFKD", preprocessed_text)
        
        return preprocessed_text

    def text_to_tokens(self, text: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
        text_tokens = self.encode(text, language_id=language_id, lowercase=lowercase, nfkd_normalize=nfkd_normalize)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
        txt = self.preprocess_text(txt, language_id=language_id, lowercase=lowercase, nfkd_normalize=nfkd_normalize)
        
        # Language-specific text processing
        if language_id == 'zh':
            txt = self.cangjie_converter(txt)
        elif language_id == 'ja':
            txt = hiragana_normalize(txt)
        elif language_id == 'he':
            txt = add_hebrew_diacritics(txt)
        elif language_id == 'ko':
            txt = korean_normalize(txt)
        elif language_id == 'ru':
            txt = add_russian_stress(txt)
        elif language_id == 'it':
            txt = italian_text_normalize(txt)

        # Prepend language token
        if language_id:
            txt = f"[{language_id.lower()}]{txt}"
        
        txt = txt.replace(' ', SPACE)
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '').replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')
        return txt
