"""
Text normalization for European languages: French, German, Spanish, Portuguese, English.
Converts numbers, dates, times, currency, abbreviations to spoken words.

Follows the same pattern as italian_text_normalize in tokenizer.py.
"""
import re
import logging

logger = logging.getLogger(__name__)

try:
    from num2words import num2words
    _HAS_NUM2WORDS = True
except ImportError:
    _HAS_NUM2WORDS = False
    logger.warning("num2words not available - number normalization disabled")


# ============================================================================
# FRENCH
# ============================================================================

_FRENCH_ABBREVIATIONS = {
    r'\bM\.\s': 'monsieur ',
    r'\bMme\.?\s': 'madame ',
    r'\bMlle\.?\s': 'mademoiselle ',
    r'\bDr\.?\s': 'docteur ',
    r'\bPr\.?\s': 'professeur ',
    r'\bMe\.?\s': 'maître ',
    r'\bSt\.?\s': 'saint ',
    r'\bSte\.?\s': 'sainte ',
    r'\bav\.?\s': 'avenue ',
    r'\bbd\.?\s': 'boulevard ',
    r'\bpl\.?\s': 'place ',
    r'\brue\.?\s': 'rue ',
    r'\btél\.?\s': 'téléphone ',
    r'\bn°\s?': 'numéro ',
    r'\betc\.': 'et cetera',
    r'\bcàd\.?': "c'est-à-dire",
    r'\bex\.?\s': 'exemple ',
    r'\benv\.?\s': 'environ ',
    r'\bréf\.?\s': 'référence ',
}

_FRENCH_SYMBOLS = {
    '%': ' pour cent',
    '&': ' et ',
    '+': ' plus ',
    '=': ' égal ',
    '@': ' arobase ',
    '«': '',
    '»': '',
    '"': '',
    '"': '',
}

_FRENCH_MONTHS = {
    '1': 'janvier', '01': 'janvier', 'janvier': 'janvier',
    '2': 'février', '02': 'février', 'février': 'février',
    '3': 'mars', '03': 'mars', 'mars': 'mars',
    '4': 'avril', '04': 'avril', 'avril': 'avril',
    '5': 'mai', '05': 'mai', 'mai': 'mai',
    '6': 'juin', '06': 'juin', 'juin': 'juin',
    '7': 'juillet', '07': 'juillet', 'juillet': 'juillet',
    '8': 'août', '08': 'août', 'août': 'août',
    '9': 'septembre', '09': 'septembre', 'septembre': 'septembre',
    '10': 'octobre', 'octobre': 'octobre',
    '11': 'novembre', 'novembre': 'novembre',
    '12': 'décembre', 'décembre': 'décembre',
}


def french_text_normalize(text: str) -> str:
    """French text normalization."""
    if not _HAS_NUM2WORDS:
        return text
    try:
        # 1. Abbreviations
        for pattern, replacement in _FRENCH_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 1b. Thousands separator: 425.000 → 425000
        text = re.sub(r'\b(\d{1,3})(?:\.(\d{3}))+\b', lambda m: m.group(0).replace('.', ''), text)

        # 2. Phone numbers
        def replace_phone(match):
            full = match.group(0).replace('+', '')
            parts = re.split(r'[\s\-\.]+', full)
            try:
                return ', '.join(num2words(int(p), lang='fr') for p in parts if p)
            except Exception:
                return match.group(0)
        text = re.sub(r'\+?\d{2,4}[\s\-\.]\d{2,4}(?:[\s\-\.]\d{2,7}){1,3}', replace_phone, text)

        # 3. Currency
        def _fr_currency(amount_str, symbol):
            names = {'€': 'euro', '$': 'dollar', '£': 'livre'}
            plurals = {'€': 'euros', '$': 'dollars', '£': 'livres'}
            amount = int(amount_str)
            words = num2words(amount, lang='fr')
            name = names.get(symbol, symbol) if amount <= 1 else plurals.get(symbol, symbol)
            return f"{words} {name}"

        text = re.sub(r'(\d+)\s*([€$£])', lambda m: _fr_currency(m.group(1), m.group(2)), text)
        text = re.sub(r'([€$£])\s*(\d+)', lambda m: _fr_currency(m.group(2), m.group(1)), text)

        # 4. Symbols
        for sym, rep in _FRENCH_SYMBOLS.items():
            text = text.replace(sym, rep)

        # 5. Ordinals: 1er, 2e, 3ème
        def replace_ordinal(match):
            n = int(match.group(1))
            try:
                return num2words(n, to='ordinal', lang='fr')
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+)(?:er|ère|e|ème)(?=\s|$|[,.])', replace_ordinal, text)

        # 6. Time: 14h30, 14:30
        def replace_time(match):
            h, m = int(match.group(1)), int(match.group(2))
            try:
                h_w = num2words(h, lang='fr')
                if m == 0:
                    return f"{h_w} heures"
                m_w = num2words(m, lang='fr')
                return f"{h_w} heures {m_w}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2})[h:](\d{2})\b', replace_time, text)

        # 7. Dates: DD/MM/YYYY
        def replace_date(match):
            day, month, year = match.group(1), match.group(2), match.group(3)
            try:
                month_name = _FRENCH_MONTHS.get(month.lstrip('0'), month)
                day_w = 'premier' if int(day) == 1 else num2words(int(day), lang='fr')
                if year:
                    year_w = num2words(int(year), lang='fr')
                    return f"{day_w} {month_name} {year_w}"
                return f"{day_w} {month_name}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b', replace_date, text)

        # 8. Decimals: 3,14 → trois virgule quatorze
        def replace_decimal(match):
            try:
                return f"{num2words(int(match.group(1)), lang='fr')} virgule {num2words(int(match.group(2)), lang='fr')}"
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+),(\d+)', replace_decimal, text)

        # 9. Cardinal numbers
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='fr'), text)

        # 10. Prosody
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r'\s*[—–]\s*', ', ', text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r',(?=\S)', ', ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logger.warning(f"French normalization failed: {e}")
        return text


# ============================================================================
# GERMAN
# ============================================================================

_GERMAN_ABBREVIATIONS = {
    r'\bHr\.?\s': 'Herr ',
    r'\bFr\.?\s': 'Frau ',
    r'\bDr\.?\s': 'Doktor ',
    r'\bProf\.?\s': 'Professor ',
    r'\bNr\.?\s': 'Nummer ',
    r'\bStr\.?\s': 'Straße ',
    r'\bz\.?\s?B\.': 'zum Beispiel',
    r'\bd\.?\s?h\.': 'das heißt',
    r'\busw\.': 'und so weiter',
    r'\bbzw\.': 'beziehungsweise',
    r'\bca\.': 'circa',
    r'\bggf\.': 'gegebenenfalls',
    r'\bevtl\.': 'eventuell',
    r'\bTel\.?\s': 'Telefon ',
    r'\bAbt\.?\s': 'Abteilung ',
    r'\bMio\.': 'Millionen',
    r'\bMrd\.': 'Milliarden',
}

_GERMAN_SYMBOLS = {
    '%': ' Prozent',
    '&': ' und ',
    '+': ' plus ',
    '=': ' gleich ',
    '@': ' at ',
    '«': '',
    '»': '',
    '"': '',
    '"': '',
}

_GERMAN_MONTHS = {
    '1': 'Januar', '01': 'Januar',
    '2': 'Februar', '02': 'Februar',
    '3': 'März', '03': 'März',
    '4': 'April', '04': 'April',
    '5': 'Mai', '05': 'Mai',
    '6': 'Juni', '06': 'Juni',
    '7': 'Juli', '07': 'Juli',
    '8': 'August', '08': 'August',
    '9': 'September', '09': 'September',
    '10': 'Oktober',
    '11': 'November',
    '12': 'Dezember',
}


def german_text_normalize(text: str) -> str:
    """German text normalization."""
    if not _HAS_NUM2WORDS:
        return text
    try:
        # 1. Abbreviations
        for pattern, replacement in _GERMAN_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 1b. Thousands separator: 425.000 → 425000 (must be before dates/decimals)
        text = re.sub(r'\b(\d{1,3})(?:\.(\d{3}))+\b', lambda m: m.group(0).replace('.', ''), text)

        # 2. Phone numbers
        def replace_phone(match):
            full = match.group(0).replace('+', '')
            parts = re.split(r'[\s\-]+', full)
            try:
                return ', '.join(num2words(int(p), lang='de') for p in parts if p)
            except Exception:
                return match.group(0)
        text = re.sub(r'\+?\d{2,4}[\s\-]\d{2,4}(?:[\s\-]\d{2,7}){1,3}', replace_phone, text)

        # 3. Currency
        def _de_currency(amount_str, symbol):
            names = {'€': 'Euro', '$': 'Dollar', '£': 'Pfund'}
            amount = int(amount_str)
            words = num2words(amount, lang='de')
            name = names.get(symbol, symbol)
            return f"{words} {name}"

        text = re.sub(r'(\d+)\s*([€$£])', lambda m: _de_currency(m.group(1), m.group(2)), text)
        text = re.sub(r'([€$£])\s*(\d+)', lambda m: _de_currency(m.group(2), m.group(1)), text)

        # 4. Symbols
        for sym, rep in _GERMAN_SYMBOLS.items():
            text = text.replace(sym, rep)

        # 5. Ordinals: 1., 2., 3.
        def replace_ordinal(match):
            n = int(match.group(1))
            try:
                return num2words(n, to='ordinal', lang='de')
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+)\.(?=\s|$|[,])', replace_ordinal, text)

        # 6. Time: 14:30 Uhr, 14 Uhr
        def replace_time(match):
            h = int(match.group(1))
            m = int(match.group(2)) if match.group(2) else 0
            try:
                h_w = num2words(h, lang='de')
                if m == 0:
                    return f"{h_w} Uhr"
                m_w = num2words(m, lang='de')
                return f"{h_w} Uhr {m_w}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2}):(\d{2})\s*(?:Uhr)?', replace_time, text)

        # 7. Dates: DD.MM.YYYY (German uses dots)
        def replace_date(match):
            day, month, year = match.group(1), match.group(2), match.group(3)
            try:
                month_name = _GERMAN_MONTHS.get(month.lstrip('0'), month)
                day_w = num2words(int(day), to='ordinal', lang='de') + 'r' if int(day) == 1 else num2words(int(day), to='ordinal', lang='de')
                if year:
                    year_w = num2words(int(year), lang='de')
                    return f"{day_w} {month_name} {year_w}"
                return f"{day_w} {month_name}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b', replace_date, text)

        # 8. Decimals: 3,14 → drei Komma vierzehn
        def replace_decimal(match):
            try:
                return f"{num2words(int(match.group(1)), lang='de')} Komma {num2words(int(match.group(2)), lang='de')}"
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+),(\d+)', replace_decimal, text)

        # 9. Cardinal numbers
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='de'), text)

        # 10. Prosody
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r'\s*[—–]\s*', ', ', text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r',(?=\S)', ', ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logger.warning(f"German normalization failed: {e}")
        return text


# ============================================================================
# SPANISH
# ============================================================================

_SPANISH_ABBREVIATIONS = {
    r'\bSr\.?\s': 'señor ',
    r'\bSra\.?\s': 'señora ',
    r'\bSrta\.?\s': 'señorita ',
    r'\bDr\.?\s': 'doctor ',
    r'\bDra\.?\s': 'doctora ',
    r'\bProf\.?\s': 'profesor ',
    r'\bLic\.?\s': 'licenciado ',
    r'\bIng\.?\s': 'ingeniero ',
    r'\bArq\.?\s': 'arquitecto ',
    r'\bAv\.?\s': 'avenida ',
    r'\bC/\s?': 'calle ',
    r'\bNº\.?\s?': 'número ',
    r'\bTel\.?\s': 'teléfono ',
    r'\betc\.': 'etcétera',
    r'\bp\.?\s?ej\.': 'por ejemplo',
    r'\bUd\.?\s': 'usted ',
    r'\bUds\.?\s': 'ustedes ',
}

_SPANISH_SYMBOLS = {
    '%': ' por ciento',
    '&': ' y ',
    '+': ' más ',
    '=': ' igual ',
    '@': ' arroba ',
    '«': '',
    '»': '',
    '"': '',
    '"': '',
    '¿': '',
    '¡': '',
}

_SPANISH_MONTHS = {
    '1': 'enero', '01': 'enero',
    '2': 'febrero', '02': 'febrero',
    '3': 'marzo', '03': 'marzo',
    '4': 'abril', '04': 'abril',
    '5': 'mayo', '05': 'mayo',
    '6': 'junio', '06': 'junio',
    '7': 'julio', '07': 'julio',
    '8': 'agosto', '08': 'agosto',
    '9': 'septiembre', '09': 'septiembre',
    '10': 'octubre',
    '11': 'noviembre',
    '12': 'diciembre',
}


def spanish_text_normalize(text: str) -> str:
    """Spanish text normalization."""
    if not _HAS_NUM2WORDS:
        return text
    try:
        # 1. Abbreviations
        for pattern, replacement in _SPANISH_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 1b. Thousands separator: 425.000 → 425000
        text = re.sub(r'\b(\d{1,3})(?:\.(\d{3}))+\b', lambda m: m.group(0).replace('.', ''), text)

        # 2. Phone numbers
        def replace_phone(match):
            full = match.group(0).replace('+', '')
            parts = re.split(r'[\s\-]+', full)
            try:
                return ', '.join(num2words(int(p), lang='es') for p in parts if p)
            except Exception:
                return match.group(0)
        text = re.sub(r'\+?\d{2,4}[\s\-]\d{2,4}(?:[\s\-]\d{2,7}){1,3}', replace_phone, text)

        # 3. Currency
        def _es_currency(amount_str, symbol):
            names = {'€': 'euro', '$': 'dólar', '£': 'libra'}
            plurals = {'€': 'euros', '$': 'dólares', '£': 'libras'}
            amount = int(amount_str)
            words = num2words(amount, lang='es')
            name = names.get(symbol, symbol) if amount == 1 else plurals.get(symbol, symbol)
            return f"{words} {name}"

        text = re.sub(r'(\d+)\s*([€$£])', lambda m: _es_currency(m.group(1), m.group(2)), text)
        text = re.sub(r'([€$£])\s*(\d+)', lambda m: _es_currency(m.group(2), m.group(1)), text)

        # 4. Symbols
        for sym, rep in _SPANISH_SYMBOLS.items():
            text = text.replace(sym, rep)

        # 5. Ordinals: 1°, 2ª
        def replace_ordinal(match):
            n = int(match.group(1))
            suffix = match.group(2)
            try:
                word = num2words(n, to='ordinal', lang='es')
                if suffix in ('ª', 'a'):
                    word = word.rstrip('o') + 'a' if word.endswith('o') else word
                return word
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+)([°ºªa])(?=\s|$|[,.])', replace_ordinal, text)

        # 6. Time: 14:30 (context-aware: skip article if preceded by "la/las/a las")
        def replace_time(match):
            h, m = int(match.group(2)), int(match.group(3))
            prefix = match.group(1) or ''
            has_article = bool(re.search(r'\b(?:las?)\s*$', prefix, re.IGNORECASE))
            try:
                h_w = num2words(h, lang='es')
                if has_article:
                    art = ''
                else:
                    art = ('la ' if h == 1 else 'las ')
                if m == 0:
                    return f"{prefix}{art}{h_w}"
                m_w = num2words(m, lang='es')
                return f"{prefix}{art}{h_w} y {m_w}"
            except Exception:
                return match.group(0)
        text = re.sub(r'((?:\b(?:las?|a\s+las?)\s+)?)(\d{1,2}):(\d{2})\b', replace_time, text)

        # 7. Dates: DD/MM/YYYY (context-aware: skip "el" if preceded by "del/el")
        def replace_date(match):
            prefix = match.group(1) or ''
            day, month, year = match.group(2), match.group(3), match.group(4)
            has_article = bool(re.search(r'\b(?:del|el)\s*$', prefix, re.IGNORECASE))
            try:
                month_name = _SPANISH_MONTHS.get(month.lstrip('0'), month)
                day_w = 'primero' if int(day) == 1 else num2words(int(day), lang='es')
                if has_article:
                    day_prefix = day_w + ' de '
                else:
                    day_prefix = 'el ' + day_w + ' de '
                if year:
                    year_w = num2words(int(year), lang='es')
                    return f"{prefix}{day_prefix}{month_name} de {year_w}"
                return f"{prefix}{day_prefix}{month_name}"
            except Exception:
                return match.group(0)
        text = re.sub(r'((?:\b(?:del|el)\s+)?)(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b', replace_date, text)

        # 8. Decimals: 3,14 → tres coma catorce
        def replace_decimal(match):
            try:
                return f"{num2words(int(match.group(1)), lang='es')} coma {num2words(int(match.group(2)), lang='es')}"
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+),(\d+)', replace_decimal, text)

        # 9. Cardinal numbers
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='es'), text)

        # 10. Prosody
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r'\s*[—–]\s*', ', ', text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r',(?=\S)', ', ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logger.warning(f"Spanish normalization failed: {e}")
        return text


# ============================================================================
# PORTUGUESE
# ============================================================================

_PORTUGUESE_ABBREVIATIONS = {
    r'\bSr\.?\s': 'senhor ',
    r'\bSra\.?\s': 'senhora ',
    r'\bDr\.?\s': 'doutor ',
    r'\bDra\.?\s': 'doutora ',
    r'\bProf\.?\s': 'professor ',
    r'\bEng\.?\s': 'engenheiro ',
    r'\bArq\.?\s': 'arquiteto ',
    r'\bAv\.?\s': 'avenida ',
    r'\bR\.?\s': 'rua ',
    r'\bNº\.?\s?': 'número ',
    r'\bTel\.?\s': 'telefone ',
    r'\betc\.': 'et cetera',
    r'\bp\.?\s?ex\.': 'por exemplo',
    r'\bObrig\.': 'obrigado',
}

_PORTUGUESE_SYMBOLS = {
    '%': ' por cento',
    '&': ' e ',
    '+': ' mais ',
    '=': ' igual ',
    '@': ' arroba ',
    '«': '',
    '»': '',
    '"': '',
    '"': '',
}

_PORTUGUESE_MONTHS = {
    '1': 'janeiro', '01': 'janeiro',
    '2': 'fevereiro', '02': 'fevereiro',
    '3': 'março', '03': 'março',
    '4': 'abril', '04': 'abril',
    '5': 'maio', '05': 'maio',
    '6': 'junho', '06': 'junho',
    '7': 'julho', '07': 'julho',
    '8': 'agosto', '08': 'agosto',
    '9': 'setembro', '09': 'setembro',
    '10': 'outubro',
    '11': 'novembro',
    '12': 'dezembro',
}


def portuguese_text_normalize(text: str) -> str:
    """Portuguese text normalization."""
    if not _HAS_NUM2WORDS:
        return text
    try:
        # 1. Abbreviations
        for pattern, replacement in _PORTUGUESE_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 1b. Thousands separator: 425.000 → 425000
        text = re.sub(r'\b(\d{1,3})(?:\.(\d{3}))+\b', lambda m: m.group(0).replace('.', ''), text)

        # 2. Phone numbers
        def replace_phone(match):
            full = match.group(0).replace('+', '')
            parts = re.split(r'[\s\-]+', full)
            try:
                return ', '.join(num2words(int(p), lang='pt') for p in parts if p)
            except Exception:
                return match.group(0)
        text = re.sub(r'\+?\d{2,4}[\s\-]\d{2,4}(?:[\s\-]\d{2,7}){1,3}', replace_phone, text)

        # 3. Currency
        def _pt_currency(amount_str, symbol):
            names = {'€': 'euro', '$': 'dólar', '£': 'libra'}
            plurals = {'€': 'euros', '$': 'dólares', '£': 'libras'}
            amount = int(amount_str)
            words = num2words(amount, lang='pt')
            name = names.get(symbol, symbol) if amount == 1 else plurals.get(symbol, symbol)
            return f"{words} {name}"

        text = re.sub(r'(\d+)\s*([€$£])', lambda m: _pt_currency(m.group(1), m.group(2)), text)
        text = re.sub(r'([€$£])\s*(\d+)', lambda m: _pt_currency(m.group(2), m.group(1)), text)

        # 4. Symbols
        for sym, rep in _PORTUGUESE_SYMBOLS.items():
            text = text.replace(sym, rep)

        # 5. Ordinals: 1º, 2ª
        def replace_ordinal(match):
            n = int(match.group(1))
            suffix = match.group(2)
            try:
                word = num2words(n, to='ordinal', lang='pt')
                if suffix in ('ª', 'a'):
                    word = word.rstrip('o') + 'a' if word.endswith('o') else word
                return word
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+)([°ºªa])(?=\s|$|[,.])', replace_ordinal, text)

        # 6. Time: 14:30, 14h30
        def replace_time(match):
            h, m = int(match.group(1)), int(match.group(2))
            try:
                h_w = num2words(h, lang='pt')
                if m == 0:
                    return f"{h_w} horas"
                m_w = num2words(m, lang='pt')
                return f"{h_w} horas e {m_w}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2})[h:](\d{2})\b', replace_time, text)

        # 7. Dates: DD/MM/YYYY
        def replace_date(match):
            day, month, year = match.group(1), match.group(2), match.group(3)
            try:
                month_name = _PORTUGUESE_MONTHS.get(month.lstrip('0'), month)
                day_w = 'primeiro' if int(day) == 1 else num2words(int(day), lang='pt')
                if year:
                    year_w = num2words(int(year), lang='pt')
                    return f"{day_w} de {month_name} de {year_w}"
                return f"{day_w} de {month_name}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b', replace_date, text)

        # 8. Decimals: 3,14 → três vírgula catorze
        def replace_decimal(match):
            try:
                return f"{num2words(int(match.group(1)), lang='pt')} vírgula {num2words(int(match.group(2)), lang='pt')}"
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+),(\d+)', replace_decimal, text)

        # 9. Cardinal numbers
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='pt'), text)

        # 10. Prosody
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r'\s*[—–]\s*', ', ', text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r',(?=\S)', ', ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logger.warning(f"Portuguese normalization failed: {e}")
        return text


# ============================================================================
# ENGLISH
# ============================================================================

_ENGLISH_ABBREVIATIONS = {
    r'\bMr\.?\s': 'mister ',
    r'\bMrs\.?\s': 'misses ',
    r'\bMs\.?\s': 'miss ',
    r'\bDr\.?\s': 'doctor ',
    r'\bProf\.?\s': 'professor ',
    r'\bSt\.?\s': 'saint ',
    r'\bAve\.?\s': 'avenue ',
    r'\bBlvd\.?\s': 'boulevard ',
    r'\bDept\.?\s': 'department ',
    r'\bTel\.?\s': 'telephone ',
    r'\bNo\.?\s': 'number ',
    r'\betc\.': 'et cetera',
    r'\be\.g\.': 'for example',
    r'\bi\.e\.': 'that is',
    r'\bvs\.': 'versus',
    r'\bapprox\.': 'approximately',
    r'\bgovt\.': 'government',
}

_ENGLISH_SYMBOLS = {
    '%': ' percent',
    '&': ' and ',
    '+': ' plus ',
    '=': ' equals ',
    '@': ' at ',
    '"': '',
    '"': '',
}

_ENGLISH_MONTHS = {
    '1': 'January', '01': 'January',
    '2': 'February', '02': 'February',
    '3': 'March', '03': 'March',
    '4': 'April', '04': 'April',
    '5': 'May', '05': 'May',
    '6': 'June', '06': 'June',
    '7': 'July', '07': 'July',
    '8': 'August', '08': 'August',
    '9': 'September', '09': 'September',
    '10': 'October',
    '11': 'November',
    '12': 'December',
}


def english_text_normalize(text: str) -> str:
    """English text normalization."""
    if not _HAS_NUM2WORDS:
        return text
    try:
        # 1. Abbreviations
        for pattern, replacement in _ENGLISH_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 2. Phone numbers
        def replace_phone(match):
            full = match.group(0).replace('+', '').replace('-', ' ').replace('.', ' ')
            parts = full.split()
            try:
                return ', '.join(num2words(int(p), lang='en') for p in parts if p)
            except Exception:
                return match.group(0)
        text = re.sub(r'\+?\d{1,4}[\s\-\.]\d{2,4}(?:[\s\-\.]\d{2,7}){1,3}', replace_phone, text)

        # 3. Currency (English puts symbol before amount, handle thousands with comma)
        def _en_currency(amount_str, symbol):
            names = {'€': 'euro', '$': 'dollar', '£': 'pound'}
            plurals = {'€': 'euros', '$': 'dollars', '£': 'pounds'}
            clean = amount_str.replace(',', '')
            amount = int(clean)
            words = num2words(amount, lang='en')
            name = names.get(symbol, symbol) if amount == 1 else plurals.get(symbol, symbol)
            return f"{words} {name}"

        text = re.sub(r'([€$£])\s*([\d,]+)', lambda m: _en_currency(m.group(2), m.group(1)), text)
        text = re.sub(r'([\d,]+)\s*([€$£])', lambda m: _en_currency(m.group(1), m.group(2)), text)

        # 4. Symbols
        for sym, rep in _ENGLISH_SYMBOLS.items():
            text = text.replace(sym, rep)

        # 5. Ordinals: 1st, 2nd, 3rd
        def replace_ordinal(match):
            n = int(match.group(1))
            try:
                return num2words(n, to='ordinal', lang='en')
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+)(?:st|nd|rd|th)(?=\s|$|[,.])', replace_ordinal, text)

        # 6. Time: 2:30 PM, 14:30
        def replace_time(match):
            h, m = int(match.group(1)), int(match.group(2))
            ampm = match.group(3) or ''
            try:
                h_w = num2words(h, lang='en')
                if m == 0:
                    return f"{h_w} o'clock {ampm}".strip()
                m_w = num2words(m, lang='en')
                return f"{h_w} {m_w} {ampm}".strip()
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', replace_time, text)

        # 7. Dates: MM/DD/YYYY (US format)
        def replace_date(match):
            month, day, year = match.group(1), match.group(2), match.group(3)
            try:
                month_name = _ENGLISH_MONTHS.get(month.lstrip('0'), month)
                day_w = num2words(int(day), to='ordinal', lang='en')
                if year:
                    year_w = num2words(int(year), lang='en')
                    return f"{month_name} {day_w}, {year_w}"
                return f"{month_name} {day_w}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', replace_date, text)

        # 8a. Thousands separator: 1,000,000 → 1000000 (EN uses comma)
        text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', lambda m: m.group(0).replace(',', ''), text)

        # 8. Decimals: 3.14 → three point one four
        def replace_decimal(match):
            try:
                int_w = num2words(int(match.group(1)), lang='en')
                dec_digits = ' '.join(num2words(int(d), lang='en') for d in match.group(2))
                return f"{int_w} point {dec_digits}"
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+)\.(\d+)', replace_decimal, text)

        # 9. Cardinal numbers
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='en'), text)

        # 10. Prosody
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r'\s*[—–]\s*', ', ', text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r',(?=\S)', ', ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logger.warning(f"English normalization failed: {e}")
        return text


# ============================================================================
# ITALIAN
# ============================================================================

_ITALIAN_ABBREVIATIONS = {
    r'\bsig\.?\s': 'signore ',
    r'\bsig\.ra\.?\s': 'signora ',
    r'\bdott\.?\s': 'dottore ',
    r'\bdott\.ssa\.?\s': 'dottoressa ',
    r'\bprof\.?\s': 'professore ',
    r'\bprof\.ssa\.?\s': 'professoressa ',
    r'\bavv\.?\s': 'avvocato ',
    r'\bing\.?\s': 'ingegnere ',
    r'\barch\.?\s': 'architetto ',
    r'\bgeom\.?\s': 'geometra ',
    r'\brag\.?\s': 'ragioniere ',
    r'\bcomm\.?\s': 'commendatore ',
    r'\bon\.?\s': 'onorevole ',
    r'\bspett\.?\s': 'spettabile ',
    r'\bgent\.?\s': 'gentile ',
    r'\btel\.?\s': 'telefono ',
    r'\bvia\.?\s': 'via ',
    r'\bp\.zza\.?\s': 'piazza ',
    r'\bc\.a\.p\.?\s?': 'codice di avviamento postale ',
    r'\bn°?\s?': 'numero ',
    r'\becc\.': 'eccetera',
    r'\bad\s?es\.': 'ad esempio',
}

_ITALIAN_SYMBOLS = {
    '%': ' per cento',
    '&': ' e ',
    '+': ' più ',
    '=': ' uguale ',
    '@': ' chiocciola ',
    '«': '',
    '»': '',
    '\u201c': '',
    '\u201d': '',
}

_ITALIAN_MONTHS = {
    '1': 'gennaio', '01': 'gennaio',
    '2': 'febbraio', '02': 'febbraio',
    '3': 'marzo', '03': 'marzo',
    '4': 'aprile', '04': 'aprile',
    '5': 'maggio', '05': 'maggio',
    '6': 'giugno', '06': 'giugno',
    '7': 'luglio', '07': 'luglio',
    '8': 'agosto', '08': 'agosto',
    '9': 'settembre', '09': 'settembre',
    '10': 'ottobre',
    '11': 'novembre',
    '12': 'dicembre',
}


def italian_text_normalize(text: str) -> str:
    """Italian text normalization."""
    if not _HAS_NUM2WORDS:
        return text
    try:
        # 1. Abbreviations
        for pattern, replacement in _ITALIAN_ABBREVIATIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 1b. Thousands separator: 425.000 → 425000
        text = re.sub(r'\b(\d{1,3})(?:\.(\d{3}))+\b', lambda m: m.group(0).replace('.', ''), text)

        # 2. Phone numbers
        def replace_phone(match):
            full = match.group(0).replace('+', '')
            parts = re.split(r'[\s\-]+', full)
            try:
                return ', '.join(num2words(int(p), lang='it') for p in parts if p)
            except Exception:
                return match.group(0)
        text = re.sub(r'\+?\d{2,4}[\s\-]\d{2,4}(?:[\s\-]\d{2,7}){1,3}', replace_phone, text)

        # 3. Currency
        def _it_currency(amount_str, symbol):
            names = {'€': 'euro', '$': 'dollaro', '£': 'sterlina'}
            plurals = {'€': 'euro', '$': 'dollari', '£': 'sterline'}
            amount = int(amount_str)
            words = num2words(amount, lang='it')
            if amount == 1:
                words = 'un'
            name = names.get(symbol, symbol) if amount == 1 else plurals.get(symbol, symbol)
            return f"{words} {name}"

        text = re.sub(r'(\d+)\s*([€$£])', lambda m: _it_currency(m.group(1), m.group(2)), text)
        text = re.sub(r'([€$£])\s*(\d+)', lambda m: _it_currency(m.group(2), m.group(1)), text)

        # 4. Symbols
        for sym, rep in _ITALIAN_SYMBOLS.items():
            text = text.replace(sym, rep)

        # 5. Ordinals: 1°, 2ª
        def replace_ordinal(match):
            n = int(match.group(1))
            suffix = match.group(2)
            try:
                word = num2words(n, to='ordinal', lang='it')
                if suffix in ('\u00aa', 'a'):
                    word = word.rstrip('o') + 'a' if word.endswith('o') else word
                return word
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+)([°ºªa])(?=\s|$|[,.])', replace_ordinal, text)

        # 6. Time: 14:30 (don't add "le" — user text may already have it)
        def replace_time(match):
            h, m = int(match.group(1)), int(match.group(2))
            try:
                h_w = num2words(h, lang='it')
                if m == 0:
                    return h_w
                m_w = num2words(m, lang='it')
                return f"{h_w} e {m_w}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2}):(\d{2})\b', replace_time, text)

        # 7. Dates: DD/MM/YYYY
        def replace_date(match):
            day, month, year = match.group(1), match.group(2), match.group(3)
            try:
                month_name = _ITALIAN_MONTHS.get(month.lstrip('0'), month)
                day_w = 'primo' if int(day) == 1 else num2words(int(day), lang='it')
                if year:
                    year_w = num2words(int(year), lang='it')
                    return f"{day_w} {month_name} {year_w}"
                return f"{day_w} {month_name}"
            except Exception:
                return match.group(0)
        text = re.sub(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b', replace_date, text)

        # 9. Decimals: 3,14 → tre virgola quattordici
        def replace_decimal(match):
            try:
                return f"{num2words(int(match.group(1)), lang='it')} virgola {num2words(int(match.group(2)), lang='it')}"
            except Exception:
                return match.group(0)
        text = re.sub(r'(\d+),(\d+)', replace_decimal, text)

        # 10. Cardinal numbers
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0)), lang='it'), text)

        # 10. Prosody
        text = re.sub(r',\s*(vero|no|eh|già|giusto)\s*([.!]|$)', r', \1?', text)
        text = re.sub(r'\.{3,}', '…', text)
        text = re.sub(r'\s*[—–]\s*', ', ', text)
        text = re.sub(r'([!?])\1+', r'\1', text)
        text = re.sub(r',(?=\S)', ', ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logger.warning(f"Italian normalization failed: {e}")
        return text


# ============================================================================
# DISPATCHER
# ============================================================================

NORMALIZERS = {
    'it': italian_text_normalize,
    'fr': french_text_normalize,
    'de': german_text_normalize,
    'es': spanish_text_normalize,
    'pt': portuguese_text_normalize,
    'en': english_text_normalize,
}


def normalize_text_for_language(text: str, language_id: str) -> str:
    """Apply language-specific text normalization if available."""
    normalizer = NORMALIZERS.get(language_id)
    if normalizer:
        return normalizer(text)
    return text
