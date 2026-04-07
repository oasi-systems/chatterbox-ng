#!/usr/bin/env python3
"""
Demo: G2P pipeline in action with realistic telephony sentences.

Shows the before/after for each language, with custom dictionaries loaded.
Run: python3 tests/test_g2p_demo.py
"""
import sys, os
import importlib.util

# Direct import to avoid heavy deps
_g2p_path = os.path.join(os.path.dirname(__file__), "..", "src", "chatterbox", "g2p.py")
_spec = importlib.util.spec_from_file_location("chatterbox.g2p", _g2p_path)
_g2p_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g2p_mod)

G2PPipeline = _g2p_mod.G2PPipeline
CustomDictionary = _g2p_mod.CustomDictionary

# --- Load dictionaries ---
dict_dir = os.path.join(os.path.dirname(__file__), "..", "dictionaries")

dictionary = CustomDictionary()
common_path = os.path.join(dict_dir, "common_foreign_words.yaml")

if os.path.exists(common_path):
    dictionary.load_yaml(common_path)

# Load per-language dictionaries
for lang_code, filename in [
    ("it", "italian_telephony.yaml"),
    ("fr", "french_telephony.yaml"),
    ("de", "german_telephony.yaml"),
    ("es", "spanish_telephony.yaml"),
    ("pt", "portuguese_telephony.yaml"),
]:
    lang_path = os.path.join(dict_dir, filename)
    if os.path.exists(lang_path):
        dictionary.load_yaml(lang_path, language_id=lang_code)

g2p = G2PPipeline(custom_dict=dictionary)

# --- Test sentences ---
TESTS = {
    "it": [
        "Buongiorno, parlo con il signor Schmidt?",
        "Il suo codice IBAN è stato verificato per il bonifico SEPA.",
        "Il signor McDonald ha richiesto un callback dal nostro helpdesk.",
        "La password del suo account WiFi deve essere aggiornata.",
        "Ho verificato la sua policy di compliance con il nostro CEO.",
        "Il tecnico Schwartz ha configurato il bluetooth dello smartphone.",
        "Prego, si colleghi alla nostra hotline per il roaming.",
        "La pratica del signor Müller è stata approvata.",
    ],
    "fr": [
        "Bonjour, je parle avec monsieur Schmidt?",
        "Votre code IBAN a été vérifié pour le virement SEPA.",
        "Le WiFi de votre smartphone a été configuré.",
    ],
    "de": [
        "Guten Tag, Ihr IBAN wurde für die SEPA-Überweisung verifiziert.",
        "Der CEO hat das Meeting per WiFi durchgeführt.",
    ],
    "es": [
        "Buenos días, el código IBAN ha sido verificado para la transferencia SEPA.",
        "El señor Schmidt ha solicitado un callback.",
    ],
    "pt": [
        "Bom dia, o código IBAN foi verificado para a transferência SEPA.",
        "O senhor Schmidt solicitou um callback.",
    ],
    "en": [
        "Good morning, your IBAN code has been verified for the SEPA transfer.",
        "Mr. Schäfer has requested a callback from our helpdesk.",
    ],
}


def main():
    print("=" * 80)
    print("G2P Pipeline Demo — Respelling for BPE-based TTS")
    print("=" * 80)

    has_espeak = _g2p_mod._HAS_PHONEMIZER
    print(f"\nespeak-ng available: {'YES' if has_espeak else 'NO (dictionary-only mode)'}")
    print(f"Custom dictionary entries: global={len(dictionary._global)}, "
          f"it={len(dictionary._entries.get('it', {}))}")
    print()

    total_changes = 0
    total_sentences = 0

    for lang, sentences in TESTS.items():
        print(f"\n{'─' * 80}")
        print(f"  [{lang.upper()}]")
        print(f"{'─' * 80}")

        for sent in sentences:
            result = g2p.process(sent, lang=lang)
            total_sentences += 1

            if result != sent:
                total_changes += 1
                # Highlight changes
                print(f"\n  IN:  {sent}")
                print(f"  OUT: {result}")
            else:
                print(f"\n  [=]  {sent}")

    print(f"\n{'=' * 80}")
    print(f"Summary: {total_changes}/{total_sentences} sentences modified by G2P")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
