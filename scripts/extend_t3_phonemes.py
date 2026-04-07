#!/usr/bin/env python3
"""
Extend T3 checkpoint with phoneme token embeddings.

Takes the original t3_mtl23ls_v2.safetensors (2454 BPE tokens) and creates
t3_phoneme.safetensors with extended text_emb and text_head weights
(2454 + N_phoneme_tokens = 2528 tokens).

New embedding rows are initialized from related grapheme embeddings
(e.g., phoneme /k/ gets mean of 'k','c' grapheme embeddings).

Usage:
    python scripts/extend_t3_phonemes.py \
        --ckpt-dir /path/to/chatterbox/weights \
        --output t3_phoneme.safetensors

The original checkpoint is NOT modified. A new file is created.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.phoneme_tokens import (
    get_all_new_tokens,
    initialize_phoneme_embeddings,
    N_PHONEMES,
    PHONEME_LIST,
    PHONEME_MODE_TOKEN,
    phoneme_token_name,
)
from chatterbox.models.t3.modules.t3_config import BPE_VOCAB_SIZE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_bpe_vocab(vocab_path: str) -> dict:
    """Load the BPE vocab from the HuggingFace tokenizer JSON."""
    with open(vocab_path) as f:
        data = json.load(f)

    if "model" in data and "vocab" in data["model"]:
        vocab_data = data["model"]["vocab"]
        if isinstance(vocab_data, dict):
            return vocab_data
        elif isinstance(vocab_data, list):
            return {tok: i for i, (tok, _score) in enumerate(vocab_data)}

    raise ValueError(f"Cannot parse vocab from {vocab_path}")


def main():
    parser = argparse.ArgumentParser(description="Extend T3 with phoneme embeddings")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Directory containing t3_mtl23ls_v2.safetensors and vocab JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: t3_phoneme.safetensors in ckpt-dir)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the extended checkpoint after creation")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    t3_path = ckpt_dir / "t3_mtl23ls_v2.safetensors"
    vocab_path = ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"

    if not t3_path.exists():
        logger.error(f"T3 checkpoint not found: {t3_path}")
        sys.exit(1)
    if not vocab_path.exists():
        logger.error(f"Vocab file not found: {vocab_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else ckpt_dir / "t3_phoneme.safetensors"

    # --- Load ---
    logger.info(f"Loading T3 checkpoint: {t3_path}")
    state = load_safetensors(str(t3_path))
    if "model" in state:
        state = state["model"][0]

    old_emb = state["text_emb.weight"]
    old_head = state["text_head.weight"]
    old_size, dim = old_emb.shape
    logger.info(f"Original text_emb: [{old_size}, {dim}]")
    logger.info(f"Original text_head: [{old_head.shape[0]}, {old_head.shape[1]}]")

    assert old_size == BPE_VOCAB_SIZE, f"Expected {BPE_VOCAB_SIZE} tokens, got {old_size}"

    # --- Load BPE vocab ---
    logger.info(f"Loading BPE vocab: {vocab_path}")
    bpe_vocab = load_bpe_vocab(str(vocab_path))
    logger.info(f"BPE vocab size: {len(bpe_vocab)}")

    # --- Phoneme tokens ---
    new_tokens = get_all_new_tokens()
    n_new = len(new_tokens)
    new_vocab_size = old_size + n_new
    logger.info(f"Adding {n_new} phoneme tokens ({PHONEME_MODE_TOKEN} + {N_PHONEMES} IPA symbols)")
    logger.info(f"New vocab size: {new_vocab_size}")

    # --- Initialize phoneme embeddings ---
    logger.info("Initializing phoneme embeddings from grapheme mappings...")
    new_emb_rows = initialize_phoneme_embeddings(old_emb, bpe_vocab)

    # Stats
    emb_std = old_emb.std().item()
    new_std = new_emb_rows.std().item()
    logger.info(f"Original emb std: {emb_std:.4f}, New emb std: {new_std:.4f}")

    # Count how many phonemes got grapheme-based init vs random
    n_grapheme_init = 0
    n_random_init = 0
    for i, ph in enumerate(PHONEME_LIST):
        row = new_emb_rows[i + 1]  # +1 for PHON marker
        if row.abs().max() > 0.001:  # non-trivial init
            n_grapheme_init += 1
        else:
            n_random_init += 1
    logger.info(f"Grapheme-initialized: {n_grapheme_init}, Random-initialized: {n_random_init}")

    # --- Extend state dict ---
    state["text_emb.weight"] = torch.cat([old_emb, new_emb_rows], dim=0)

    # text_head: random init with matching std
    head_std = old_head.std().item()
    new_head_rows = torch.zeros(n_new, dim)
    new_head_rows.normal_(0, head_std)
    state["text_head.weight"] = torch.cat([old_head, new_head_rows], dim=0)

    logger.info(f"Extended text_emb: [{state['text_emb.weight'].shape[0]}, {dim}]")
    logger.info(f"Extended text_head: [{state['text_head.weight'].shape[0]}, {dim}]")

    # --- Save ---
    logger.info(f"Saving extended checkpoint: {output_path}")
    save_safetensors(state, str(output_path))

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved: {output_path} ({file_size:.1f} MB)")

    # --- Verify ---
    if args.verify:
        logger.info("Verifying extended checkpoint...")
        loaded = load_safetensors(str(output_path))

        # Check shapes
        assert loaded["text_emb.weight"].shape == (new_vocab_size, dim), \
            f"text_emb shape mismatch: {loaded['text_emb.weight'].shape}"
        assert loaded["text_head.weight"].shape == (new_vocab_size, dim), \
            f"text_head shape mismatch: {loaded['text_head.weight'].shape}"

        # Check original weights are preserved exactly
        orig_emb = load_safetensors(str(t3_path))["text_emb.weight"]
        assert torch.equal(loaded["text_emb.weight"][:old_size], orig_emb), \
            "Original text_emb weights were modified!"

        orig_head = load_safetensors(str(t3_path))["text_head.weight"]
        assert torch.equal(loaded["text_head.weight"][:old_size], orig_head), \
            "Original text_head weights were modified!"

        # Check new rows are not all zeros
        new_emb_check = loaded["text_emb.weight"][old_size:]
        assert new_emb_check.abs().sum() > 0, "New embeddings are all zeros!"

        logger.info("Verification PASSED — original weights preserved, new tokens initialized")

    # --- Print token mapping ---
    print(f"\n{'='*60}")
    print(f"Phoneme Token Mapping")
    print(f"{'='*60}")
    print(f"  [PHON] mode marker → ID {old_size}")
    for i, ph in enumerate(PHONEME_LIST[:10]):
        token_name = phoneme_token_name(ph)
        token_id = old_size + 1 + i
        print(f"  {token_name:12s} → ID {token_id}")
    print(f"  ... ({N_PHONEMES - 10} more)")
    print(f"  Total: {n_new} new tokens (IDs {old_size}–{new_vocab_size - 1})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
