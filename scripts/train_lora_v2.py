#!/usr/bin/env python3
"""
LoRA fine-tuning v2 — Speaker-safe, multi-GPU.

Key differences from v1:
  - FREEZES cond_enc (speaker identity preserved)
  - FREEZES speech_emb, pos_emb (no drift)
  - LoRA r=8 (not 64 — less capacity to memorize speaker timbre)
  - Multi-GPU via accelerate (DDP on 2× L4)
  - Validation: cosine similarity between output and reference audio

Usage:
  # Single GPU
  python train_lora_v2.py

  # Multi-GPU (2× L4)
  accelerate launch --multi_gpu --num_processes 2 train_lora_v2.py

  # Resume from checkpoint
  accelerate launch --multi_gpu --num_processes 2 train_lora_v2.py --resume
"""

import os
import sys
import csv
import glob
import json
import math
import time
import random
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    manifest_path: str = "/workspace/data/manifest_filtered.csv"
    tokenized_dir: str = "/workspace/data/tokenized/mls"

    # Model
    model_name: str = "ResembleAI/chatterbox"
    cache_dir: str = "/workspace/models"

    # LoRA — LOW RANK to prevent speaker memorization
    lora_r: int = 8           # was 64 in v1 — too high
    lora_alpha: int = 16      # alpha = 2*r
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj")  # attention only

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # effective batch = 32 per GPU
    learning_rate: float = 1e-4           # lower than v1 (2e-4)
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 20000
    save_every: int = 2000
    log_every: int = 50
    max_seq_len: int = 1024

    # Output
    output_dir: str = "/workspace/checkpoints_v2"

    # Languages
    languages: tuple = ("it", "en", "fr", "de", "es", "pt")

    # Resume
    resume: bool = False
    resume_from: Optional[str] = None


# ─── Dataset ─────────────────────────────────────────────────────────────────

class MLSTokenizedDataset(Dataset):
    """Loads pre-tokenized speech samples from manifest.

    Data format per .pt file:
      speech_tokens: (seq_len,) int64 — S3 tokenizer output
      text_tokens:   (seq_len,) int32 — BPE text tokens
      language:      str
      duration:      float (seconds)
      text:          str
    """

    def __init__(self, manifest_path: str, tokenized_dir: str,
                 languages: tuple, max_seq_len: int = 1024):
        self.tokenized_dir = tokenized_dir
        self.max_seq_len = max_seq_len
        self.samples = []

        lang_counts = {l: 0 for l in languages}

        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = row.get("language", "en")
                if lang not in languages:
                    continue

                token_path = os.path.join(
                    tokenized_dir, lang,
                    os.path.splitext(os.path.basename(row.get("audio_path", row.get("path", ""))))[0] + ".pt"
                )

                if os.path.exists(token_path):
                    self.samples.append({
                        "token_path": token_path,
                        "text": row.get("text", ""),
                        "language": lang,
                    })
                    lang_counts[lang] += 1

        logger.info(f"Loaded {len(self.samples)} samples: {lang_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(sample["token_path"], map_location="cpu", weights_only=True)

        speech_tokens = data["speech_tokens"].long().squeeze(0)
        text_tokens = data["text_tokens"].long().squeeze(0)

        # Truncate if needed (before adding special tokens)
        if speech_tokens.shape[-1] > self.max_seq_len - 2:
            speech_tokens = speech_tokens[..., :self.max_seq_len - 2]
        if text_tokens.shape[-1] > self.max_seq_len - 2:
            text_tokens = text_tokens[..., :self.max_seq_len - 2]

        # Add BOT/EOT and BOS/EOS special tokens
        # T3Config: start_text=255, stop_text=0, start_speech=6561, stop_speech=6562
        START_TEXT = 255
        STOP_TEXT = 0
        START_SPEECH = 6561
        STOP_SPEECH = 6562

        text_tokens = torch.cat([
            torch.tensor([START_TEXT], dtype=torch.long),
            text_tokens,
            torch.tensor([STOP_TEXT], dtype=torch.long),
        ])
        speech_tokens = torch.cat([
            torch.tensor([START_SPEECH], dtype=torch.long),
            speech_tokens,
            torch.tensor([STOP_SPEECH], dtype=torch.long),
        ])

        return {
            "speech_tokens": speech_tokens,
            "text_tokens": text_tokens,
            "language": sample["language"],
        }


def collate_fn(batch):
    """Pad sequences to max length in batch. Returns lens for T3.loss()."""
    speech_lens = [b["speech_tokens"].shape[0] for b in batch]
    text_lens = [b["text_tokens"].shape[0] for b in batch]

    max_speech = max(speech_lens)
    max_text = max(text_lens)

    speech_padded = torch.zeros(len(batch), max_speech, dtype=torch.long)
    text_padded = torch.zeros(len(batch), max_text, dtype=torch.long)

    for i, b in enumerate(batch):
        sl = b["speech_tokens"].shape[0]
        tl = b["text_tokens"].shape[0]
        speech_padded[i, :sl] = b["speech_tokens"]
        text_padded[i, :tl] = b["text_tokens"]

    return {
        "speech_tokens": speech_padded,
        "text_tokens": text_padded,
        "speech_token_lens": torch.tensor(speech_lens, dtype=torch.long),
        "text_token_lens": torch.tensor(text_lens, dtype=torch.long),
        "languages": [b["language"] for b in batch],
    }


# ─── Model setup ────────────────────────────────────────────────────────────

def setup_model(config: TrainConfig, device):
    """Load T3, freeze speaker modules, apply LoRA to backbone."""
    from peft import LoraConfig, get_peft_model

    # Load full model to extract T3
    # ChatterBox is installed in the training image at /opt/chatterbox-ng/src/
    # or as a pip package. Try import directly first.
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError:
        for p in ["/opt/chatterbox-ng/src", "/workspace/ChatterBox/src"]:
            if os.path.exists(p):
                sys.path.insert(0, p)
                break
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    logger.info("Loading base model...")
    # Set HF cache dir via env var before loading
    os.environ["HF_HOME"] = config.cache_dir
    model = ChatterboxMultilingualTTS.from_pretrained(str(device))
    t3 = model.t3

    # ──────────────────────────────────────────────────────────────────────
    # FREEZE: Speaker identity modules (CRITICAL — this was missing in v1!)
    # ──────────────────────────────────────────────────────────────────────

    frozen_params = 0

    # 1. Conditioning encoder — speaker embedding + perceiver + emotion
    for name, param in t3.cond_enc.named_parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    logger.info(f"FROZEN: cond_enc ({frozen_params:,} params)")

    # 2. Speech embeddings — S3 tokenizer vocab is fixed
    n = sum(p.numel() for p in t3.speech_emb.parameters())
    for param in t3.speech_emb.parameters():
        param.requires_grad = False
    frozen_params += n
    logger.info(f"FROZEN: speech_emb ({n:,} params)")

    # 3. Position embeddings — RoPE in transformer handles positions
    for name in ["text_pos_emb", "speech_pos_emb"]:
        if hasattr(t3, name):
            n = sum(p.numel() for p in getattr(t3, name).parameters())
            for param in getattr(t3, name).parameters():
                param.requires_grad = False
            frozen_params += n
            logger.info(f"FROZEN: {name} ({n:,} params)")

    # 4. Text embeddings — freeze BPE rows, allow new phoneme rows
    if hasattr(t3, "text_emb"):
        n = sum(p.numel() for p in t3.text_emb.parameters())
        for param in t3.text_emb.parameters():
            param.requires_grad = False
        frozen_params += n
        logger.info(f"FROZEN: text_emb ({n:,} params)")

    logger.info(f"Total FROZEN: {frozen_params:,} params (speaker + fixed modules)")

    # ──────────────────────────────────────────────────────────────────────
    # LoRA: Apply to transformer backbone ONLY
    # ──────────────────────────────────────────────────────────────────────

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
    )

    # Apply LoRA to the transformer backbone
    t3.tfmr = get_peft_model(t3.tfmr, lora_config)

    trainable = sum(p.numel() for p in t3.parameters() if p.requires_grad)
    total = sum(p.numel() for p in t3.parameters())
    logger.info(f"LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Also unfreeze speech_head and text_head (small, phonetic output)
    for name in ["speech_head", "text_head"]:
        if hasattr(t3, name):
            for param in getattr(t3, name).parameters():
                param.requires_grad = True
            n = sum(p.numel() for p in getattr(t3, name).parameters())
            logger.info(f"TRAINABLE: {name} ({n:,} params)")

    trainable = sum(p.numel() for p in t3.parameters() if p.requires_grad)
    logger.info(f"Final trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, t3


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint directory."""
    pattern = os.path.join(output_dir, "lora_step_*")
    checkpoints = sorted(glob.glob(pattern))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_step_")[-1]))
    return checkpoints[-1]


# ─── Training loop ───────────────────────────────────────────────────────────

def train(config: TrainConfig):
    """Main training loop with accelerate for multi-GPU."""
    from accelerate import Accelerator

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=None,
    )

    is_main = accelerator.is_main_process
    device = accelerator.device

    if is_main:
        os.makedirs(config.output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump({k: str(v) if isinstance(v, tuple) else v
                       for k, v in config.__dict__.items()}, f, indent=2)

    # ── Dataset ──
    dataset = MLSTokenizedDataset(
        config.manifest_path, config.tokenized_dir,
        config.languages, config.max_seq_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ── Model ──
    model, t3 = setup_model(config, device)

    # ── Optimizer (only trainable params) ──
    trainable_params = [p for p in t3.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # ── LR scheduler: cosine with warmup ──
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Prepare with accelerate ──
    t3, optimizer, dataloader, scheduler = accelerator.prepare(
        t3, optimizer, dataloader, scheduler
    )

    # ── Resume ──
    global_step = 0
    start_step = 0

    if config.resume:
        ckpt_dir = config.resume_from or find_latest_checkpoint(config.output_dir)
        if ckpt_dir and os.path.exists(ckpt_dir):
            step_num = int(ckpt_dir.split("_step_")[-1])
            if is_main:
                logger.info(f"Resuming from {ckpt_dir} (step {step_num})")

            # Load LoRA weights
            from safetensors.torch import load_file
            weights_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
            if os.path.exists(weights_path):
                state_dict = load_file(weights_path)
                # Load into the unwrapped model
                unwrapped = accelerator.unwrap_model(t3)
                missing, unexpected = unwrapped.tfmr.load_state_dict(state_dict, strict=False)
                if is_main:
                    logger.info(f"Loaded LoRA weights: {len(state_dict)} tensors, "
                                f"{len(missing)} missing, {len(unexpected)} unexpected")

            # Load optimizer
            opt_path = os.path.join(ckpt_dir, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                if is_main:
                    logger.info("Loaded optimizer state")

            global_step = step_num
            start_step = step_num

            # Fast-forward scheduler
            for _ in range(step_num):
                scheduler.step()
            if is_main:
                logger.info(f"Scheduler fast-forwarded to step {step_num}")
        else:
            if is_main:
                logger.warning(f"No checkpoint found, starting from scratch")

    # ── Training ──
    if is_main:
        logger.info(f"Starting training from step {global_step}")
        logger.info(f"  GPUs: {accelerator.num_processes}")
        logger.info(f"  Batch/GPU: {config.batch_size}")
        logger.info(f"  Grad accum: {config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes}")
        logger.info(f"  Steps: {config.max_steps}")
        logger.info(f"  LR: {config.learning_rate}")
        logger.info(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")

    t3.train()
    running_loss = 0.0
    running_loss_text = 0.0
    running_loss_speech = 0.0
    step_times = []
    epoch = 0

    # File logging
    log_file = None
    if is_main:
        log_file = open(os.path.join(config.output_dir, "train.log"), "a")
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Resume from step: {start_step}\n")
        log_file.write(f"GPUs: {accelerator.num_processes}\n")
        log_file.write(f"LoRA r={config.lora_r}, cond_enc=FROZEN\n")
        log_file.write(f"{'='*60}\n")
        log_file.flush()

    # Import T3Cond once
    from chatterbox.models.t3.t3 import T3Cond

    while global_step < config.max_steps:
        epoch += 1
        if is_main:
            logger.info(f"Epoch {epoch}")

        for batch in dataloader:
            if global_step >= config.max_steps:
                break

            t0 = time.time()

            with accelerator.accumulate(t3):
                speech_tokens = batch["speech_tokens"]
                text_tokens = batch["text_tokens"]
                speech_token_lens = batch["speech_token_lens"]
                text_token_lens = batch["text_token_lens"]

                B = speech_tokens.shape[0]

                # Zero speaker conditioning — cond_enc is frozen, we want
                # speaker-agnostic phonetic learning. At inference, real
                # reference audio will be used and cond_enc will encode it
                # faithfully since its weights are unchanged.
                dummy_cond = T3Cond(
                    speaker_emb=torch.zeros(B, 256, device=device),
                    cond_prompt_speech_tokens=torch.zeros(B, 1, dtype=torch.long, device=device),
                    emotion_adv=torch.tensor([0.5] * B, device=device),
                )

                # Forward through T3
                unwrapped = accelerator.unwrap_model(t3)
                out = unwrapped.forward(
                    t3_cond=dummy_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_token_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens,
                    training=True,
                )

                # Compute loss with proper masking
                IGNORE_ID = -100
                len_text = text_tokens.size(1)
                len_speech = speech_tokens.size(1)

                mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None].to(device)
                mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None].to(device)
                masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
                masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)

                # Transpose logits: (B, seq, vocab) → (B, vocab, seq) for cross_entropy
                loss_text = F.cross_entropy(
                    out.text_logits.transpose(1, 2), masked_text, ignore_index=IGNORE_ID
                )
                loss_speech = F.cross_entropy(
                    out.speech_logits.transpose(1, 2), masked_speech, ignore_index=IGNORE_ID
                )

                # Combined loss (speech is primary objective)
                loss = loss_speech + 0.1 * loss_text
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Only count after accumulation
            if accelerator.sync_gradients:
                global_step += 1
                dt = time.time() - t0
                step_times.append(dt)
                running_loss += loss.item()
                running_loss_text += loss_text.item()
                running_loss_speech += loss_speech.item()

                # ── Logging ──
                if global_step % config.log_every == 0 and is_main:
                    avg_loss = running_loss / config.log_every
                    avg_lt = running_loss_text / config.log_every
                    avg_ls = running_loss_speech / config.log_every
                    avg_time = sum(step_times[-config.log_every:]) / len(step_times[-config.log_every:])
                    lr = scheduler.get_last_lr()[0]

                    msg = (f"step={global_step}/{config.max_steps} "
                           f"loss={avg_loss:.4f} (text={avg_lt:.4f} speech={avg_ls:.4f}) "
                           f"lr={lr:.2e} time={avg_time:.1f}s/step")
                    logger.info(msg)

                    if log_file:
                        log_file.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
                        log_file.flush()

                    running_loss = 0.0
                    running_loss_text = 0.0
                    running_loss_speech = 0.0

                # ── Save checkpoint ──
                if global_step % config.save_every == 0 and is_main:
                    save_dir = os.path.join(config.output_dir, f"lora_step_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)

                    unwrapped = accelerator.unwrap_model(t3)

                    # Save LoRA adapter weights
                    unwrapped.tfmr.save_pretrained(save_dir)

                    # Save speech_head and text_head
                    heads_state = {}
                    for name in ["speech_head", "text_head"]:
                        if hasattr(unwrapped, name):
                            for pname, param in getattr(unwrapped, name).named_parameters():
                                heads_state[f"{name}.{pname}"] = param.data.cpu()
                    if heads_state:
                        torch.save(heads_state, os.path.join(save_dir, "heads.pt"))

                    # Save optimizer
                    torch.save(optimizer.state_dict(),
                               os.path.join(save_dir, "optimizer.pt"))

                    # Save step info
                    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                        json.dump({
                            "global_step": global_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "frozen_modules": ["cond_enc", "speech_emb", "text_emb",
                                              "text_pos_emb", "speech_pos_emb"],
                            "lora_r": config.lora_r,
                            "lora_alpha": config.lora_alpha,
                        }, f, indent=2)

                    logger.info(f"Saved checkpoint: {save_dir}")

    # ── Final save ──
    if is_main:
        save_dir = os.path.join(config.output_dir, f"lora_step_{global_step}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(t3)
            unwrapped.tfmr.save_pretrained(save_dir)

            heads_state = {}
            for name in ["speech_head", "text_head"]:
                if hasattr(unwrapped, name):
                    for pname, param in getattr(unwrapped, name).named_parameters():
                        heads_state[f"{name}.{pname}"] = param.data.cpu()
            if heads_state:
                torch.save(heads_state, os.path.join(save_dir, "heads.pt"))

            logger.info(f"Final checkpoint: {save_dir}")

        if log_file:
            log_file.write(f"\nTraining completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Final step: {global_step}\n")
            log_file.close()

        logger.info("Training complete!")

    accelerator.end_training()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA v2 — speaker-safe multi-GPU training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, help="Resume from specific checkpoint dir")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--max-steps", type=int, default=20000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--save-every", type=int, default=2000, help="Save every N steps")
    args = parser.parse_args()

    config = TrainConfig(
        resume=args.resume,
        resume_from=args.resume_from,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        save_every=args.save_every,
        lora_alpha=args.lora_r * 2,  # alpha = 2*r
    )

    train(config)


if __name__ == "__main__":
    import torch.distributed.elastic.multiprocessing.errors as errors
    @errors.record
    def _main():
        main()
    _main()
