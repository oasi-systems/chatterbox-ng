# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils.mask import make_pad_mask
from .configs import CFM_PARAMS
from .transformer.upsample_encoder import EncoderCaches
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def _repeat_batch_dim(tnsr, B, ndim):
    "repeat batch dimension if it's equal to 1"
    if tnsr is not None:
        # add missing batch dim if needed
        while tnsr.ndim < ndim:
            tnsr = tnsr[None]
        # repeat batch dim as needed
        if B > 1 and tnsr.size(0) == 1:
            tnsr = tnsr.repeat(B, *([1] * (ndim - 1)))
        assert tnsr.ndim == ndim, f"Expected {ndim=}, got {tnsr.ndim=}"
    return tnsr


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 6561,
                 input_frame_rate: int = 25,
                 only_mask_loss: bool = True,
                 token_mel_ratio: int = 2,
                 pre_lookahead_len: int = 3,
                 encoder: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig(
                                           {'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                            'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7,
                                            'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0,
                                                          'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8,
                                                          'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    # NOTE: copied in from cosyvoice repo
    def compute_loss(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)  # (B, 80, T)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        # streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)  # (B, T, 1)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask  # (B, T, emb)

        # text encode
        h, h_lengths = self.encoder(token, token_len)  # (B, T, C) -> (B, 2T, C)
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :, :index] = feat[i, :, :index]

        mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
        loss, _ = self.decoder.compute_loss(
            feat.contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            # streaming=streaming,
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize,
                  n_timesteps=10,
                  noised_mels=None,
                  meanflow=False):
        # token: (B, n_toks)
        # token_len: (B,)
        B = token.size(0)

        # xvec projection
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)  # (1 or B, emb_dim)

        # adjust shapes (batching logic)
        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)  # (B, n_prompt)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)  # (B,)
        prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)  # (B, n_feat, feat_dim=80)
        prompt_feat_len = _repeat_batch_dim(prompt_feat_len, B, ndim=1)  # (B,) or None
        embedding = _repeat_batch_dim(embedding, B, ndim=2)  # (B, emb_dim)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        # make_pad_mask must match token tensor shape — extra positions (from
        # prompt padding) are naturally masked to zero. No truncation needed.
        mask = (~make_pad_mask(token_len, max_len=token.shape[1])).unsqueeze(-1).to(embedding)

        if (token >= self.vocab_size).any():
            logger.error(f"{token.max()}>{self.vocab_size}\n out-of-range special tokens found in flow, fix inputs!")
        token = self.input_embedding(token.long()) * mask

        # text encode
        h, h_masks = self.encoder(token, token_len)
        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        if finalize is False:
            trim = self.pre_lookahead_len * self.token_mel_ratio
            h = h[:, :-trim]
            h_lengths = h_lengths - trim
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # # get conditions
        conds = torch.zeros([B, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths, max_len=h.shape[1])).unsqueeze(1).to(h)

        if mask.shape[0] != B:
            mask = mask.repeat(B, 1, 1)

        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat, None  # NOTE jrm: why are they returning None here?

    @torch.inference_mode()
    def encode_only(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        embedding,
        finalize,
    ):
        """Run encoder only (no CFM). Returns projected encoder output for ALL positions
        (prompt + speech) plus the prompt mel length so the caller can split them.

        This is critical because the CFM decoder needs the full [prompt | speech] sequence
        with prompt_feat as conditioning — without it, voice quality degrades severely.

        Args:
            token: (B, n_speech_toks) — speech tokens (without prompt)
            token_len: (B,)
            prompt_token, prompt_token_len: reference audio tokens
            prompt_feat: (B, mel_len1, 80) — reference mel features
            embedding: (1 or B, spk_dim) — speaker embedding (raw, will be projected)
            finalize: whether generation is complete

        Returns:
            full_mu: (B, 80, prompt_mel + speech_mel) — projected encoder output for FULL sequence
            spk_embedding: (B, 80) — projected speaker embedding
            prompt_mel_len: int — number of prompt mel frames (for splitting)
        """
        B = token.size(0)

        # xvec projection
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # adjust shapes
        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)
        embedding = _repeat_batch_dim(embedding, B, ndim=2)

        # concat [prompt | speech]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len, max_len=token.shape[1])).unsqueeze(-1).to(embedding)

        if (token >= self.vocab_size).any():
            logger.error(f"{token.max()}>{self.vocab_size}\n out-of-range special tokens found in flow, fix inputs!")
        token = self.input_embedding(token.long()) * mask

        # text encode
        h, h_masks = self.encoder(token, token_len)
        if finalize is False:
            trim = self.pre_lookahead_len * self.token_mel_ratio
            h = h[:, :-trim]
        mel_len1 = prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # Return FULL sequence (prompt + speech) — caller splits as needed
        full_mu = h.transpose(1, 2).contiguous()  # (B, 80, prompt_mel + speech_mel)
        return full_mu, embedding, mel_len1

    @torch.inference_mode()
    def inference_cached(
        self,
        token: torch.Tensor,
        token_len: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_token_len: torch.Tensor,
        embedding: torch.Tensor,
        finalize: bool,
        encoder_caches: EncoderCaches,
    ) -> Tuple[torch.Tensor, EncoderCaches]:
        """Encoder forward with KV-cache. Returns projected encoder output for new positions.

        On the first call (empty cache), the encoder processes [prompt | speech] and
        returns output for ALL positions. We strip the prompt portion so that only
        speech mel frames are returned — matching flow.inference() which does
        ``feat = feat[:, :, mel_len1:]``.

        Args:
            token: (B, n_speech_toks) — speech tokens (without prompt)
            token_len: (B,)
            prompt_token: (B, n_prompt)
            prompt_token_len: (B,)
            embedding: (1 or B, spk_dim) — speaker embedding (already projected)
            finalize: whether EOS reached
            encoder_caches: KV-cache state

        Returns:
            new_h_proj: (B, new_mel_frames, 80) — projected encoder output for NEW SPEECH positions only
            encoder_caches: updated caches
        """
        B = token.size(0)
        is_first_call = encoder_caches.enc_cached_len == 0

        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)

        # Full token sequence: [prompt | speech]
        all_token = torch.concat([prompt_token, token], dim=1)
        all_token_len = prompt_token_len + token_len
        mask = (~make_pad_mask(all_token_len, max_len=all_token.shape[1])).unsqueeze(-1).to(embedding)

        if (all_token >= self.vocab_size).any():
            logger.error(f"{all_token.max()}>{self.vocab_size}\n out-of-range special tokens found in flow, fix inputs!")
        all_token_emb = self.input_embedding(all_token.long()) * mask

        # Encoder with KV-cache: embed+PreLookaheadLayer on full, attention on new only
        new_h, h_masks, encoder_caches = self.encoder.forward_cached(
            all_token_emb, all_token_len, encoder_caches
        )

        # On first call, strip prompt mel frames from output.
        # The encoder upsamples by token_mel_ratio, so prompt produces
        # prompt_token_len * token_mel_ratio mel frames.
        if is_first_call and new_h.size(1) > 0:
            prompt_mel_len = int(prompt_token_len[0].item()) * self.token_mel_ratio
            new_h = new_h[:, prompt_mel_len:]

        # Trim lookahead if not finalizing
        if finalize is False and new_h.size(1) > 0:
            trim = self.pre_lookahead_len * self.token_mel_ratio
            if new_h.size(1) > trim:
                new_h = new_h[:, :-trim]

        new_h_proj = self.encoder_proj(new_h)  # (B, new_mel, 80)
        return new_h_proj, encoder_caches

    @torch.inference_mode()
    def inference_windowed(
        self,
        full_mu: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        prompt_mel_len: int,
        prev_speech_mel_len: int,
        n_timesteps: int = 10,
        meanflow: bool = False,
        cached_mel: torch.Tensor = None,
    ) -> torch.Tensor:
        """CFM inference that mirrors the standard inference() path exactly.

        Runs CFM on the FULL [prompt | speech] sequence with proper prompt_feat
        conditioning, then extracts only the NEW speech frames. Previous speech
        frames are frozen via freeze_len to avoid recomputing them.

        This produces identical quality to the monolithic path because the CFM
        decoder sees the same inputs: full mu, prompt_feat as cond, speaker embedding.

        Args:
            full_mu: (B, 80, prompt_mel + total_speech_mel) — full encoder output
            prompt_feat: (B, prompt_mel, 80) — reference mel features (row-major)
            embedding: (B, 80) — projected speaker embedding
            prompt_mel_len: number of prompt mel frames
            prev_speech_mel_len: speech mel frames already generated (to freeze)
            n_timesteps: ODE steps
            meanflow: whether to use meanflow mode
            cached_mel: (B, 80, prompt_mel + prev_speech_mel) — accumulated mel to freeze

        Returns:
            new_mel: (B, 80, new_speech_frames) — generated mel for new positions only
        """
        B = full_mu.size(0)
        total_frames = full_mu.size(2)  # prompt_mel + total_speech_mel

        # Frames to freeze: prompt + previously generated speech
        freeze_len = prompt_mel_len + prev_speech_mel_len

        # Build initial z: [cached_mel | random_noise_for_new]
        z_init = torch.randn(B, 80, total_frames, device=full_mu.device, dtype=full_mu.dtype)
        if cached_mel is not None and cached_mel.size(2) > 0:
            n_cached = min(cached_mel.size(2), freeze_len)
            z_init[:, :, :n_cached] = cached_mel[:, :, :n_cached]

        # Build conditioning: prompt_feat in prompt positions, zeros for speech
        # This matches exactly what inference() does:
        #   conds[:, :mel_len1] = prompt_feat
        conds = torch.zeros(B, 80, total_frames, device=full_mu.device, dtype=full_mu.dtype)
        conds[:, :, :prompt_mel_len] = prompt_feat.transpose(1, 2).contiguous()

        # Mask: all valid
        mask = torch.ones(B, 1, total_frames, device=full_mu.device, dtype=full_mu.dtype)

        # Run CFM decoder — freeze prompt + previous speech frames
        feat, _ = self.decoder(
            mu=full_mu,
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            noised_mels=z_init,
            meanflow=meanflow,
            freeze_len=freeze_len,
        )

        # Extract only NEW speech frames (after prompt + previously generated)
        return feat[:, :, freeze_len:]

    @torch.inference_mode()
    def decode_cfm_windowed(
        self,
        mu_window: torch.Tensor,
        spk_embedding: torch.Tensor,
        context_mel: torch.Tensor,
        n_timesteps: int = 2,
        meanflow: bool = True,
    ) -> torch.Tensor:
        """Run CFM decoder on a WINDOWED [context | new] speech sequence.

        Instead of processing the full [prompt | all_speech] mel sequence
        (O(N) per ODE step × N steps), processes only [context | new] frames
        (O(1) per step). This is the key optimization for O(N) streaming.

        For speech positions, the CFM conditioning (cond) is always zeros —
        voice identity comes from the global speaker embedding (spks), not
        from prompt_feat. So windowed CFM on speech-only frames produces
        equivalent results when context >= ~30 frames (300ms).

        Args:
            mu_window: (B, 80, C+N) — encoder output for context + new positions
            spk_embedding: (B, 80) — projected speaker embedding
            context_mel: (B, 80, C) — previously generated mel for context frames
            n_timesteps: ODE steps (default 2 for meanflow)
            meanflow: whether to use meanflow mode

        Returns:
            new_mel: (B, 80, N) — generated mel for new frames only
        """
        C = context_mel.size(2)
        total = mu_window.size(2)
        N = total - C

        # Build noised_mels covering the FULL window [context_mel | noise].
        # CausalConditionalCFM.forward() does: prompt_len = mu.size(2) - noised_mels.size(2)
        # By passing noised_mels of full size, prompt_len=0, so z = noised_mels entirely.
        # freeze_len=C then correctly freezes context_mel during ODE integration.
        noised_mels = torch.randn(1, 80, total, device=mu_window.device, dtype=mu_window.dtype)
        noised_mels[:, :, :C] = context_mel

        # cond: zeros for all positions (speech frames have no prompt_feat)
        cond = torch.zeros_like(mu_window)

        # mask: all valid
        mask = torch.ones(1, 1, total, device=mu_window.device, dtype=mu_window.dtype)

        # Call decoder forward — it builds z with context + noise, runs ODE
        # with freeze_len to reset context frames after each step.
        feat, _ = self.decoder(
            mu=mu_window,
            mask=mask,
            spks=spk_embedding,
            cond=cond,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
            freeze_len=C,
        )

        # Return only new frames
        return feat[:, :, C:]
