# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import torch
from torch import nn
from torch.nn import functional as F

from .convolution import ConvolutionModule
from .encoder_layer import ConformerEncoderLayer
from .positionwise_feed_forward import PositionwiseFeedForward
from ..utils.class_utils import (
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_ACTIVATION_CLASSES,
)
from ..utils.mask import make_pad_mask
from ..utils.mask import add_optional_chunk_mask


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        # look ahead
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs
        return outputs


@dataclass
class EncoderCaches:
    """KV-cache state for streaming encoder inference."""
    enc_att: List[torch.Tensor] = field(default_factory=list)   # per-layer att cache
    enc_cnn: List[torch.Tensor] = field(default_factory=list)   # per-layer cnn cache
    up_att: List[torch.Tensor] = field(default_factory=list)    # per-layer up-encoder att cache
    up_cnn: List[torch.Tensor] = field(default_factory=list)    # per-layer up-encoder cnn cache
    enc_cached_len: int = 0      # cached positions in main encoder
    up_cached_len: int = 0       # cached positions in up-encoder
    enc_output_cache: Optional[torch.Tensor] = None  # (B, cached_len, D) stage-1 output


class UpsampleConformerEncoder(torch.nn.Module):

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)
        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)
        self.up_embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size,
                                                      positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(4)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        # lookahead + conformer encoder
        xs = self.pre_lookahead_layer(xs)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        # upsample + conformer encoder
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size * self.up_layer.stride,
                                              num_decoding_left_chunks)
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                          pos_emb: torch.Tensor,
                          mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def init_caches(self, device: torch.device, dtype: torch.dtype) -> EncoderCaches:
        """Initialize empty encoder caches for streaming."""
        zero4 = lambda: torch.zeros(0, 0, 0, 0, device=device, dtype=dtype)
        return EncoderCaches(
            enc_att=[zero4() for _ in self.encoders],
            enc_cnn=[zero4() for _ in self.encoders],
            up_att=[zero4() for _ in self.up_encoders],
            up_cnn=[zero4() for _ in self.up_encoders],
            enc_cached_len=0,
            up_cached_len=0,
            enc_output_cache=None,
        )

    def forward_cached(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        caches: EncoderCaches,
    ) -> Tuple[torch.Tensor, torch.Tensor, EncoderCaches]:
        """Encoder forward with KV-cache for streaming.

        Runs embed + PreLookaheadLayer on the full sequence (cheap O(N)),
        then processes only NEW positions through attention layers using
        cached K/V from previous chunks.

        Args:
            xs: (B, T_full, D) — full token embeddings (all accumulated tokens)
            xs_lens: (B,) — sequence lengths
            caches: EncoderCaches from previous chunk (or init_caches())

        Returns:
            new_output: (B, new_up_len, D) — encoder output for new positions only
            masks: (B, 1, total_up_len) — full output masks
            caches: updated EncoderCaches
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # Embed + PreLookaheadLayer on FULL sequence (O(N), cheap)
        xs, pos_emb_full, masks = self.embed(xs, masks)
        xs = self.pre_lookahead_layer(xs)

        # --- Stage 1: Main encoder layers with KV-cache ---
        new_len = T - caches.enc_cached_len
        new_xs = xs[:, caches.enc_cached_len:]  # only new positions

        # Positional encoding for asymmetric Q(new) / K(all) attention
        pos_enc = self.embed.pos_enc
        if caches.enc_cached_len > 0:
            pos_emb = pos_enc.position_encoding_cached(
                query_len=new_len, key_len=T
            )
        else:
            pos_emb = pos_emb_full

        # Mask: new queries can attend to all positions
        mask_new = torch.ones(1, new_len, T, dtype=torch.bool, device=xs.device)
        mask_pad_new = torch.ones(1, 1, new_len, dtype=torch.bool, device=xs.device)

        new_enc_att = []
        new_enc_cnn = []
        for i, layer in enumerate(self.encoders):
            att_c = caches.enc_att[i] if i < len(caches.enc_att) else torch.zeros(0, 0, 0, 0, device=xs.device)
            cnn_c = caches.enc_cnn[i] if i < len(caches.enc_cnn) else torch.zeros(0, 0, 0, 0, device=xs.device)
            new_xs, mask_new, new_att_c, new_cnn_c = layer(
                new_xs, mask_new, pos_emb, mask_pad_new, att_c, cnn_c
            )
            new_enc_att.append(new_att_c)
            new_enc_cnn.append(new_cnn_c)

        # --- Stage 2: Upsample ---
        # Upsampler Conv1d(kernel=5, left-pad=4) needs context from previous output.
        # Prepend last few cached output frames for correct boundary.
        if caches.enc_output_cache is not None:
            # Need ceil(kernel_size / stride) = ceil(5/2) = 3 pre-upsample frames as context
            ctx_len = min(3, caches.enc_output_cache.size(1))
            ctx = caches.enc_output_cache[:, -ctx_len:]
            up_input = torch.cat([ctx, new_xs], dim=1)
        else:
            ctx_len = 0
            up_input = new_xs

        up_input_t = up_input.transpose(1, 2).contiguous()  # (B, D, ctx+new)
        up_lens = torch.tensor([up_input.size(1)], device=xs.device)
        up_output, up_out_lens = self.up_layer(up_input_t, up_lens)
        up_output = up_output.transpose(1, 2).contiguous()  # (B, 2*(ctx+new), D)

        # Remove context frames from upsampled output (they were just for Conv1d boundary)
        up_ctx_frames = ctx_len * self.up_layer.stride  # 2x upsampled context
        new_up = up_output[:, up_ctx_frames:]  # only new upsampled frames
        new_up_len = new_up.size(1)

        # Update cached encoder output (pre-upsample, for next chunk's context)
        if caches.enc_output_cache is not None:
            caches.enc_output_cache = torch.cat([caches.enc_output_cache, new_xs], dim=1)
        else:
            caches.enc_output_cache = new_xs

        # --- Stage 3: Up-encoder layers with KV-cache ---
        total_up_len = caches.up_cached_len + new_up_len
        up_masks = torch.ones(1, 1, total_up_len, dtype=torch.bool, device=xs.device)

        # Embed upsampled features (re-embed for pos encoding)
        up_mask_for_embed = torch.ones(1, 1, new_up_len, dtype=torch.bool, device=xs.device)
        new_up, up_pos_emb, _ = self.up_embed(new_up, up_mask_for_embed)

        # Positional encoding for asymmetric attention
        up_pos_enc = self.up_embed.pos_enc
        if caches.up_cached_len > 0:
            up_pos_emb = up_pos_enc.position_encoding_cached(
                query_len=new_up_len, key_len=total_up_len
            )

        up_mask_attn = torch.ones(1, new_up_len, total_up_len, dtype=torch.bool, device=xs.device)
        up_mask_pad = torch.ones(1, 1, new_up_len, dtype=torch.bool, device=xs.device)

        new_up_att = []
        new_up_cnn = []
        for i, layer in enumerate(self.up_encoders):
            att_c = caches.up_att[i] if i < len(caches.up_att) else torch.zeros(0, 0, 0, 0, device=xs.device)
            cnn_c = caches.up_cnn[i] if i < len(caches.up_cnn) else torch.zeros(0, 0, 0, 0, device=xs.device)
            new_up, up_mask_attn, new_att_c, new_cnn_c = layer(
                new_up, up_mask_attn, up_pos_emb, up_mask_pad, att_c, cnn_c
            )
            new_up_att.append(new_att_c)
            new_up_cnn.append(new_cnn_c)

        if self.normalize_before:
            new_up = self.after_norm(new_up)

        # Update caches
        caches.enc_att = new_enc_att
        caches.enc_cnn = new_enc_cnn
        caches.up_att = new_up_att
        caches.up_cnn = new_up_cnn
        caches.enc_cached_len = T
        caches.up_cached_len = total_up_len

        return new_up, up_masks, caches
