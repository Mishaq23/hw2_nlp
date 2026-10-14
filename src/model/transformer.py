"""Encoder-decoder Transformer trained from scratch for code generation."""

import math

import torch
from torch import nn

from src.datasets.tokenizer import CodeTokenizer
from src.model.transformer_layers import DecoderLayer, EncoderLayer, RotaryEmbedding


class CodeGenerationTransformer(nn.Module):
    def __init__(
        self,
        tokenizer_path,
        d_model=256,
        n_heads=4,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=1024,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        bias=True,
        share_embeddings=True,
        tie_output_embedding=True,
        scale_embeddings=True,
        init_std=0.02,
    ):
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError(
                "d_model must be divisible by n_heads, with even head_dim for RoPE."
            )
        if n_encoder_layers < 1 or n_decoder_layers < 1:
            raise ValueError(
                "Encoder and decoder must each contain at least one layer."
            )

        tokenizer = CodeTokenizer.from_file(tokenizer_path)
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.embedding_scale = math.sqrt(d_model) if scale_embeddings else 1.0

        self.encoder_embedding = nn.Embedding(
            self.vocab_size, d_model, padding_idx=self.pad_id
        )
        self.decoder_embedding = (
            self.encoder_embedding
            if share_embeddings
            else nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)
        )
        self.embedding_dropout = nn.Dropout(dropout)
        rope = RotaryEmbedding(d_model // n_heads, rope_theta)
        layer_kwargs = dict(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
            rope=rope,
        )
        self.encoder = nn.ModuleList(
            EncoderLayer(**layer_kwargs) for _ in range(n_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(**layer_kwargs) for _ in range(n_decoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.output_projection = nn.Linear(d_model, self.vocab_size, bias=False)

        # Initialize each shared module once, then tie output weights.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=init_std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
                if isinstance(module, nn.Embedding):
                    with torch.no_grad():
                        module.weight[self.pad_id].zero_()
        if tie_output_embedding:
            self.output_projection.weight = self.decoder_embedding.weight

    @staticmethod
    def _causal_mask(length, device, offset=0):
        queries = torch.arange(offset, offset + length, device=device)
        keys = torch.arange(offset + length, device=device)
        return keys[None, :] <= queries[:, None]

    def encode(self, encoder_input_ids, encoder_attention_mask=None):
        """Encode the source once; the returned memory is reused during generation."""
        if encoder_attention_mask is None:
            encoder_attention_mask = encoder_input_ids.ne(self.pad_id)
        mask = encoder_attention_mask[:, None, None, :]
        x = self.embedding_dropout(
            self.encoder_embedding(encoder_input_ids) * self.embedding_scale
        )
        for layer in self.encoder:
            x = layer(x, mask)
        return self.encoder_norm(x)

    def _decode(self, decoder_input_ids, memory, self_mask, memory_mask, cache=None):
        x = self.embedding_dropout(
            self.decoder_embedding(decoder_input_ids) * self.embedding_scale
        )
        next_cache = []
        for index, layer in enumerate(self.decoder):
            layer_cache = None if cache is None else cache[index]
            x, layer_cache = layer(x, memory, self_mask, memory_mask, layer_cache)
            next_cache.append(layer_cache)
        return self.output_projection(self.decoder_norm(x)), next_cache

    def decode(
        self,
        decoder_input_ids,
        memory,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
    ):
        """Teacher-forced decoding; target shifting is already done by the dataset."""
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(self.pad_id)
        self_mask = (
            self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)[
                None, None, :, :
            ]
            & decoder_attention_mask[:, None, None, :]
        )
        memory_mask = (
            None
            if encoder_attention_mask is None
            else encoder_attention_mask[:, None, None, :]
        )
        logits, _ = self._decode(decoder_input_ids, memory, self_mask, memory_mask)
        return logits

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        **batch,
    ):
        if encoder_attention_mask is None:
            encoder_attention_mask = encoder_input_ids.ne(self.pad_id)
        memory = self.encode(encoder_input_ids, encoder_attention_mask)
        return {
            "logits": self.decode(
                decoder_input_ids,
                memory,
                encoder_attention_mask,
                decoder_attention_mask,
            )
        }

    @torch.no_grad()
    def decode_step(self, token_ids, memory, encoder_attention_mask=None, cache=None):
        """In eval mode, consume new tokens and return next-token logits [B, V].

        Start with BOS and cache=None, then pass the previous cache each step.
        New tokens must have no padding. Encoder K/V are projected on the first
        step and reused; cached self-attention K already include RoPE positions.
        """
        offset = 0 if cache is None else cache[0][0][0].size(-2)
        self_mask = (
            None
            if token_ids.size(1) == 1
            else self._causal_mask(token_ids.size(1), token_ids.device, offset)
        )
        memory_mask = (
            None
            if encoder_attention_mask is None
            else encoder_attention_mask[:, None, None, :]
        )
        logits, cache = self._decode(token_ids, memory, self_mask, memory_mask, cache)
        return logits[:, -1], cache
