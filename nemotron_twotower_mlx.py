#!/usr/bin/env python3
"""
MLX port of nvidia/Nemotron-Labs-TwoTower-30B-A3B (NemotronHTwoTowerForCausalLM):
a block-wise autoregressive *diffusion* LM.

  context_tower  — frozen AR NemotronH backbone (KV + Mamba states for the prompt)
  denoiser_tower — diffusion decoder, adaLN-conditioned on the diffusion timestep
  mask-diffusion generation: per block, iteratively denoise a fully-masked block,
  commit high-confidence tokens, remask the rest, then commit the block to context.

Reuses mlx-lm's `nemotron_h` building blocks (NemotronHModel, mixers, caches).
Only the two-tower orchestration + adaLN + diffusion sampling are new here.

Reference (CUDA-only, cannot run on this Mac): modeling_nemotron_twotower.py.
"""
import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.nemotron_h import ModelArgs, NemotronHModel
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.base import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Time conditioning (PixArt-alpha adaLN-single)
# ---------------------------------------------------------------------------

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=1000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        # keys: mlp.0 (Linear), mlp.1 (SiLU, no params), mlp.2 (Linear)
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]

    def __call__(self, t):
        t_scaled = t * self.max_period
        h = timestep_embedding(t_scaled, self.frequency_embedding_size)
        h = self.mlp[0](h)
        h = self.mlp[1](h)
        return self.mlp[2](h)


def quantization_predicate(bits, group_size=64):
    """Mixed-precision scheme 'mixed_v1' for the diffusion two-tower.

    Diffusion compounds quantization error across denoising steps, so the tiny
    but critical timestep-conditioning MLPs stay bf16 and the embeddings / LM
    heads stay >=8-bit; the bulk (MoE experts, attention & Mamba projections)
    uses the target bits. Used identically at pack time and load time so the
    quantized module structure matches the saved tensors."""
    def pred(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if "t_embedder" in path or "t_block" in path:
            return False  # keep bf16
        if "embeddings" in path or "lm_head" in path:
            return {"group_size": group_size, "bits": max(bits, 8)}
        return {"group_size": group_size, "bits": bits}
    return pred


def modulate(x, shift, scale):
    # x:(B,L,D) shift/scale:(B,D)  ->  x*(1+scale) + shift
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def get_mod_params(t_emb, table):
    # t_emb:(B,3D) table:(3,D) -> shift,scale,gate each (B,D)
    B = t_emb.shape[0]
    D = table.shape[1]
    combined = table[None] + t_emb.reshape(B, 3, D)
    shift = combined[:, 0, :]
    scale = combined[:, 1, :]
    gate = combined[:, 2, :]
    return shift, scale, gate


# ---------------------------------------------------------------------------
# Two-Tower diffusion model
# ---------------------------------------------------------------------------

class TwoTowerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        H = args.hidden_size
        N = args.num_hidden_layers
        self.context_tower = NemotronHModel(args)
        self.context_lm_head = nn.Linear(H, args.vocab_size, bias=False)
        self.denoiser_tower = NemotronHModel(args)
        self.lm_head = nn.Linear(H, args.vocab_size, bias=False)
        # time conditioning
        self.t_embedder = TimestepEmbedder(H)
        self.t_block = [nn.SiLU(), nn.Linear(H, 3 * H, bias=True)]  # t_block.0/.1
        self.scale_shift_tables = [
            mx.zeros((3, H)) for _ in range(N)
        ]
        # single-char block-type pattern, e.g. "M","E","*"
        self.pattern = list(args.hybrid_override_pattern)
        # NemotronHModel keeps a COMPACT cache list (one slot per M/* layer only).
        # Map each layer index -> its slot in that compact list (None for E/-).
        self.cache_index = []
        c = 0
        for t in self.pattern:
            if t in ("M", "*"):
                self.cache_index.append(c)
                c += 1
            else:
                self.cache_index.append(None)

    # -- weight loading -----------------------------------------------------

    def sanitize(self, weights):
        # conv1d weight layout + per-tower expert stacking (mirror nemotron_h)
        out = {}
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            out[k] = v
        weights = out
        for tower in ("context_tower", "denoiser_tower"):
            for l in range(self.args.num_hidden_layers):
                prefix = f"{tower}.layers.{l}.mixer"
                for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
                    k0 = f"{prefix}.experts.0.{m}.weight"
                    if k0 in weights:
                        stack = [
                            weights.pop(f"{prefix}.experts.{e}.{m}.weight")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(stack)
        return weights

    # -- context tower cache (prefill / extend) -----------------------------

    def _make_ctx_cache(self):
        # COMPACT: one entry per M/* layer, matching NemotronHModel.__call__.
        caches = []
        for t in self.pattern:
            if t == "M":
                caches.append(ArraysCache(size=2))
            elif t == "*":
                caches.append(KVCache())
        return caches

    def build_context_cache(self, prompt_ids):
        """Prefill the whole prompt through the context tower, populating per-layer
        Mamba (conv+ssm) and attention (KV) caches. Returns the cache list."""
        caches = self._make_ctx_cache()
        # NemotronHModel.__call__ populates the caches in place; return is unused.
        self.context_tower(prompt_ids, cache=caches)
        return caches

    def extend_context_cache(self, block_ids, caches):
        """Advance the context cache by a committed block (standard causal forward)."""
        self.context_tower(block_ids, cache=caches)
        return caches

    # -- denoiser bidirectional attention -----------------------------------

    @staticmethod
    def _repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        b, h, l, d = x.shape
        x = mx.broadcast_to(x[:, :, None, :, :], (b, h, n_rep, l, d))
        return x.reshape(b, h * n_rep, l, d)

    def _denoiser_attention(self, mixer, hidden, ctx_k, ctx_v):
        """Bidirectional self-attention over [context_KV | block_KV] (NoPE, is_causal=False)."""
        B, L, _ = hidden.shape
        nH = mixer.num_heads
        nKV = mixer.num_key_value_heads
        hd = mixer.head_dim
        q = mixer.q_proj(hidden).reshape(B, L, nH, hd).transpose(0, 2, 1, 3)
        k = mixer.k_proj(hidden).reshape(B, L, nKV, hd).transpose(0, 2, 1, 3)
        v = mixer.v_proj(hidden).reshape(B, L, nKV, hd).transpose(0, 2, 1, 3)
        if ctx_k is not None and ctx_k.shape[2] > 0:
            k = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2)
            v = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2)
        n_rep = nH // nKV
        k = self._repeat_kv(k, n_rep)
        v = self._repeat_kv(v, n_rep)
        # full (non-causal) attention, no mask
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=mixer.scale, mask=None)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, nH * hd)
        return mixer.o_proj(o)

    def _denoiser_mamba(self, mixer, hidden, ctx_conv, ctx_ssm):
        """Forward-only chunk-scan of the block seeded from the context Mamba state.
        Reuses the mixer's own conv+ssm via a fresh seeded cache (context state is
        immutable; the fresh cache absorbs the writes)."""
        seed = ArraysCache(size=2)
        seed[0] = ctx_conv
        seed[1] = ctx_ssm
        return mixer(hidden, mask=None, cache=seed)

    # -- one diffusion denoiser forward over a full (masked) block ----------

    def denoiser_forward(self, block_ids, ctx_caches, t):
        tower = self.denoiser_tower
        t_repr = self.t_embedder(t.astype(mx.float32))
        t_emb = self.t_block[1](self.t_block[0](t_repr))  # (B,3H)

        hidden = tower.embeddings(block_ids)
        for i, block in enumerate(tower.layers):
            residual = hidden
            shift, scale, gate = get_mod_params(t_emb, self.scale_shift_tables[i])
            bt = block.block_type
            if bt in ("M", "*"):
                # norm is fused after modulate in mcore -> modulate THEN norm
                h = modulate(hidden, shift, scale)
                h = block.norm(h.astype(block.norm.weight.dtype))
            else:  # E / - : separate pre-norm -> norm THEN modulate
                h = block.norm(hidden.astype(block.norm.weight.dtype))
                h = modulate(h, shift, scale)

            if bt == "M":
                c = ctx_caches[self.cache_index[i]]
                h = self._denoiser_mamba(block.mixer, h, c[0], c[1])
            elif bt == "*":
                c = ctx_caches[self.cache_index[i]]
                ctx_k = c.keys[..., : c.offset, :] if c.offset > 0 else None
                ctx_v = c.values[..., : c.offset, :] if c.offset > 0 else None
                h = self._denoiser_attention(block.mixer, h, ctx_k, ctx_v)
            else:
                h = block.mixer(h)

            h = gate[:, None, :] * h
            hidden = residual + h

        hidden = tower.norm_f(hidden)
        return self.lm_head(hidden.astype(self.lm_head.weight.dtype)).astype(mx.float32)

    # -- mask-diffusion generation ------------------------------------------

    @staticmethod
    def _mdlm_logprobs(logits, xt, mask_token_id):
        logits = mx.array(logits)
        neg = mx.full(logits.shape[-1:], -1e12)
        # mask token -> -inf
        logits = mx.concatenate([
            logits[..., :mask_token_id],
            mx.full(logits[..., mask_token_id:mask_token_id + 1].shape, -1e12),
            logits[..., mask_token_id + 1:],
        ], axis=-1)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # unmasked positions predict themselves w.p. 1
        unmasked = (xt != mask_token_id)  # (B,L)
        onehot = mx.zeros(log_probs.shape)
        # scatter 0.0 at xt for unmasked, -1e12 elsewhere
        forced = mx.full(log_probs.shape, -1e12)
        idx = mx.clip(xt, 0, log_probs.shape[-1] - 1)
        forced = _set_at_last(forced, idx, 0.0)
        log_probs = mx.where(unmasked[..., None], forced, log_probs)
        return log_probs

    def generate_mask_diffusion(
        self, input_ids, max_new_tokens=128, block_size=16, steps_per_block=16,
        mask_token_id=3, confidence_threshold=0.9, eos_token_id=None, verbose=False,
    ):
        assert max_new_tokens % block_size == 0
        B = input_ids.shape[0]
        num_blocks = max_new_tokens // block_size
        caches = self.build_context_cache(input_ids)
        context_ids = input_ids
        nfe = 0

        for blk in range(num_blocks):
            xt = mx.full((B, block_size), mask_token_id, dtype=mx.int32)
            for step in range(steps_per_block):
                is_masked = (xt == mask_token_id)
                n_masked = int(is_masked.sum().item())
                if n_masked == 0:
                    break
                t_model = is_masked.astype(mx.float32).mean()
                t_vec = mx.broadcast_to(t_model.reshape(1), (B,))
                logits = self.denoiser_forward(xt, caches, t_vec)
                nfe += 1
                log_xt = self._mdlm_logprobs(logits, xt, mask_token_id)
                x_theta = mx.exp(log_xt)
                predicted = mx.argmax(log_xt, axis=-1).astype(mx.int32)
                conf = mx.take_along_axis(
                    x_theta, predicted[..., None].astype(mx.int32), axis=-1
                )[..., 0]
                conf = mx.where(is_masked, conf, mx.array(float("inf")))

                is_last = (step == steps_per_block - 1)
                n_masked_row = is_masked.sum(-1)  # (B,)
                if is_last:
                    commit = n_masked_row
                else:
                    remaining = max(1, steps_per_block - step)
                    num_above = ((conf > confidence_threshold) & is_masked).sum(-1)
                    commit = mx.where(num_above > 0, num_above, mx.ones_like(num_above))
                    min_commit = mx.ceil(n_masked_row.astype(mx.float32) / remaining).astype(commit.dtype)
                    commit = mx.minimum(mx.maximum(commit, min_commit), n_masked_row)

                output = mx.where(is_masked, predicted, xt)
                # remask lowest-confidence (per row)
                new_xt = []
                for b in range(B):
                    row = output[b]
                    nrem = int((n_masked_row[b] - commit[b]).item())
                    if nrem > 0:
                        cb = conf[b]
                        order = mx.argsort(cb)  # ascending; masked lowest first
                        remask_idx = order[:nrem]
                        row = _set_at_index(row, remask_idx, mask_token_id)
                    new_xt.append(row)
                xt = mx.stack(new_xt)
                mx.eval(xt)

            context_ids = mx.concatenate([context_ids, xt], axis=1)
            caches = self.extend_context_cache(xt, caches)
            if verbose:
                print(f"[block {blk+1}/{num_blocks}] nfe={nfe}")
            if eos_token_id is not None:
                eos = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
                if any(bool((xt == e).any().item()) for e in eos):
                    break

        self.last_nfe = nfe
        return context_ids


# helpers for scatter on last axis / by index (mlx has no in-place scatter sugar)
def _set_at_last(arr, idx, value):
    """arr:(...,V) idx:(...) -> set arr[...,idx]=value along last axis."""
    oh = nn.losses._make_one_hot if False else None
    V = arr.shape[-1]
    onehot = (mx.arange(V) == idx[..., None])
    return mx.where(onehot, mx.array(value, dtype=arr.dtype), arr)


def _set_at_index(vec, indices, value):
    """vec:(L,) set vec[indices]=value."""
    L = vec.shape[0]
    mask = mx.zeros((L,), dtype=mx.bool_)
    mask = mask + (mx.arange(L)[:, None] == indices[None, :]).any(axis=1)
    return mx.where(mask, mx.array(value, dtype=vec.dtype), vec)
