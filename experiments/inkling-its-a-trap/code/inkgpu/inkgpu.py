"""Inkling 2 Mini "readable" on GPUs: forward + gradients with respect to activations (not weights), experts kept in the
checkpoint format (NVFP4 layers 0-19, MXFP8 layers 20-23) and dequantized on the fly, model split by layers over 2 GPUs.

Reuses the transformers modeling from PR #51050 (Inkling2LoopForCausalLM), built on the meta device and filled directly
from the safetensors: all non-expert modules are the library's, with three substitutions that leave the math unchanged:
  - MoE: QuantMoE (library router; experts dequantized on the fly in groups of 8; an autograd.Function that dequantizes
    again in the backward and saves no weights);
  - attention core: in query blocks (same sequence of roundings as the library's eager path), with per-block checkpointing,
    to avoid materializing [24, T, T];
  - attention residuals on the blocks (attention_input, mlp_input, final_input, message_input): in chunks along T (they are
    per-position), with per-chunk checkpointing.
The 48-visit loop is written here (same order as Inkling2LoopTextModel.forward) to control device placement,
per-visit checkpointing, reads and steering. API: see README.md.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# name in the file (after "model.llm.layers.{L}.") -> library module name (after "model.layers.{L}.")
LAYER_MAP = {
    "attn.attn_sink_proj.weight": "self_attn.sink_proj.weight",
    "attn.gate_attn.weight": "self_attn.o_proj.gate_proj.weight",
    "attn.wo_ud.weight": "self_attn.o_proj.linear.weight",
    "attn.wq_du.weight": "self_attn.q_proj.linear.weight",
    "attn.q_l2_clip.scale": "self_attn.q_proj.norm.scale",
    "attn.wk_dv.weight": "self_attn.k_proj.linear.weight",
    "attn.wv_dv.weight": "self_attn.v_proj.linear.weight",
    "attn.k_sconv.weight": "self_attn.k_proj.sconv.conv1d.weight",
    "attn.v_sconv.weight": "self_attn.v_proj.sconv.conv1d.weight",
    "attn.k_sconv.static_w": "self_attn.k_proj.sconv.static_weight",
    "attn.v_sconv.static_w": "self_attn.v_proj.sconv.static_weight",
    "attn.k_sconv.weight_proj": "self_attn.k_proj.sconv.weight_proj.weight",
    "attn.v_sconv.weight_proj": "self_attn.v_proj.sconv.weight_proj.weight",
    "attn.wr_du.weight": "self_attn.relative_position_bias.linear.weight",
    "attn.rel_l2_clip.scale": "self_attn.relative_position_bias.norm.scale",
    "attn.rel_logits_proj.proj": "self_attn.relative_position_bias.proj",
    "attn_norm.weight": "input_layernorm.weight",
    "q_input_norm.weight": "q_input_layernorm.weight",
    "k_input_norm.weight": "k_input_layernorm.weight",
    "v_input_norm.weight": "v_input_layernorm.weight",
    "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
    "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
}
for _c, _m in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
    for _a, _b in (("down_weight", "down_proj"), ("forget_weight", "forget_gate_proj"), ("gain_weight", "message_gate_proj")):
        LAYER_MAP[f"controllers.{_c}.{_a}"] = f"{_m}_message_gate.{_b}.weight"


def visit_table(cfg):
    """[(visit, layer, pass)] in execution order: 0-5 prelude, 6-41 core x3, 42-47 tail."""
    s, e, n, P = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.num_hidden_layers, cfg.loop_num_passes
    order = [(L, 0) for L in range(s)] + [(L, p) for p in range(P) for L in range(s, e)] + [(L, 0) for L in range(e, n)]
    return [(v, L, p) for v, (L, p) in enumerate(order)]


# ----------------------------------------------------------------------------------------------------------------------
# Quantized experts
# ----------------------------------------------------------------------------------------------------------------------
class QMat:
    """One expert matrix of a layer in the checkpoint format (all experts), dequantizable per expert.

    nvfp4: q = U8 bytes [E, R, C/2] (low nibble = even column), s = E4M3 scale [E, R, C/16] (1x16 blocks along the
           input), f = per-expert amax/denominator (fp32). Value = cast( (e2m1 * s) * f ), products in fp32.
    mxfp8: q = E4M3 [E, R, C], s = UE8M0 U8 [E, R, C/32]. Value = cast( f8 * 2^(s-127) ) (exact in bf16).
    FP8 tensors are kept as uint8 (same bytes): index_select and the edit do not depend on the device's FP8 support.
    """

    def __init__(self, fmt, q, s, f=None):
        self.fmt, self.q, self.s, self.f = fmt, q, s, f
        self.E, self.rows = q.shape[0], q.shape[1]
        self.cols = q.shape[2] * 2 if fmt == "nvfp4" else q.shape[2]
        if fmt == "nvfp4":
            e2m1 = torch.tensor(E2M1, dtype=torch.float32, device=q.device)
            b = torch.arange(256, device=q.device)
            self.lut = torch.stack((e2m1[b & 0xF], e2m1[b >> 4]), dim=-1).contiguous()  # [256, 2]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.q, self.s, self.f) if t is not None)

    def dq(self, ids, dtype):
        """[n, rows, cols] in `dtype` for the experts `ids` (LongTensor on the device)."""
        q, s = self.q.index_select(0, ids), self.s.index_select(0, ids)
        n = q.shape[0]
        if self.fmt == "nvfp4":
            v = F.embedding(q.int(), self.lut).view(n, self.rows, self.cols // 16, 16)
            v = v * s.view(torch.float8_e4m3fn).float().unsqueeze(-1)
            v = v * self.f.index_select(0, ids).view(n, 1, 1, 1)
        else:
            e = s.int()
            sc = (e << 23).view(torch.float32)  # 2^(e-127), exact for 1 <= e <= 254
            sc = torch.where(e == 0, torch.full_like(sc, 2.0**-127), sc)
            v = q.view(torch.float8_e4m3fn).float().view(n, self.rows, self.cols // 32, 32) * sc.unsqueeze(-1)
        return v.view(n, self.rows, self.cols).to(dtype)


class _State:
    """Shared state: backward tap, forward capture, routing recording, recomputation in progress."""

    def __init__(self):
        self.recomputing = False
        self.tap = None
        self.capture = None
        self.route = None
        self.freeze = None  # {visit: idx [N, k]}: frozen routing (expert choice fixed, weights recomputed)
        self.prefix = None  # {visit: {"k","v","k_conv","v_conv","attn_conv","mlp_conv"}}: prefix state
        self.record = None  # dict to fill with the state for the next chunk (prefill)
        self.pos0 = 0  # absolute position of the first current token
        self.cur_visit = -1


class Tap:
    """Accumulates in the backward pass sum_token h*dm/dh per (visit, expert, neuron) over the routed tokens, filtered by
    position; optionally keeps the raw h and dm/dh at the positions `raw_pos`."""

    def __init__(self, T, E, Fdim, devices, positions=None, raw_pos=None):
        self.E, self.F = E, Fdim
        mask = torch.zeros(T, dtype=torch.bool)
        mask[positions if positions is not None else slice(None)] = True
        self.mask = {d: mask.to(d) for d in devices}
        self.raw_pos = None if raw_pos is None else {d: torch.as_tensor(raw_pos, dtype=torch.long).to(d) for d in devices}
        self.acc, self.cnt, self.raw = {}, {}, []

    def add(self, layer, visit, e, tok, h, g):
        d = h.device
        sel = self.mask[d][tok]
        prod = h.float() * g.float()
        if visit not in self.acc:
            self.acc[visit] = torch.zeros(self.E, self.F, dtype=torch.float32, device=d)
            self.cnt[visit] = torch.zeros(self.E, dtype=torch.int64, device=d)
        self.acc[visit][e] += (prod * sel.unsqueeze(-1)).sum(0)
        self.cnt[visit][e] += sel.sum()
        if self.raw_pos is not None:
            r = torch.isin(tok, self.raw_pos[d])
            if bool(r.any()):
                self.raw.append((visit, layer, e, tok[r].cpu(), h[r].float().cpu(), g[r].float().cpu()))


class Capture:
    """Captures in the forward pass h of every expert at the tokens of the given positions: list of (visit, layer, expert, pos, h)."""

    def __init__(self, positions, devices):
        self.pos = {d: torch.as_tensor(positions, dtype=torch.long).to(d) for d in devices}
        self.out = []

    def __call__(self, moe, visit, e, tok, h):
        r = torch.isin(tok, self.pos[h.device])
        if bool(r.any()):
            self.out.append((visit, moe.layer_idx, e, tok[r].cpu(), h[r].float().cpu()))


class _MoEFn(torch.autograd.Function):
    """y = MoE(x; w, idx) with quantized experts: saves only x, w, idx; the backward dequantizes again expert by expert
    and returns the gradients with respect to x and the routing weights (no gradient on the model weights)."""

    @staticmethod
    def forward(ctx, x, w, idx, moe, visit):
        ctx.moe, ctx.visit = moe, visit
        ctx.save_for_backward(x, w, idx)
        return moe._fwd(x, w, idx, visit)

    @staticmethod
    def backward(ctx, gout):
        x, w, idx = ctx.saved_tensors
        gx, gw = ctx.moe._bwd(x, w, idx, gout, ctx.visit, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return gx, gw, None, None, None


class QuantMoE(nn.Module):
    """Replaces Inkling2LoopSparseMoeBlock. Router = the library module (fp32). Expert e:
        gu = cast(x @ W13_e^T) with the w13 rows in FILE order (row 2j = gate j, 2j+1 = up j)
        h  = cast( softcap(g,4)*sigma(g)*softcap(a,25) ) in fp32, g = gu[:, 0::2], a = gu[:, 1::2]   (the library's activation)
        o  = cast(h @ W2_e^T);  y = cast( sum_k fp32(o_k)*w_k )
    Per-neuron runtime scales: scale_up[e] (multiplies a before the softcap) and scale_h[e] (multiplies h)."""

    def __init__(self, gate, layer_idx, cfg, state):
        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx
        self.state = state
        self.E, self.F = cfg.n_routed_experts, cfg.moe_intermediate_size
        self.c_gate, self.c_up = float(cfg.mlp_gate_softcap), float(cfg.mlp_up_softcap)
        self.w13 = self.w2 = None
        self.dtype = torch.bfloat16
        self.chunk = 4
        self.scale_h, self.scale_up = {}, {}
        self.cur_visit = -1
        self.plain = False

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        st = self.state
        if st.freeze is not None:  # same experts as the recorded forward; weights as in the library router
            g = self.gate
            idx = st.freeze[self.cur_visit].to(x.device)
            logits = F.linear(x.float(), g.weight)
            w = g.score_fn(logits).gather(1, idx)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) * g.routed_scaling_factor
        else:
            logits, w, idx = self.gate(x)
        if st.route is not None and not st.recomputing:
            st.route(self, logits, w, idx)
        out = self._plain(x, w, idx) if self.plain else _MoEFn.apply(x, w, idx, self, self.cur_visit)
        return out.view(B, T, H)

    def _groups(self, idx):
        k = idx.shape[1]
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.E).tolist()
        tok, slot = order // k, order % k
        out, o = [], 0
        for e, c in enumerate(counts):
            if c:
                out.append((e, tok[o : o + c], slot[o : o + c]))
                o += c
        return out

    def _chunks(self, groups, device):
        for c0 in range(0, len(groups), self.chunk):
            grp = groups[c0 : c0 + self.chunk]
            ids = torch.tensor([g[0] for g in grp], dtype=torch.long, device=device)
            yield grp, self.w13.dq(ids, self.dtype), self.w2.dq(ids, self.dtype)

    def _act32(self, g, a, e):
        su = self.scale_up.get(e)
        if su is not None:
            a = a * su
        cg, cu = self.c_gate, self.c_up
        return torch.tanh(g / cg) * cg * torch.sigmoid(g) * (torch.tanh(a / cu) * cu)

    def _h(self, gu, e):
        h = self._act32(gu[:, 0::2].float(), gu[:, 1::2].float(), e).to(gu.dtype)
        sh = self.scale_h.get(e)
        return h if sh is None else (h.float() * sh).to(gu.dtype)

    def _fwd(self, x, w, idx, visit):
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        cap = None if self.state.recomputing else self.state.capture
        for grp, W13, W2 in self._chunks(self._groups(idx), x.device):
            for i, (e, tok, slot) in enumerate(grp):
                h = self._h(F.linear(x[tok], W13[i]), e)
                if cap is not None:
                    cap(self, visit, e, tok, h)
                o = F.linear(h, W2[i])
                out.index_add_(0, tok, o.float() * w[tok, slot].unsqueeze(-1))
        return out.to(x.dtype)

    def _bwd(self, x, w, idx, gout, visit, need_x, need_w):
        g32 = gout.float()
        gx = torch.zeros(x.shape, dtype=torch.float32, device=x.device) if need_x else None
        gw = torch.zeros_like(w) if need_w else None
        tap = self.state.tap
        for grp, W13, W2 in self._chunks(self._groups(idx), x.device):
            for i, (e, tok, slot) in enumerate(grp):
                gu = F.linear(x[tok], W13[i])
                g = gu[:, 0::2].float().requires_grad_(True)
                a = gu[:, 1::2].float().requires_grad_(True)
                with torch.enable_grad():
                    h = self._act32(g, a, e).to(gu.dtype)
                    sh = self.scale_h.get(e)
                    if sh is not None:
                        h = (h.float() * sh).to(gu.dtype)
                go = g32[tok]
                if need_w:
                    o = F.linear(h.detach(), W2[i])
                    gw[tok, slot] = (go * o.float()).sum(-1)
                gh = (go * w[tok, slot].unsqueeze(-1)).to(gu.dtype) @ W2[i]  # [n, F]
                if tap is not None:
                    tap.add(self.layer_idx, visit, e, tok, h.detach(), gh)
                if need_x:
                    dg, da = torch.autograd.grad(h, (g, a), gh)
                    ggu = torch.empty_like(gu)
                    ggu[:, 0::2] = dg.to(gu.dtype)
                    ggu[:, 1::2] = da.to(gu.dtype)
                    gx.index_add_(0, tok, (ggu @ W13[i]).float())
        return (gx.to(x.dtype) if need_x else None), gw

    def _plain(self, x, w, idx):
        """Same computation with ordinary autograd (tests only: keeps the dequantized weights in the graph)."""
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        tap, visit = self.state.tap, self.cur_visit
        with torch.no_grad():
            chunks = list(self._chunks(self._groups(idx), x.device))
        for grp, W13, W2 in chunks:
            for i, (e, tok, slot) in enumerate(grp):
                h = self._h(F.linear(x[tok], W13[i]), e)
                if tap is not None and h.requires_grad:
                    h.register_hook(lambda g, e=e, tok=tok, h=h: tap.add(self.layer_idx, visit, e, tok, h.detach(), g))
                o = F.linear(h, W2[i])
                out = out.index_add(0, tok, o.float() * w[tok, slot].unsqueeze(-1))
        return out.to(x.dtype)


# ----------------------------------------------------------------------------------------------------------------------
# Attention in query blocks (same arithmetic as the library's eager_attention_forward + RelativePositionBias)
# ----------------------------------------------------------------------------------------------------------------------
def _attn_block(q, k, v, rs, proj, sinks, *, q0, k0, scaling, window, rel_extent):
    B, nh, C, D = q.shape
    nkv, K = k.shape[1], k.shape[2]
    G = nh // nkv
    s = torch.matmul(q.reshape(B, nkv, G * C, D), k.transpose(-1, -2)).view(B, nh, C, K)
    s = s * scaling
    dev = q.device
    dist = (torch.arange(C, device=dev) + q0)[:, None] - (torch.arange(K, device=dev) + k0)[None, :]
    rl = (rs @ proj).transpose(1, 2)  # [B, nh, C, R]
    bias = rl.gather(-1, dist.clamp(0, rel_extent - 1).expand(B, nh, C, K))
    bias = bias.masked_fill((dist < 0) | (dist >= rel_extent), 0.0)
    s = s + bias
    allowed = dist >= 0
    if window:
        allowed = allowed & (dist < window)
    s = s.masked_fill(~allowed, torch.finfo(s.dtype).min)  # = s + the library's additive mask (0 / min)
    p = F.softmax(torch.cat([s, sinks], dim=-1).float(), dim=-1)[..., :-1].to(q.dtype)
    o = torch.matmul(p.reshape(B, nkv, G * C, K), v)
    return o.view(B, nh, C, D)


def _chunk_len(T, Ktot, window, budget):
    """Query rows per block: power-of-2 blocks with (rows x keys of the block) <= budget."""
    C = 1 << max(0, (T - 1).bit_length())
    while C > 16:
        K = min(Ktot, C + window - 1) if window else Ktot
        if min(C, T) * K <= budget:
            break
        C //= 2
    return min(C, T)


def sink_attention(q, k, v, rs, proj, sinks, scaling, window, rel_extent, budget, q_start=0, k_start=0):
    """q [B,nh,T,D] at absolute positions q_start.., k/v [B,nkv,K,D] at positions k_start.. (prefix included),
    rs [B,T,nh,dr], proj [dr,R], sinks [B,nh,T,1] -> [B,nh,T,D]."""
    T, Ktot = q.shape[2], k.shape[2]
    C = _chunk_len(T, Ktot, window, budget)
    outs = []
    for q0 in range(0, T, C):
        q1 = min(q0 + C, T)
        a0, a1 = q_start + q0, q_start + q1  # absolute positions of the block's queries
        ka = max(k_start, a0 - window + 1) if window else k_start
        j0, j1 = ka - k_start, a1 - k_start
        fn = functools.partial(_attn_block, q0=a0, k0=ka, scaling=scaling, window=window, rel_extent=rel_extent)
        args = (q[:, :, q0:q1], k[:, :, j0:j1], v[:, :, j0:j1], rs[:, q0:q1], proj, sinks[:, :, q0:q1])
        if torch.is_grad_enabled() and C < T:
            outs.append(checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False))
        else:
            outs.append(fn(*args))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)


def _attn_forward(self, hidden_states, key_hidden_states=None, value_hidden_states=None, attention_mask=None,
                  relative_distance=None, conv_mask=None, cache=None, cache_offset=0, **kwargs):
    """Replaces forward of Inkling2LoopSlidingAttention / Inkling2LoopFullAttention (without the library's cache).
    With a prefix (state.prefix) the prefix keys/values of this visit precede those of the current tokens;
    with state.record the keys/values to pass to the next chunk are recorded (all of them in global layers, the last W-1
    in local ones, like the library's cache)."""
    assert cache is None
    st = self._ink_state
    pos = kwargs["position_ids"]
    kh = hidden_states if key_hidden_states is None else key_hidden_states
    vh = hidden_states if value_hidden_states is None else value_hidden_states
    B, T, _ = hidden_states.shape
    q = self.q_proj(hidden_states, pos)
    k = self.k_proj(kh)
    v = self.v_proj(vh)
    pre = st.prefix.get(st.cur_visit) if st.prefix is not None else None
    q_start = st.pos0
    if pre is not None and "k" in pre:
        k = torch.cat([pre["k"].unsqueeze(0).to(k.dtype), k], dim=2)
        v = torch.cat([pre["v"].unsqueeze(0).to(v.dtype), v], dim=2)
    k_start = q_start + T - k.shape[2]
    W = self.sliding_window or 0
    if st.record is not None:
        rec = st.record.setdefault(st.cur_visit, {})
        rec["k"] = (k[0, :, -(W - 1):] if W else k[0]).detach().clone(memory_format=torch.contiguous_format)
        rec["v"] = (v[0, :, -(W - 1):] if W else v[0]).detach().clone(memory_format=torch.contiguous_format)
    rpb = self.relative_position_bias
    rs = rpb.norm(rpb.linear(hidden_states).view(B, T, rpb.num_heads, -1), pos)
    sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
    o = sink_attention(q, k, v, rs, rpb.proj, sinks, self.scaling, W, rpb.rel_extent, self._ink_budget,
                       q_start=q_start, k_start=k_start)
    o = o.transpose(1, 2).reshape(B, T, -1)
    return self.o_proj(o, hidden_states), None


def _conv_tail(orig, state, name, dynamic, K=4):
    """Short causal conv with the prefix tail: the last K-1 recorded inputs (the conv state) are prepended and their
    outputs discarded; with state.record the new tail is recorded. Dynamic conv: the dynamic weights of the prepended
    rows are not needed (outputs discarded), their dynamic input is zero."""

    def f(hidden_states, dynamic_input=None, *args, **kwargs):
        pre = state.prefix.get(state.cur_visit) if state.prefix is not None else None
        tail = None if pre is None else pre.get(name)
        if state.record is not None:
            h0 = hidden_states[0].detach()
            full = h0 if tail is None else torch.cat([tail.to(h0.dtype), h0], dim=0)
            # clone: a view would keep the whole input matrix [T, C] of the visit alive
            state.record.setdefault(state.cur_visit, {})[name] = full[-(K - 1):].clone()
        if tail is None:
            return orig(hidden_states, dynamic_input, *args, **kwargs)
        n = tail.shape[0]
        hs = torch.cat([tail.unsqueeze(0).to(hidden_states.dtype), hidden_states], dim=1)
        di = dynamic_input
        if dynamic:
            di = torch.cat([dynamic_input.new_zeros(dynamic_input.shape[0], n, dynamic_input.shape[2]), dynamic_input],
                           dim=1)
        return orig(hs, di, *args, **kwargs)[:, n:]

    return f


def _chunked_res(orig, chunk):
    """Per-position module (attention residual on the blocks, message gate) in chunks along T with per-chunk checkpointing:
    same output, bounded memory. The first two arguments are [B, T, ...]."""

    def f(block_states, partial, *rest):
        T = partial.shape[1]
        if T <= chunk:
            return orig(block_states, partial, *rest)
        outs = []
        for t0 in range(0, T, chunk):
            fn = lambda b, p: orig(b, p, *rest)  # noqa: E731
            a = (block_states[:, t0 : t0 + chunk], partial[:, t0 : t0 + chunk])
            outs.append(checkpoint(fn, *a, use_reentrant=False, preserve_rng_state=False)
                        if torch.is_grad_enabled() else fn(*a))
        if isinstance(outs[0], tuple):
            return tuple(torch.cat([o[i] for o in outs], dim=1) for i in range(len(outs[0])))
        return torch.cat(outs, dim=1)

    return f


def _ckpt_long(orig, min_len, dim):
    """Checkpoint a submodule when the length (dimension `dim` of the first tensor) is >= min_len and the gradient is
    needed: in the visit's backward only its inputs are kept (peak = the largest submodule, not the sum). Same math."""

    def f(*args, **kwargs):
        t = next((a for a in list(args) + list(kwargs.values()) if torch.is_tensor(a)), None)
        if t is None or not torch.is_grad_enabled() or t.shape[dim] < min_len:
            return orig(*args, **kwargs)
        return checkpoint(orig, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)

    return f


class _Head(torch.autograd.Function):
    """logit = fp32(h) @ fp32(W)^T in vocabulary chunks (lm_head in FP32 like the reference), without an FP32 copy of W."""

    @staticmethod
    def forward(ctx, h, W, chunk):
        ctx.W, ctx.chunk, ctx.hdtype = W, chunk, h.dtype
        h32 = h.float()
        out = torch.empty(h.shape[0], W.shape[0], dtype=torch.float32, device=h.device)
        for c in range(0, W.shape[0], chunk):
            out[:, c : c + chunk] = h32 @ W[c : c + chunk].float().T
        return out

    @staticmethod
    def backward(ctx, g):
        W, chunk = ctx.W, ctx.chunk
        gh = torch.zeros(g.shape[0], W.shape[1], dtype=torch.float32, device=g.device)
        for c in range(0, W.shape[0], chunk):
            gh += g[:, c : c + chunk] @ W[c : c + chunk].float()
        return gh.to(ctx.hdtype), None, None


@dataclass
class Out:
    logits: torch.Tensor | None = None  # [P, V] fp32 after the softcap, rows = rows
    rows: torch.Tensor | None = None
    reads: dict = field(default_factory=dict)  # visit -> [P, H] fp32 (CPU); "embed", "final"


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class PrefixState:
    """State of a prefix computed by InkGPU.prefill_state: P, metadata, prefix ids and, per visit, the tensors
    k, v [8, n, 128] and the tails k_conv, v_conv [3, 1024], attn_conv, mlp_conv [3, 2304]."""

    def __init__(self, P, visits, meta, ids):
        self.P, self.visits, self.meta, self.ids = P, visits, meta, ids

    def nbytes(self):
        return sum(t.numel() * t.element_size() for d in self.visits.values() for t in d.values())

    def save(self, path):
        from safetensors.torch import save_file

        t0 = time.time()
        tensors = {f"V{v:02d}.{k}": t.detach().cpu().contiguous() for v, d in self.visits.items() for k, t in d.items()}
        tensors["ids"] = torch.tensor(self.ids, dtype=torch.int64)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={"inkgpu_prefix": json.dumps(self.meta)})
        self.meta["save_s"] = round(time.time() - t0, 2)

    @classmethod
    def load(cls, path, model):
        from safetensors import safe_open

        table = {v: L for v, L, _ in visit_table(model.cfg)}
        visits = {}
        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["inkgpu_prefix"])
            ids = f.get_tensor("ids").tolist()
            for key in f.keys():
                if key == "ids":
                    continue
                vname, name = key.split(".", 1)
                v = int(vname[1:])
                visits.setdefault(v, {})[name] = f.get_tensor(key).to(model.dev_of[table[v]])
        return cls(meta["P"], visits, meta, ids)


class InkGPU:
    """See README.md. Build with InkGPU.load(...)."""

    @classmethod
    def load(cls, ckpt, devices=("cuda:0", "cuda:1"), split=14, dtype=torch.bfloat16, amax="hf", embed_device="cpu",
             attn_budget=2**20, res_chunk=1024, ckpt_min_len=4096, ckpt_layers="none", head_device=None, verbose=True):
        """ckpt: directory with the original safetensors (model-*.safetensors, names model.llm.*).
        split: layers 0..split-1 on devices[0], the rest (+ final_input, norm, lm_head) on devices[1].
        amax: "hf" = per-expert factor as in the library loader (amax rounded to `dtype` before the division:
        this is the reference of the bf16 dumps); "exact" = fp32 amax.
        ckpt_min_len / ckpt_layers: checkpoint the visit's submodules (attention, conv, norms, router) when
        T >= ckpt_min_len, on the layers of ckpt_layers ("none" default, "dev1" = those on devices[1], "all"): at 16k it lowers
        the peak of devices[1] (G5); on devices[0] it raised it.
        head_device: where lm_head lives (default devices[1]); "cpu" frees 0.86 GiB on devices[1] (logits on CPU, FP32)."""
        from safetensors import safe_open
        from transformers import Inkling2LoopConfig, Inkling2LoopForCausalLM
        from transformers import initialization as hfinit

        t0 = time.time()
        self = cls()
        cfg = Inkling2LoopConfig.from_pretrained(str(ckpt)).text_config
        cfg._attn_implementation = "eager"
        self.cfg, self.dtype, self.state = cfg, dtype, _State()
        self.devices = [torch.device(d) for d in devices]
        self.dev_of = [self.devices[0] if L < split else self.devices[1] for L in range(cfg.num_hidden_layers)]
        self.head_dev = self.devices[1]
        self.embed_dev = torch.device(embed_device)
        self.split, self.amax_mode = split, amax
        self.H = cfg.hidden_size
        self.ckpt = True
        self.offload = False
        self.attn_budget = attn_budget
        with hfinit.no_init_weights(), torch.device("meta"):
            model = Inkling2LoopForCausalLM(cfg)
        lm = model.model
        keep32 = tuple(type(model)._keep_in_fp32_modules_strict)
        for L, layer in enumerate(lm.layers):
            layer.mlp = QuantMoE(layer.mlp.gate, L, cfg, self.state)
            layer.mlp.dtype = dtype
        expected = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}

        def dev_for(name):
            if name.startswith("model.layers."):
                return self.dev_of[int(name.split(".")[2])]
            if name == "model.embed_tokens.weight":
                return self.embed_dev
            if name == "lm_head.weight" and head_device is not None:
                return torch.device(head_device)
            if name.startswith(("model.embed_tokens.", "model.message_input.")):
                return self.devices[0]
            return self.head_dev

        def assign(name, t):
            want = torch.float32 if any(k in name for k in keep32) else dtype
            t = t.to(device=dev_for(name), dtype=want)
            mod_name, _, leaf = name.rpartition(".")
            mod = model.get_submodule(mod_name)
            if leaf in mod._parameters:
                mod._parameters[leaf] = nn.Parameter(t, requires_grad=False)
            elif leaf in mod._buffers:
                mod._buffers[leaf] = t
            else:
                raise KeyError(name)
            expected.discard(name)

        experts = {}
        mq = {}
        files = sorted(Path(ckpt).glob("model-*.safetensors"))
        nbytes = 0
        for path in files:
            with safe_open(str(path), framework="pt") as f:
                for key in f.keys():
                    if not key.startswith("model.llm."):
                        continue
                    k = key[len("model.llm."):]
                    t = f.get_tensor(key)
                    nbytes += t.numel() * t.element_size()
                    if k.startswith("layers."):
                        L, rest = k.split(".", 2)[1:]
                        L = int(L)
                        if rest.startswith("mlp.experts."):
                            name = rest[len("mlp.experts."):]
                            if not name.endswith(("_amax", "_denominator")):
                                t = t.view(torch.uint8).to(self.dev_of[L])
                            experts.setdefault(L, {})[name] = t
                            continue
                        if rest not in LAYER_MAP:
                            raise KeyError(key)
                        assign(f"model.layers.{L}.{LAYER_MAP[rest]}", t)
                    elif k.startswith("attn_res.attn_queries."):
                        assign(f"model.layers.{int(k.rsplit('.', 1)[1]) + 1}.attention_input.query", t)
                    elif k.startswith("attn_res.mlp_queries."):
                        assign(f"model.layers.{int(k.rsplit('.', 1)[1])}.mlp_input.query", t)
                    elif k == "attn_res.output_query":
                        assign("model.final_input.query", t)
                    elif k in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
                        mq[k] = t
                    elif k == "embed.weight":
                        assign("model.embed_tokens.weight", t)
                    elif k == "embed_norm.weight":
                        assign("model.embed_tokens.embed_norm.weight", t)
                    elif k == "norm.weight":
                        assign("model.norm.weight", t)
                    elif k == "unembed.weight":
                        assign("lm_head.weight", t[: cfg.unpadded_vocab_size])
                    else:
                        raise KeyError(key)
        # Qm[i] = [q[2i], kv[2i], kv[2i+1], q[2i+1]]  (Q, K, V, MLP of layer 6+i)
        q = mq["middle_cycle_cross_pass_queries"].unflatten(0, (-1, 2))
        kv = mq["middle_cycle_cross_pass_kv_queries"].unflatten(0, (-1, 2))
        assign("model.message_input.query", torch.cat([q[:, :1], kv, q[:, 1:]], dim=1))
        for L, layer in enumerate(lm.layers):
            ex, d, moe = experts[L], self.dev_of[L], layer.mlp
            for m in ("w13", "w2"):
                qq, ss = ex[f"{m}_weight"], ex[f"{m}_weight_scale"]
                if f"{m}_weight_amax" in ex:
                    am, den = ex[f"{m}_weight_amax"], ex[f"{m}_weight_outer_scale_denominator"]
                    if amax == "hf":
                        fac = am.to(dtype).float() / den.to(dtype).float()
                    else:
                        fac = am.float() / den.float()
                    setattr(moe, m, QMat("nvfp4", qq, ss, fac.to(d)))
                else:
                    setattr(moe, m, QMat("mxfp8", qq, ss))
        left = {n for n in expected if not n.startswith(tuple(f"model.layers.{L}.mlp.experts" for L in range(99)))}
        if left:
            raise RuntimeError(f"parameters not loaded: {sorted(left)[:10]} ({len(left)})")
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            if p.is_meta:
                raise RuntimeError(f"{n} is still on meta")
        # substitutions that leave the math unchanged
        for layer in lm.layers:
            layer.self_attn.forward = types.MethodType(_attn_forward, layer.self_attn)
            layer.self_attn._ink_budget = attn_budget
            layer.self_attn._ink_state = self.state
            sa = layer.self_attn
            for proj, name in ((sa.k_proj, "k_conv"), (sa.v_proj, "v_conv")):
                dyn = not hasattr(proj.sconv, "conv1d")
                proj.sconv.forward = _conv_tail(proj.sconv.forward, self.state, name, dyn)
            layer.post_attention_convolution.forward = _conv_tail(layer.post_attention_convolution.forward, self.state,
                                                                  "attn_conv", False)
            layer.post_mlp_convolution.forward = _conv_tail(layer.post_mlp_convolution.forward, self.state,
                                                            "mlp_conv", False)
            for name in ("attention_input", "mlp_input", "q_message_gate", "k_message_gate", "v_message_gate",
                         "mlp_message_gate"):
                mod = getattr(layer, name, None)
                if mod is not None:
                    mod.forward = _chunked_res(mod.forward, res_chunk)
            Li = layer.mlp.layer_idx
            if ckpt_layers == "none" or (ckpt_layers == "dev1" and self.dev_of[Li] != self.devices[1]):
                continue
            for name in ("self_attn", "post_attention_convolution", "post_mlp_convolution", "input_layernorm",
                         "q_input_layernorm", "k_input_layernorm", "v_input_layernorm", "post_attention_layernorm"):
                mod = getattr(layer, name, None)
                if mod is not None:
                    mod.forward = _ckpt_long(mod.forward, ckpt_min_len, 1)
            layer.mlp.gate.forward = _ckpt_long(layer.mlp.gate.forward, ckpt_min_len, 0)
        lm.final_input.forward = _chunked_res(lm.final_input.forward, res_chunk)
        self.msg_in = {}
        for d in dict.fromkeys(self.devices):
            mi = copy.deepcopy(lm.message_input).to(d)
            mi.forward = _chunked_res(mi.forward, res_chunk)
            self.msg_in[d] = mi
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model, self.lm, self.layers = model, lm, lm.layers
        self.vchunk = 32768
        self.load_s = time.time() - t0
        self.read_gb = nbytes / 1e9
        if verbose:
            print(f"[inkgpu] loaded in {self.load_s:.1f} s ({self.read_gb:.2f} GB read); split {split}; "
                  f"dtype {dtype}; amax {amax}; " + self.static_bytes_str(), flush=True)
        return self

    # ------------------------------------------------------------------------------------------------- memory
    def static_bytes(self):
        out = {}
        seen = set()
        for t in list(self.model.parameters()) + list(self.model.buffers()):
            if id(t) in seen:
                continue
            seen.add(id(t))
            out[str(t.device)] = out.get(str(t.device), 0) + t.numel() * t.element_size()
        for layer in self.layers:
            for m in (layer.mlp.w13, layer.mlp.w2):
                out[str(m.q.device)] = out.get(str(m.q.device), 0) + m.nbytes()
        for d, mi in self.msg_in.items():
            if d != self.devices[0]:
                out[str(d)] += sum(p.numel() * p.element_size() for p in mi.parameters())
        return out

    def static_bytes_str(self):
        return "weights per device: " + ", ".join(f"{d} {b / 2**30:.2f} GiB" for d, b in self.static_bytes().items())

    # ------------------------------------------------------------------------------------------------- neurons
    def set_neuron_scale(self, spec, where="h"):
        """spec {(layer, expert, j): alpha}: h_j (where="h") or the up pre-activation a_j (where="up") of the expert
        multiplied by alpha at all visits of the layer. Replaces the previous scales."""
        self.clear_neuron_scale()
        for (L, e, j), a in spec.items():
            moe = self.layers[L].mlp
            d = moe.scale_h if where == "h" else moe.scale_up
            if e not in d:
                d[e] = torch.ones(moe.F, dtype=torch.float32, device=self.dev_of[L])
            d[e][j] = float(a)

    def clear_neuron_scale(self):
        for layer in self.layers:
            layer.mlp.scale_h.clear()
            layer.mlp.scale_up.clear()

    # ------------------------------------------------------------------------------------------------- weight edits
    def up_row(self, L, e, j):
        """Row of w13 in the file for the up of neuron j (the gate is 2j)."""
        return 2 * j + 1

    def edit_zero_up(self, L, e, j):
        """Zero the up row of neuron (L, e, j) in the quantized tensors: NVFP4 = the row's E4M3 block scales set to 0
        (144 bytes, private to the neuron); MXFP8 = the row's FP8 values set to 0 (the UE8M0 scale cannot represent zero).
        Returns a function that restores the original bytes."""
        m = self.layers[L].mlp.w13
        r = self.up_row(L, e, j)
        t = m.s if m.fmt == "nvfp4" else m.q
        old = t[e, r].clone()
        t[e, r] = 0

        def undo():
            t[e, r] = old

        return undo

    # ------------------------------------------------------------------------------------------------- forward
    def _embed(self, ids):
        e = F.embedding(ids.to(self.embed_dev), self.lm.embed_tokens.weight)
        return self.lm.embed_tokens.embed_norm(e.to(self.devices[0]))

    def _run(self, ids, grad, logit_pos=None, read=None, hooks=None, pos0=0):
        cfg, lm = self.cfg, self.lm
        ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
        T = ids.shape[1]
        devs = list(dict.fromkeys(self.devices + [self.head_dev]))
        pos = {d: (torch.arange(T, device=d) + pos0).view(1, -1) for d in devs}
        self.state.pos0 = pos0
        cache = {}

        def on(t, d):
            if t.device == d:
                return t
            key = (id(t), str(d))
            if key not in cache:
                cache[key] = (t, t.to(d))
            return cache[key][1]

        read = read or {}
        reads = {}

        def rd(name, y):
            if name in read:
                p = None if name == "final" else read[name]
                yy = y[0] if p is None else y[0, torch.as_tensor(p, device=y.device)]
                reads[name] = yy.detach().float().cpu()

        x = self._embed(ids)
        if grad:
            x = x.detach().requires_grad_(True)
            self.leaf = x
        rd("embed", x)
        bs = x.new_zeros(1, T, 0, self.H)
        counter = [0]

        def visit(L, x, bs, prev=None, li=None):
            v = counter[0]
            counter[0] += 1
            layer, d = lm.layers[L], self.dev_of[L]
            bd, P = on(bs, d), pos[d]
            if prev is None:
                def fn(x_, b_):
                    layer.mlp.cur_visit = self.state.cur_visit = v
                    return layer(x_.to(d), b_, position_ids=P)
                args = (x, bd)
            else:
                mi = self.msg_in[d]

                def fn(x_, b_, pb_, ph_):
                    layer.mlp.cur_visit = self.state.cur_visit = v
                    return layer(x_.to(d), b_, mi(pb_, ph_, li), position_ids=P)
                args = (x, bd, on(prev[0], d), on(prev[1], d))
            if grad and self.ckpt:
                y = checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
            else:
                y = fn(*args)
            if layer.is_block_boundary:
                bs = torch.cat([bd, on(x, d).unsqueeze(-2)], dim=-2)
            if hooks and v in hooks:
                y2 = hooks[v](y)
                y = y if y2 is None else y2
            rd(v, y)
            if grad and self.offload:  # visit outputs (saved checkpoint inputs) kept in RAM
                y = y.to("cpu")
            return y, bs

        s, e, n = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.num_hidden_layers
        for L in range(s):
            x, bs = visit(L, x, bs)
        pre_x, pre_bs = x, bs
        npre = pre_bs.shape[-2]
        for L in range(s, e):
            x, bs = visit(L, x, bs)
        for _ in range(1, cfg.loop_num_passes):
            prev = (bs[..., npre + 1 :, :], x)
            x, bs = pre_x, pre_bs
            for i, L in enumerate(range(s, e)):
                x, bs = visit(L, x, bs, prev, i)
        for L in range(e, n):
            x, bs = visit(L, x, bs)
        out = Out(reads=reads)
        if logit_pos is not None:
            rows = torch.as_tensor(logit_pos, dtype=torch.long).view(-1)
            r = rows.to(self.head_dev)
            h = lm.final_input(on(bs, self.head_dev)[:, r], on(x, self.head_dev)[:, r])[0]
            h = lm.norm(h).to(self.dtype)
            rd("final", h)
            c = float(cfg.final_logit_softcapping)
            W = self.model.lm_head.weight
            lg = _Head.apply(h[0].to(W.device), W, self.vchunk)
            out.logits, out.rows = torch.tanh(lg / c) * c, rows
        return out

    @torch.no_grad()
    def forward(self, ids, logit_pos=None, read=None, hooks=None, capture=None, route=None, routes=None,
                prefix_state=None):
        """Forward without gradient. logit_pos: logit positions (None = no logits; "all" = all).
        prefix_state: PrefixState or path (prefill_state): ids is then only the suffix S, at absolute positions
        P..P+len(S)-1; logit_pos, read, capture and the results are indexed inside S (0 = first token of S).
        read: {visit | "embed" | "final": positions or None (all)} -> out.reads[...] [P, H] fp32 CPU (output of the visit).
        hooks: {visit: f(y) -> y'} applied to the visit output (steering). capture: Capture (h per expert).
        route: f(moe, logits, w, idx) called at every router. routes: frozen routing (capture_routing)."""
        if isinstance(logit_pos, str) and logit_pos == "all":
            logit_pos = range(len(ids))
        self.state.capture, self.state.route, self.state.freeze = capture, route, routes
        pre = self._use_prefix(prefix_state)
        try:
            return self._run(ids, False, logit_pos, read, hooks, pos0=0 if pre is None else pre.P)
        finally:
            self.state.capture = self.state.route = self.state.freeze = None
            self.state.prefix, self.state.pos0 = None, 0

    def capture_routing(self, ids, prefix_state=None):
        """Experts chosen at every visit for the sequence ids: {visit: idx [T, k]} (for forward/attribution with routes=)."""
        routes = {}

        def rec(moe, logits, w, idx):
            routes[moe.cur_visit] = idx.clone()

        self.forward(ids, route=rec, prefix_state=prefix_state)
        return routes

    def logprobs(self, ids, positions, targets, prefix_state=None):
        """log p(targets[i] | ids[: positions[i] + 1]) for every i (fp32, CPU)."""
        out = self.forward(ids, logit_pos=positions, prefix_state=prefix_state)
        lp = torch.log_softmax(out.logits, -1)
        return lp[torch.arange(len(positions)), torch.as_tensor(targets, device=lp.device)].cpu()

    def attribution(self, ids, margin_fn, logit_pos, positions=None, raw_pos=None, read=None, hooks=None,
                    grad_read=None, routes=None, prefix_state=None):
        """Forward + backward of a scalar margin m = margin_fn(logits [P, V] fp32 at the positions logit_pos).
        Returns a dict: m, logits (CPU), attr [24, E, F] = sum_visits sum_tokens h*dm/dh (tokens at `positions`, None =
        all), attr_visit {visit: [E, F]}, count_visit {visit: [E] routed tokens counted}, raw (if raw_pos: list of
        (visit, layer, expert, pos, h, dm/dh)), reads, grad_reads {visit: dm/dy [P, H]} for grad_read {visit: pos}.
        prefix_state: as in forward; ids = suffix S; everything (logit_pos, positions, raw_pos) is indexed inside S and
        the attribution counts only the tokens of S (the prefix is a constant: no gradient towards P)."""
        T = len(ids)
        devs = list(dict.fromkeys(self.devices + [self.head_dev]))
        tap = Tap(T, self.cfg.n_routed_experts, self.cfg.moe_intermediate_size, devs, positions, raw_pos)
        grad_reads = {}
        hk = dict(hooks or {})
        for v, p in (grad_read or {}).items():
            def f(y, v=v, p=p, inner=hk.get(v)):
                if inner is not None:
                    y2 = inner(y)
                    y = y if y2 is None else y2
                if y.requires_grad:
                    def gh(g):
                        gg = g[0] if p is None else g[0, torch.as_tensor(p, device=g.device)]
                        grad_reads[v] = gg.detach().float().cpu()
                    y.register_hook(gh)
                return y
            hk[v] = f
        self.state.tap, self.state.freeze = tap, routes
        pre = self._use_prefix(prefix_state)
        try:
            with torch.enable_grad():
                out = self._run(ids, True, logit_pos, read, hk, pos0=0 if pre is None else pre.P)
                m = margin_fn(out.logits)
                self.state.recomputing = True
                m.backward()
        finally:
            self.state.recomputing = False
            self.state.tap = self.state.freeze = None
            self.state.prefix, self.state.pos0 = None, 0
            self.leaf = None
        L = self.cfg.num_hidden_layers
        attr = torch.zeros(L, tap.E, tap.F, dtype=torch.float32)
        table = {v: Lv for v, Lv, _ in visit_table(self.cfg)}
        av = {v: a.cpu() for v, a in tap.acc.items()}
        for v, a in av.items():
            attr[table[v]] += a
        return {
            "m": float(m.detach()),
            "logits": out.logits.detach().cpu(),
            "attr": attr,
            "attr_visit": av,
            "count_visit": {v: c.cpu() for v, c in tap.cnt.items()},
            "raw": tap.raw,
            "reads": out.reads,
            "grad_reads": grad_reads,
        }

    # ------------------------------------------------------------------------------------------------- prefix
    def scale_spec(self):
        """Active runtime scales: [layer, expert, j, alpha, where] entries, recorded in the prefix metadata."""
        out = []
        for L, layer in enumerate(self.layers):
            for where, d in (("h", layer.mlp.scale_h), ("up", layer.mlp.scale_up)):
                for e, vec in d.items():
                    for j in torch.nonzero(vec != 1).flatten().tolist():
                        out.append([L, e, j, float(vec[j]), where])
        return out

    @torch.no_grad()
    def prefill_state(self, ids, path=None, chunk=8192):
        """Compute without gradient the state of the prefix ids (P tokens) in chunks of `chunk` and save it to `path`
        (safetensors) if given. State per visit: attention keys and values (global: all P; local: the last
        W-1 = 1023) and the last 3 inputs of the 4 short convs (K, V, post-attention, post-MoE). Nothing else passes between
        tokens (attention residuals, messages and routing are per token). Uses the runtime scales ACTIVE at that moment
        (recorded in the metadata): to measure an edit on the prefix it must be recomputed with the scales active.
        Returns a PrefixState already on the devices."""
        ids = [int(i) for i in ids]
        P = len(ids)
        prev = None
        t0 = time.time()
        try:
            for c0 in range(0, P, chunk):
                self.state.prefix, self.state.record = prev, {}
                self._run(ids[c0 : c0 + chunk], False, None, None, None, pos0=c0)
                prev = self.state.record
        finally:
            self.state.prefix = self.state.record = None
            self.state.pos0 = 0
        meta = {"P": P, "chunk": chunk, "dtype": str(self.dtype), "amax": self.amax_mode, "scales": self.scale_spec(),
                "prefill_s": round(time.time() - t0, 2)}
        st = PrefixState(P, prev, meta, ids)
        if path is not None:
            st.save(path)
        return st

    def load_state(self, path):
        """PrefixState from a prefill_state file, with the tensors on the visits' devices."""
        return PrefixState.load(path, self)

    def _use_prefix(self, prefix_state):
        if prefix_state is None:
            self.state.prefix = None
            return None
        pre = prefix_state if isinstance(prefix_state, PrefixState) else self.load_state(prefix_state)
        if pre.meta.get("scales", []) != self.scale_spec():
            print(f"[inkgpu] warning: prefix computed with scales {pre.meta.get('scales')} but now active "
                  f"{self.scale_spec()}: the scales act only on the suffix", flush=True)
        self.state.prefix = pre.visits
        return pre

    # ------------------------------------------------------------------------------------------------- tests
    def set_plain(self, on=True):
        """Reference path for tests: no checkpointing, non-blocked attention, MoE with ordinary autograd."""
        self.ckpt = not on
        for layer in self.layers:
            layer.mlp.plain = on
            layer.self_attn._ink_budget = 2**62 if on else self.attn_budget

    def dequant_layer(self, L):
        """(gate_up [E, 2F, H], down [E, H, F]) in the library layout (gate first, then up), for tests."""
        moe = self.layers[L].mlp
        ids = torch.arange(moe.E, device=self.dev_of[L])
        w13 = moe.w13.dq(ids, self.dtype)
        gate_up = torch.cat([w13[:, 0::2], w13[:, 1::2]], dim=1)
        return gate_up, moe.w2.dq(ids, self.dtype)


def peak_gib(devices):
    return {str(d): round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in devices if torch.device(d).type == "cuda"}


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def fmt_s(x):
    return f"{x:.2f}" if x < 100 else f"{x:.0f}"


__all__ = ["InkGPU", "Capture", "Tap", "visit_table", "peak_gib", "save_json", "math"]
