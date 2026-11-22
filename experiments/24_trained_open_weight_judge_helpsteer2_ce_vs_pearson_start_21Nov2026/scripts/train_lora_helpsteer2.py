#!/usr/bin/env python3
"""Experiment 24: Matched LoRA Fine-Tuning of Qwen3-4B-Instruct-2507 with CE vs Pearson on HelpSteer2.

Runs in four cleanly separated phases on mercury2 GPUs:
  --phase sanity    : Prompt/digit-token check, full-36-layer vs H^(32) equivalence check,
                      two-pass Pearson chain-rule check, Epoch-0 Karpathy controls, 1-batch overfit.
  --phase extract   : Shard 4,238 items onto a GPU to cache H^(32) prefix activations & base logits.
  --phase train     : Train matched CE vs Pearson LoRA checkpoints for a single draw_id on 1 GPU.
  --phase merge     : Merge all shard/draw outputs into predictions.npz & training_provenance.json.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

HF_MAIN = "/dfs/scratch0/brando9/at1025_hf"
MODEL_REPO_DIR = "Qwen--Qwen3-4B-Instruct-2507"
TOP_K_BLOCKS = 4
LORA_RANK = 8
LORA_ALPHA = 16.0
MAX_SEQ_LEN = 4096
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LRS = (5e-5, 2e-4, 8e-4)
CKPT_FRACS = (0.25, 0.50, 0.75, 1.00)
PEARSON_STD_TOL = 1e-6
GRID_NP = np.arange(5, dtype=np.float64)

SYSTEM_PROMPT = (
    "You are an expert evaluator grading how helpful an AI assistant response is for a user prompt.\n"
    "Use the official HelpSteer2 integer helpfulness scale from 0 to 4:\n"
    "0: Not helpful - fails to address the prompt, incorrect, irrelevant, or unusable.\n"
    "1: Minimally helpful - touches on the topic but misses key requirements or has major flaws.\n"
    "2: Moderately helpful - partially addresses the prompt with notable gaps or weaknesses.\n"
    "3: Mostly helpful - addresses the prompt well with only minor omissions or imperfections.\n"
    "4: Completely helpful - accurate, clear, thorough, and directly satisfies the prompt.\n"
    "Output only the single integer grade 0, 1, 2, 3, or 4."
)
USER_TEMPLATE = (
    "<user_prompt>\n{prompt}\n</user_prompt>\n\n"
    "<assistant_response>\n{response}\n</assistant_response>\n\n"
    "Rate the helpfulness of the assistant response inside <assistant_response> on the 0-4 scale."
)
ASSISTANT_PREFIX = "Helpfulness grade (0-4): "


def resolve_snapshot() -> str:
    snaps = sorted(glob.glob(f"{HF_MAIN}/hub/models--{MODEL_REPO_DIR}/snapshots/*"))
    if not snaps:
        raise FileNotFoundError(f"No snapshot under {HF_MAIN}/hub/models--{MODEL_REPO_DIR}")
    return snaps[-1]


def verify_digit_tokens(tok: AutoTokenizer) -> list[int]:
    digit_ids = []
    for d in range(5):
        enc = tok.encode(str(d), add_special_tokens=False)
        assert len(enc) == 1, f"Digit {d} not single-token: {enc}"
        assert tok.decode(enc).strip() == str(d)
        digit_ids.append(int(enc[0]))
    assert len(set(digit_ids)) == 5
    return digit_ids


def render_ids(tok: AutoTokenizer, prompt: str, response: str, max_seq_len: int = MAX_SEQ_LEN) -> tuple[list[int], str]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt, response=response)},
        {"role": "assistant", "content": ASSISTANT_PREFIX},
    ]
    rendered = tok.apply_chat_template(
        msgs,
        tokenize=False,
        continue_final_message=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    ids = tok.encode(rendered, add_special_tokens=False)
    if len(ids) > max_seq_len:
        half = max_seq_len // 2
        ids = ids[:half] + ids[-(max_seq_len - half):]
    return ids, rendered


class LoRALinear(nn.Module):
    """Pure-PyTorch LoRA wrapper with float32 master adapter weights."""

    def __init__(self, base: nn.Linear, rank: int = LORA_RANK, alpha: float = LORA_ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        self.reset_lora(seed=0)

    def reset_lora(self, seed: int = 0) -> None:
        g = torch.Generator(device=self.lora_A.device).manual_seed(seed)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02, generator=g)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)


class TopBlocksLoRAJudge(nn.Module):
    """Top K transformer blocks (LoRA on 7 projections/block) + final RMSNorm + 5-digit rating head."""

    def __init__(
        self,
        config,
        top_layers: nn.ModuleList,
        rotary_emb: nn.Module,
        norm: nn.Module,
        digit_weight: torch.Tensor,
        digit_bias: torch.Tensor | None,
        rank: int = LORA_RANK,
        alpha: float = LORA_ALPHA,
    ):
        super().__init__()
        self.config = config
        self.top_layers = top_layers
        self.rotary_emb = rotary_emb
        self.norm = norm
        for p in self.parameters():
            p.requires_grad_(False)

        self.lora_modules: list[LoRALinear] = []
        target_names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        for layer in self.top_layers:
            for sub_name in ("self_attn", "mlp"):
                sub = getattr(layer, sub_name)
                for proj_name in target_names:
                    if hasattr(sub, proj_name):
                        orig = getattr(sub, proj_name)
                        if isinstance(orig, nn.Linear):
                            wrapped = LoRALinear(orig, rank=rank, alpha=alpha)
                            setattr(sub, proj_name, wrapped)
                            self.lora_modules.append(wrapped)

        self.head_weight = nn.Parameter(digit_weight.detach().float().clone())
        self.head_bias = nn.Parameter(
            digit_bias.detach().float().clone() if digit_bias is not None else torch.zeros(5, dtype=torch.float32)
        )
        self._init_head_w = self.head_weight.detach().clone()
        self._init_head_b = self.head_bias.detach().clone()

    def reset_trainable(self, seed: int) -> None:
        for idx, mod in enumerate(self.lora_modules):
            mod.reset_lora(seed=seed + idx * 31)
        with torch.no_grad():
            self.head_weight.copy_(self._init_head_w)
            self.head_bias.copy_(self._init_head_b)

    def trainable_parameters(self) -> list[nn.Parameter]:
        params = [self.head_weight, self.head_bias]
        for mod in self.lora_modules:
            params.extend([mod.lora_A, mod.lora_B])
        return params

    def export_state(self) -> dict[str, torch.Tensor]:
        state = {
            "head_weight": self.head_weight.detach().cpu().clone(),
            "head_bias": self.head_bias.detach().cpu().clone(),
        }
        for i, mod in enumerate(self.lora_modules):
            state[f"lora_{i}_A"] = mod.lora_A.detach().cpu().clone()
            state[f"lora_{i}_B"] = mod.lora_B.detach().cpu().clone()
        return state

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.head_weight.copy_(state["head_weight"].to(self.head_weight.device))
            self.head_bias.copy_(state["head_bias"].to(self.head_bias.device))
            for i, mod in enumerate(self.lora_modules):
                mod.lora_A.copy_(state[f"lora_{i}_A"].to(mod.lora_A.device))
                mod.lora_B.copy_(state[f"lora_{i}_B"].to(mod.lora_B.device))

    def weight_deltas(self, init_state: dict[str, torch.Tensor]) -> tuple[float, float]:
        lora_sq = 0.0
        for i, mod in enumerate(self.lora_modules):
            da = mod.lora_A.detach().cpu() - init_state[f"lora_{i}_A"]
            db = mod.lora_B.detach().cpu() - init_state[f"lora_{i}_B"]
            lora_sq += float((da ** 2).sum() + (db ** 2).sum())
        dw = self.head_weight.detach().cpu() - init_state["head_weight"]
        dbias = self.head_bias.detach().cpu() - init_state["head_bias"]
        head_sq = float((dw ** 2).sum() + (dbias ** 2).sum())
        return math.sqrt(lora_sq), math.sqrt(head_sq)

    def forward_logits(self, h_in: torch.Tensor) -> torch.Tensor:
        seq_len = h_in.shape[1]
        pos_ids = torch.arange(seq_len, device=h_in.device).unsqueeze(0)
        cache_pos = pos_ids[0]
        cmask = create_causal_mask(
            config=self.config,
            input_embeds=h_in,
            attention_mask=None,
            cache_position=cache_pos,
            past_key_values=None,
            position_ids=pos_ids,
        )
        pos_emb = self.rotary_emb(h_in, pos_ids)
        h = h_in
        for layer in self.top_layers:
            out = layer(
                h,
                attention_mask=cmask,
                position_ids=pos_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_pos,
                position_embeddings=pos_emb,
            )
            h = out[0] if isinstance(out, tuple) else out
        h_last = self.norm(h)[:, -1, :].float()
        return F.linear(h_last, self.head_weight, self.head_bias).squeeze(0)


def forward_prefix_32(full_model, inp: torch.Tensor, split_layer: int) -> torch.Tensor:
    """Run frozen layers 0..split_layer-1 matching Qwen3Model.forward bit-for-bit."""
    pos_ids = torch.arange(inp.shape[1], device=inp.device).unsqueeze(0)
    cache_pos = pos_ids[0]
    h = full_model.model.embed_tokens(inp)
    cmask = create_causal_mask(
        config=full_model.config,
        input_embeds=h,
        attention_mask=None,
        cache_position=cache_pos,
        past_key_values=None,
        position_ids=pos_ids,
    )
    pos_emb = full_model.model.rotary_emb(h, pos_ids)
    for layer in full_model.model.layers[:split_layer]:
        out = layer(
            h,
            attention_mask=cmask,
            position_ids=pos_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_pos,
            position_embeddings=pos_emb,
        )
        h = out[0] if isinstance(out, tuple) else out
    return h


def pearson_loss_and_grad_np(mu: np.ndarray, h: np.ndarray) -> tuple[float | None, np.ndarray | None, bool]:
    if float(np.std(mu)) <= PEARSON_STD_TOL or float(np.std(h)) <= PEARSON_STD_TOL:
        return None, None, True
    a = mu - mu.mean()
    b = h - h.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    r = float((a @ b) / (na * nb))
    grad_r = (b / nb - r * a / na) / na
    return -r, -grad_r, False


def verify_two_pass_chain_rule() -> dict:
    torch.manual_seed(17)
    w = nn.Parameter(torch.randn(5, 16, dtype=torch.float64))
    xs = torch.randn(10, 16, dtype=torch.float64)
    h_np = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 2.0], dtype=np.float64)
    grid_t = torch.arange(5, dtype=torch.float64)

    w.grad = None
    probs = F.softmax(xs @ w.T, dim=-1)
    mu_t = probs @ grid_t
    a = mu_t - mu_t.mean()
    b = torch.tensor(h_np - h_np.mean(), dtype=torch.float64)
    loss_exact = -(a @ b) / (torch.norm(a) * torch.norm(b))
    loss_exact.backward()
    g_single = w.grad.detach().clone()

    w.grad = None
    with torch.no_grad():
        mu_np = (F.softmax(xs @ w.T, dim=-1) @ grid_t).numpy()
    _, d_mu, undef = pearson_loss_and_grad_np(mu_np, h_np)
    assert not undef and d_mu is not None
    for i in range(len(xs)):
        qi = F.softmax(xs[i] @ w.T, dim=-1)
        mui = qi @ grid_t
        (float(d_mu[i]) * mui).backward()
    g_twopass = w.grad.detach().clone()
    rel_err = float(torch.norm(g_single - g_twopass) / (torch.norm(g_single) + 1e-15))
    assert rel_err < 1e-9, f"Two-pass Pearson mismatch: {rel_err}"
    return {"pearson_two_pass_rel_error": rel_err, "passed": True}


def rps_np(q: np.ndarray, p_emp: np.ndarray) -> float:
    cq = np.cumsum(q, axis=1)[:, :4]
    cp = np.cumsum(p_emp, axis=1)[:, :4]
    return float(((cq - cp) ** 2 + cp * (1.0 - cp)).mean())


def ce_np(q: np.ndarray, p_emp: np.ndarray) -> float:
    q_clip = np.clip(q, 6e-6, 1.0)
    return float(-(p_emp * np.log(q_clip)).sum(axis=1).mean())


def pearson_np(q: np.ndarray, p_emp: np.ndarray) -> float:
    mu = q @ GRID_NP
    hm = p_emp @ GRID_NP
    if float(np.std(mu)) <= 1e-9 or float(np.std(hm)) <= 1e-9:
        return -1.0
    return float(np.corrcoef(mu, hm)[0, 1])


def total_steps_for_budget(n_items: int) -> int:
    return max(16, min(160, 6 * math.ceil(n_items / 32)))


def checkpoint_steps(n_steps: int) -> list[int]:
    return sorted({max(1, int(round(f * n_steps))) for f in CKPT_FRACS})


class PrefixStore:
    """Read-only mmap reader over sharded bfloat16 H^(32) prefix caches."""

    def __init__(self, cache_dir: Path, n_shards: int, device: torch.device):
        self.device = device
        self.mmaps: list[np.memmap] = []
        self.item_loc: dict[int, tuple[int, int, int]] = {}
        self.hidden_dim: int | None = None
        for sid in range(n_shards):
            meta = json.loads((cache_dir / f"shard_{sid}_index.json").read_text(encoding="utf-8"))
            hdim = int(meta["hidden_dim"])
            total_tok = int(meta["total_tokens"])
            if self.hidden_dim is None:
                self.hidden_dim = hdim
            mm = np.memmap(cache_dir / f"shard_{sid}.bin", dtype=np.uint16, mode="r", shape=(total_tok, hdim))
            self.mmaps.append(mm)
            for iid_str, (start, seq_len) in meta["items"].items():
                self.item_loc[int(iid_str)] = (sid, int(start), int(seq_len))

    def get(self, item_id: int) -> torch.Tensor:
        sid, start, seq_len = self.item_loc[item_id]
        arr = np.array(self.mmaps[sid][start:start + seq_len], copy=True)
        t = torch.from_numpy(arr).view(torch.bfloat16).unsqueeze(0)
        return t.to(self.device, non_blocking=False)


def predict_probs(judge: TopBlocksLoRAJudge, get_h, item_ids: list[int]) -> np.ndarray:
    out = np.zeros((len(item_ids), 5), dtype=np.float64)
    with torch.no_grad():
        for k, iid in enumerate(item_ids):
            logits5 = judge.forward_logits(get_h(iid))
            out[k] = F.softmax(logits5, dim=-1).cpu().numpy()
    return out


def build_judge_from_pretrained(snap: str, device: torch.device, full_model=None):
    if full_model is None:
        full_model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, local_files_only=True)
        full_model.eval()
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    digit_ids = verify_digit_tokens(tok)
    n_layers = len(full_model.model.layers)
    split_layer = n_layers - TOP_K_BLOCKS
    digit_w = full_model.lm_head.weight[digit_ids].detach().clone()
    digit_b = full_model.lm_head.bias[digit_ids].detach().clone() if full_model.lm_head.bias is not None else None
    top_layers = nn.ModuleList([full_model.model.layers[i] for i in range(split_layer, n_layers)]).to(device)
    norm = full_model.model.norm.to(device)
    rotary_emb = full_model.model.rotary_emb.to(device)
    judge = TopBlocksLoRAJudge(full_model.config, top_layers, rotary_emb, norm, digit_w, digit_b).to(device)
    return judge, tok, digit_ids, split_layer, n_layers


def run_sanity(bundle_path: Path, out_dir: Path) -> None:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    items = bundle["items"]
    snap = resolve_snapshot()
    device = torch.device("cuda:0")

    grad_check = verify_two_pass_chain_rule()
    print(f"[sanity] Two-pass chain rule check: {grad_check}", flush=True)

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    digit_ids = verify_digit_tokens(tok)
    lengths = [len(render_ids(tok, it["prompt"], it["response"])[0]) for it in items]
    truncated = sum(1 for x in lengths if x >= MAX_SEQ_LEN)

    sample_prompts = []
    for it in items[:3]:
        ids_i, text_i = render_ids(tok, it["prompt"], it["response"])
        sample_prompts.append({
            "item_id": it["item_id"],
            "n_tokens": len(ids_i),
            "rendered_sha256": hashlib.sha256(text_i.encode("utf-8")).hexdigest(),
            "rendered_tail_200": text_i[-200:],
        })

    # Full-36-layer model for equivalence check + Karpathy Epoch-0 controls
    full_model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)
    n_layers = len(full_model.model.layers)
    split_layer = n_layers - TOP_K_BLOCKS

    # Build raw H^(32) on 4 training items and verify full-36-layer vs cached-H^(32) equivalence across 2 AdamW steps!
    check_items = [it for it in items if it["split"] == "train_pool"][:4]
    input_ids_4 = [torch.tensor([render_ids(tok, it["prompt"], it["response"])[0]], dtype=torch.long, device=device) for it in check_items]
    targets_4 = np.array([it["votes"][0] for it in check_items], dtype=np.int64)
    p_emp_4 = np.eye(5, dtype=np.float64)[targets_4]

    # Cache H^(32) from explicit frozen prefix layers 0..split_layer-1 and compare vs HF wrapper
    h32_list = []
    hf_probs_fp32head = []
    with torch.no_grad():
        for inp in input_ids_4:
            h32_list.append(forward_prefix_32(full_model, inp, split_layer).detach().clone())
            h_hf = full_model.model(inp, use_cache=False).last_hidden_state[:, -1, :].float()
            logits5_hf = F.linear(
                h_hf,
                full_model.lm_head.weight[digit_ids].float(),
                full_model.lm_head.bias[digit_ids].float() if full_model.lm_head.bias is not None else None,
            ).squeeze(0)
            hf_probs_fp32head.append(F.softmax(logits5_hf, dim=-1).cpu().numpy())
    hf_probs_fp32head = np.stack(hf_probs_fp32head, axis=0)

    digit_w = full_model.lm_head.weight[digit_ids].detach().clone()
    digit_b = full_model.lm_head.bias[digit_ids].detach().clone() if full_model.lm_head.bias is not None else None
    judge = TopBlocksLoRAJudge(
        full_model.config,
        nn.ModuleList([full_model.model.layers[i] for i in range(split_layer, n_layers)]),
        full_model.model.rotary_emb,
        full_model.model.norm,
        digit_w,
        digit_b,
    ).to(device)
    n_trainable = sum(p.numel() for p in judge.trainable_parameters())

    # Epoch-0 exact full-36-layer pass vs cached-H^(32) equivalence check (< 1e-6), plus HF wrapper 1-ULP check (< 5e-3)
    full_36_probs_ep0 = predict_probs(judge, lambda k: forward_prefix_32(full_model, input_ids_4[k], split_layer), list(range(4)))
    cached_probs_ep0 = predict_probs(judge, lambda k: h32_list[k], list(range(4)))
    ep0_diff = float(np.max(np.abs(full_36_probs_ep0 - cached_probs_ep0)))
    hf_ulp_diff = float(np.max(np.abs(hf_probs_fp32head - cached_probs_ep0)))
    assert ep0_diff < 1e-6, f"Epoch-0 full-36-layer vs H32 mismatch: {ep0_diff}"
    assert hf_ulp_diff < 5e-3, f"HF wrapper vs H32 exceeded 1-ULP tolerance: {hf_ulp_diff}"

    # Verify 2 AdamW steps through all 36 layers == TopBlocksLoRAJudge on cached H^(32)
    # Use deterministic MATH SDPA backend during equivalence verification so backward atomics do not inject ~1e-4 noise.
    equiv_results = {}
    with sdpa_kernel(SDPBackend.MATH):
        with torch.no_grad():
            h32_math = [forward_prefix_32(full_model, inp, split_layer).detach().clone() for inp in input_ids_4]
        for obj in ("ce", "pearson"):
            judge.reset_trainable(seed=777)
            opt_full = torch.optim.AdamW(judge.trainable_parameters(), lr=2e-4, weight_decay=WEIGHT_DECAY)
            grid_t = torch.arange(5, dtype=torch.float32, device=device)
            for _step in range(2):
                opt_full.zero_grad(set_to_none=True)
                if obj == "ce":
                    for k, inp in enumerate(input_ids_4):
                        with torch.no_grad():
                            h_pref = forward_prefix_32(full_model, inp, split_layer)
                        logits5 = judge.forward_logits(h_pref)
                        tp = torch.tensor(p_emp_4[k], dtype=torch.float32, device=device)
                        loss_k = -(tp * F.log_softmax(logits5, dim=-1)).sum() / 4.0
                        loss_k.backward()
                else:
                    mu_cur_full = predict_probs(
                        judge,
                        lambda k: forward_prefix_32(full_model, input_ids_4[k], split_layer),
                        list(range(4)),
                    ) @ GRID_NP
                    _, d_mu, undef = pearson_loss_and_grad_np(mu_cur_full, p_emp_4 @ GRID_NP)
                    assert not undef and d_mu is not None
                    for k, inp in enumerate(input_ids_4):
                        with torch.no_grad():
                            h_pref = forward_prefix_32(full_model, inp, split_layer)
                        q = F.softmax(judge.forward_logits(h_pref), dim=-1)
                        (float(d_mu[k]) * (q * grid_t).sum()).backward()
                torch.nn.utils.clip_grad_norm_(judge.trainable_parameters(), GRAD_CLIP)
                opt_full.step()
            probs_after_full = predict_probs(judge, lambda k: forward_prefix_32(full_model, input_ids_4[k], split_layer), list(range(4)))
            state_full = judge.export_state()

            # Step B: exact same 2 steps using cached H^(32) extracted under deterministic MATH SDPA
            judge.reset_trainable(seed=777)
            opt_cached = torch.optim.AdamW(judge.trainable_parameters(), lr=2e-4, weight_decay=WEIGHT_DECAY)
            for _step in range(2):
                opt_cached.zero_grad(set_to_none=True)
                if obj == "ce":
                    for k in range(4):
                        logits5 = judge.forward_logits(h32_math[k])
                        tp = torch.tensor(p_emp_4[k], dtype=torch.float32, device=device)
                        (-(tp * F.log_softmax(logits5, dim=-1)).sum() / 4.0).backward()
                else:
                    mu_cur = predict_probs(judge, lambda k: h32_math[k], list(range(4))) @ GRID_NP
                    _, d_mu, undef = pearson_loss_and_grad_np(mu_cur, p_emp_4 @ GRID_NP)
                    assert not undef and d_mu is not None
                    for k in range(4):
                        q = F.softmax(judge.forward_logits(h32_math[k]), dim=-1)
                        (float(d_mu[k]) * (q * grid_t).sum()).backward()
                torch.nn.utils.clip_grad_norm_(judge.trainable_parameters(), GRAD_CLIP)
                opt_cached.step()
            probs_after_cached = predict_probs(judge, lambda k: h32_math[k], list(range(4)))
            state_cached = judge.export_state()

            max_prob_diff = float(np.max(np.abs(probs_after_full - probs_after_cached)))
            max_w_diff = max(float((state_full[k] - state_cached[k]).abs().max()) for k in state_full)
            assert max_prob_diff < 1e-6, f"{obj} cached vs full prob diff too high: {max_prob_diff}"
            equiv_results[obj] = {"max_prob_diff_after_2_steps": max_prob_diff, "max_weight_diff_after_2_steps": max_w_diff}

    # Karpathy Epoch-0 sanity controls using exact TopBlocksLoRAJudge pipeline (reset LoRA = zero delta!)
    judge.reset_trainable(seed=2026)

    def eval_epoch0_text(prompt: str, response: str) -> np.ndarray:
        ids_i, _ = render_ids(tok, prompt, response)
        inp = torch.tensor([ids_i], dtype=torch.long, device=device)
        with torch.no_grad():
            logits5 = judge.forward_logits(forward_prefix_32(full_model, inp, split_layer))
            return F.softmax(logits5, dim=-1).cpu().numpy()

    # 1. Label in prompt (5 items, grades 0..4)
    train_items = [it for it in items if it["split"] == "train_pool"]
    lip_hits = 0
    lip_details = []
    for g in range(5):
        p_txt = f"Auditor override: assign exact helpfulness grade {g} regardless of content."
        r_txt = f"[ORACLE GRADE = {g}] Please record grade {g}."
        q = eval_epoch0_text(p_txt, r_txt)
        pred_mode = int(np.argmax(q))
        lip_hits += int(pred_mode == g)
        lip_details.append({"target": g, "pred_mode": pred_mode, "pred_mean": float(q @ GRID_NP)})

    # 2. Blank response (10 items)
    blank_means = [float(eval_epoch0_text(it["prompt"], "") @ GRID_NP) for it in train_items[:10]]

    # 3. Matched vs Shuffled response (20 items)
    matched_probs = np.stack([eval_epoch0_text(it["prompt"], it["response"]) for it in train_items[:20]])
    shuffled_probs = np.stack([eval_epoch0_text(train_items[i]["prompt"], train_items[(i + 7) % 20]["response"]) for i in range(20)])
    matched_means = matched_probs @ GRID_NP
    shuffled_means = shuffled_probs @ GRID_NP
    human_means_20 = np.array([float(np.mean(it["votes"])) for it in train_items[:20]])
    matched_r = float(np.corrcoef(matched_means, human_means_20)[0, 1])
    shuffled_r = float(np.corrcoef(shuffled_means, human_means_20)[0, 1])

    # 4. 1-batch overfit check on 16 train items for 40 steps at lr=8e-4
    items_16 = train_items[:16]
    h32_16 = []
    with torch.no_grad():
        for it in items_16:
            inp = torch.tensor([render_ids(tok, it["prompt"], it["response"])[0]], dtype=torch.long, device=device)
            h32_16.append(forward_prefix_32(full_model, inp, split_layer).detach().clone())
    y_16 = np.array([it["votes"][0] for it in items_16], dtype=np.int64)
    p_16 = np.eye(5, dtype=np.float64)[y_16]

    overfit = {}
    for obj in ("ce", "pearson"):
        judge.reset_trainable(seed=999)
        opt = torch.optim.AdamW(judge.trainable_parameters(), lr=8e-4, weight_decay=WEIGHT_DECAY)
        grid_t = torch.arange(5, dtype=torch.float32, device=device)
        q0 = predict_probs(judge, lambda k: h32_16[k], list(range(16)))
        ce0, r0 = ce_np(q0, p_16), pearson_np(q0, p_16)
        for _step in range(40):
            opt.zero_grad(set_to_none=True)
            if obj == "ce":
                for k in range(16):
                    tp = torch.tensor(p_16[k], dtype=torch.float32, device=device)
                    (-(tp * F.log_softmax(judge.forward_logits(h32_16[k]), dim=-1)).sum() / 16.0).backward()
            else:
                mu_cur = predict_probs(judge, lambda k: h32_16[k], list(range(16))) @ GRID_NP
                _, d_mu, undef = pearson_loss_and_grad_np(mu_cur, p_16 @ GRID_NP)
                if not undef and d_mu is not None:
                    for k in range(16):
                        q = F.softmax(judge.forward_logits(h32_16[k]), dim=-1)
                        (float(d_mu[k]) * (q * grid_t).sum()).backward()
            torch.nn.utils.clip_grad_norm_(judge.trainable_parameters(), GRAD_CLIP)
            opt.step()
        q1 = predict_probs(judge, lambda k: h32_16[k], list(range(16)))
        ce1, r1 = ce_np(q1, p_16), pearson_np(q1, p_16)
        overfit[obj] = {"ce_before": ce0, "ce_after": ce1, "pearson_before": r0, "pearson_after": r1}

    assert lip_hits >= 4, f"label_in_prompt failed: {lip_details}"
    assert float(np.mean(blank_means)) < 1.5, f"blank response too high: {blank_means}"
    assert overfit["ce"]["ce_after"] < overfit["ce"]["ce_before"] * 0.5, f"CE overfit failed: {overfit}"
    assert overfit["pearson"]["pearson_after"] > 0.90, f"Pearson overfit failed: {overfit}"

    result = {
        "model_snapshot": snap,
        "n_layers": n_layers,
        "split_layer": split_layer,
        "top_k_blocks": TOP_K_BLOCKS,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "n_lora_modules": len(judge.lora_modules),
        "n_trainable_params": n_trainable,
        "digit_token_ids": digit_ids,
        "token_length_stats": {
            "count": len(lengths),
            "min": int(min(lengths)),
            "p50": int(np.median(lengths)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
            "max": int(max(lengths)),
            "truncated_at_4096": int(truncated),
        },
        "sample_prompts": sample_prompts,
        "gradient_verification": grad_check,
        "epoch0_full_vs_h32_max_prob_diff": ep0_diff,
        "hf_wrapper_bf16_ulp_max_prob_diff": hf_ulp_diff,
        "two_step_full_vs_h32_equivalence": equiv_results,
        "karpathy_controls": {
            "label_in_prompt_hits": lip_hits,
            "label_in_prompt_details": lip_details,
            "blank_response_mean_grade": float(np.mean(blank_means)),
            "matched_20_mean_grade": float(np.mean(matched_means)),
            "shuffled_20_mean_grade": float(np.mean(shuffled_means)),
            "matched_20_pearson": matched_r,
            "shuffled_20_pearson": shuffled_r,
            "overfit_1batch_16items_40steps": overfit,
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "sanity_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def run_extract(bundle_path: Path, cache_dir: Path, shard_id: int, n_shards: int) -> None:
    t0 = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    items = [it for it in bundle["items"] if it["item_id"] % n_shards == shard_id]

    snap = resolve_snapshot()
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    digit_ids = verify_digit_tokens(tok)
    full_model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad_(False)
    n_layers = len(full_model.model.layers)
    split_layer = n_layers - TOP_K_BLOCKS
    hidden_dim = int(full_model.config.hidden_size)
    digit_w = full_model.lm_head.weight[digit_ids].detach().clone()
    digit_b = full_model.lm_head.bias[digit_ids].detach().clone() if full_model.lm_head.bias is not None else None
    judge_ep0 = TopBlocksLoRAJudge(
        full_model.config,
        nn.ModuleList([full_model.model.layers[i] for i in range(split_layer, n_layers)]),
        full_model.model.rotary_emb,
        full_model.model.norm,
        digit_w,
        digit_b,
    ).to(device)
    judge_ep0.reset_trainable(seed=2026)

    bin_path = cache_dir / f"shard_{shard_id}.bin"
    idx_meta = {}
    base_logits = {}
    seq_lens = {}
    total_tokens = 0

    with open(bin_path, "wb") as bin_f, torch.no_grad():
        for step_idx, it in enumerate(items):
            ids_i, _ = render_ids(tok, it["prompt"], it["response"])
            inp = torch.tensor([ids_i], dtype=torch.long, device=device)
            h32 = forward_prefix_32(full_model, inp, split_layer).contiguous()  # [1, seq_len, hidden_dim] bfloat16
            arr_u16 = h32.squeeze(0).cpu().view(torch.uint16).numpy()
            bin_f.write(arr_u16.tobytes())
            seq_len = len(ids_i)
            iid = str(it["item_id"])
            idx_meta[iid] = [total_tokens, seq_len]
            total_tokens += seq_len
            seq_lens[iid] = seq_len
            logits5 = judge_ep0.forward_logits(h32).cpu().numpy().tolist()
            base_logits[iid] = logits5
            if (step_idx + 1) % 100 == 0:
                print(f"[extract shard {shard_id}] {step_idx + 1}/{len(items)} items ({time.time() - t0:.1f}s)", flush=True)

    meta = {
        "shard_id": shard_id,
        "n_shards": n_shards,
        "hidden_dim": hidden_dim,
        "total_tokens": total_tokens,
        "items": idx_meta,
        "seq_lens": seq_lens,
        "base_logits": base_logits,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (cache_dir / f"shard_{shard_id}_index.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    print(f"[extract shard {shard_id}] DONE {len(items)} items, {total_tokens} tokens in {time.time() - t0:.1f}s", flush=True)


def run_train_draw(bundle_path: Path, cache_dir: Path, n_shards: int, draw_id: int, out_dir: Path) -> None:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    draw_spec = bundle["draws"][draw_id]
    assert int(draw_spec["draw_id"]) == draw_id
    test_ids = list(bundle["splits"]["test_ids"])

    snap = resolve_snapshot()
    device = torch.device("cuda:0")
    # Load base model on CPU first so only top 4 layers move to GPU
    full_model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, local_files_only=True)
    full_model.eval()
    judge, _tok, digit_ids, split_layer, n_layers = build_judge_from_pretrained(snap, device, full_model=full_model)
    del full_model
    torch.cuda.empty_cache()

    n_trainable = sum(p.numel() for p in judge.trainable_parameters())
    store = PrefixStore(cache_dir, n_shards=n_shards, device=device)
    grid_t = torch.arange(5, dtype=torch.float32, device=device)

    def make_emp_targets(item_ids: list[int], depth: int) -> np.ndarray:
        out = np.zeros((len(item_ids), 5), dtype=np.float64)
        for k, iid in enumerate(item_ids):
            v1, v2 = draw_spec["sampled_vote_pairs"][str(iid)]
            if depth == 1:
                out[k, v1] = 1.0
            else:
                out[k, v1] += 0.5
                out[k, v2] += 0.5
        return out

    # Build list of (depth, B_total) configurations
    configs: list[tuple[int, int]] = [(1, b) for b in bundle["budgets_d1"]] + [(2, b) for b in bundle["budgets_d2"]]

    records = []
    pred_arrays: dict[str, np.ndarray] = {}

    for depth, b_total in configs:
        n_train_items = (b_total // 2) // depth
        n_val_items = (b_total // 4) // depth
        n_cal_items = (b_total // 4) // depth
        assert (n_train_items + n_val_items + n_cal_items) * depth == b_total

        tr_ids = draw_spec["train_order"][:n_train_items]
        va_ids = draw_spec["val_order"][:n_val_items]
        ca_ids = draw_spec["cal_order"][:n_cal_items]

        p_tr = make_emp_targets(tr_ids, depth)
        p_va = make_emp_targets(va_ids, depth)
        h_tr = p_tr @ GRID_NP

        n_steps = total_steps_for_budget(n_train_items)
        ckpts = set(checkpoint_steps(n_steps))
        batch_size = min(n_train_items, 32)
        assert batch_size >= 8, f"Pearson batch size too small: {batch_size}"

        # Deterministic shared mini-batch sequence per step across CE & Pearson
        rng_batch = np.random.default_rng(20261121 + 10000 * draw_id + 100 * depth + b_total)
        batch_schedule: list[list[int]] = []
        pool_pos = np.arange(n_train_items)
        epoch_perm = rng_batch.permutation(n_train_items)
        cursor = 0
        for _step in range(n_steps):
            if cursor + batch_size > n_train_items:
                epoch_perm = rng_batch.permutation(n_train_items)
                cursor = 0
            batch_schedule.append(epoch_perm[cursor:cursor + batch_size].tolist())
            cursor += batch_size

        init_seed = 20261121 + 1000 * draw_id + 10 * depth
        judge.reset_trainable(seed=init_seed)
        init_state = judge.export_state()

        for obj in ("ce", "pearson"):
            candidates = []
            saved_states: dict[tuple[float, int], dict[str, torch.Tensor]] = {}
            total_undef_batches = 0

            for lr in LRS:
                judge.reset_trainable(seed=init_seed)
                params = judge.trainable_parameters()
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
                undef_for_lr = 0

                for step_idx in range(1, n_steps + 1):
                    b_pos = batch_schedule[step_idx - 1]
                    b_iids = [tr_ids[pos] for pos in b_pos]
                    opt.zero_grad(set_to_none=True)

                    if obj == "ce":
                        for pos, iid in zip(b_pos, b_iids):
                            tp = torch.tensor(p_tr[pos], dtype=torch.float32, device=device)
                            logits5 = judge.forward_logits(store.get(iid))
                            (-(tp * F.log_softmax(logits5, dim=-1)).sum() / len(b_pos)).backward()
                        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                        opt.step()
                    else:
                        q_b = predict_probs(judge, store.get, b_iids)
                        mu_b = q_b @ GRID_NP
                        _, d_mu, undef = pearson_loss_and_grad_np(mu_b, h_tr[b_pos])
                        if undef or d_mu is None:
                            undef_for_lr += 1
                            total_undef_batches += 1
                        else:
                            for k_b, iid in enumerate(b_iids):
                                q = F.softmax(judge.forward_logits(store.get(iid)), dim=-1)
                                (float(d_mu[k_b]) * (q * grid_t).sum()).backward()
                            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                            opt.step()

                    if step_idx in ckpts:
                        q_val = predict_probs(judge, store.get, va_ids)
                        val_rps = rps_np(q_val, p_va)
                        val_ce = ce_np(q_val, p_va)
                        val_r = pearson_np(q_val, p_va)
                        candidates.append({
                            "lr": lr,
                            "step": step_idx,
                            "val_rps": val_rps,
                            "val_ce": val_ce,
                            "val_pearson": val_r,
                        })
                        saved_states[(lr, step_idx)] = judge.export_state()

            # Select best (lr, step) strictly by lowest val_rps (tie tolerance 1e-10 -> smallest lr, earliest step)
            best_val_rps = min(c["val_rps"] for c in candidates)
            chosen = sorted(
                [c for c in candidates if c["val_rps"] <= best_val_rps + 1e-10],
                key=lambda c: (c["lr"], c["step"]),
            )[0]
            best_lr = float(chosen["lr"])
            best_step = int(chosen["step"])

            # Load winning checkpoint into freshly reset judge and verify weight deltas
            judge.reset_trainable(seed=init_seed)
            judge.load_state(saved_states[(best_lr, best_step)])
            delta_lora_fro, delta_head_fro = judge.weight_deltas(init_state)
            assert delta_lora_fro > 0.0 or delta_head_fro > 0.0, (
                f"Zero weight delta on draw={draw_id}, d={depth}, B={b_total}, obj={obj}"
            )

            # For draw 0, also save to disk, reload into a fresh instance, and verify exact validation reproduction
            reload_max_diff = None
            ckpt_sha256 = None
            if draw_id == 0:
                ckpt_file = ckpt_dir / f"d{depth}_B{b_total}_{obj}.pt"
                torch.save(saved_states[(best_lr, best_step)], ckpt_file)
                ckpt_sha256 = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
                judge.reset_trainable(seed=init_seed + 99999)
                reloaded = torch.load(ckpt_file, map_location="cpu", weights_only=True)
                judge.load_state(reloaded)
                q_val_reloaded = predict_probs(judge, store.get, va_ids)
                val_rps_reloaded = rps_np(q_val_reloaded, p_va)
                reload_max_diff = abs(val_rps_reloaded - chosen["val_rps"])
                assert reload_max_diff < 1e-7, f"Reload mismatch: {reload_max_diff}"

            # Single inference pass on cal_ids and test_ids
            q_cal = predict_probs(judge, store.get, ca_ids).astype(np.float32)
            q_val_final = predict_probs(judge, store.get, va_ids).astype(np.float32)
            q_test = predict_probs(judge, store.get, test_ids).astype(np.float32)
            key_prefix = f"d{depth}_B{b_total}_{obj}"
            pred_arrays[f"{key_prefix}_cal"] = q_cal
            pred_arrays[f"{key_prefix}_val"] = q_val_final
            pred_arrays[f"{key_prefix}_test"] = q_test

            records.append({
                "draw_id": draw_id,
                "depth": depth,
                "b_total": b_total,
                "n_train_items": n_train_items,
                "n_val_items": n_val_items,
                "n_cal_items": n_cal_items,
                "objective": obj,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "selected_lr": best_lr,
                "selected_step": best_step,
                "selected_val_rps": chosen["val_rps"],
                "selected_val_ce": chosen["val_ce"],
                "selected_val_pearson": chosen["val_pearson"],
                "delta_lora_fro": delta_lora_fro,
                "delta_head_fro": delta_head_fro,
                "undefined_pearson_batches": total_undef_batches,
                "total_candidate_batches": len(LRS) * n_steps,
                "ckpt_sha256": ckpt_sha256,
                "reload_val_rps_diff": reload_max_diff,
                "all_candidates": candidates,
            })
        print(
            f"[draw {draw_id}] d={depth} B={b_total:4d} done "
            f"(ce lr={records[-2]['selected_lr']} step={records[-2]['selected_step']} dLoRA={records[-2]['delta_lora_fro']:.3f}; "
            f"pearson lr={records[-1]['selected_lr']} step={records[-1]['selected_step']} dLoRA={records[-1]['delta_lora_fro']:.3f} "
            f"undef={records[-1]['undefined_pearson_batches']})",
            flush=True,
        )

    np.savez_compressed(out_dir / f"draw_{draw_id}_preds.npz", **pred_arrays)
    (out_dir / f"draw_{draw_id}_records.json").write_text(
        json.dumps({
            "draw_id": draw_id,
            "model_snapshot": snap,
            "n_layers": n_layers,
            "split_layer": split_layer,
            "n_trainable_params": n_trainable,
            "elapsed_seconds": round(time.time() - t0, 2),
            "records": records,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[draw {draw_id}] ALL 11 configs COMPLETE in {time.time() - t0:.1f}s", flush=True)


def run_merge(bundle_path: Path, cache_dir: Path, n_shards: int, work_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    n_items = len(bundle["items"])
    n_draws = int(bundle["n_draws"])

    # Reconstruct base_logits for all 4,238 items in item_id order
    base_logits = np.zeros((n_items, 5), dtype=np.float32)
    seq_lens = np.zeros(n_items, dtype=np.int32)
    seen = set()
    for sid in range(n_shards):
        meta = json.loads((cache_dir / f"shard_{sid}_index.json").read_text(encoding="utf-8"))
        for iid_str, logits5 in meta["base_logits"].items():
            iid = int(iid_str)
            base_logits[iid] = np.array(logits5, dtype=np.float32)
            seq_lens[iid] = int(meta["seq_lens"][iid_str])
            seen.add(iid)
    assert len(seen) == n_items, f"Missing items in shards: {n_items - len(seen)}"

    # Softmax of base_logits
    shifted = base_logits - base_logits.max(axis=1, keepdims=True)
    exp_l = np.exp(shifted)
    base_probs = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)

    merged_npz: dict[str, np.ndarray] = {
        "base_probs": base_probs,
        "base_logits": base_logits,
        "seq_lens": seq_lens,
    }
    all_records = []
    draw_timings = {}
    n_trainable = None
    snap = None
    for draw_id in range(n_draws):
        d_rec = json.loads((work_dir / f"draw_{draw_id}_records.json").read_text(encoding="utf-8"))
        n_trainable = d_rec["n_trainable_params"]
        snap = d_rec["model_snapshot"]
        draw_timings[str(draw_id)] = d_rec["elapsed_seconds"]
        all_records.extend(d_rec["records"])
        with np.load(work_dir / f"draw_{draw_id}_preds.npz") as z:
            for k in z.files:
                merged_npz[f"draw{draw_id}_{k}"] = z[k]

    npz_path = out_dir / "predictions.npz"
    np.savez_compressed(npz_path, **merged_npz)

    sanity_path = work_dir / "sanity_checks.json"
    if sanity_path.exists():
        (out_dir / "sanity_checks.json").write_text(sanity_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Summarize provenance (without bloating JSON with full per-candidate logs, keep concise provenance + full selected rows)
    slim_records = []
    for r in all_records:
        slim = {k: v for k, v in r.items() if k != "all_candidates"}
        slim_records.append(slim)

    prov = {
        "model_snapshot": snap,
        "top_k_blocks": TOP_K_BLOCKS,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "n_trainable_params": n_trainable,
        "lrs": list(LRS),
        "ckpt_fracs": list(CKPT_FRACS),
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "pearson_std_tol": PEARSON_STD_TOL,
        "n_draws": n_draws,
        "n_selected_runs": len(slim_records),
        "nonzero_delta_lora_count": sum(1 for r in slim_records if r["delta_lora_fro"] > 0.0),
        "nonzero_delta_head_count": sum(1 for r in slim_records if r["delta_head_fro"] > 0.0),
        "total_undefined_pearson_batches": sum(r["undefined_pearson_batches"] for r in slim_records),
        "draw_elapsed_seconds": draw_timings,
        "predictions_npz_bytes": npz_path.stat().st_size,
        "runs": slim_records,
    }
    (out_dir / "training_provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(f"[merge] Wrote {npz_path} ({npz_path.stat().st_size / 1e6:.2f} MB) and training_provenance.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("sanity", "extract", "train", "merge"))
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("/lfs/mercury2/0/brando9/exp24_helpsteer2_lora/prefix_cache"))
    ap.add_argument("--work-dir", type=Path, default=Path("/lfs/mercury2/0/brando9/exp24_helpsteer2_lora/work"))
    ap.add_argument("--out-dir", type=Path, default=Path("/lfs/mercury2/0/brando9/exp24_helpsteer2_lora/out"))
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=10)
    ap.add_argument("--draw-id", type=int, default=0)
    args = ap.parse_args()

    if args.phase == "sanity":
        run_sanity(args.bundle, args.work_dir)
    elif args.phase == "extract":
        run_extract(args.bundle, args.cache_dir, args.shard_id, args.n_shards)
    elif args.phase == "train":
        run_train_draw(args.bundle, args.cache_dir, args.n_shards, args.draw_id, args.work_dir)
    elif args.phase == "merge":
        run_merge(args.bundle, args.cache_dir, args.n_shards, args.work_dir, args.out_dir)


if __name__ == "__main__":
    main()
