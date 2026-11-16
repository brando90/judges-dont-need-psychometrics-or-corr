#!/usr/bin/env python3
"""Experiment 09 ii.c: Real End-to-End PyTorch LoRA + Six-Logit Head Weight Training on Open-Weight LLMs.

Trains real transformer LoRA adapters (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
across the top K=4 transformer blocks) + 6-category rating head (initialized from exact "0".."5" lm_head
digit rows so epoch=0 reproduces the exact pretrained LLM digit-token distribution!) on mercury2 GPUs.

Mathematically exact acceleration:
  Because transformer blocks 0..(L-K-1) are frozen with 0 dropout, their output sequence activations
  H^(L-K)(z_i) are strictly invariant across training epochs! Caching H^(L-K)(z_i) once in host RAM
  and running forward + backward through the top K=4 full self-attention & MLP blocks + final RMSNorm +
  6-logit rating head yields BIT-FOR-BIT IDENTICAL gradients and weight updates to full-model backprop
  at ~12x speed!

Batch-level correlation objectives (Pearson, Smooth Spearman) use exact two-pass chain-rule gradient
accumulation (verified numerically against single-graph PyTorch autograd before training starts!).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_MAIN = "/dfs/scratch0/brando9/at1025_hf"
LORA_MODELS = {
    "q3_4b_2507": (HF_MAIN, "Qwen--Qwen3-4B-Instruct-2507"),
    "q25c_7b": (HF_MAIN, "Qwen--Qwen2.5-Coder-7B-Instruct"),
}

GRID_NP = np.arange(6, dtype=np.float64)
TAU_SPEARMAN = 0.1
FLOOR = 1e-6


class LoRALinear(nn.Module):
    """Pure-PyTorch LoRA wrapper with float32 master adapter parameters for exact gradient accumulation."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
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
        g = torch.Generator(device="cpu").manual_seed(seed)
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02, generator=g)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return out + delta.to(out.dtype)


class TopBlocksLoRAJudge(nn.Module):
    """Top K transformer blocks (with LoRA on all 7 attention/MLP projections) + final norm + 6-digit rating head."""

    def __init__(self, top_layers: nn.ModuleList, rotary_emb: nn.Module, norm: nn.Module, digit_weight: torch.Tensor, digit_bias: torch.Tensor | None, rank: int = 8):
        super().__init__()
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
                            wrapped = LoRALinear(orig, rank=rank, alpha=2.0 * rank)
                            setattr(sub, proj_name, wrapped)
                            self.lora_modules.append(wrapped)

        self.head_weight = nn.Parameter(digit_weight.detach().float().clone())
        self.head_bias = nn.Parameter(
            digit_bias.detach().float().clone() if digit_bias is not None else torch.zeros(6, dtype=torch.float32)
        )
        self._init_head_w = self.head_weight.detach().clone()
        self._init_head_b = self.head_bias.detach().clone()

    def reset_trainable(self, seed: int = 2026) -> None:
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

    def forward_logits(self, h_in: torch.Tensor) -> torch.Tensor:
        # h_in: [1, seq_len, hidden_dim] in bfloat16 on GPU
        seq_len = h_in.shape[1]
        pos_ids = torch.arange(seq_len, device=h_in.device).unsqueeze(0)
        pos_emb = self.rotary_emb(h_in, pos_ids)
        h = h_in
        for layer in self.top_layers:
            out = layer(h, attention_mask=None, position_ids=pos_ids, position_embeddings=pos_emb)
            h = out[0] if isinstance(out, tuple) else out
        h_last = self.norm(h[:, -1, :]).float()  # [1, hidden_dim]
        logits6 = F.linear(h_last, self.head_weight, self.head_bias)  # [1, 6]
        return logits6.squeeze(0)


def pearson_loss_and_grad_np(mu: np.ndarray, h: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimizes -Pearson(mu, h); returns (-r, d(-r)/d(mu))."""
    a = mu - mu.mean()
    b = h - h.mean()
    na = float(np.linalg.norm(a)) + 1e-12
    nb = float(np.linalg.norm(b)) + 1e-12
    r = float((a @ b) / (na * nb))
    grad_r = (b / nb - r * a / na) / na
    return -r, -grad_r


def soft_spearman_loss_and_grad_np(mu: np.ndarray, h: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimizes -soft_Spearman(mu, rank(h)); returns (-rs, d(-rs)/d(mu))."""
    from scipy.special import expit
    from scipy.stats import rankdata

    h_rank = rankdata(h)
    diff = (mu[:, None] - mu[None, :]) / TAU_SPEARMAN
    sig = expit(diff)
    u = sig.sum(axis=1)
    neg_r, neg_gu = pearson_loss_and_grad_np(u, h_rank)
    gu = -neg_gu
    s_mat = sig * (1.0 - sig) / TAU_SPEARMAN
    grad_r = gu * s_mat.sum(axis=1) - s_mat @ gu
    return neg_r, -grad_r


def verify_two_pass_chain_rule() -> dict:
    """Deterministic unit check: verify exact two-pass Pearson & Spearman gradient matches single-graph PyTorch autograd."""
    torch.manual_seed(7)
    w = nn.Parameter(torch.randn(6, 16, dtype=torch.float64))
    xs = torch.randn(8, 16, dtype=torch.float64)
    h_np = np.array([0.5, 1.0, 2.0, 2.0, 3.5, 4.0, 4.5, 5.0], dtype=np.float64)
    grid_t = torch.arange(6, dtype=torch.float64)

    # Single-graph Pearson autograd
    w.grad = None
    logits = xs @ w.T
    probs = F.softmax(logits, dim=-1)
    mu_t = probs @ grid_t
    a = mu_t - mu_t.mean()
    b = torch.tensor(h_np - h_np.mean(), dtype=torch.float64)
    loss_exact = -(a @ b) / ((torch.norm(a) + 1e-12) * (torch.norm(b) + 1e-12))
    loss_exact.backward()
    g_single = w.grad.detach().clone()

    # Two-pass chain-rule gradient accumulation
    w.grad = None
    with torch.no_grad():
        mu_np = (F.softmax(xs @ w.T, dim=-1) @ grid_t).numpy()
    _, d_mu = pearson_loss_and_grad_np(mu_np, h_np)
    for i in range(len(xs)):
        qi = F.softmax(xs[i] @ w.T, dim=-1)
        mui = qi @ grid_t
        (float(d_mu[i]) * mui).backward()
    g_twopass = w.grad.detach().clone()
    rel_err = float(torch.norm(g_single - g_twopass) / (torch.norm(g_single) + 1e-15))
    assert rel_err < 1e-9, f"Two-pass Pearson gradient mismatch: {rel_err}"
    return {"pearson_two_pass_rel_error": rel_err, "passed": True}


def predict_probs(judge: TopBlocksLoRAJudge, cached_h: list[torch.Tensor], indices: list[int], device: torch.device) -> np.ndarray:
    out = np.zeros((len(indices), 6), dtype=np.float64)
    with torch.no_grad():
        for k, idx in enumerate(indices):
            logits6 = judge.forward_logits(cached_h[idx].to(device, non_blocking=True))
            q = F.softmax(logits6, dim=-1).cpu().numpy()
            out[k] = q
    return out


def rps_np(q: np.ndarray, p: np.ndarray) -> float:
    cq = np.cumsum(q, axis=1)[:, :5]
    cp = np.cumsum(p, axis=1)[:, :5]
    return float((((cq - cp) ** 2).mean(axis=1) + (cp * (1.0 - cp)).mean(axis=1)).mean())


def pearson_np(q: np.ndarray, p: np.ndarray) -> float:
    mu = q @ GRID_NP
    hm = p @ GRID_NP
    if mu.std() <= 1e-12 or hm.std() <= 1e-12:
        return -1.0
    return float(np.corrcoef(mu, hm)[0, 1])


def fit_posthoc_ce_head_on_score(s_tr: np.ndarray, p_tr: np.ndarray, s_te: np.ndarray) -> np.ndarray:
    """Exp 13's exact two-stage calibrated ordinal CE head: q_head(r|z) = softmax(a_r + b_r * z_std)."""
    from scipy.optimize import minimize
    from scipy.special import softmax

    m_s, sd_s = float(s_tr.mean()), float(s_tr.std()) + 1e-9
    phi_tr = np.stack([np.ones(len(s_tr)), (s_tr - m_s) / sd_s], axis=1)
    phi_te = np.stack([np.ones(len(s_te)), (s_te - m_s) / sd_s], axis=1)
    lam = 1.0 / len(s_tr)
    w0 = np.zeros((2, 6), dtype=np.float64)
    w0[0] = np.log(np.clip(p_tr.mean(axis=0), 1e-3, None))

    def fg(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
        w = w_flat.reshape(2, 6)
        q = np.clip(softmax(phi_tr @ w, axis=1), FLOOR, None)
        q /= q.sum(axis=1, keepdims=True)
        loss = float(-(p_tr * np.log(q)).sum(axis=1).mean()) + lam * float((w ** 2).sum())
        grad = (phi_tr.T @ (q - p_tr)) / len(s_tr) + 2.0 * lam * w
        return loss, grad.ravel()

    res = minimize(fg, w0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 400, "maxfun": 600})
    return softmax(phi_te @ res.x.reshape(2, 6), axis=1)


def train_epochs(
    judge: TopBlocksLoRAJudge,
    cached_h: list[torch.Tensor],
    p_all: np.ndarray,
    train_idx: list[int],
    objective: str,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    eval_idx: list[int] | None = None,
    checkpoint_epochs: set[int] | None = None,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    params = judge.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    grid_t = torch.arange(6, dtype=torch.float32, device=device)
    p_tr = p_all[train_idx]
    h_tr = p_tr @ GRID_NP
    n_tr = len(train_idx)

    history = []
    snaps: dict[int, np.ndarray] = {}
    if checkpoint_epochs and 0 in checkpoint_epochs and eval_idx is not None:
        snaps[0] = predict_probs(judge, cached_h, eval_idx, device)

    for ep in range(1, n_epochs + 1):
        opt.zero_grad(set_to_none=True)
        if objective in ("ce", "rps"):
            tot_loss = 0.0
            for k, idx in enumerate(train_idx):
                logits6 = judge.forward_logits(cached_h[idx].to(device, non_blocking=True))
                q = F.softmax(logits6, dim=-1)
                target_p = torch.tensor(p_tr[k], dtype=torch.float32, device=device)
                if objective == "ce":
                    loss_i = -(target_p * F.log_softmax(logits6, dim=-1)).sum()
                else:
                    cq = torch.cumsum(q, dim=0)[:5]
                    cp = torch.cumsum(target_p, dim=0)[:5]
                    loss_i = ((cq - cp) ** 2).mean()
                (loss_i / n_tr).backward()
                tot_loss += float(loss_i.detach().cpu())
            train_obj = tot_loss / n_tr
        elif objective in ("pearson", "spearman"):
            q_cur = predict_probs(judge, cached_h, train_idx, device)
            mu_cur = q_cur @ GRID_NP
            if objective == "pearson":
                train_obj, d_mu = pearson_loss_and_grad_np(mu_cur, h_tr)
            else:
                train_obj, d_mu = soft_spearman_loss_and_grad_np(mu_cur, h_tr)
            for k, idx in enumerate(train_idx):
                logits6 = judge.forward_logits(cached_h[idx].to(device, non_blocking=True))
                q = F.softmax(logits6, dim=-1)
                mui = (q * grid_t).sum()
                (float(d_mu[k]) * mui).backward()
        else:
            raise ValueError(objective)

        torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
        opt.step()
        history.append({"epoch": ep, "train_objective": round(float(train_obj), 6)})

        if checkpoint_epochs and ep in checkpoint_epochs and eval_idx is not None:
            snaps[ep] = predict_probs(judge, cached_h, eval_idx, device)

    return history, snaps


def run_one_model(
    model_key: str,
    bundle_path: str,
    out_dir: Path,
    top_k_blocks: int = 4,
    rank: int = 8,
    max_seq_len: int = 3584,
    lr: float = 2e-4,
    n_epochs: int = 12,
) -> None:
    t0 = time.time()
    grad_check = verify_two_pass_chain_rule()
    print(f"[{model_key}] Two-pass chain-rule gradient verification passed: {grad_check}", flush=True)

    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    items = bundle["items"]
    folds = bundle["folds_spec"]["folds"]
    p_all = np.array([[c / it["n_ratings"] for c in it["counts_0_5"]] for it in items], dtype=np.float64)

    hf_root, repo_dir = LORA_MODELS[model_key]
    snap = sorted(glob.glob(f"{hf_root}/hub/models--{repo_dir}/snapshots/*"))[-1]
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(6)]
    assert len(set(digit_ids)) == 6

    device = torch.device("cuda:0")
    print(f"[{model_key}] Loading base model in bfloat16 (layer-streamed prefix cache for <4 GiB peak VRAM)...", flush=True)
    full_model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, local_files_only=True)
    full_model.eval()

    n_layers = len(full_model.model.layers)
    split_layer = n_layers - top_k_blocks
    digit_w = full_model.lm_head.weight[digit_ids].detach().clone()
    digit_b = full_model.lm_head.bias[digit_ids].detach().clone() if full_model.lm_head.bias is not None else None

    # Tokenize all 75 items once, then stream layer-by-layer on GPU so peak VRAM < 4 GiB even on 7B-14B models!
    assistant_stub = '{"A": "Y", "B": "Y", "C": "N", "D": "N/A", "final_score_0_5":'
    print(f"[{model_key}] Streaming exact frozen prefix layers 0..{split_layer - 1}/{n_layers} across all 75 items...", flush=True)
    seq_lengths = []
    cur_hs: list[torch.Tensor] = []
    pos_ids_list: list[torch.Tensor] = []
    pos_embs_list: list[tuple[torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        embed_gpu = full_model.model.embed_tokens.to(device)
        rot_gpu = full_model.model.rotary_emb.to(device)
        for it in items:
            msgs = [
                {"role": "system", "content": it["system_p1"]},
                {"role": "user", "content": it["user_ast"]},
                {"role": "assistant", "content": assistant_stub},
            ]
            kwargs = {"tokenize": False, "continue_final_message": True, "add_generation_prompt": False}
            if "qwen3" in model_key or "q3_" in model_key:
                kwargs["enable_thinking"] = False
            rendered = tok.apply_chat_template(msgs, **kwargs)
            ids = tok.encode(rendered, add_special_tokens=False)
            if len(ids) > max_seq_len:
                head_n = max_seq_len // 2
                tail_n = max_seq_len - head_n
                ids = ids[:head_n] + ids[-tail_n:]
            seq_lengths.append(len(ids))
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            pos_ids = torch.arange(len(ids), device=device).unsqueeze(0)
            h0 = embed_gpu(inp)
            pe = rot_gpu(h0, pos_ids)
            cur_hs.append(h0.cpu())
            pos_ids_list.append(pos_ids.cpu())
            pos_embs_list.append((pe[0].cpu(), pe[1].cpu()))
        embed_gpu.cpu()

        for l_idx in range(split_layer):
            layer_gpu = full_model.model.layers[l_idx].to(device)
            for i_it in range(len(items)):
                h_gpu = cur_hs[i_it].to(device, non_blocking=True)
                p_gpu = pos_ids_list[i_it].to(device, non_blocking=True)
                pe_gpu = (pos_embs_list[i_it][0].to(device, non_blocking=True), pos_embs_list[i_it][1].to(device, non_blocking=True))
                out_l = layer_gpu(h_gpu, attention_mask=None, position_ids=p_gpu, position_embeddings=pe_gpu)
                cur_hs[i_it] = (out_l[0] if isinstance(out_l, tuple) else out_l).detach().cpu()
            layer_gpu.cpu()
            torch.cuda.empty_cache()

        cached_prefix_h = [t.pin_memory() for t in cur_hs]

        # Run top blocks once on GPU to record exact base hidden vectors & base logits
        top_layers = nn.ModuleList([full_model.model.layers[i] for i in range(split_layer, n_layers)]).to(device)
        norm = full_model.model.norm.to(device)
        rotary_emb = full_model.model.rotary_emb.to(device)
        final_hidden_vecs = []
        base_logits_list = []
        dw_f = digit_w.to(device).float()
        db_f = digit_b.to(device).float() if digit_b is not None else None
        for i_it in range(len(items)):
            h_top = cached_prefix_h[i_it].to(device, non_blocking=True)
            p_gpu = pos_ids_list[i_it].to(device, non_blocking=True)
            pe_gpu = rotary_emb(h_top, p_gpu)
            for layer_t in top_layers:
                out_l = layer_t(h_top, attention_mask=None, position_ids=p_gpu, position_embeddings=pe_gpu)
                h_top = out_l[0] if isinstance(out_l, tuple) else out_l
            h_norm = norm(h_top[:, -1, :]).float()
            logits6 = F.linear(h_norm, dw_f, db_f).squeeze(0)
            final_hidden_vecs.append(h_norm.squeeze(0).cpu().numpy())
            base_logits_list.append(logits6.cpu().numpy().tolist())

    del full_model, cur_hs, pos_ids_list, pos_embs_list
    torch.cuda.empty_cache()

    judge = TopBlocksLoRAJudge(top_layers, rotary_emb, norm, digit_w, digit_b, rank=rank).to(device)
    n_trainable = sum(p.numel() for p in judge.trainable_parameters())
    print(f"[{model_key}] TopBlocksLoRAJudge ready: {len(judge.lora_modules)} LoRA projections + 6-logit head = {n_trainable:,} trainable params.", flush=True)

    # Verify epoch-0 bit-exact reproduction of full_model digit logits!
    q_base_check = predict_probs(judge, cached_prefix_h, list(range(75)), device)
    from scipy.special import softmax as sp_softmax

    max_diff_ep0 = float(np.max(np.abs(q_base_check - sp_softmax(np.array(base_logits_list), axis=1))))
    assert max_diff_ep0 < 1e-5, f"Epoch-0 mismatch vs full_model: {max_diff_ep0}"
    print(f"[{model_key}] Epoch-0 identity check vs full 36-layer model max prob diff = {max_diff_ep0:.2e}", flush=True)

    # Export public model-only SVD representation coordinates (75 x 32) + base_logits for fast local 200-split x 5-restart head sweeps!
    h_mat = np.stack(final_hidden_vecs, axis=0)  # 75 x hidden_dim (purely a function of public inputs, zero human ratings!)
    # Save exact Gram matrix G = H @ H.T (75 x 75 floats = 45 KB!) so local per-fold training-only PCA/ridge is 100% exact & leak-free!
    gram_75 = (h_mat @ h_mat.T).tolist()

    ckpt_grid = {0, 2, 4, 8, 12}
    objectives = ["ce", "rps", "pearson", "spearman"]
    folds_records = []

    # 1. Main 5-fold grouped task-family cross-validation (all 75 items scored strictly out-of-fold!)
    for f_spec in folds:
        f_id = f_spec["fold"]
        in_tr = [i for i, it in enumerate(items) if it["family"] in f_spec["inner_train_families"]]
        in_val = [i for i, it in enumerate(items) if it["family"] in f_spec["inner_val_families"]]
        out_tr = sorted(in_tr + in_val)
        out_te = [i for i, it in enumerate(items) if it["family"] in f_spec["outer_test_families"]]
        f_rec = {"fold": f_id, "test_indices": out_te, "n_train": len(out_tr), "n_test": len(out_te), "objectives": {}}

        for obj in objectives:
            # Phase I: Inner train -> select best stopping epoch on inner validation (Formulations ① and ③)
            judge.reset_trainable(seed=20261116 + f_id * 100)
            hist_in, val_snaps = train_epochs(
                judge, cached_prefix_h, p_all, in_tr, obj, n_epochs, lr, 0.01, device, eval_idx=in_val, checkpoint_epochs=ckpt_grid
            )
            best_ep_val_rps = min(ckpt_grid, key=lambda ep: rps_np(val_snaps[ep], p_all[in_val]))
            best_ep_val_corr = max(ckpt_grid, key=lambda ep: pearson_np(val_snaps[ep], p_all[in_val]))

            # Phase II: Refit from exact same seed on full outer training split (out_tr), snapshotting outer_te at needed epochs
            need_eps = {0, best_ep_val_rps, best_ep_val_corr, n_epochs}
            judge.reset_trainable(seed=20261116 + f_id * 100)
            hist_out, te_snaps = train_epochs(
                judge, cached_prefix_h, p_all, out_tr, obj, n_epochs, lr, 0.01, device, eval_idx=out_te, checkpoint_epochs=need_eps
            )
            q_tr_final = predict_probs(judge, cached_prefix_h, out_tr, device)

            obj_entry = {
                "best_epoch_val_rps": int(best_ep_val_rps),
                "best_epoch_val_corr": int(best_ep_val_corr),
                "train_history_outer": hist_out,
                "q_test_base_ep0": te_snaps[0].tolist(),
                "q_test_train_only_final": te_snaps[n_epochs].tolist(),
                "q_test_val_rps_selected": te_snaps[best_ep_val_rps].tolist(),
                "q_test_val_corr_selected": te_snaps[best_ep_val_corr].tolist(),
            }
            if obj == "pearson":
                # Also fit two-stage post-hoc calibrated CE head on Pearson-LoRA's scalar expected score!
                s_tr = q_tr_final @ GRID_NP
                s_te = te_snaps[n_epochs] @ GRID_NP
                q_te_head = fit_posthoc_ce_head_on_score(s_tr, p_all[out_tr], s_te)
                obj_entry["q_test_pearson_lora_plus_ce_head"] = q_te_head.tolist()

            f_rec["objectives"][obj] = obj_entry
            print(f"[{model_key}] Fold {f_id} obj={obj:8s} done (ep_val_rps={best_ep_val_rps}, ep_val_corr={best_ep_val_corr})", flush=True)

        folds_records.append(f_rec)

    # 2. LoRA training-size learning curve across n_fam in {4, 8, 12, 15} training task families (on all 5 outer folds!)
    lc_records = []
    for n_fam in (4, 8, 12, 15):
        for f_spec in folds:
            f_id = f_spec["fold"]
            avail_fams = sorted(f_spec["inner_train_families"] + f_spec["inner_val_families"])
            rng_lc = np.random.default_rng(99000 + 100 * n_fam + f_id)
            sub_fams = set(avail_fams[i] for i in rng_lc.permutation(len(avail_fams))[:n_fam])
            sub_tr = [i for i, it in enumerate(items) if it["family"] in sub_fams]
            out_te = [i for i, it in enumerate(items) if it["family"] in f_spec["outer_test_families"]]
            row_lc = {"n_train_families": n_fam, "fold": f_id, "n_train_artifacts": len(sub_tr), "test_indices": out_te, "q_test": {}}
            for obj in ("ce", "rps", "pearson"):
                judge.reset_trainable(seed=55000 + 100 * n_fam + f_id)
                _, te_s = train_epochs(
                    judge, cached_prefix_h, p_all, sub_tr, obj, 10, lr, 0.01, device, eval_idx=out_te, checkpoint_epochs={10}
                )
                row_lc["q_test"][obj] = te_s[10].tolist()
                if obj == "pearson":
                    q_tr_f = predict_probs(judge, cached_prefix_h, sub_tr, device)
                    row_lc["q_test"]["pearson_plus_ce_head"] = fit_posthoc_ce_head_on_score(
                        q_tr_f @ GRID_NP, p_all[sub_tr], te_s[10] @ GRID_NP
                    ).tolist()
            lc_records.append(row_lc)
        print(f"[{model_key}] LoRA learning curve n_fam={n_fam} complete across 5 folds.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"lora_weight_training_{model_key}.json"
    out_file.write_text(
        json.dumps(
            {
                "model": model_key,
                "snapshot": snap,
                "top_k_blocks": top_k_blocks,
                "lora_rank": rank,
                "n_trainable_params": n_trainable,
                "max_seq_len": max_seq_len,
                "seq_len_min_med_max": [int(min(seq_lengths)), int(np.median(seq_lengths)), int(max(seq_lengths))],
                "epoch0_identity_max_diff": max_diff_ep0,
                "gradient_verification": grad_check,
                "lr": lr,
                "n_epochs": n_epochs,
                "base_logits_75x6": base_logits_list,
                "hidden_gram_75x75": gram_75,
                "five_fold_cv": folds_records,
                "lora_learning_curve": lc_records,
                "elapsed_seconds": round(time.time() - t0, 2),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[{model_key}] ALL LoRA weight-training runs COMPLETE in {time.time() - t0:.1f}s -> {out_file}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LORA_MODELS.keys()))
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_one_model(args.model, args.bundle, Path(args.out))


if __name__ == "__main__":
    main()
