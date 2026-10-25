"""Model loading, exact sampling, and sequence log-probabilities.

Sampling is from the unmodified policy: temperature 1, no top-k / top-p /
repetition penalty (Qwen's generation_config sets all three, so every call
passes an explicit GenerationConfig). A completion keeps its EOS token, so the
sum of its token log-probabilities is log pi(y | x) of the whole response.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, dtype: torch.dtype, device: torch.device, train: bool = False):
    kw = dict(attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    model.to(device)
    model.train(train)
    if not train:
        model.requires_grad_(False)
    return model


def eos_ids(model, tok) -> list[int]:
    e = model.generation_config.eos_token_id
    ids = [e] if isinstance(e, int) else list(e or [])
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


def format_prompt(tok, text: str, prefix: list[dict] | None = None) -> str:
    """Chat-formatted prompt; prefix holds few-shot turns (role/content dicts)."""
    msgs = list(prefix or []) + [{"role": "user", "content": text}]
    if tok.chat_template:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return "\n".join(m["content"] for m in msgs) + "\n"


def encode_prompts(tok, texts: list[str], prefix: list[dict] | None = None) -> list[list[int]]:
    return [tok(format_prompt(tok, t, prefix), add_special_tokens=False)["input_ids"] for t in texts]


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


@dataclass
class Sample:
    prompt_idx: int          # index into the caller's prompt list
    completion_ids: list[int]
    text: str
    truncated: bool          # no EOS within max_new_tokens


@torch.no_grad()
def generate(model, tok, prompt_ids: list[list[int]], n_per_prompt: int, *,
             max_new_tokens: int, seed: int, batch_size: int = 256,
             device: torch.device | None = None) -> list[Sample]:
    """n_per_prompt i.i.d. samples per prompt from the exact policy."""
    device = device or next(model.parameters()).device
    eos = eos_ids(model, tok)
    gcfg = GenerationConfig(
        do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
        max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=eos,
    )
    jobs = [i for i in range(len(prompt_ids)) for _ in range(n_per_prompt)]
    out: list[Sample] = []
    was_training = model.training
    model.eval()
    for b0 in range(0, len(jobs), batch_size):
        idx = jobs[b0:b0 + batch_size]
        P = max(len(prompt_ids[i]) for i in idx)
        ids = torch.full((len(idx), P), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), P), dtype=torch.long)
        for r, i in enumerate(idx):
            p = prompt_ids[i]
            ids[r, P - len(p):] = torch.tensor(p)
            att[r, P - len(p):] = 1
        torch.manual_seed(seed * 1_000_003 + b0)
        gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device),
                             generation_config=gcfg)
        gen = gen[:, P:].tolist()
        for r, i in enumerate(idx):
            c = gen[r]
            cut = next((t + 1 for t, v in enumerate(c) if v in eos), None)
            comp = c[:cut] if cut is not None else c
            out.append(Sample(i, comp, tok.decode(comp, skip_special_tokens=True), cut is None))
    model.train(was_training)
    return out


def _batch(prompt_ids: list[list[int]], comp_ids: list[list[int]], pad: int, device):
    P = max(len(p) for p in prompt_ids)
    L = max(len(c) for c in comp_ids)
    B = len(prompt_ids)
    ids = torch.full((B, P + L), pad, dtype=torch.long)
    att = torch.zeros((B, P + L), dtype=torch.long)
    cmask = torch.zeros((B, L), dtype=torch.float32)
    for r, (p, c) in enumerate(zip(prompt_ids, comp_ids)):
        ids[r, P - len(p):P] = torch.tensor(p)
        att[r, P - len(p):P] = 1
        ids[r, P:P + len(c)] = torch.tensor(c)
        att[r, P:P + len(c)] = 1
        cmask[r, :len(c)] = 1
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), att.to(device), pos.to(device), cmask.to(device), L


def token_logprobs(model, prompt_ids: list[list[int]], comp_ids: list[list[int]],
                   pad_id: int, fp32: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log pi(y_t | x, y_<t) for one micro-batch.

    Returns (logp [B, L], mask [B, L]); gradients flow iff grad is enabled.
    Only the completion positions go through the LM head (logits_to_keep).
    fp32 turns off the CUDA bf16 autocast.
    """
    device = next(model.parameters()).device
    ids, att, pos, cmask, L = _batch(prompt_ids, comp_ids, pad_id, device)
    with torch.autocast(device.type, enabled=False) if fp32 else autocast(device):
        logits = model(input_ids=ids, attention_mask=att, position_ids=pos,
                       logits_to_keep=L + 1, use_cache=False).logits[:, :-1]
    logits = logits.float()
    tgt = ids[:, -L:]
    logp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    return logp * cmask, cmask


@torch.no_grad()
def sequence_logprobs(model, prompt_ids, comp_ids, pad_id, micro_batch: int = 16,
                      fp32: bool = False) -> torch.Tensor:
    """log pi(y | x) for each (prompt, completion) pair, on CPU."""
    out = []
    for b0 in range(0, len(comp_ids), micro_batch):
        lp, _ = token_logprobs(model, prompt_ids[b0:b0 + micro_batch],
                               comp_ids[b0:b0 + micro_batch], pad_id, fp32)
        out.append(lp.sum(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0)


class Timer:
    """Timer()() returns the seconds since construction or the previous call."""

    def __init__(self):
        self.t = time.time()

    def __call__(self) -> float:
        now = time.time()
        dt, self.t = now - self.t, now
        return dt
