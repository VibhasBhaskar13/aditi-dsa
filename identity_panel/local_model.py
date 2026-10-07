"""Run the judges on an open-weights model that you load yourself (e.g. on a Kaggle GPU).

No API, no rate limits, no key: the model lives in this process. To keep a GPU busy the
panel runs are executed by many worker threads at once (see scripts/run_suite.py); each
call to `client.messages.create(...)` hands its prompt to `BatchedGenerator`, which collects
whatever requests are waiting and sends them through the model as ONE batch.

    worker threads --create()--> BatchedGenerator --batch--> model.generate --> answers

`LocalHFClient` exposes the same `.messages.create(...)` surface as every other client, so
nothing else in the pipeline knows a local model is behind it. The only model-specific code
is `build_hf_generate_fn`; pass your own `generate_fn` to use another engine (or a fake in tests).

Output format: small models do not reliably follow a JSON schema, so the answer is *pre-filled*
with `{"score": ` (the model just continues it) and then parsed tolerantly. The score is
therefore produced before the reasoning, matching the key order of the schema.

Sampling: judges use temperature > 0 (see suite.py), otherwise every judge with the same
prompt would give the identical answer and a panel of identical judges would be pointless.
Answers are stored by CachedClient, so a given (seat, prompt) is sampled once and then shared
by every experiment condition that sends it - that is what makes conditions comparable.
"""

from __future__ import annotations

import concurrent.futures
import json
import queue
import re
import threading
import time
from collections import defaultdict
from typing import Callable, Optional, Sequence

from .client import _Response, _extract_json

PREFILL = '{"score": '


class BatchedGenerator:
    """Collects concurrent requests into batches for `generate_fn(specs) -> list[str]`.

    A spec is a dict: system, user, max_new_tokens, temperature. Requests with different
    temperatures are never mixed in one `generate_fn` call. If a batch raises (typically a
    CUDA out-of-memory) it is split in half and retried, down to single requests.
    """

    def __init__(self, generate_fn: Callable[[Sequence[dict]], Sequence[str]], *,
                 max_batch: int = 16, max_wait: float = 0.05) -> None:
        self._fn, self.max_batch, self.max_wait = generate_fn, max_batch, max_wait
        self._q: "queue.Queue" = queue.Queue()
        self.batches = 0
        self.requests = 0
        self._t = threading.Thread(target=self._loop, name="batched-generator", daemon=True)
        self._t.start()

    def generate(self, spec: dict) -> str:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put((spec, fut))
        return fut.result()

    def close(self) -> None:
        self._q.put(None)

    def _loop(self) -> None:
        while True:
            first = self._q.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._q.put(None)           # finish this batch, then stop
                    break
                batch.append(nxt)
            groups = defaultdict(list)
            for item in batch:
                groups[item[0].get("temperature")].append(item)
            for group in groups.values():
                self._call(group)

    def _call(self, group: list) -> None:
        try:
            outs = self._fn([spec for spec, _ in group])
            if len(outs) != len(group):
                raise RuntimeError(f"generate_fn returned {len(outs)} answers for {len(group)} requests")
        except Exception as exc:  # noqa: BLE001
            if len(group) > 1:
                mid = len(group) // 2
                self._call(group[:mid])
                self._call(group[mid:])
            else:
                group[0][1].set_exception(exc)
            return
        self.batches += 1
        self.requests += len(group)
        for (_, fut), out in zip(group, outs):
            fut.set_result(out)


class LocalHFClient:
    """`client.messages.create(...)` backed by a locally loaded model.

    `model` is a Hugging Face id (downloaded) or a folder on disk (e.g. a Kaggle input path).
    Panels using several different models are not supported: all seats share this one model.
    """

    def __init__(self, model: str, *, generate_fn: Optional[Callable] = None, max_batch: int = 16,
                 max_wait: float = 0.05, max_new_tokens: int = 256, **hf_kwargs) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._gen = BatchedGenerator(generate_fn or build_hf_generate_fn(model, **hf_kwargs),
                                     max_batch=max_batch, max_wait=max_wait)
        self.messages = _LocalMessages(self)

    @property
    def stats(self) -> dict:
        g = self._gen
        return {"requests": g.requests, "batches": g.batches,
                "avg_batch": round(g.requests / g.batches, 2) if g.batches else 0.0}

    def close(self) -> None:
        self._gen.close()


class _LocalMessages:
    def __init__(self, outer: LocalHFClient) -> None:
        self._o = outer

    def create(self, *, model=None, max_tokens=256, system="", messages, output_config=None,
               temperature=None, **_):
        o = self._o
        schema = (output_config or {}).get("format", {}).get("schema")
        if schema:
            keep = {k: v for k, v in schema.items() if k != "additionalProperties"}
            system = (f"{system}\n\nRespond with ONLY a single JSON object matching this JSON "
                      f"schema, with no other text:\n{json.dumps(keep)}")
        spec = {"system": system, "user": messages[-1]["content"],
                "max_new_tokens": min(int(max_tokens or o.max_new_tokens), o.max_new_tokens),
                "temperature": temperature}
        continuation = o._gen.generate(spec)
        text = repair_json(PREFILL + continuation)
        resp = _Response(text, request_id=None)
        if "}" not in continuation and not text.rstrip().endswith("}"):
            resp.stop_reason = "max_tokens"
        return resp


_SCORE_RE = re.compile(r'"score"\s*:\s*"?(-?\d+)')


def repair_json(text: str) -> str:
    """Return valid `{"score", "reasoning"}` JSON if at all recoverable, else the text unchanged."""
    candidate = _extract_json(text)
    try:
        json.loads(candidate)
        return candidate
    except ValueError:
        pass
    m = _SCORE_RE.search(text)
    if not m:
        return text
    rest = text[m.end():]
    r = re.search(r'"reasoning"\s*:\s*"(.*)', rest, re.S)
    reasoning = r.group(1) if r else rest
    reasoning = re.sub(r'"\s*}?\s*$', "", reasoning.strip()).replace('\\"', '"').strip(' ",}\n')
    return json.dumps({"score": int(m.group(1)), "reasoning": reasoning})


def build_hf_generate_fn(model_id: str, *, dtype: str = "float16", device_map: str = "auto",
                         load_in_4bit: bool = False, trust_remote_code: bool = False,
                         top_p: float = 0.95) -> Callable[[Sequence[dict]], list[str]]:
    """Load `model_id` with transformers and return a batched `generate_fn`. Needs a GPU for speed.

    Not exercised by the offline tests (no GPU / no model download there).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = dict(device_map=device_map, trust_remote_code=trust_remote_code)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                       bnb_4bit_compute_dtype=getattr(torch, dtype))
    else:
        kw["torch_dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()

    def render(spec: dict) -> str:
        msgs = [{"role": "system", "content": spec["system"]}, {"role": "user", "content": spec["user"]}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001 - template without a system role (e.g. Gemma): merge it in
            merged = [{"role": "user", "content": spec["system"] + "\n\n" + spec["user"]}]
            prompt = tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        return prompt + PREFILL

    def generate(specs: Sequence[dict]) -> list[str]:
        enc = tok([render(s) for s in specs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        temp = specs[0].get("temperature")
        sample = bool(temp and temp > 0)
        gen_kw = dict(do_sample=sample, max_new_tokens=max(s["max_new_tokens"] for s in specs),
                      pad_token_id=tok.pad_token_id)
        if sample:
            gen_kw.update(temperature=float(temp), top_p=top_p)
        try:
            with torch.no_grad():
                out = model.generate(**enc, **gen_kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise                       # BatchedGenerator retries with a smaller batch
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate
