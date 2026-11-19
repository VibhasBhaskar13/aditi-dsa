"""Per-case author labels (popquorn_biases.jsonl), case ranges, and model-size tags.

popquorn_biases.jsonl has one line per case, in the same order and with the same item_id
as popquorn.jsonl:

    {"item_id": "q1", "gender": "Man", "race": "White", "age": "35-39",
     "occupation": "Unemployed", "education": "High school diploma or equivalent"}

Every judge sees ALL of these attributes for the case. Gender is the main attribute (it is
the one the biased judge reacts to), the rest are listed after it:

    "... was written by a man. Other details about the author: race: White; age: 35-39;
     occupation: Unemployed; education: High school diploma or equivalent."
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .data import IdentityLabel

MAIN_ATTRIBUTE = "gender"
DETAIL_ATTRIBUTES = ("race", "age", "occupation", "education")
DEMOGRAPHIC_FIELDS = (MAIN_ATTRIBUTE, *DETAIL_ATTRIBUTES)


def load_profiles(path: Path) -> dict[str, dict]:
    """item_id -> {gender, race, age, occupation, education}."""
    out = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        missing = [f for f in DEMOGRAPHIC_FIELDS if not d.get(f)]
        if missing:
            raise ValueError(f"{Path(path).name} line {n} ({d.get('item_id')}): missing {missing}")
        out[str(d["item_id"])] = {f: str(d[f]).strip() for f in DEMOGRAPHIC_FIELDS}
    return out


def profile_label(profile: dict) -> IdentityLabel:
    """The label a judge sees for one case. Gender is lower-cased ("Woman" -> "woman") so it
    matches the biased judge's instructions; the other values are shown as written."""
    return IdentityLabel(
        attribute=MAIN_ATTRIBUTE,
        value=profile[MAIN_ATTRIBUTE].lower(),
        details=tuple((f, profile[f]) for f in DETAIL_ATTRIBUTES),
    )


def parse_cases(spec: Optional[str], n_cases: int) -> tuple[int, int]:
    """'0-2000' -> (0, 2000). Case numbers are 0-based row numbers of popquorn.jsonl and the
    range is inclusive at both ends. None / 'all' -> every case. An end past the last row
    is clipped to the last row."""
    if n_cases <= 0:
        raise ValueError("the items file has no cases")
    if spec is None or spec.strip().lower() == "all":
        return 0, n_cases - 1
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", spec)
    if not m:
        raise ValueError(f"--cases must look like 0-2000 (got {spec!r})")
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) is not None else start
    if end < start:
        raise ValueError(f"--cases {spec!r}: end is before start")
    if start >= n_cases:
        raise ValueError(f"--cases {spec!r}: the file only has cases 0-{n_cases - 1}")
    return start, min(end, n_cases - 1)


def model_size_tag(model: str) -> str:
    """'Qwen/Qwen2.5-7B-Instruct' -> '7B'; a name without a size -> a file-safe version of it."""
    found = re.findall(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])", str(model))
    if found:
        return f"{found[-1]}B"
    return re.sub(r"[^A-Za-z0-9.]+", "-", str(model).split("/")[-1]).strip("-") or "model"


def default_out_name(model: str, start: int, end: int) -> str:
    """results_7b_cases_0-2000.jsonl - one file per model and case range."""
    return f"results_{model_size_tag(model).lower()}_cases_{start}-{end}.jsonl"
