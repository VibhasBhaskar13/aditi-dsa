#!/usr/bin/env python3
"""Run the experiment on a Kaggle Notebook, keeping all progress in a Hugging Face dataset.

Why: a Kaggle session is temporary (its disk is wiped when it ends), but the experiment takes
many sessions. This wrapper makes every session pick up where the last one stopped:

    1. PULL   results / answer cache (and quota counter) from your HF dataset repo
    2. RUN    scripts/run_suite.py (all experiments, local model; resumes by itself), stopping
              cleanly before Kaggle's time limit
    3. PUSH   the same files back, every few minutes while running and once more at the end

Secrets (Kaggle: Add-ons -> Secrets; elsewhere: environment variables):
    HF_TOKEN         Hugging Face token with WRITE access (state sync; also downloads gated models)
    GEMINI_API_KEY   only for --entry run_experiment (API providers)
Plus --hf-repo user/name (a dataset repo; created private if missing).

Everything after `--` goes straight to the entry script (run_suite.py by default), e.g.

    python scripts/kaggle_runner.py --hf-repo me/identity-panel-results -- \
        --items examples/popquorn.jsonl --n-scenarios 300 --seed 0 --model Qwen/Qwen2.5-7B-Instruct

--out/--max-runtime-minutes (and --resume for run_experiment) are set for you.
Run only ONE session at a time against the same HF repo, or they will overwrite each other.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # run_experiment.py
sys.path.insert(0, str(HERE.parent))             # identity_panel/

# Files that make up "progress", per entry script (names follow each script's defaults for --out results.jsonl).
ENTRIES = {
    "run_suite": dict(files=("results.jsonl", "results.cache.jsonl", "results_summary.csv"), resume_flag=False),
    "run_experiment": dict(files=("results.jsonl", "results.csv", "results_gaps.csv",
                                  "results.cache.jsonl", "results.quota.json"), resume_flag=True),
}
STATE_FILES = ENTRIES["run_experiment"]["files"]      # kept for older callers/tests


def get_secret(name: str):
    """Kaggle Secrets first (if we are on Kaggle), then the environment."""
    try:
        from kaggle_secrets import UserSecretsClient  # only exists inside Kaggle

        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 - not on Kaggle, or the secret isn't attached
        return os.environ.get(name)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HubSync:
    """Pull/push the state files to a Hugging Face dataset repo. `api` is injectable for tests."""

    def __init__(self, repo: str, workdir: Path, token: str, api=None, files=STATE_FILES) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.api, self.repo, self.dir, self.files = api, repo, Path(workdir), tuple(files)
        self._pushed: dict[str, str] = {}
        self._lock = threading.Lock()

    def ensure_repo(self) -> None:
        self.api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)

    def pull(self) -> list[str]:
        """Download whichever state files exist remotely. Returns the names fetched."""
        self.dir.mkdir(parents=True, exist_ok=True)
        have = set(self.api.list_repo_files(self.repo, repo_type="dataset"))
        got = []
        for name in self.files:
            if name in have:
                local = self.api.hf_hub_download(self.repo, name, repo_type="dataset",
                                                 local_dir=str(self.dir))
                got.append(name)
                self._pushed[name] = _digest(Path(local))      # unchanged => no re-upload
        return got

    def push(self, message: str) -> list[str]:
        """Upload files that changed since the last push, in ONE commit. Returns names sent."""
        with self._lock:
            changed = [n for n in self.files
                       if (self.dir / n).exists() and self._pushed.get(n) != _digest(self.dir / n)]
            if not changed:
                return []
            self.api.upload_folder(folder_path=str(self.dir), repo_id=self.repo, repo_type="dataset",
                                   allow_patterns=changed, commit_message=message)
            for n in changed:
                self._pushed[n] = _digest(self.dir / n)
            return changed


def start_periodic_push(sync: HubSync, every_minutes: float):
    """Background thread: push every N minutes so a hard kill loses at most N minutes."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(every_minutes * 60):
            try:
                sent = sync.push("progress (periodic)")
                if sent:
                    print(f"  [hf] pushed {', '.join(sent)}", flush=True)
            except Exception as exc:  # noqa: BLE001 - never kill the run over a failed sync
                print(f"  [hf] push failed (will retry): {type(exc).__name__}: {exc}", flush=True)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-repo", required=True, help="HF dataset repo id, e.g. username/identity-panel-results")
    p.add_argument("--workdir", type=Path, default=Path("/kaggle/working/run")
                   if Path("/kaggle/working").exists() else Path("run_state"),
                   help="local folder for results (default: /kaggle/working/run on Kaggle)")
    p.add_argument("--max-runtime-minutes", type=float, default=600,
                   help="stop cleanly after this long. Kaggle's CPU session cap is ~12 h, "
                        "so 600 (10 h) leaves room to upload (default: %(default)s)")
    p.add_argument("--push-every", type=float, default=15, help="minutes between uploads (default: %(default)s)")
    p.add_argument("--entry", choices=sorted(ENTRIES), default="run_suite",
                   help="which script to run: run_suite (all experiments, local model; default) or "
                        "run_experiment (API providers)")
    p.add_argument("--no-pull", action="store_true", help="start fresh, ignoring what is on the Hub")
    p.add_argument("--dry-run", action="store_true", help="pass --dry-run to the experiment (no API calls)")
    a, rest = p.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]
    return a, rest


def main(argv=None, api=None) -> int:
    """`api`: inject a Hugging Face client (tests); default builds a real HfApi from HF_TOKEN."""
    a, rest = parse_args(argv)
    hf_token = get_secret("HF_TOKEN")
    if not hf_token:
        sys.exit("error: HF_TOKEN not found (Kaggle: Add-ons -> Secrets, attach it to this notebook)")
    if not a.dry_run and a.entry == "run_experiment":
        key = get_secret("GEMINI_API_KEY")
        if not key:
            sys.exit("error: GEMINI_API_KEY not found (Kaggle: Add-ons -> Secrets)")
        os.environ["GEMINI_API_KEY"] = key           # the client reads it from the environment

    entry = ENTRIES[a.entry]
    sync = HubSync(a.hf_repo, a.workdir, hf_token, api=api, files=entry["files"])
    sync.ensure_repo()
    if a.no_pull:
        a.workdir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"pulled from hub: {sync.pull() or 'nothing yet (fresh start)'}")

    import importlib

    script = importlib.import_module(a.entry)

    args = [*rest, "--out", str(a.workdir / "results.jsonl"),
            "--max-runtime-minutes", str(a.max_runtime_minutes)]
    if entry["resume_flag"]:
        args.append("--resume")
    if a.dry_run:
        args.append("--dry-run")
    stop = start_periodic_push(sync, a.push_every)
    code = 1
    try:
        code = script.main(args)
    finally:
        stop.set()
        try:
            print(f"final push: {sync.push('progress (end of session)') or 'nothing new'}")
        except Exception as exc:  # noqa: BLE001
            print(f"FINAL PUSH FAILED: {type(exc).__name__}: {exc}\n"
                  f"Your results are still in {a.workdir} - download them from the notebook's Output tab.")
            code = code or 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
