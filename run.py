from __future__ import annotations

import argparse
from pathlib import Path

from src.memory_runner import run_memory
from src.action_runner import run_actions
import subprocess, sys


def main() -> None:
    p = argparse.ArgumentParser(description="Candor Memory Engine & dry-run action system")
    sub = p.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("memory", help="answer JSONL memory questions")
    m.add_argument("--input", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--data", default="data")

    v = sub.add_parser("verify", help="run the benchmark scorers against generated outputs")
    v.add_argument("--data", default="data")
    v.add_argument("--evals", default="evals")
    v.add_argument("--out", default="out")

    a = sub.add_parser("actions", help="predict dry-run actions")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--data", default="data")

    ns = p.parse_args()
    if ns.mode == "memory":
        run_memory(ns.input, ns.output, ns.data)
    elif ns.mode == "actions":
        run_actions(ns.input, ns.output, ns.data)
    else:
        out_dir = Path(ns.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Every split is regenerated and scored the same way, so a fresh clone is fully reproducible.
        #   train, heldout : DEVELOPMENT sets. Their failures were read and fixed against; treat as upper bounds.
        #   fresh          : written before the answer-composer fixes (baseline in README) but read afterwards.
        #   blind          : written last and run once, never tuned on. The closest thing here to a clean test.
        splits = [("train", ""), ("heldout", "_heldout"), ("fresh", "_fresh"), ("blind", "_blind")]
        commands = []
        for name, suf in splits:
            mem_in, act_in = f"{ns.evals}/memory_{name}.jsonl", f"{ns.evals}/actions_{name}.jsonl"
            if not (Path(mem_in).exists() and Path(act_in).exists()):
                continue
            mem_out, act_out = str(out_dir / f"memory_answers{suf}.jsonl"), str(out_dir / f"action_predictions{suf}.jsonl")
            print(f"\n== {name} ==")
            run_memory(mem_in, mem_out, ns.data)
            run_actions(act_in, act_out, ns.data)
            tag = suf or "_train"
            commands += [
                [sys.executable, "eval_harness/score_retrieval.py", "--gold", mem_in, "--answers", mem_out, "--out", str(out_dir / f"retrieval{tag}.json")],
                [sys.executable, "eval_harness/score_memory.py", "--gold", mem_in, "--answers", mem_out, "--judge", "none", "--out", str(out_dir / f"memory{tag}.json")],
                [sys.executable, "eval_harness/score_actions.py", "--gold", act_in, "--predictions", act_out, "--out", str(out_dir / f"actions{tag}.json")],
            ]
        for cmd in commands:
            print("\n$", " ".join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
