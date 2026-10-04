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

        # Regenerate outputs first so a fresh clone is fully reproducible.
        memory_input = f"{ns.evals}/memory_train.jsonl"
        memory_output = str(out_dir / "memory_answers.jsonl")
        actions_input = f"{ns.evals}/actions_train.jsonl"
        actions_output = str(out_dir / "action_predictions.jsonl")

        print("\n$", sys.executable, "run.py", "memory", "--input", memory_input, "--output", memory_output, "--data", ns.data)
        run_memory(memory_input, memory_output, ns.data)
        print("\n$", sys.executable, "run.py", "actions", "--input", actions_input, "--output", actions_output, "--data", ns.data)
        run_actions(actions_input, actions_output, ns.data)

        commands = [
            [sys.executable, "eval_harness/score_retrieval.py", "--gold", memory_input, "--answers", memory_output, "--out", str(out_dir / "results_retrieval.json")],
            [sys.executable, "eval_harness/score_memory.py", "--gold", memory_input, "--answers", memory_output, "--judge", "none", "--out", str(out_dir / "results_memory.json")],
            [sys.executable, "eval_harness/score_actions.py", "--gold", actions_input, "--predictions", actions_output, "--out", str(out_dir / "results_actions.json")],
        ]
        for cmd in commands:
            print("\n$", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
