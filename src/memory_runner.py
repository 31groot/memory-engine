from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .answer import Answer, extractive_answer, llm_answer, llm_config, load_env, supported
from .indexer import CorpusIndex
from .retrieval import HybridRetriever


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class MemoryEngine:
    """retrieve -> (support check) -> answer, with the as_of time enforced at every step."""

    def __init__(self, data_dir: str, use_llm: bool = True):
        self.corpus = CorpusIndex(data_dir)
        self.retriever = HybridRetriever(self.corpus)
        env = load_env(Path(data_dir).resolve().parent / ".env")
        self.llm = llm_config(env) if use_llm else None
        self.stats = {"llm": 0, "extractive": 0, "llm_fallback": 0}

    def ask(self, question: str, as_of: str) -> dict[str, Any]:
        hits = self.retriever.search(question, as_of, 20)
        retrieved = [h.record.id for h in hits]
        ans: Answer | None = None
        if self.llm:
            ok, why = supported(question, hits, self.retriever, as_of)
            ans = llm_answer(self.llm, question, as_of, hits) if hits else None
            if ans is None or ans.mode == "llm-error" or not ans.text:
                self.stats["llm_fallback"] += 1
                if ans is not None and ans.note:
                    print(f"[llm fallback] {ans.note}", file=sys.stderr)
                ans = None
            else:
                self.stats["llm"] += 1
        if ans is None:
            ans = extractive_answer(question, as_of, hits, self.retriever)
            self.stats["extractive"] += 1
        return {"answer": ans.text, "sources": ans.sources, "retrieved": retrieved, "abstained": ans.abstained}


def run_memory(input_path: str, output_path: str, data_dir: str) -> None:
    engine = MemoryEngine(data_dir)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in load_jsonl(input_path):
            row = {"id": item["id"], **engine.ask(item["question"], item["as_of"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    mode = "LLM" if engine.llm else "offline extractive"
    print(f"memory: answered with {mode} back end {engine.stats}")


if __name__ == "__main__":
    raise SystemExit("Use `python3 run.py memory ...` from the project root.")
