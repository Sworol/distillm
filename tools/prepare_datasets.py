import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


NEWLINE_TOKEN = "<@x(x!>"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _iter_take(it: Iterable[Any], n: int) -> Iterable[Any]:
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x


def prepare_dolly(out_dir: str, limit: int, dataset_name: str) -> str:
    """
    Writes data/dolly/raw.jsonl with {instruction,input,output}.
    Uses a publicly available Dolly-style dataset by default.
    """
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "raw.jsonl")

    # Prefer a dataset with instruction/context/response fields.
    ds = load_dataset(dataset_name, split="train")

    def to_example(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
        # databricks/databricks-dolly-15k schema: instruction, context, response, category
        instruction = row.get("instruction") or row.get("prompt") or row.get("question")
        if not instruction:
            return None
        example_input = row.get("context") or row.get("input") or ""
        output = row.get("response") or row.get("output") or row.get("answer")
        if output is None:
            return None
        return {
            "instruction": str(instruction),
            "input": str(example_input),
            "output": str(output),
        }

    wrote = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in _iter_take(ds, limit if limit > 0 else len(ds)):
            ex = to_example(row)
            if ex is None:
                continue
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            wrote += 1

    if wrote == 0:
        raise RuntimeError(f"Wrote 0 examples to {out_path}; dataset schema may be unexpected: {dataset_name}")

    return out_path


def prepare_openwebtext(out_dir: str, max_docs: int, streaming: bool) -> str:
    """
    Writes data/openwebtext/data.txt where each line is a document with newlines replaced by NEWLINE_TOKEN.
    Uses streaming by default to avoid downloading the full dataset.
    """
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "data.txt")

    ds = load_dataset("openwebtext", split="train", streaming=streaming)
    if streaming:
        iterator = ds
    else:
        iterator = iter(ds)

    wrote = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in _iter_take(iterator, max_docs):
            text = row.get("text")
            if not text:
                continue
            f.write(re.sub(r"\n+", NEWLINE_TOKEN, text) + "\n")
            wrote += 1

    if wrote == 0:
        raise RuntimeError(f"Wrote 0 docs to {out_path}")

    return out_path


def warm_model_cache(model_id: str) -> None:
    """
    Downloads tokenizer + model weights into the local HF cache.
    """
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default=".", help="Repo base path")

    parser.add_argument("--prepare-dolly", action="store_true")
    parser.add_argument("--dolly-dataset", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--dolly-limit", type=int, default=15000)

    parser.add_argument("--prepare-openwebtext", action="store_true")
    parser.add_argument("--openwebtext-max-docs", type=int, default=200000)
    parser.add_argument("--openwebtext-streaming", action="store_true", default=True)
    parser.add_argument("--openwebtext-no-streaming", dest="openwebtext_streaming", action="store_false")

    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument("--model-id", type=str, default="gpt2-large")

    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)

    # HuggingFace/datasets + PyTorch can spawn background threads; in some environments
    # the interpreter may abort during finalization. Exiting via os._exit avoids that.
    # We still flush any prints before exiting.
    def _hard_exit_ok() -> None:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    if args.prepare_dolly:
        out = prepare_dolly(os.path.join(base_path, "data", "dolly"), args.dolly_limit, args.dolly_dataset)
        print(f"[ok] wrote dolly jsonl: {out}")

    if args.prepare_openwebtext:
        out = prepare_openwebtext(
            os.path.join(base_path, "data", "openwebtext"),
            max_docs=args.openwebtext_max_docs,
            streaming=args.openwebtext_streaming,
        )
        print(f"[ok] wrote openwebtext txt: {out}")

    if args.warm_cache:
        warm_model_cache(args.model_id)
        print(f"[ok] warmed HF cache for: {args.model_id}")

    _hard_exit_ok()


if __name__ == "__main__":
    main()
