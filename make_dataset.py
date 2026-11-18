#!/usr/bin/env python3
"""Collect LLMmap training/test traces for a list of LLMs.

The output JSONL is resumable: re-running the same command skips the
(llm, split, prompt-conf) triples already present, so a crash, a rate limit,
or adding models to the list later only costs the missing calls.

    python make_dataset.py soc_v1 ./confs/LLMs/market_2026-11.json ./confs/queries/soc_pool.json \
        --prompt_conf_path ./confs/prompt_configurations_soc --workers 16 --dry_run
"""
import argparse
import json
import logging
import os
from pathlib import Path

from LLMmap.prompt_configuration import PromptConfFactory
from LLMmap.dataset_maker import DatasetMaker
from LLMmap.llm import parse_llm_spec, llm_label, DEFAULT_MAX_NEW_TOKENS


def get_root_dir(default="./data/datasets"):
    return os.getenv("DATASET_DIR", default)


def load_json(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_name", help="Base name for the output dataset file (no extension).")
    p.add_argument("llms_to_use_path", help="JSON list of LLM specs: [name, type] pairs or {name, backend, ...} dicts.")
    p.add_argument("query_strategy_path", help="JSON list of query strings.")
    p.add_argument("--num_prompt_conf_train", type=int, default=150)
    p.add_argument("--num_prompt_conf_test", type=int, default=20)
    p.add_argument("--prompt_conf_path", default="./confs/prompt_configurations/")
    p.add_argument("--dataset_root", default=get_root_dir())
    p.add_argument("--fresh", action="store_true", help="Delete an existing output file and start over (default: resume).")
    p.add_argument("--overwrite", action="store_true", help="(legacy, now the default) extend an existing file.")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--workers", type=int, default=8, help="Concurrent requests per API model.")
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help="Visible-answer token budget. Keep it >= max_number_chars_response/3 of the training conf.")
    p.add_argument("--seed", type=int, default=0, help="Seed for prompt-conf sampling (same confs for every LLM).")
    p.add_argument("--only", nargs="*", help="Only run LLMs whose label contains one of these substrings.")
    p.add_argument("--dry_run", action="store_true", help="Print the plan and an upper-bound cost estimate, then exit.")
    return p


def estimate_cost(dm, specs, max_new_tokens):
    """Upper bound: every answer uses the full token budget (+ reasoning budget for reasoning models)."""
    from LLMmap.prompt_configuration import PromptConf
    import statistics

    class _Echo:
        def make_prompt(self, system, user):
            return (system or '') + user

    echo = _Echo()
    confs = dm.train_confs + dm.test_confs
    sample = [len(pc(q, echo)[0]) / 3.5 for pc in confs[:200] for q in dm.queries]
    avg_in = statistics.mean(sample) if sample else 0

    plan = {label: (a, b) for label, a, b in dm.plan()}
    total, total_calls, unknown = 0.0, 0, []
    rows = []
    for spec in specs:
        label = llm_label(spec)
        a, b = plan[label]
        calls = (a + b) * len(dm.queries)
        total_calls += calls
        _, _, extra = parse_llm_spec(spec)
        pricing = spec.get('pricing') if isinstance(spec, dict) else None
        out_tok = max_new_tokens + (extra.get('reasoning_budget', 1024) if extra.get('reasoning') else 0)
        if pricing:
            cost = calls * (avg_in * float(pricing['prompt']) + out_tok * float(pricing['completion']))
            total += cost
            rows.append((label, calls, f"${cost:,.2f}"))
        else:
            unknown.append(label)
            rows.append((label, calls, "n/a (local or unpriced)"))
    for r in rows:
        print(f"\t{r[0]:<60} {r[1]:>8} calls  {r[2]}")
    print(f"\nTotal calls: {total_calls:,}   avg prompt ≈ {avg_in:.0f} tokens   "
          f"upper-bound API cost ≈ ${total:,.2f}  (+{len(unknown)} unpriced/local models)")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    pc = PromptConfFactory(args.prompt_conf_path, seed=args.seed)

    out_path = Path(args.dataset_root) / f"{args.dataset_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if args.fresh and not args.dry_run:
            out_path.unlink()
            print(f"Deleted existing {out_path}")
        else:
            print(f"Resuming {out_path}")
    print(f'Exporting to {out_path}')

    queries = load_json(args.query_strategy_path, args.encoding)
    llms = load_json(args.llms_to_use_path, args.encoding)
    if args.only:
        llms = [s for s in llms if any(o in llm_label(s) for o in args.only)]
    labels = [llm_label(s) for s in llms]
    dup = {l for l in labels if labels.count(l) > 1}
    if dup:
        parser.error(f"Duplicate LLM labels in {args.llms_to_use_path}: {sorted(dup)}")

    dm = DatasetMaker(pc, llms, queries, args.num_prompt_conf_train, args.num_prompt_conf_test, str(out_path),
                      encoding=args.encoding, workers=args.workers, max_new_tokens=args.max_new_tokens)

    if args.dry_run:
        print(f"\n{len(llms)} LLMs x {len(queries)} queries x "
              f"({args.num_prompt_conf_train} train + {args.num_prompt_conf_test} test) prompt confs\n")
        estimate_cost(dm, llms, args.max_new_tokens)
        return

    dm()


if __name__ == "__main__":
    main()
