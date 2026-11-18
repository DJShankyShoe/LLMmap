#!/usr/bin/env python3
"""Build an LLM list for make_dataset.py from the live OpenRouter catalog.

OpenRouter exposes (almost) every commercial and open-weights chat model behind one
OpenAI-compatible API and one key, which makes it the cheapest way to cover "the
whole market". The catalog endpoint is public, so the list can be regenerated at
any time to pick up new releases:

    python utility/build_llm_list.py -o confs/LLMs/market_$(date +%Y-%m).json
    python utility/build_llm_list.py --max_completion_price 20 --since 2024-01-01 -o confs/LLMs/cheap.json

Each entry looks like
    {"name": "openai/gpt-4.1", "backend": "openrouter", "reasoning": false,
     "family": "openai", "pricing": {"prompt": "2e-06", "completion": "8e-06"}}
`pricing` (USD per token) is only used by `make_dataset.py --dry_run` for cost estimates.
"""
import argparse
import datetime as dt
import json
import re
import sys
import urllib.request

CATALOG = "https://openrouter.ai/api/v1/models"

# Not chat LLMs in the sense we need (routers, aliases, image/audio/music generators,
# guard/classifier models, apply/edit models, translation-only models, deep-research agents).
EXCLUDE_PATTERNS = [
    r"^openrouter/", r"^~", r"router", r"/auto", r"switchyard", r"pareto", r"fusion", r"bodybuilder",
    r"-image", r"image-", r"nano-banana", r"lyria", r"audio", r"voxtral", r"omni",
    r"guard", r"content-safety", r"safeguard", r"schematron", r"morph-", r"relace", r"scaledown",
    r"/hy-mt", r"deep-research", r"ui-tars", r"-customtools", r"-multi-agent", r"-contributor",
    r"ultraspeed", r"legal$", r"-vision-exp", r"moondream", r"perceptron", r"-vl", r"vl-", r"5v", r"4\.\dv",
]
# Role-play / uncensored fine-tunes: very unlikely behind an enterprise SOC assistant.
ROLEPLAY_PATTERNS = [r"mythomax", r"magnum", r"euryale", r"lunaris", r"cydonia", r"skyfall", r"unslop",
                     r"remm-", r"weaver", r"dolphin", r"aion-rp", r"thedrummer", r"sao10k", r"undi95", r"anthracite", r"mancer/"]


def fetch_catalog():
    with urllib.request.urlopen(CATALOG, timeout=60) as r:
        return json.load(r)["data"]


def is_text_chat(m):
    arch = m.get("architecture", {}) or {}
    out = arch.get("output_modalities") or ["text"]
    return out == ["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--catalog_file", help="Use a saved copy of the catalog JSON instead of downloading it.")
    ap.add_argument("--since", help="Only models added to OpenRouter after this date (YYYY-MM-DD).")
    ap.add_argument("--max_completion_price", type=float, default=None,
                    help="Drop models whose completion price exceeds this many USD per 1M tokens (e.g. skip *-pro tiers).")
    ap.add_argument("--include_roleplay", action="store_true")
    ap.add_argument("--include_free", action="store_true", help="Keep ':free' variants (rate-limited duplicates).")
    ap.add_argument("--vendors", nargs="*", help="Only these vendor prefixes (e.g. openai anthropic google meta-llama).")
    args = ap.parse_args()

    if args.catalog_file:
        data = json.load(open(args.catalog_file))
        data = data.get("data", data)
    else:
        data = fetch_catalog()

    pats = EXCLUDE_PATTERNS + ([] if args.include_roleplay else ROLEPLAY_PATTERNS)
    since = dt.datetime.fromisoformat(args.since).timestamp() if args.since else None

    by_id = {m["id"]: m for m in data}
    out, skipped = [], []
    for m in sorted(data, key=lambda m: (m["id"].split("/")[0], m.get("created", 0))):
        mid = m["id"]
        if ":" in mid:
            base = mid.split(":")[0]
            # keep a ':free' variant only if no paid base model exists
            if not (args.include_free and mid.endswith(":free") and base not in by_id):
                continue
        if any(re.search(p, mid) for p in pats) or not is_text_chat(m):
            skipped.append(mid)
            continue
        pricing = m.get("pricing", {}) or {}
        try:
            comp = float(pricing.get("completion", 0))
            prompt = float(pricing.get("prompt", 0))
        except (TypeError, ValueError):
            skipped.append(mid)
            continue
        if comp < 0 or prompt < 0:
            skipped.append(mid)
            continue
        if args.max_completion_price is not None and comp * 1e6 > args.max_completion_price:
            skipped.append(mid)
            continue
        if since and m.get("created", 0) < since:
            continue
        vendor = mid.split("/")[0]
        if args.vendors and vendor not in args.vendors:
            continue
        params = m.get("supported_parameters") or []
        entry = {
            "name": mid,
            "backend": "openrouter",
            "reasoning": ("reasoning" in params) or ("include_reasoning" in params),
            "family": vendor,
            "pricing": {"prompt": pricing.get("prompt"), "completion": pricing.get("completion")},
        }
        if m.get("created"):
            entry["note"] = "added " + dt.datetime.fromtimestamp(m["created"], dt.timezone.utc).strftime("%Y-%m-%d")
        out.append(entry)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    vendors = sorted({e["family"] for e in out})
    print(f"{len(out)} models from {len(vendors)} vendors written to {args.output} ({len(skipped)} filtered out)")
    print("Vendors:", ", ".join(vendors))


if __name__ == "__main__":
    sys.exit(main())
