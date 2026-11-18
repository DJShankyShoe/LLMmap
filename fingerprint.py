#!/usr/bin/env python3
"""Fingerprint a chat-only target (no API, no logprobs) with a trained LLMmap model.

Workflow for a monitored chat UI:

 1. Print the queries and an answers template:
        python fingerprint.py ./data/pretrained_models/soc --print_queries --template answers.json
 2. Send each query to the target as a normal message, ideally each one in a *fresh*
    conversation (LLMmap assumes independent single-turn queries) and spread over time,
    between real work. Paste each full answer into answers.json.
    Running the whole set in 2-3 independent sessions and putting them in the file as a
    list of lists makes the result noticeably more robust.
 3. Classify:
        python fingerprint.py ./data/pretrained_models/soc --answers answers.json

answers.json may be:
    ["answer to q1", "answer to q2", ...]                        one session
    [["s1 q1", "s1 q2", ...], ["s2 q1", ...]]                     several sessions
    {"<query text>": "answer", ...}  or  [{"<query>": "answer"}, ...]

Without --answers an interactive paste session starts (end each answer with a line containing only ###).
"""
import argparse
import json
import sys

from LLMmap.inference import load_LLMmap


def read_multiline(prompt):
    print(prompt)
    lines = []
    for line in sys.stdin:
        if line.rstrip('\n').strip() == '###':
            break
        lines.append(line.rstrip('\n'))
    return '\n'.join(lines)


def to_sessions(obj, queries):
    def from_dict(d):
        missing = [q for q in queries if q not in d]
        if missing:
            raise SystemExit(f"answers file lacks answers for {len(missing)} queries, e.g. {missing[0][:80]!r}")
        return [d[q] for q in queries]

    if isinstance(obj, dict):
        return [from_dict(obj)]
    if obj and isinstance(obj[0], dict):
        return [from_dict(d) for d in obj]
    if obj and isinstance(obj[0], list):
        return obj
    return [obj]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model_dir')
    ap.add_argument('--answers', help='JSON file with the target answers')
    ap.add_argument('--print_queries', action='store_true')
    ap.add_argument('--template', help='with --print_queries: write an answers template {query: ""} here')
    ap.add_argument('--sessions', type=int, default=1, help='interactive mode: number of independent sessions to paste')
    ap.add_argument('-k', type=int, default=8, help='how many candidates to print')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    conf, inf = load_LLMmap(args.model_dir, device=args.device, verbose=False)
    queries = inf.queries

    if args.print_queries:
        for i, q in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {q}\n")
        if args.template:
            with open(args.template, 'w') as f:
                json.dump({q: "" for q in queries}, f, indent=2, ensure_ascii=False)
            print(f"Template written to {args.template}")
        return

    if args.answers:
        with open(args.answers) as f:
            sessions = to_sessions(json.load(f), queries)
        empty = [(s, i) for s, ans in enumerate(sessions) for i, a in enumerate(ans) if not str(a).strip()]
        if empty:
            print(f"[warning] {len(empty)} empty answers (session, query): {empty[:5]}")
    else:
        print(f"Send each query to the target (fresh conversation each time), paste the full answer, "
              f"then a line with only ### .")
        sessions = []
        for s in range(args.sessions):
            print(f"\n===== session {s + 1}/{args.sessions} =====")
            sessions.append([read_multiline(f"\n[Query {i}/{len(queries)}]\n{q}\n\n[Answer]:")
                             for i, q in enumerate(queries, 1)])

    print(f"\n### RESULTS ({len(sessions)} session(s), {len(getattr(inf, 'llms_supported', []) or inf.label_map)} known LLMs) ###")
    out = inf(sessions)
    if conf['is_open']:
        inf.print_result(out, k=args.k)
        if len(sessions) > 1:
            print("\nPer-session top-1 (consistency check):")
            for i, s in enumerate(sessions, 1):
                d = inf([s])
                print(f"\tsession {i}: {inf.label_map[int(d.argmin())]}  ({d.min():.4f})")
    else:
        inf.print_result(out, k=args.k)


if __name__ == '__main__':
    main()
