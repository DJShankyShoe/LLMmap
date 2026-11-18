import json
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

from .llm import load_llm, parse_llm_spec, llm_label, DEFAULT_MAX_NEW_TOKENS
from .prompt_configuration import TRAIN, TEST

log = logging.getLogger(__name__)


def read_dataset(path, encoding='utf-8', shuffle=True, seed=None):
    import random
    rng = random.Random(seed)
    train, test = [], []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            pool = entry.pop('dataset')
            if pool == TRAIN:
                train.append(entry)
            elif pool == TEST:
                test.append(entry)
    if shuffle:
        rng.shuffle(train)
        rng.shuffle(test)
    return train, test


def _entry(pool, llm_name, conf_id, prompt_conf, traces):
    return {'dataset': pool, 'llm': llm_name, 'conf_id': conf_id,
            'traces': traces, 'prompt_conf': prompt_conf.to_dict()}


def make_dataset_entries_for_new_llm(llm, queries, prompt_confs, pool=TRAIN, workers=1,
                                     max_new_tokens=DEFAULT_MAX_NEW_TOKENS, conf_ids=None, on_entry=None,
                                     max_consecutive_failures=5):
    """Query `llm` with every (prompt_conf, query) pair.

    * HF models: the queries of one prompt conf are generated as one padded batch.
    * API models: (conf, query) calls run on a thread pool of `workers` threads.

    An entry is emitted only when all its queries succeeded (no partial traces).
    `on_entry(entry)` is called as soon as each entry is complete (used for streaming to disk / resume).
    """
    conf_ids = list(conf_ids) if conf_ids is not None else list(range(len(prompt_confs)))
    entries = []

    def emit(e):
        entries.append(e)
        if on_entry:
            on_entry(e)

    if llm.is_hf or not llm.supports_concurrency or workers <= 1:
        consecutive_failures = 0
        for cid, pc in tqdm.tqdm(list(zip(conf_ids, prompt_confs)), desc=llm.llm_name):
            try:
                prompts, params = zip(*(pc(q, llm) for q in queries))
                if llm.is_hf:
                    outs = llm.generate_batch(list(prompts), dict(params[0]), max_new_tokens=max_new_tokens)
                else:
                    outs = [llm.generate(p, dict(params[0]), max_new_tokens=max_new_tokens)[0] for p in prompts]
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                log.error("[%s] conf %s failed: %s", llm.llm_name, cid, e)
                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(f"{llm.llm_name}: {consecutive_failures} consecutive failures, skipping model") from e
                continue
            emit(_entry(pool, llm.llm_name, cid, pc, [list(t) for t in zip(queries, outs)]))
        return entries

    # ---- concurrent path (API backends) --------------------------------------------------------
    results = defaultdict(dict)
    failed = set()
    lock = threading.Lock()
    n_q = len(queries)
    by_cid = dict(zip(conf_ids, prompt_confs))
    stats = {'fail': 0, 'ok': 0}

    def job(cid, qi):
        pc = by_cid[cid]
        prompt, params = pc(queries[qi], llm)
        return cid, qi, llm.generate(prompt, dict(params), max_new_tokens=max_new_tokens)[0]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, cid, qi) for cid in conf_ids for qi in range(n_q)]
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc=llm.llm_name):
            try:
                cid, qi, out = fut.result()
            except Exception as e:
                with lock:
                    stats['fail'] += 1
                    # bail out early on a model that fails systematically (wrong id, no access, ...)
                    if stats['ok'] == 0 and stats['fail'] >= max_consecutive_failures * 2:
                        for f in futs:
                            f.cancel()
                        raise RuntimeError(f"{llm.llm_name}: every request fails ({e}); skipping model") from e
                log.error("[%s] request failed: %s", llm.llm_name, e)
                continue
            with lock:
                stats['ok'] += 1
                results[cid][qi] = out
                if len(results[cid]) == n_q and cid not in failed:
                    traces = [[queries[i], results[cid][i]] for i in range(n_q)]
                    emit(_entry(pool, llm.llm_name, cid, by_cid[cid], traces))
                    del results[cid]
    if results:
        log.warning("[%s] %d prompt confs incomplete (dropped)", llm.llm_name, len(results))
    return entries


class DatasetMaker:
    """Collect traces for many LLMs into one JSONL.

    Differences w.r.t. the original version:
      * the same (seeded) train / test prompt confs are used for every LLM, so models are compared
        under identical conditions, and test confs really come from the held-out split;
      * resumable: entries already in `output_path` (matched by llm + pool + conf_id) are skipped;
      * concurrent API calls, batched HF generation, retries, and a failing model doesn't kill the run.
    """

    def __init__(self, pc, llms, queries, num_prompt_conf_train, num_prompt_conf_test, output_path,
                 encoding='utf8', workers=8, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        self.pc = pc
        self.llms = llms
        self.queries = queries
        self.num_prompt_conf_train = num_prompt_conf_train
        self.num_prompt_conf_test = num_prompt_conf_test
        self.output_path = output_path
        self.encoding = encoding
        self.workers = workers
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()

        self.train_confs = pc.sample(num_prompt_conf_train, pool=TRAIN) if num_prompt_conf_train else []
        self.test_confs = pc.sample(num_prompt_conf_test, pool=TEST) if num_prompt_conf_test else []
        self.done = self._scan_existing()
        self.failed_llms = []

    def _scan_existing(self):
        done = defaultdict(set)
        try:
            with open(self.output_path, 'r', encoding=self.encoding) as f:
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if [t[0] for t in e['traces']] != self.queries:
                        raise ValueError(f"{self.output_path} was built with a different query set; use a new dataset name")
                    done[(e['llm'], e['dataset'])].add(e.get('conf_id'))
        except FileNotFoundError:
            pass
        return done

    def _todo(self, label, pool, confs):
        done = self.done.get((label, pool), set())
        return [(i, c) for i, c in enumerate(confs) if i not in done]

    def plan(self):
        """[(label, n_train_missing, n_test_missing)] -- what a run would still query."""
        out = []
        for spec in self.llms:
            label = llm_label(spec)
            out.append((label, len(self._todo(label, TRAIN, self.train_confs)), len(self._todo(label, TEST, self.test_confs))))
        return out

    def run_on_an_llm(self, spec):
        name, backend, extra = parse_llm_spec(spec)
        label = extra.get('label') or name
        todo = {TRAIN: self._todo(label, TRAIN, self.train_confs), TEST: self._todo(label, TEST, self.test_confs)}
        if not todo[TRAIN] and not todo[TEST]:
            print(f"[skip] {label}: already complete")
            return
        print(f"Loading {label} ({backend})...")
        llm = load_llm(name, backend, **extra)
        for pool in (TRAIN, TEST):
            if not todo[pool]:
                continue
            ids, confs = zip(*todo[pool])
            print(f"\tRunning on {label} {pool} ({len(ids)} prompt confs x {len(self.queries)} queries)...")
            make_dataset_entries_for_new_llm(
                llm, self.queries, list(confs), pool=pool, workers=self.workers,
                max_new_tokens=self.max_new_tokens, conf_ids=ids, on_entry=self.dump_one)
        del llm

    def __call__(self):
        for spec in self.llms:
            try:
                self.run_on_an_llm(spec)
            except Exception as e:
                label = llm_label(spec)
                log.error("[%s] skipped: %s", label, e)
                print(f"[ERROR] {label} skipped: {e}")
                self.failed_llms.append((label, str(e)))
        if self.failed_llms:
            print("\nModels that failed (re-run the same command to retry them):")
            for label, err in self.failed_llms:
                print(f"\t{label}: {err[:200]}")

    def dump_one(self, entry):
        with self._lock, open(self.output_path, 'a', encoding=self.encoding) as f:
            print(json.dumps(entry, ensure_ascii=False), file=f)

    def dump(self, entries):
        for e in entries:
            self.dump_one(e)
