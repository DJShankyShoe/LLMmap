import os
import hashlib
import random
from typing import Dict, List, Any, Optional

import torch
import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler

from .dataset_maker import read_dataset
from .embedding_model import load_model
from .text_norm import normalizer_from_conf


def _h(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8', 'surrogatepass')).hexdigest()


class EmbeddingCache:
    """Embeds every (normalised) query/response string once.

    With `disk_path` set, embeddings are persisted (keyed by model + text hash) so that
    re-training, template building and query selection don't re-embed the corpus.
    """

    def __init__(self, emb_model, batch_size: int = 128, normalizer=None, disk_path: Optional[str] = None) -> None:
        self.batch_size = max(1, batch_size)
        self._cache: Dict[str, torch.Tensor] = {}
        self.emb_model = emb_model
        self.normalizer = normalizer or (lambda s: s)
        self.llms_map = None
        self.queries = None
        self.embedding_size = None
        self.llms = set()
        self.disk_path = disk_path
        self._disk: Dict[str, torch.Tensor] = {}
        if disk_path and os.path.isfile(disk_path):
            self._disk = torch.load(disk_path, map_location='cpu')
        self._disk_dirty = False

    def get_embedding(self, texts: List[str]):
        emb = self.emb_model.get_embedding(texts).cpu()
        if self.embedding_size is None:
            self.embedding_size = emb.shape[-1]
        return emb

    def precompute(self, dataset, query_idx=None) -> None:
        pending: List[str] = []
        pending_set = set()

        def flush():
            if pending:
                embs = self.get_embedding(pending)
                for t, e in zip(pending, embs):
                    self._cache[t] = e
                    if self.disk_path is not None:
                        self._disk[_h(t)] = e.half()
                        self._disk_dirty = True
                pending.clear()
                pending_set.clear()

        def handle_one(t: str):
            if t in self._cache or t in pending_set:
                return
            if self.disk_path is not None:
                hit = self._disk.get(_h(t))
                if hit is not None:
                    self._cache[t] = hit.float()
                    if self.embedding_size is None:
                        self.embedding_size = hit.shape[-1]
                    return
            pending.append(t)
            pending_set.add(t)
            if len(pending) >= self.batch_size:
                flush()

        for entry in tqdm.tqdm(dataset, desc='embedding'):
            self.add_llm(entry['llm'])
            traces = select_traces(entry['traces'], query_idx)
            queries = [t[0] for t in traces]
            if self.queries is None:
                self.queries = queries
            elif self.queries != queries:
                raise ValueError("Inconsistent query sets across dataset entries")
            for query, resp in traces:
                handle_one(query)
                handle_one(self.normalizer(resp))
        flush()
        self.set_llms_map()
        self.save()

    def save(self):
        if self.disk_path and self._disk_dirty:
            os.makedirs(os.path.dirname(os.path.abspath(self.disk_path)), exist_ok=True)
            torch.save(self._disk, self.disk_path)
            self._disk_dirty = False

    def add_llm(self, llm):
        self.llms.add(llm)

    def __call__(self, key):
        return self._cache[key]

    def set_llms_map(self):
        llms = sorted(self.llms)
        self.llms_map = dict(zip(llms, range(len(llms))))


def select_traces(traces, query_idx):
    if query_idx is None:
        return traces
    return [traces[i] for i in query_idx]


def resolve_query_subset(conf, all_queries):
    """`conf['query_subset']` may list indices or query strings of the dataset's query pool."""
    subset = conf.get('query_subset')
    if not subset:
        return None
    idx = []
    for q in subset:
        if isinstance(q, int):
            idx.append(q)
        else:
            idx.append(all_queries.index(q))
    return idx


# ---------------------------------------------------------------------

class DatasetFactory(Dataset):
    def __init__(self, dataset_raw, cache, *args, query_idx=None, **k):
        self.dataset_raw = dataset_raw
        self.cache = cache
        self.query_idx = query_idx
        self.num_labels = len(self.cache.llms_map)
        self.labels = [self.cache.llms_map[e['llm']] for e in dataset_raw]
        # normalise once (regex work), not on every __getitem__
        norm = self.cache.normalizer
        self.keys = [[(q, norm(o)) for q, o in select_traces(e['traces'], query_idx)] for e in dataset_raw]

    def __len__(self):
        return len(self.dataset_raw)

    def pack(self, idx):
        return torch.stack([torch.cat([self.cache(q), self.cache(o)]) for q, o in self.keys[idx]])

    def __getitem__(self, idx):
        return self.pack(idx), self.labels[idx]


# ---------------------------------------------------------------------

class DatasetFactorySiamese(DatasetFactory):
    """Random positive / negative pairs.

    The original implementation reseeded python's global RNG with `idx` on every item, so every
    epoch saw exactly the same pairs. Now training pairs are fresh each epoch; evaluation pairs
    stay fixed (deterministic=True) so val_loss is comparable across epochs.
    """

    def __init__(self, dataset_raw, cache, num_pairs_per_epoch, *args, deterministic=False, seed=0, **kargs):
        super().__init__(dataset_raw, cache, *args, **kargs)
        self.num_pairs_per_epoch = num_pairs_per_epoch
        self.deterministic = deterministic
        self.seed = seed
        self.traces_per_llm = [[] for _ in range(self.num_labels)]
        for i, label_id in enumerate(self.labels):
            self.traces_per_llm[label_id].append(i)
        self.present = [c for c in range(self.num_labels) if self.traces_per_llm[c]]
        self._rng = random.Random(seed)

    def __len__(self):
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx) if self.deterministic else self._rng

        llm_a = rng.choice(self.present)
        trace_a_id = rng.choice(self.traces_per_llm[llm_a])

        if rng.random() < 0.5 and len(self.traces_per_llm[llm_a]) > 1:
            trace_b_id = trace_a_id
            while trace_b_id == trace_a_id:
                trace_b_id = rng.choice(self.traces_per_llm[llm_a])
            label = 1  # same LLM
        else:
            llm_b = llm_a
            while llm_b == llm_a:
                llm_b = rng.choice(self.present)
            trace_b_id = rng.choice(self.traces_per_llm[llm_b])
            label = 0  # different LLMs

        trace_a = self.pack(trace_a_id)
        trace_b = self.pack(trace_b_id)
        return torch.stack([trace_a, trace_b]), label


class PKBatchSampler(Sampler):
    """Batches of P classes x K samples each (for batch-hard triplet / metric learning)."""

    def __init__(self, labels, p, k, num_batches, seed=0):
        self.by_label = {}
        for i, l in enumerate(labels):
            self.by_label.setdefault(l, []).append(i)
        self.classes = sorted(self.by_label)
        self.p = min(p, len(self.classes))
        self.k = k
        self.num_batches = num_batches
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c in self.rng.sample(self.classes, self.p):
                pool = self.by_label[c]
                if len(pool) >= self.k:
                    batch += self.rng.sample(pool, self.k)
                else:
                    batch += [self.rng.choice(pool) for _ in range(self.k)]
            yield batch


# ---------------------------------------------------------------------

def build_cache(conf, train, test):
    emb_model = load_model(conf['embedding_model_id'])
    cache = EmbeddingCache(emb_model, conf['embedding_batch_size'], normalizer=normalizer_from_conf(conf),
                           disk_path=conf.get('embedding_cache_path'))
    all_queries = [t[0] for t in (train or test)[0]['traces']]
    query_idx = resolve_query_subset(conf, all_queries)
    cache.precompute(train + test, query_idx=query_idx)
    return cache, query_idx


def load_datasets(conf, siamese=True, ks=None, mode=None):
    """mode: 'pairs' (original siamese contrastive), 'triplet' (PK batches), or 'classify'.
    For backwards compatibility `siamese=True` -> 'pairs' unless conf['open_set_loss'] == 'triplet'."""
    if mode is None:
        if siamese:
            mode = 'triplet' if conf.get('open_set_loss', 'contrastive') == 'triplet' else 'pairs'
        else:
            mode = 'classify'

    train, test = read_dataset(conf['dataset_path'], seed=conf.get('seed', 0))
    if ks:
        train, test = train[:ks[0]], test[:ks[1]]
    if conf.get('exclude_llms'):
        ex = set(conf['exclude_llms'])
        train = [e for e in train if e['llm'] not in ex]
        test = [e for e in test if e['llm'] not in ex]

    cache, query_idx = build_cache(conf, train, test)

    train_llms = {e['llm'] for e in train}
    test = [e for e in test if e['llm'] in train_llms]

    if mode == 'pairs':
        dataset_train = DatasetFactorySiamese(train, cache, conf['num_pairs_per_epoch'], query_idx=query_idx)
        dataset_test = DatasetFactorySiamese(test, cache, conf['num_pairs_per_eval'], query_idx=query_idx, deterministic=True)
        loader_train = DataLoader(dataset_train, batch_size=conf['batch_size'], shuffle=False)
        loader_test = DataLoader(dataset_test, batch_size=conf['batch_size'], shuffle=False)
    else:
        dataset_train = DatasetFactory(train, cache, query_idx=query_idx)
        dataset_test = DatasetFactory(test, cache, query_idx=query_idx)
        if mode == 'triplet':
            p, k = conf.get('pk_batch', [32, 4])
            nb = max(1, conf.get('num_pairs_per_epoch', 100000) // (p * k))
            loader_train = DataLoader(dataset_train, batch_sampler=PKBatchSampler(dataset_train.labels, p, k, nb, seed=conf.get('seed', 0)))
            loader_test = DataLoader(dataset_test, batch_sampler=PKBatchSampler(dataset_test.labels, p, k, max(1, len(dataset_test) // (p * k)) * 4, seed=1))
        else:
            loader_train = DataLoader(dataset_train, batch_size=conf['batch_size'], shuffle=True)
            loader_test = DataLoader(dataset_test, batch_size=conf['batch_size'], shuffle=False)

    conf['llms_map'] = cache.llms_map
    conf['queries'] = cache.queries
    conf['inference_model']['num_classes'] = dataset_train.num_labels
    conf['inference_model']['num_queries'] = len(cache.queries)
    conf['inference_model']['emb_size'] = cache.embedding_size

    return (loader_train, loader_test), cache, (dataset_train, dataset_test)
