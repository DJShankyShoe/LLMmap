"""LLM backends used to collect behavioural traces.

Every backend exposes the same small interface:

    llm.llm_name                     -> label stored in the dataset
    llm.make_prompt(system, user)    -> backend specific prompt object
    llm.generate(prompt, gen_kargs)  -> [str]
    llm.generate_batch(prompts, gen_kargs) -> [str, ...]   (one per prompt)
    llm.is_hf                        -> True only for local Hugging Face models
    llm.supports_concurrency         -> True if generate() can be called from threads

Backends
--------
    hf / 0            local Hugging Face model (transformers)
    openai / 1        OpenAI API
    anthropic / 2     Anthropic API
    openrouter / 3    OpenRouter (one key, hundreds of hosted models)
    openai_compat / 4 any OpenAI-compatible server (vLLM, Ollama, LM Studio,
                      Together, Groq, Fireworks, DeepInfra, Azure OpenAI,
                      Gemini's OpenAI endpoint, ...). Requires `base_url`.

Heavy dependencies (torch / transformers / openai / anthropic) are imported
lazily so that API-only collection does not need a GPU stack and vice-versa.
"""

import os
import time
import random
import logging

DEFAULT_MAX_NEW_TOKENS = 256
CACHE_DIR = os.environ.get('HF_MODEL_CACHE', None)

log = logging.getLogger(__name__)

BACKENDS = {
    'hf': 0, 'huggingface': 0,
    'openai': 1,
    'anthropic': 2,
    'openrouter': 3,
    'openai_compat': 4, 'vllm': 4, 'ollama': 4,
}


class GenerationError(RuntimeError):
    pass


def _retry(fn, *, retries=6, base_delay=2.0, max_delay=60.0, fatal=()):
    """Call fn() with exponential backoff + jitter. Exceptions in `fatal` are re-raised immediately."""
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except fatal:
            raise
        except Exception as e:  # network errors, 429, 5xx, ...
            last = e
            delay = min(max_delay, base_delay * (2 ** attempt)) * (0.5 + random.random())
            log.warning("generation failed (%s: %s), retry %d/%d in %.1fs",
                        type(e).__name__, str(e)[:200], attempt + 1, retries, delay)
            time.sleep(delay)
    raise GenerationError(f"giving up after {retries} attempts: {last}") from last


def _convert_gen_kargs(gen_kargs):
    """Map HF-style sampling params to chat-API params (without mutating the input)."""
    gen_kargs = dict(gen_kargs)
    do_sample = gen_kargs.pop('do_sample', True)
    if not do_sample:
        gen_kargs['temperature'] = 0
    # keys only meaningful for local HF generation
    for k in ('top_k', 'repetition_penalty'):
        gen_kargs.pop(k, None)
    return gen_kargs


class _BaseLLM:
    is_hf = False
    supports_concurrency = True

    def generate_batch(self, prompts, gen_kargs, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        return [self.generate(p, gen_kargs, max_new_tokens=max_new_tokens)[0] for p in prompts]


#####################################################################################################################

class LLM_huggingface(_BaseLLM):
    is_hf = True
    supports_concurrency = False

    def __init__(
        self,
        llm_name,
        model_class=None,
        tokenizer_class=None,
        model_load_kargs=None,
        tokenizer_only=False,
        label=None,
    ):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_class = model_class or AutoModelForCausalLM
        tokenizer_class = tokenizer_class or AutoTokenizer
        model_load_kargs = dict(model_load_kargs or {})

        # Token is only needed for gated repos; don't force it.
        api_key = os.environ.get('HUGGINGFACE_API_KEY') or os.environ.get('HF_TOKEN')

        self.llm_name = label or llm_name
        self.model_class = model_class

        tok_kargs = {k: v for k, v in model_load_kargs.items() if k in ('cache_dir', 'trust_remote_code')}
        self.tokenizer = tokenizer_class.from_pretrained(llm_name, padding_side='left', token=api_key, **tok_kargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = None
        if not tokenizer_only:
            import transformers
            dtype_key = 'dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'
            model_load_kargs.setdefault(dtype_key, 'auto')
            self.model = model_class.from_pretrained(llm_name, token=api_key, **model_load_kargs)
            self.model.eval()

    @staticmethod
    def _does_template_have_system(tokenizer):
        chat_template = getattr(tokenizer, 'chat_template', None)
        if chat_template is None:
            return False
        return "system" in chat_template

    def make_prompt(self, system, user):
        messages = []
        if system:
            if self._does_template_have_system(self.tokenizer):
                messages.append({'role': 'system', 'content': system})
            else:
                user = f'{system}\n\n{user}'
        messages.append({'role': 'user', 'content': user})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _hf_kargs(self, gen_kargs):
        gen_kargs = dict(gen_kargs)
        for k in ('reasoning_effort',):
            gen_kargs.pop(k, None)
        if not gen_kargs.get('do_sample', True):
            gen_kargs.pop('temperature', None)
            gen_kargs.pop('top_p', None)
        return gen_kargs

    def generate_batch(self, prompts, gen_kargs, skip_special_tokens=True, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        import torch
        with torch.no_grad():
            in_toks = self.tokenizer(
                prompts, padding=True, return_tensors="pt",
                add_special_tokens=False, return_token_type_ids=False,
            ).to(self.model.device)
            out_toks = self.model.generate(
                **in_toks,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **self._hf_kargs(gen_kargs),
            )
            n_in = in_toks.input_ids.shape[1]  # left padding -> same offset for all rows
            return self.tokenizer.batch_decode(out_toks[:, n_in:], skip_special_tokens=skip_special_tokens)

    def generate(self, prompt, gen_kargs, skip_special_tokens=True, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        prompts = prompt if isinstance(prompt, list) else [prompt]
        return self.generate_batch(prompts, gen_kargs, skip_special_tokens, max_new_tokens)


#####################################################################################################################

class LLM_OpenAI(_BaseLLM):
    """OpenAI chat-completions API, and anything speaking the same protocol."""

    api_key_env = 'OPENAI_API_KEY'
    default_base_url = None

    def __init__(self, llm_name, base_url=None, api_key_env=None, label=None, extra_body=None,
                 reasoning=False, reasoning_budget=1024, timeout=120, system_role='system'):
        from openai import OpenAI

        api_key_env = api_key_env or self.api_key_env
        api_key = os.environ.get(api_key_env)
        if api_key is None:
            if base_url and any(h in base_url for h in ('localhost', '127.0.0.1', '0.0.0.0')):
                api_key = 'EMPTY'  # local vLLM / Ollama servers don't check it
            else:
                raise RuntimeError(f'Missing API key. Export "{api_key_env}" and try again.')

        self.client = OpenAI(api_key=api_key, base_url=base_url or self.default_base_url, timeout=timeout, max_retries=0)
        self.model_id = llm_name
        self.llm_name = label or llm_name
        self.extra_body = dict(extra_body or {})
        self.reasoning = reasoning
        self.reasoning_budget = reasoning_budget
        self.system_role = system_role
        self._dropped_params = set()

    def make_prompt(self, system, user):
        messages = []
        if system:
            messages.append({'role': self.system_role, 'content': system})
        messages.append({'role': 'user', 'content': user})
        return messages

    def _request(self, prompt, gen_kargs, max_new_tokens):
        import openai
        gen_kargs = _convert_gen_kargs(gen_kargs)
        effort = gen_kargs.pop('reasoning_effort', None)
        for p in self._dropped_params:
            gen_kargs.pop(p, None)

        kargs = dict(model=self.model_id, messages=prompt, **gen_kargs)
        extra_body = dict(self.extra_body)
        if self.reasoning:
            # reasoning tokens count against the budget: leave room for the visible answer
            kargs['max_tokens'] = max_new_tokens + self.reasoning_budget
            if effort:
                self._set_reasoning_effort(kargs, extra_body, effort)
        else:
            kargs['max_tokens'] = max_new_tokens
        if extra_body:
            kargs['extra_body'] = extra_body

        try:
            response = self.client.chat.completions.create(**kargs)
        except openai.BadRequestError as e:
            # Some models reject temperature/top_p (e.g. reasoning models). Drop it once and retry.
            msg = str(e).lower()
            for p in ('temperature', 'top_p'):
                if p in msg and p in kargs and p not in self._dropped_params:
                    log.warning("%s rejected '%s'; dropping it for this model", self.model_id, p)
                    self._dropped_params.add(p)
                    return self._request(prompt, gen_kargs, max_new_tokens)
            raise
        if not response.choices:
            raise GenerationError(f"empty response: {response}")
        return response.choices[0].message.content or ''

    def _set_reasoning_effort(self, kargs, extra_body, effort):
        kargs['reasoning_effort'] = effort

    def generate(self, prompt, gen_kargs, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        import openai
        fatal = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError, openai.BadRequestError)
        return [_retry(lambda: self._request(prompt, gen_kargs, max_new_tokens), fatal=fatal)]


class LLM_OpenRouter(LLM_OpenAI):
    api_key_env = 'OPENROUTER_API_KEY'
    default_base_url = 'https://openrouter.ai/api/v1'

    def _set_reasoning_effort(self, kargs, extra_body, effort):
        # OpenRouter's unified reasoning parameter; `exclude` keeps the chain of thought out of `content`
        extra_body['reasoning'] = {'effort': effort, 'exclude': True}


#####################################################################################################################

class LLM_Anthropic(_BaseLLM):
    def __init__(self, llm_name, label=None, **_):
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise RuntimeError('Missing Anthropic API key. Export "ANTHROPIC_API_KEY" and try again.')
        self.client = Anthropic(api_key=api_key, max_retries=0)
        self.model_id = llm_name
        self.llm_name = label or llm_name

    def make_prompt(self, system, user):
        return (system, [{'role': 'user', 'content': user}])

    def _request(self, prompt, gen_kargs, max_new_tokens):
        system, messages = prompt
        gen_kargs = _convert_gen_kargs(gen_kargs)
        gen_kargs.pop('reasoning_effort', None)
        kargs = dict(max_tokens=max_new_tokens, messages=messages, model=self.model_id, **gen_kargs)
        if system:
            kargs['system'] = system
        message = self.client.messages.create(**kargs)
        return ''.join(b.text for b in message.content if getattr(b, 'type', '') == 'text')

    def generate(self, prompt, gen_kargs, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        import anthropic
        fatal = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError, anthropic.BadRequestError)
        return [_retry(lambda: self._request(prompt, gen_kargs, max_new_tokens), fatal=fatal)]


#####################################################################################################################

def parse_llm_spec(spec):
    """Accepts the legacy ["name", type] pair or a dict
    {"name": ..., "backend": ..., "label": ..., "base_url": ..., "api_key_env": ..., "reasoning": bool, ...}.
    Returns (name, backend_id, extra_kargs)."""
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop('name')
        backend = spec.pop('backend', 'hf')
        spec.pop('family', None)
        spec.pop('pricing', None)
        spec.pop('note', None)
        return name, backend, spec
    name, backend = spec[0], spec[1]
    return name, backend, {}


def llm_label(spec):
    name, _, extra = parse_llm_spec(spec)
    return extra.get('label') or name


def load_llm(llm_name, llm_type, cache_dir=CACHE_DIR, **kargs):
    if isinstance(llm_type, str):
        if llm_type not in BACKENDS:
            raise ValueError(f"Unknown backend '{llm_type}'. Known: {sorted(BACKENDS)}")
        llm_type = BACKENDS[llm_type]

    if llm_type == 0:
        mlk = {'device_map': "auto", 'cache_dir': cache_dir, 'trust_remote_code': True}
        mlk.update(kargs.pop('model_load_kargs', {}))
        return LLM_huggingface(llm_name, model_load_kargs=mlk, label=kargs.get('label'))
    if llm_type == 1:
        return LLM_OpenAI(llm_name, **kargs)
    if llm_type == 2:
        return LLM_Anthropic(llm_name, **kargs)
    if llm_type == 3:
        return LLM_OpenRouter(llm_name, **kargs)
    if llm_type == 4:
        if 'base_url' not in kargs:
            raise ValueError("openai_compat backend requires 'base_url'")
        kargs.setdefault('api_key_env', 'OPENAI_COMPAT_API_KEY')
        return LLM_OpenAI(llm_name, **kargs)
    raise ValueError(f"Unknown llm_type {llm_type}")
