"""Response normalisation shared by training and inference.

When answers are copied out of a chat UI, markdown is already *rendered*:
`**bold**` becomes bold, `# Title` becomes a heading, code fences disappear,
bullets become "•" or nothing. Raw API responses used for training still carry
the markdown syntax. Embedding the two forms differently is a train/test
distribution shift, so both sides go through `normalize_response` which maps
raw and rendered markdown to (approximately) the same plain text.

The same function also applies the character budget, so the training traces
and the live answers are cut at exactly the same point.
"""

import re
import unicodedata

_FENCE = re.compile(r"^\s*(```|~~~)[^\n]*$", re.M)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+•●▪◦‣–]|\d+[.)])\s+", re.M)
_BLOCKQUOTE = re.compile(r"^\s*>+\s?", re.M)
_HR = re.compile(r"^\s*([-*_=])(\s*\1){2,}\s*$", re.M)
# emphasis markers only count at word boundaries, so svc_backup / file_name / 2*3*4 survive
_EMPH = re.compile(r"(?<![\w*~])(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1(?![\w*~])", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", re.M)
_TABLE_PIPE = re.compile(r"\s*\|\s*")
_HTML_TAG = re.compile(r"</?(br|p|div|span|strong|em|b|i|code|pre|ul|ol|li)\b[^>]*>", re.I)
_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n\s*\n+")


def strip_markdown(text: str) -> str:
    t = text.replace('\r\n', '\n')
    t = _FENCE.sub('', t)
    t = _HR.sub('', t)
    t = _TABLE_SEP.sub('', t)
    t = _HEADING.sub('', t)
    t = _BLOCKQUOTE.sub('', t)
    t = _BULLET.sub('', t)
    t = _LINK.sub(r'\1', t)
    t = _INLINE_CODE.sub(r'\1', t)
    for _ in range(2):  # nested emphasis like ***x***
        t = _EMPH.sub(r'\2', t)
    t = _HTML_TAG.sub(' ', t)
    t = '\n'.join(_TABLE_PIPE.sub(' ', line).strip() if line.count('|') >= 2 else line for line in t.split('\n'))
    return t


def normalize_response(text, max_chars=None, markdown=True):
    if text is None:
        text = ''
    t = unicodedata.normalize('NFKC', str(text))
    if markdown:
        t = strip_markdown(t)
    t = '\n'.join(_WS.sub(' ', line).strip() for line in t.split('\n'))
    t = _NL.sub('\n', t).strip()
    if max_chars:
        t = t[:max_chars]
    return t


def normalizer_from_conf(conf):
    max_chars = conf.get('max_number_chars_response')
    markdown = conf.get('normalize_markdown', False)
    if conf.get('normalize_text', markdown):
        return lambda s: normalize_response(s, max_chars=max_chars, markdown=markdown)
    # legacy behaviour: plain truncation only
    return lambda s: (s or '')[:max_chars] if max_chars else (s or '')
