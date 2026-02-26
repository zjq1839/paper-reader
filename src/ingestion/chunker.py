import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class TokenChunk:
    text: str
    start_token: int
    end_token: int
    index: int


def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


def _iter_tokens(text: str) -> Iterable[Tuple[str, int, int]]:
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if _is_cjk(ch):
            yield ch, i, i + 1
            i += 1
            continue
        if ch.isalnum() or ch == "_":
            j = i + 1
            while j < n:
                cj = text[j]
                if cj.isalnum() or cj == "_":
                    j += 1
                    continue
                break
            yield text[i:j], i, j
            i = j
            continue
        yield ch, i, i + 1
        i += 1


def chunk_text_by_tokens(
    text: str,
    chunk_tokens: int = 600,
    overlap_tokens: int = 100,
) -> List[TokenChunk]:
    raw = (text or "").strip()
    if not raw:
        return []
    if chunk_tokens <= 0:
        return []
    if overlap_tokens < 0:
        overlap_tokens = 0
    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens - 1)

    tokens: List[Tuple[str, int, int]] = list(_iter_tokens(raw))
    if not tokens:
        return []

    chunks: List[TokenChunk] = []
    step = max(1, chunk_tokens - overlap_tokens)
    total = len(tokens)
    idx = 0
    start = 0
    while start < total:
        end = min(total, start + chunk_tokens)
        start_char = tokens[start][1]
        end_char = tokens[end - 1][2]
        chunk_str = raw[start_char:end_char].strip()
        if chunk_str:
            chunks.append(TokenChunk(text=chunk_str, start_token=start, end_token=end, index=idx))
            idx += 1
        if end >= total:
            break
        start += step
    return chunks


def get_chunking_config_from_env() -> Tuple[int, int]:
    size = int(os.environ.get("INGEST_CHUNK_TOKENS", "600"))
    overlap = int(os.environ.get("INGEST_CHUNK_OVERLAP_TOKENS", "100"))
    return size, overlap

