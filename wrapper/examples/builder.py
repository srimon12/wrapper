# wrapper/examples/builder.py
from __future__ import annotations
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

STOPWORDS = {
    "a","an","the","and","or","but","if","to","of","in","for","on","with","as","by","at",
    "is","are","was","were","be","been","being","this","that","it","its","from","into",
    "about","over","after","before","between","within","without","via","per","not"
}

def _clean(s: str) -> str:
    return (s or "").strip()

def _auto_title(inp: str, out: str, max_words: int = 12) -> str:
    base = _clean(inp) or _clean(out)
    words = re.sub(r"\s+", " ", base).split()
    return " ".join(words[:max_words]) if words else "Example"

def _compress_output(s: str, budget_chars: int = 400) -> str:
    s = _clean(s)
    # keep at most one fenced block fragment, cap by budget
    s = re.sub(r"```[\s\S]*?```", lambda m: m.group(0)[:budget_chars], s)
    if len(s) <= budget_chars:
        return s
    return s[:budget_chars].rstrip() + " …"

def _local_tags(text: str, k: int = 5) -> List[str]:
    text = text.lower()
    toks = [t for t in re.findall(r"[a-z0-9\-]+", text) if t and not t.isdigit()]
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    bigrams = [" ".join([toks[i], toks[i+1]]) for i in range(len(toks)-1)]
    counts = Counter(toks + bigrams)
    scored = []
    for term, c in counts.items():
        bonus = 1.5 if "-" in term or " " in term else 1.0
        scored.append((c * bonus, term))
    scored.sort(reverse=True)
    out: List[str] = []
    for _, term in scored:
        if term in out:
            continue
        out.append(term)
        if len(out) >= k:
            break
    return out

def suggest_tags_local(*chunks: str, k: int = 5, hints: Optional[List[str]] = None) -> List[str]:
    text = " ".join(_clean(x) for x in chunks if x)
    tags = _local_tags(text, k=k)
    if hints:
        seen = set()
        merged: List[str] = []
        for h in hints:
            h = h.strip()
            if not h or h in seen:
                continue
            merged.append(h); seen.add(h)
        for t in tags:
            if t not in seen:
                merged.append(t); seen.add(t)
        return merged[:k]
    return tags

def build_example_from_segments(
    segments: List[Tuple[str, str]],
    title: Optional[str] = None,
    out_budget_chars: int = 400
) -> Dict[str, str]:
    """Return a compact {'title','input','output'} from chat segments."""
    user_texts = [t for r, t in segments if r == "user" and _clean(t)]
    asst_texts = [t for r, t in segments if r in ("assistant","model") and _clean(t)]
    inp = user_texts[-1] if user_texts else (_clean(segments[-1][1]) if segments else "")
    out = _compress_output(asst_texts[-1] if asst_texts else "", budget_chars=out_budget_chars)
    return {"title": title or _auto_title(inp, out), "input": _clean(inp), "output": _clean(out)}
