# wrapper/memory/json_store.py
from __future__ import annotations
import json, time, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from .store import MemoryStore, Message

MEM_DIR = Path(".memory")

class JSONMemoryStore(MemoryStore):
    def __init__(self) -> None:
        MEM_DIR.mkdir(exist_ok=True)

    # ---------- internals ----------
    def _path(self, user_id: str) -> Path:
        return MEM_DIR / f"{user_id}.json"

    def _blank(self) -> Dict[str, Any]:
        return {"threads": {}, "profile": {}, "profiles": {}, "examples": {}, "examples_counter": 0}

    def _load(self, user_id: str) -> Dict[str, Any]:
        p = self._path(user_id)
        if not p.exists():
            return self._blank()
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            backup = p.with_suffix(f".{int(time.time())}.bak")
            try: p.rename(backup)
            except Exception: pass
            return self._blank()

    def _save(self, user_id: str, data: Dict[str, Any]) -> None:
        p = self._path(user_id)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    
    # ---------- short-term ----------
    def get_window(self, user_id: str, conversation_id: str, k: int) -> List[Message]:
        d = self._load(user_id)
        t = d.setdefault("threads", {}).setdefault(conversation_id, {"messages": [], "summary": ""})
        msgs: List[Message] = t.get("messages", [])
        return msgs[-k:] if k and k > 0 else msgs

    def append_message(self, user_id: str, conversation_id: str, role: str, content: str) -> None:
        d = self._load(user_id)
        t = d.setdefault("threads", {}).setdefault(conversation_id, {"messages": [], "summary": ""})
        t["messages"].append({"role": role, "content": content, "ts": int(time.time())})
        self._save(user_id, d)

    # ---------- profile (legacy default) ----------
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        return self._load(user_id).get("profile", {}) or {}

    def upsert_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        d = self._load(user_id)
        p = d.setdefault("profile", {})
        for k, v in (updates or {}).items():
            p[k] = v
        self._save(user_id, d)
        return p

    # ---------- named profiles ----------
    def get_profile_named(self, user_id: str, profile_name: str) -> Dict[str, Any]:
        d = self._load(user_id)
        return d.setdefault("profiles", {}).get(profile_name, {}) or {}

    def upsert_profile_named(self, user_id: str, profile_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        d = self._load(user_id)
        profs = d.setdefault("profiles", {})
        p = profs.setdefault(profile_name, {})
        for k, v in (updates or {}).items():
            p[k] = v
        # ensure fields used by wrapper exist (but keep thin)
        p.setdefault("example_ids", [])
        p.setdefault("example_tags", [])
        self._save(user_id, d)
        return p

    # ---------- examples (global) ----------
    def save_example(self, user_id: str, tags: List[str], input_text: str, output_text: str, title: Optional[str] = None) -> str:
        d = self._load(user_id)
        ex_store: Dict[str, Any] = d.setdefault("examples", {})
        d["examples_counter"] = int(d.get("examples_counter", 0)) + 1
        ex_num = d["examples_counter"]
        ex_id = f"ex_{ex_num:06d}"  # deterministic numeric id
        title = title or self._auto_title(input_text, output_text)
        ex_store[ex_id] = {
            "num": ex_num,
            "tags": sorted({t.strip() for t in (tags or []) if t.strip()}),
            "title": title,
            "input": (input_text or "").strip(),
            "output": (output_text or "").strip(),
            "ts": int(time.time()),
        }
        self._save(user_id, d)
        return ex_id

    def get_examples_by_tags(self, user_id: str, tags: List[str], limit: int = 2) -> List[Dict[str, Any]]:
        d = self._load(user_id)
        ex_store: Dict[str, Any] = d.get("examples", {}) or {}
        tagset = {t.strip() for t in (tags or []) if t.strip()}
        scored = []
        for ex_id, ex in ex_store.items():
            overlap = len(tagset.intersection(set(ex.get("tags", []))))
            if overlap == 0:
                continue
            scored.append((overlap, ex.get("ts", 0), ex_id, ex))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [{"id": ex_id, **ex} for _, _, ex_id, ex in scored[: max(1, limit)]]

    def get_examples_by_ids(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids: return []
        d = self._load(user_id)
        ex_store: Dict[str, Any] = d.get("examples", {}) or {}
        out = []
        for ex_id in ids:
            ex = ex_store.get(ex_id)
            if ex:
                out.append({"id": ex_id, **ex})
        return out

    # ---------- profile-example linking ----------
    def link_examples_to_profile(self, user_id: str, profile_name: str, example_ids: List[str], mode: str = "append") -> List[str]:
        d = self._load(user_id)
        p = d.setdefault("profiles", {}).setdefault(profile_name, {"example_ids": [], "example_tags": []})
        current = p.get("example_ids", []) or []
        if mode == "set":
            p["example_ids"] = list(dict.fromkeys(example_ids))  # de-dupe, preserve order
        else:
            p["example_ids"] = list(dict.fromkeys(current + list(example_ids)))
        self._save(user_id, d)
        return p["example_ids"]

    def get_examples_for_profile(self, user_id: str, profile_name: str, limit: int = 2) -> List[Dict[str, Any]]:
        prof = self.get_profile_named(user_id, profile_name)
        # priority: explicit example_ids, then example_tags
        ids = prof.get("example_ids") or []
        examples = self.get_examples_by_ids(user_id, ids)
        if len(examples) < limit:
            tags = prof.get("example_tags") or []
            more = self.get_examples_by_tags(user_id, tags, limit=limit - len(examples))
            # merge, avoiding dupes by id
            have = {e["id"] for e in examples}
            for e in more:
                if e["id"] not in have:
                    examples.append(e)
        return examples[:limit]

    # ---------- thread summary ----------
    def get_thread_summary(self, user_id: str, conversation_id: str) -> str:
        d = self._load(user_id)
        t = d.setdefault("threads", {}).setdefault(conversation_id, {"messages": [], "summary": ""})
        return t.get("summary") or ""

    def set_thread_summary(self, user_id: str, conversation_id: str, summary: str) -> None:
        d = self._load(user_id)
        t = d.setdefault("threads", {}).setdefault(conversation_id, {"messages": [], "summary": ""})
        t["summary"] = summary or ""
        self._save(user_id, d)

    def get_thread_length(self, user_id: str, conversation_id: str) -> int:
        d = self._load(user_id)
        t = d.setdefault("threads", {}).setdefault(conversation_id, {"messages": [], "summary": ""})
        return len(t.get("messages", []))

    # ---------- helpers ----------
    @staticmethod
    def _auto_title(inp: str, out: str, max_words: int = 12) -> str:
        text = (inp or "").strip() or (out or "").strip()
        words = text.replace("\n", " ").split()
        return " ".join(words[:max_words]) if words else "Example"
