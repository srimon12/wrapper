# wrapper/memory/sqlite_store.py
from __future__ import annotations
import sqlite3, time, json, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .store import MemoryStore, Message
from wrapper.config import SQLITE_DB_PATH, ENABLE_SQLITE_LOGGING

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS threads(
  user_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  summary TEXT DEFAULT '',
  PRIMARY KEY(user_id, conversation_id)
);
CREATE TABLE IF NOT EXISTS messages(
  user_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(user_id, conversation_id, ts);

CREATE TABLE IF NOT EXISTS profiles(
  user_id TEXT NOT NULL,
  profile JSON DEFAULT '{}',
  PRIMARY KEY(user_id)
);
CREATE TABLE IF NOT EXISTS profiles_named(
  user_id TEXT NOT NULL,
  profile_name TEXT NOT NULL,
  profile JSON NOT NULL,
  PRIMARY KEY(user_id, profile_name)
);

CREATE TABLE IF NOT EXISTS examples(
  user_id TEXT NOT NULL,
  ex_id TEXT NOT NULL,
  ex_num INTEGER NOT NULL,
  title TEXT NOT NULL,
  input TEXT NOT NULL,
  output TEXT NOT NULL,
  ts INTEGER NOT NULL,
  PRIMARY KEY(user_id, ex_id)
);
CREATE INDEX IF NOT EXISTS idx_examples_user_num ON examples(user_id, ex_num);

CREATE TABLE IF NOT EXISTS example_tags(
  user_id TEXT NOT NULL,
  ex_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  PRIMARY KEY(user_id, ex_id, tag)
);

CREATE TABLE IF NOT EXISTS profile_examples(
  user_id TEXT NOT NULL,
  profile_name TEXT NOT NULL,
  ex_id TEXT NOT NULL,
  PRIMARY KEY(user_id, profile_name, ex_id)
);

CREATE TABLE IF NOT EXISTS counters(
  user_id TEXT PRIMARY KEY,
  examples_counter INTEGER NOT NULL
);

-- optional metrics/events sink
CREATE TABLE IF NOT EXISTS events_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  user_id TEXT,
  conversation_id TEXT,
  event TEXT NOT NULL,
  meta JSON
);
"""

class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or SQLITE_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=OFF;")
        self.conn.executescript(SCHEMA)

    # ---------- helpers ----------
    def _now(self) -> int:
        return int(time.time())

    def _log(self, event: str, meta: Dict[str, Any] | None = None,
             user_id: Optional[str] = None, conversation_id: Optional[str] = None):
        if not ENABLE_SQLITE_LOGGING:
            return
        self.conn.execute(
            "INSERT INTO events_log(ts,user_id,conversation_id,event,meta) VALUES(?,?,?,?,?)",
            (self._now(), user_id, conversation_id, event, json.dumps(meta or {})),
        )
        self.conn.commit()

    # ---------- short-term ----------
    def get_window(self, user_id: str, conversation_id: str, k: int) -> List[Message]:
        cur = self.conn.execute(
            "SELECT role, content, ts FROM messages WHERE user_id=? AND conversation_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, conversation_id, max(1, k)),
        )
        rows = cur.fetchall()
        return [{"role": r, "content": c, "ts": t} for (r, c, t) in reversed(rows)]

    def append_message(self, user_id: str, conversation_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages(user_id,conversation_id,ts,role,content) VALUES(?,?,?,?,?)",
            (user_id, conversation_id, self._now(), role, content),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO threads(user_id,conversation_id,summary) VALUES(?,?,?)",
            (user_id, conversation_id, ""),
        )
        self.conn.commit()
        self._log("append_message", {"role": role, "len": len(content)}, user_id, conversation_id)

    # ---------- profiles ----------
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        cur = self.conn.execute("SELECT profile FROM profiles WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else {}

    def upsert_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        cur = self.conn.execute("SELECT profile FROM profiles WHERE user_id=?", (user_id,))
        prof = json.loads(cur.fetchone()[0]) if cur.fetchone() else {}
        prof.update(updates or {})
        self.conn.execute("INSERT OR REPLACE INTO profiles(user_id,profile) VALUES(?,?)",
                          (user_id, json.dumps(prof)))
        self.conn.commit()
        return prof

    def get_profile_named(self, user_id: str, profile_name: str) -> Dict[str, Any]:
        cur = self.conn.execute(
            "SELECT profile FROM profiles_named WHERE user_id=? AND profile_name=?",
            (user_id, profile_name),
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else {}

    def upsert_profile_named(self, user_id: str, profile_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        cur = self.conn.execute(
            "SELECT profile FROM profiles_named WHERE user_id=? AND profile_name=?",
            (user_id, profile_name),
        )
        row = cur.fetchone()
        prof = json.loads(row[0]) if row else {}
        prof.update(updates or {})
        # ensure keys used by wrapper exist
        prof.setdefault("example_ids", [])
        prof.setdefault("example_tags", [])
        self.conn.execute(
            "INSERT OR REPLACE INTO profiles_named(user_id,profile_name,profile) VALUES(?,?,?)",
            (user_id, profile_name, json.dumps(prof)))
        self.conn.commit()
        return prof

    # ---------- examples ----------
    def _next_example_num(self, user_id: str) -> int:
        cur = self.conn.execute("SELECT examples_counter FROM counters WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        n = (row[0] if row else 0) + 1
        self.conn.execute("INSERT OR REPLACE INTO counters(user_id,examples_counter) VALUES(?,?)", (user_id, n))
        return n

    def save_example(self, user_id: str, tags: List[str], input_text: str, output_text: str, title: Optional[str] = None) -> str:
        n = self._next_example_num(user_id)
        ex_id = f"ex_{n:06d}"
        ts = self._now()
        self.conn.execute(
            "INSERT INTO examples(user_id,ex_id,ex_num,title,input,output,ts) VALUES(?,?,?,?,?,?,?)",
            (user_id, ex_id, n, title or "", input_text.strip(), output_text.strip(), ts),
        )
        for t in (tags or []):
            self.conn.execute("INSERT OR IGNORE INTO example_tags(user_id,ex_id,tag) VALUES(?,?,?)",
                              (user_id, ex_id, t))
        self.conn.commit()
        self._log("save_example", {"ex_id": ex_id, "tags": tags}, user_id)
        return ex_id

    def get_examples_by_tags(self, user_id: str, tags: List[str], limit: int = 2) -> List[Dict[str, Any]]:
        if not tags:
            return []
        q = """
        SELECT e.ex_id, e.title, e.input, e.output, e.ts
        FROM examples e
        JOIN example_tags t ON e.user_id=t.user_id AND e.ex_id=t.ex_id
        WHERE e.user_id=? AND t.tag IN ({})
        GROUP BY e.ex_id ORDER BY MAX(e.ts) DESC LIMIT ?
        """.format(",".join(["?"] * len(tags)))
        cur = self.conn.execute(q, (user_id, *tags, max(1, limit)))
        rows = cur.fetchall()
        return [{"id": ex_id, "title": ttl, "input": inp, "output": out, "ts": ts} for (ex_id, ttl, inp, out, ts) in rows]

    def get_examples_by_ids(self, user_id: str, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        q = "SELECT ex_id, title, input, output, ts FROM examples WHERE user_id=? AND ex_id IN ({})".format(
            ",".join(["?"] * len(ids)))
        cur = self.conn.execute(q, (user_id, *ids))
        rows = cur.fetchall()
        return [{"id": ex_id, "title": ttl, "input": inp, "output": out, "ts": ts} for (ex_id, ttl, inp, out, ts) in rows]

    def link_examples_to_profile(self, user_id: str, profile_name: str, example_ids: List[str], mode: str = "append") -> List[str]:
        if mode == "set":
            self.conn.execute("DELETE FROM profile_examples WHERE user_id=? AND profile_name=?", (user_id, profile_name))
        for ex_id in example_ids:
            self.conn.execute("INSERT OR IGNORE INTO profile_examples(user_id,profile_name,ex_id) VALUES(?,?,?)",
                              (user_id, profile_name, ex_id))
        self.conn.commit()
        cur = self.conn.execute("SELECT ex_id FROM profile_examples WHERE user_id=? AND profile_name=?", (user_id, profile_name))
        return [r[0] for r in cur.fetchall()]

    def get_examples_for_profile(self, user_id: str, profile_name: str, limit: int = 2) -> List[Dict[str, Any]]:
        # explicit links first
        cur = self.conn.execute("""
            SELECT e.ex_id, e.title, e.input, e.output, e.ts
            FROM profile_examples px
            JOIN examples e ON e.user_id=px.user_id AND e.ex_id=px.ex_id
            WHERE px.user_id=? AND px.profile_name=?
            ORDER BY e.ts DESC LIMIT ?""", (user_id, profile_name, max(1, limit)))
        rows = cur.fetchall()
        out = [{"id": ex_id, "title": ttl, "input": inp, "output": outp, "ts": ts} for (ex_id, ttl, inp, outp, ts) in rows]
        if len(out) >= limit:
            return out
        # then tags defined inside profile JSON
        prof = self.get_profile_named(user_id, profile_name)
        tags = prof.get("example_tags") or []
        if tags:
            more = self.get_examples_by_tags(user_id, tags, limit=limit - len(out))
            # dedupe by id
            have = {e["id"] for e in out}
            out.extend([e for e in more if e["id"] not in have])
        return out[:limit]

    # ---------- summary ----------
    def get_thread_summary(self, user_id: str, conversation_id: str) -> str:
        cur = self.conn.execute("SELECT summary FROM threads WHERE user_id=? AND conversation_id=?",
                                (user_id, conversation_id))
        row = cur.fetchone()
        return row[0] if row and row[0] else ""

    def set_thread_summary(self, user_id: str, conversation_id: str, summary: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO threads(user_id,conversation_id,summary) VALUES(?,?,?)",
                          (user_id, conversation_id, ""))
        self.conn.execute("UPDATE threads SET summary=? WHERE user_id=? AND conversation_id=?",
                          (summary or "", user_id, conversation_id))
        self.conn.commit()
        self._log("set_thread_summary", {"len": len(summary or "")}, user_id, conversation_id)

    def get_thread_length(self, user_id: str, conversation_id: str) -> int:
        cur = self.conn.execute("SELECT COUNT(1) FROM messages WHERE user_id=? AND conversation_id=?",
                                (user_id, conversation_id))
        return int(cur.fetchone()[0] or 0)
