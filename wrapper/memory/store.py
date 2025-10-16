# wrapper/memory/store.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict


class Message(TypedDict, total=False):
    """
    Minimal chat turn representation used by the memory layer.

    Fields
    ------
    role : Literal["user","assistant","system"]
        Who authored the message.
    content : str
        The text content of the message.
    ts : int
        UNIX epoch seconds (set by the store when appending).
    """
    role: str
    content: str
    ts: int


class ExampleRecord(TypedDict, total=False):
    """
    A compact, reusable few-shot example.

    Fields
    ------
    id : str
        Unique example identifier (store-assigned).
    tags : List[str]
        Tags for retrieval and organization.
    title : str
        ≤ ~12 word one-line intro generated or provided by user.
    input : str
        The user-side text that produced the example.
    output : str
        The assistant-side text saved as the example.
    ts : int
        UNIX epoch seconds (store-assigned).
    """
    id: str
    tags: List[str]
    title: str
    input: str
    output: str
    ts: int


class MemoryStore(ABC):
    """
    Thin, pluggable memory interface.

    Implementations:
      - JSONMemoryStore (local, zero-deps)
      - SQLiteMemoryStore (optional; same methods)

    Concepts:
      * Short-term window: rolling K turns per (user_id, conversation_id)
      * Profiles:
          - Legacy default profile: one dict per user
          - Named profiles: many dicts per user, referenced by profile_name
      * Examples:
          - Stored once globally per user, linkable to profiles
          - Fetch by tags or by ids
      * Thread summary:
          - Rolling summary string per (user_id, conversation_id)
    """

    # ---- short-term window ----
    @abstractmethod
    def get_window(self, user_id: str, conversation_id: str, k: int) -> List[Message]:
        """
        Return the last K messages (user/assistant/system) for a thread.
        Implementations should return [] if no data exists.
        """

    @abstractmethod
    def append_message(self, user_id: str, conversation_id: str, role: str, content: str) -> None:
        """
        Append a message to a thread. Store should set `ts` internally.
        """

    # ---- long-term profile (legacy default) ----
    @abstractmethod
    def get_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Return the legacy default (unnamed) profile dict for the user, or {}.
        """

    @abstractmethod
    def upsert_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shallow-merge `updates` into the legacy default profile and return it.
        """

    # ---- named profiles ----
    @abstractmethod
    def get_profile_named(self, user_id: str, profile_name: str) -> Dict[str, Any]:
        """
        Return a named profile dict (e.g., 'sre_engineer'), or {} if missing.
        """

    @abstractmethod
    def upsert_profile_named(self, user_id: str, profile_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shallow-merge `updates` into a named profile and return it.
        Implementations should ensure the profile object exists after upsert.
        """

    # ---- tagged examples (global to the user) ----
    @abstractmethod
    def save_example(
        self,
        user_id: str,
        tags: List[str],
        input_text: str,
        output_text: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Save a new example and return its id. Store should:
          - assign `id`
          - set `ts`
          - normalize/unique tags (trim, lower if desired)
          - generate a short `title` if not provided
        """

    @abstractmethod
    def get_examples_by_tags(self, user_id: str, tags: List[str], limit: int = 2) -> List[ExampleRecord]:
        """
        Retrieve up to `limit` examples by tag overlap (tie-break by recency).
        Return [] if none found.
        """

    @abstractmethod
    def get_examples_by_ids(self, user_id: str, ids: List[str]) -> List[ExampleRecord]:
        """
        Retrieve examples by explicit ids. Missing ids are ignored.
        """

    # ---- profile-example linking (share examples across profiles) ----
    @abstractmethod
    def link_examples_to_profile(
        self,
        user_id: str,
        profile_name: str,
        example_ids: List[str],
        mode: str = "append",
    ) -> List[str]:
        """
        Link example ids to a named profile.

        Parameters
        ----------
        mode : "append" | "set"
            - "append": add (de-dupe) to existing profile links
            - "set": replace the profile's example id list entirely

        Returns
        -------
        List[str]
            The resulting list of example ids linked to the profile.
        """

    @abstractmethod
    def get_examples_for_profile(self, user_id: str, profile_name: str, limit: int = 2) -> List[ExampleRecord]:
        """
        Retrieve up to `limit` examples for a named profile.
        Implementations should prioritize:
          1) explicitly linked example_ids
          2) profile.example_tags (if present)
        """

    # ---- thread summary (for auto-summarize-after-N) ----
    @abstractmethod
    def get_thread_summary(self, user_id: str, conversation_id: str) -> str:
        """
        Return the stored rolling summary for the thread, or "".
        """

    @abstractmethod
    def set_thread_summary(self, user_id: str, conversation_id: str, summary: str) -> None:
        """
        Set/replace the rolling summary string for the thread.
        """

    @abstractmethod
    def get_thread_length(self, user_id: str, conversation_id: str) -> int:
        """
        Return the number of messages currently stored for the thread.
        """
