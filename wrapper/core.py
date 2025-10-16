# wrapper/core.py
from wrapper.providers.anthropic_provider import AnthropicProvider
from wrapper.providers.azure_provider import AzureProvider
from wrapper.providers.openai_provider import OpenAIProvider
from wrapper.providers.ollama_provider import OllamaProvider
from wrapper.providers.groq_provider import GroqProvider
from wrapper.providers.bedrock_provider import BedrockProvider
from wrapper.providers.openrouter_provider import OpenRouterProvider
from wrapper.providers.gemini_provider import GeminiProvider
from collections import defaultdict
from wrapper.utils import ColorLogger
from wrapper.config import *
from typing import Tuple

# === thin memory / prompting imports ===
from typing import Optional, Dict, Any, List, Union, Tuple
from wrapper.memory.store import MemoryStore
from wrapper.prompt.formatter import build_riceco_xml
from wrapper.examples.builder import build_example_from_segments, suggest_tags_local


log = ColorLogger(enable_debug=SHOW_LOGS)

class Wrapper:
    def __init__(self, provider: str, memory_store: Optional[MemoryStore] = None, **kwargs):
        provider = provider.lower()
        self.memory_store = memory_store

        # cache last turn to allow frictionless example-saving
        self._last_user_text: Optional[str] = None
        self._last_output_text: Optional[str] = None
        self._last_user_id: Optional[str] = None
        self._last_conversation_id: str = "default"

        if provider == "anthropic":
            self.impl = AnthropicProvider(**kwargs)
        elif provider == "azure":
            self.impl = AzureProvider(**kwargs)
        elif provider == "openai":
            self.impl = OpenAIProvider(**kwargs)
        elif provider == "ollama":
            self.impl = OllamaProvider(**kwargs)
        elif provider == "groq":
            self.impl = GroqProvider(**kwargs)
        elif provider == "bedrock":
            self.impl = BedrockProvider(**kwargs)
        elif provider == "openrouter":
            self.impl = OpenRouterProvider(**kwargs)
        elif provider == "gemini":
            self.impl = GeminiProvider(**kwargs)
        else:
            raise ValueError(f"Provider {provider} not supported yet")

    def generate(
        self,
        model: str,
        prompt: str = None,
        user: str = None,
        system: str = None,
        messages: list[dict] = None,
        temperature: float = None,
        max_tokens: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stream: bool = False,
        **kwargs
    ) -> str:

        # ---------- thin memory / prompting config ----------
        mem_cfg: Dict[str, Any] = kwargs.pop("memory", None) or {}
        prompting: Dict[str, Any] = kwargs.pop("prompting", None) or {}

        use_memory = bool(mem_cfg.get("enabled"))
        mode = (mem_cfg.get("mode") or "normal").lower()
        user_id = mem_cfg.get("user_id")
        conversation_id = mem_cfg.get("conversation_id") or "default"
        k = int(mem_cfg.get("k") or 6)
        autosummarize_n = int(mem_cfg.get("autosummarize_n") or 5)
        profile_name = mem_cfg.get("profile_name") or prompting.get("profile_name")
        profile_updates = mem_cfg.get("profile_updates") or {}

        example_tags = prompting.get("example_tags") or mem_cfg.get("example_tags") or []
        if isinstance(example_tags, str):
            example_tags = [t.strip() for t in example_tags.split(",") if t.strip()]

        history_window: List[dict] = []
        profile: Dict[str, Any] = {}
        examples: List[Dict[str, Any]] = []
        thread_summary: str = ""

        # ---------- pull memory (if enabled) ----------
        if use_memory and self.memory_store and user_id:
            self._last_user_id = user_id
            self._last_conversation_id = conversation_id

            if profile_updates:
                try:
                    self.memory_store.upsert_profile(user_id, profile_updates)
                except Exception as e:
                    log.warning(f"[memory] profile upsert failed: {e}")

            try:
                history_window = self.memory_store.get_window(user_id, conversation_id, k=k) or []
            except Exception as e:
                log.warning(f"[memory] get_window failed: {e}")

            try:
                thread_summary = self.memory_store.get_thread_summary(user_id, conversation_id) or ""
            except Exception as e:
                log.warning(f"[memory] get_thread_summary failed: {e}")

            if mode == "advanced":
                try:
                    if profile_name:
                        profile = self.memory_store.get_profile_named(user_id, profile_name) or {}
                    else:
                        profile = self.memory_store.get_profile(user_id) or {}
                except Exception as e:
                    log.warning(f"[memory] get_profile failed: {e}")

            if profile_name:
                try:
                    examples = self.memory_store.get_examples_for_profile(user_id, profile_name, limit=2) or []
                    # merge profile's example tags with caller tags
                    p_tags = profile.get("example_tags") or []
                    example_tags = list(dict.fromkeys(list(example_tags) + list(p_tags)))
                except Exception as e:
                    log.warning(f"[memory] get_examples_for_profile failed: {e}")

            if example_tags and not examples:
                try:
                    examples = self.memory_store.get_examples_by_tags(user_id, example_tags, limit=2) or []
                except Exception as e:
                    log.warning(f"[memory] get_examples_by_tags failed: {e}")

        # ---------- prepare messages (backward-compatible) ----------
        if messages is None:
            messages = []

            # system message
            if system:
                messages.append({"role": "system", "content": system})
            elif hasattr(self.impl, "default_system_prompt"):
                messages.append({"role": "system", "content": self.impl.default_system_prompt})

            # user message
            if user:
                messages.append({"role": "user", "content": user})
            elif prompt:
                messages.append({"role": "user", "content": prompt})

        # ---------- assemble XML (from profile defaults + prompting) ----------
        if prompting or profile_name or profile:
            # profile defaults -> overridden by explicit prompting params
            defaults: Dict[str, Any] = {}
            for key in ("role", "instruction", "constraints", "output_format", "context"):
                if key in profile and profile.get(key) not in (None, "", []):
                    defaults[key] = profile.get(key)
            merged = {**defaults, **prompting}

            # CONTEXT: stored summary + compact rolling window (~800 chars)
            ctx_parts: List[str] = []
            if thread_summary:
                ctx_parts.append(f"Summary: {thread_summary}")
            used = 0
            budget = 800
            for m in history_window[-k:]:
                if m.get("role") not in ("user", "assistant"):
                    continue
                seg = f'{m["role"]}: {str(m.get("content") or "").strip()}'
                if not seg:
                    continue
                if used + len(seg) > budget:
                    break
                ctx_parts.append(seg)
                used += len(seg)
            context_text = " | ".join(ctx_parts).strip()

            # current user text
            current_user_text = user or prompt
            if not current_user_text:
                for m in reversed(messages):
                    if m.get("role") == "user":
                        current_user_text = m.get("content")
                        break

            xml = build_riceco_xml(
                role=merged.get("role"),
                instruction=merged.get("instruction"),
                context=merged.get("context") or context_text,
                constraints=merged.get("constraints"),
                output_format=merged.get("output_format"),
                user_text=current_user_text,
                profile=profile if mode == "advanced" else None,
                examples=examples if examples else None,
                example_ref_tags=example_tags or None,
            )

            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = xml
            else:
                messages.insert(0, {"role": "system", "content": xml})

        # ---------- provider call ----------
        params = {"model": model, "stream": stream, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty
        params.update(kwargs)

        output = self.impl.generate(**params)

        # cache last turn for frictionless example saving
        self._last_user_text = user or prompt or None
        if not self._last_user_text:
            for m in reversed(messages):
                if m.get("role") == "user":
                    self._last_user_text = m.get("content")
                    break
        self._last_output_text = output if isinstance(output, str) else None

        # ---------- append + auto-summarize ----------
        if use_memory and self.memory_store and user_id:
            try:
                # append last user + assistant
                last_user = None
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user = m.get("content")
                        break
                if last_user:
                    self.memory_store.append_message(user_id, conversation_id, "user", last_user)
                if isinstance(output, str) and output:
                    self.memory_store.append_message(user_id, conversation_id, "assistant", output)

                # autosummarize when the thread grows
                try:
                    length = self.memory_store.get_thread_length(user_id, conversation_id)
                    if autosummarize_n and length >= autosummarize_n:
                        window = self.memory_store.get_window(user_id, conversation_id, k=10)
                        roll: List[str] = []
                        used = 0
                        budget = 600
                        for m in window:
                            if m.get("role") not in ("user", "assistant"):
                                continue
                            seg = f'{m["role"][0].upper()}: {str(m.get("content") or "").strip()}'
                            if not seg:
                                continue
                            if used + len(seg) > budget:
                                break
                            roll.append(seg)
                            used += len(seg)
                        summary = " | ".join(roll)
                        self.memory_store.set_thread_summary(user_id, conversation_id, summary)
                except Exception as e:
                    log.warning(f"[memory] autosummarize failed: {e}")

            except Exception as e:
                log.warning(f"[memory] append failed: {e}")

        return output
    def _suggest_tags_llm(self, model: str, chunks: List[str], k: int = 5) -> List[str]:
        """Ask the model for up to k comma-separated tags. Fallback happens in caller."""
        try:
            prompt = (
                "Suggest concise, domain-relevant tags (1-3 words each) for the text below. "
                f"Return only a comma-separated list of up to {k} tags, no extra text.\n\n"
                "TEXT:\n" + "\n\n".join(chunks)
            )
            out = self.impl.generate(model=model, messages=[
                {"role": "system", "content": "You output only tags as comma-separated text."},
                {"role": "user", "content": prompt}
            ])
            if not isinstance(out, str):  # be defensive
                return []
            raw = out.strip().strip(",")
            tags = [t.strip() for t in raw.split(",") if t.strip()]
            tags = [re.sub(r"\s+", " ", t) for t in tags]
            return tags[:k]
        except Exception:
            return []
    def save_last_as_example(
        self,
        tags: Union[str, List[str]] = None,
        title: Optional[str] = None,
    ) -> Optional[str]:
        """
        Frictionless saver: persist the last user/assistant turn as an example.
        - tags may be a comma string or a list; both are accepted.
        - returns the example id (e.g., 'ex_000001') or None on failure/no-op.
        """
        # normalize tags
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        tags = tags or []

        # we need a memory store and a captured last turn
        if not (self.memory_store and self._last_user_id and self._last_output_text):
            # nothing to save (e.g., generate() not called yet)
            return None

        try:
            ex_id = self.memory_store.save_example(
                user_id=self._last_user_id,
                tags=tags,
                input_text=self._last_user_text or "",
                output_text=self._last_output_text or "",
                title=title,
            )
            return ex_id
        except Exception as e:
            log.warning(f"[memory] save_last_as_example failed: {e}")
            return None
    # frictionless: save most recent turn as an example
    def save_example_from_last(
        self,
        tags: Optional[Union[str, List[str]]] = None,
        title: Optional[str] = None,
        segments: Optional[List[Tuple[str, str]]] = None,
        *,
        auto_tag: bool = True,
        tag_mode: str = "local",        
        tag_k: int = 5,
        save_to_global: bool = False,   
        link_profiles: Optional[List[str]] = None,  
        model_for_tags: Optional[str] = None,       
    ) -> Optional[str]:
        """
        Create a compact example (input/output/title) from the last turn (or provided segments),
        optionally auto-tag, save to user store (and global), and link to profiles.
        Returns the example id or None.
        """
        if not (self.memory_store and self._last_user_id and self._last_output_text):
            return None

        uid = self._last_user_id
        conv_id = self._last_conversation_id

        # Prepare segments
        if not segments:
            segs = [("user", self._last_user_text or ""), ("assistant", self._last_output_text or "")]
        else:
            segs = segments

        ex = build_example_from_segments(segs, title=title)

        # Normalize tags input
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        tags = tags or []

        # Auto-tag if needed
        if auto_tag and not tags:
            chunks = [ex["title"], ex["input"], ex["output"]]
            if tag_mode == "llm" and model_for_tags:
                llm_tags = self._suggest_tags_llm(model_for_tags, chunks, k=tag_k)
                if llm_tags:
                    tags = llm_tags
            if not tags:
                tags = suggest_tags_local(*chunks, k=tag_k)

        # Save to user store
        try:
            new_id = self.memory_store.save_example(uid, tags, ex["input"], ex["output"], title=ex["title"])
            # Optionally save to global library
            if save_to_global:
                try:
                    self.memory_store.save_example("__global__", tags, ex["input"], ex["output"], title=ex["title"])
                except Exception as e:
                    log.warning(f"[memory] save to global failed: {e}")
            # Optionally link to named profiles
            if link_profiles:
                for pname in link_profiles:
                    try:
                        self.memory_store.link_examples_to_profile(uid, pname, [new_id], mode="append")
                    except Exception as e:
                        log.warning(f"[memory] link to profile '{pname}' failed: {e}")
            return new_id
        except Exception as e:
            log.warning(f"[memory] save_example_from_last failed: {e}")
            return None

    @staticmethod
    def available_models_api(provider: str, **kwargs):
        provider = provider.lower()

        if provider == "anthropic":
            instance = AnthropicProvider(**kwargs)
            models = instance.list_models()
            log.info("\n Anthropic Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f" {i:2d}. {m}")
            return models

        elif provider == "azure":
            instance = AzureProvider(**kwargs)
            return instance.list_models()

        elif provider == "openai":
            instance = OpenAIProvider(**kwargs)
            models = instance.list_models()
            groups = defaultdict(list)
            for m in models:
                if m.startswith(("gpt-3.5", "gpt-4", "gpt-5", "o1", "o3", "o4")):
                    groups["LLMs"].append(m)
                elif "dall-e" in m or "image" in m:
                    groups["Image Models"].append(m)
                elif "tts" in m or "audio" in m or "whisper" in m:
                    groups["Audio Models"].append(m)
                elif "embedding" in m:
                    groups["Embedding Models"].append(m)
                else:
                    groups["Other"].append(m)

            log.info("\n OpenAI Models (by category):\n")
            for category, items in groups.items():
                log.info(f" {category}:")
                for i, model in enumerate(items, 1):
                    log.info(f"   {i:2d}. {model}")
            return models

        elif provider == "ollama":
            instance = OllamaProvider(**kwargs)
            return instance.list_models()

        elif provider == "groq":
            instance = GroqProvider(**kwargs)
            models = instance.list_models()
            log.info("\n Groq Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f"   {i:2d}. {m}")
            return models

        elif provider == "bedrock":
            instance = BedrockProvider(**kwargs)
            models = instance.list_models() or []
            log.info("\n Bedrock Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f"   {i:2d}. {m['modelId']} — {m['modelName']} ({m['provider']})")
            return models
        elif provider == "openrouter":
            instance = OpenRouterProvider(**kwargs)
            models = instance.list_models() or []
            log.info("\n OpenRouter Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f"   {i:2d}. {m}")
            return models

        elif provider == "gemini":
            instance = GeminiProvider(**kwargs)
            models = instance.list_models() or []
            log.info("\n Gemini Models:\n")
            for i, m in enumerate(models, 1):
                log.info(f"  {i:02d}. {m}")
            return models
        log.warning(f"No wrapper available for provider '{provider}'.")
        return []

class WrapperEmbed:
    # TODO
    def __init__(self, provider: str, **kwargs):
        provider = provider.lower()
        if provider == "openai":
            self.impl = OpenAIProvider(**kwargs)
        elif provider == "ollama":
            self.impl = OllamaProvider(**kwargs)
        else:
            raise ValueError(f"Embedding provider {provider} not supported yet")