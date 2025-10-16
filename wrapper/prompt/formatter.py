# wrapper/prompt/formatter.py
from __future__ import annotations
from typing import Any, Dict, List

def _esc(text: str) -> str:
    # Simple XML escaping to prevent broken tags
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def build_riceco_xml(
    *,
    role: str | None = None,
    instruction: str | None = None,
    context: str | List[str] | None = None,
    constraints: Dict[str, str] | None = None,
    output_format: str | None = None,
    user_text: str | None = None,
    profile: Dict[str, Any] | None = None,
    examples: List[Dict[str, Any]] | None = None,
    example_ref_tags: List[str] | None = None,
) -> str:
    """Single root <CHAT> with thin RICECO tags. Omit empty nodes."""
    parts: List[str] = ["<CHAT>"]

    if role:
        parts.append(f"  <ROLE>{_esc(role)}</ROLE>")
    if instruction:
        parts.append(f"  <INSTRUCTION>{_esc(instruction)}</INSTRUCTION>")

    if profile:
        import json
        parts.append(f"  <PROFILE>{json.dumps(profile, ensure_ascii=False)}</PROFILE>")

    if context:
        if isinstance(context, list):
            ctx = " ".join(str(c) for c in context if c)
        else:
            ctx = str(context)
        if ctx.strip():
            parts.append(f"  <CONTEXT>{_esc(ctx.strip())}</CONTEXT>")

    if examples:
        ref_attr = ""
        if example_ref_tags:
            ref_attr = f' ref="{_esc(",".join(example_ref_tags))}"'
        parts.append(f"  <EXAMPLES{ref_attr}>")
        for ex in examples:
            parts.append("    <EXAMPLE>")
            title = ex.get("title") or ""
            if title:
                parts.append(f"      <TITLE>{_esc(title)}</TITLE>")
            inp = ex.get("input") or ""
            out = ex.get("output") or ""
            if inp:
                parts.append(f"      <INPUT>{_esc(inp)}</INPUT>")
            if out:
                parts.append(f"      <OUTPUT>{_esc(out)}</OUTPUT>")
            parts.append("    </EXAMPLE>")
        parts.append("  </EXAMPLES>")

    if constraints:
        parts.append("  <CONSTRAINTS>")
        for k, v in constraints.items():
            tag = k.strip().upper().replace(" ", "_")
            parts.append(f"    <{tag}>{_esc(str(v))}</{tag}>")
        parts.append("  </CONSTRAINTS>")

    if output_format:
        parts.append(f"  <OUTPUT_FORMAT>{_esc(output_format)}</OUTPUT_FORMAT>")

    if user_text:
        parts.append(f"  <USER>{_esc(user_text)}</USER>")

    parts.append("</CHAT>")
    return "\n".join(parts)
