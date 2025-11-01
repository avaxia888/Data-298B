from typing import Any, Dict, List, Optional

import streamlit as st


def ensure_state(models: List[Any]) -> None:
    """Ensure chat history exists for each model key in session state.

    We avoid importing EndpointConfig here to keep utils independent of llm_client.
    Each model is expected to expose a unique "key" attribute.
    """
    if "history" not in st.session_state:
        st.session_state.history = {}
    for m in models:
        key = getattr(m, "key", None)
        if key is not None:
            st.session_state.history.setdefault(key, [])


def render_messages(messages: List[Dict[str, str]]) -> None:
    for msg in messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))


def build_prompt(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    lines = [f"System: {system_prompt}".strip()]
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chat: List[Dict[str, str]] = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "assistant"
        chat.append({"role": role, "content": m.get("content", "")})
    return chat


def llama3_chat_template(system_prompt: Optional[str], messages: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not messages:
        return None

    bos = "<|begin_of_text|>"

    def wrap(role: str, content: str) -> str:
        return f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

    parts: List[str] = [bos]
    if system_prompt:
        parts.append(wrap("system", system_prompt))
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role not in ("user", "assistant"):
            role = "user"
        parts.append(wrap(role, content))
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)
