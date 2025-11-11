import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_chat_role(role: str) -> str:
    return role if role in {"user", "assistant"} else "user"


def _wrap_chat_block(role: str, content: str) -> str:
    return f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"


def get_env(names: Iterable[str], default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    if required and default is None:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(names)}")
    return default


def extract_bedrock_text(body: Any) -> str:
    if not isinstance(body, dict):
        return str(body)

    if content := body.get("content"):
        if isinstance(content, list) and content:
            if text := content[0].get("text"):
                return text.strip()

    if outputs := body.get("outputs"):
        if isinstance(outputs, list) and outputs:
            if text := outputs[0].get("text"):
                return text.strip()

    if text := body.get("generated_text"):
        return text.strip()

    return json.dumps(body)


def extract_hf_text(data: Any) -> str:
    if isinstance(data, list) and data:
        return str(data[0].get("generated_text", "")).strip()
    if isinstance(data, dict):
        return str(data.get("generated_text", "")).strip()
    return json.dumps(data)


def sanitize_output(text: str) -> str:
    patterns = [
        r"^(?:\s*\*[^\n]*?\*\s*)+",
        r"^(?:\s*\([^)]{0,80}\)\s*)",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.lstrip("\n ")


def _render_retrieval_metrics(metrics: Dict[str, float]) -> None:
    with st.expander("Retrieval Metrics", expanded=False):
        st.write(f"**Average Similarity:** {metrics.get('avg', 0.0):.3f}")
        st.write(f"**Top Similarity:** {metrics.get('top', 0.0):.3f}")


def ensure_state(models: List[Any]) -> None:
    if "history" not in st.session_state:
        st.session_state.history = {}
    for m in models:
        key = getattr(m, "key", None)
        if key is not None:
            st.session_state.history.setdefault(key, [])


def build_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chat: List[Dict[str, str]] = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "assistant"
        chat.append({"role": role, "content": m.get("content", "")})
    return chat


def llama3_chat_template(system_prompt: Optional[str], messages: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not messages:
        return None

    parts: List[str] = ["<|begin_of_text|>"]
    if system_prompt:
        parts.append(_wrap_chat_block("system", system_prompt))

    parts.extend(
        _wrap_chat_block(_normalize_chat_role(m.get("role", "user")), m.get("content", ""))
        for m in messages
    )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)


def embed_query(model: Any, query: str) -> List[float]:
    return model.encode(query).tolist()


def retrieve_context(index: Any, query_vector: List[float], top_k: int = 5) -> List[str]:
    res = index.query(vector=query_vector, top_k=top_k, include_metadata=True, namespace="")
    return [m["metadata"]["text"] for m in res.get("matches", []) if "metadata" in m and "text" in m["metadata"]]


def build_rag_prompt(
    query: str,
    context: List[str],
    history: List[Dict[str, str]],
    system_prompt: str,
    *,
    include_system: bool = True,
) -> str:
    sections: List[str] = []

    if include_system:
        sections.append(system_prompt.rstrip())

    if context:
        context_block = "<<CONTEXT>>\n" + "\n\n".join(context) + "\n<</CONTEXT>>"
        sections.append(f"Context from knowledge base:\n{context_block}")

    if history:
        history_block = "".join([f"Human: {h['user']}\nAssistant: {h['assistant']}\n" for h in history])
        sections.append(f"Conversation so far:\n{history_block}".rstrip())

    sections.append(f"User: {query}\nAssistant:")
    return "\n\n".join(sections)


def evaluate_retrieval(model: Any, query: str, chunks: List[str]) -> Dict[str, float]:
    if not chunks:
        return {"avg": 0.0, "top": 0.0}
    qv = model.encode([query])[0]
    cv = model.encode(chunks)
    sims = cosine_similarity([qv], cv)[0]
    return {"avg": float(np.mean(sims)), "top": float(np.max(sims))}


def render_messages(messages: List[Dict[str, str]]) -> None:
    for msg in messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

            if msg.get("role") != "assistant":
                continue

            metrics = msg.get("metrics")
            if metrics:
                _render_retrieval_metrics(metrics)