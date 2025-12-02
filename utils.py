import base64
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Make streamlit imports conditional - only needed for UI functions
try:
    import streamlit as st
    import streamlit.components.v1 as components
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


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
    cleaned = re.sub(r"^\s*[-•]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"<\|eot_id\|>|<\|begin_of_text\|>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.lstrip("\n ")


def audio_payload_or_none(text: str, speech_service: Any) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        audio_bytes, audio_fmt = speech_service.synthesize(sanitize_output(text))
    except Exception as exc:
        if STREAMLIT_AVAILABLE:
            st.warning(f"Text generated, but TTS failed: {exc}")
        return None, None

    if not audio_bytes:
        return None, None

    mime = "audio/wav" if audio_fmt == "pcm" else f"audio/{audio_fmt}"
    return audio_bytes, mime


def _render_retrieval_metrics(metrics: Dict[str, float]) -> None:
    if not STREAMLIT_AVAILABLE:
        return
    with st.expander("Retrieval Metrics", expanded=False):
        st.write(f"**Average Similarity:** {metrics.get('avg', 0.0):.3f}")
        st.write(f"**Top Similarity:** {metrics.get('top', 0.0):.3f}")


def ensure_state(models: List[Any]) -> None:
    if not STREAMLIT_AVAILABLE:
        return
    if "history" not in st.session_state:
        st.session_state.history = {}
    for m in models:
        key = getattr(m, "key", None)
        if key is not None:
            st.session_state.history.setdefault(key, [])


def categorize_models(models: List[Any]) -> Tuple[List[Any], List[Any]]:
    finetuned: List[Any] = []
    rag_models: List[Any] = []
    for model in models:
        mode = (getattr(model, "mode", "") or "").lower()
        if mode == "rag":
            rag_models.append(model)
        elif mode != "evaluation":
            finetuned.append(model)
    return finetuned, rag_models


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


def embed_query(client: Any, query: str) -> List[float]:
    """Generate embeddings using OpenAI's text-embedding-3-small model."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding


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
    return "\n".join(sections)


def evaluate_retrieval(client: Any, query: str, chunks: List[str]) -> Dict[str, float]:
    """Evaluate retrieval metrics using OpenAI embeddings."""
    if not chunks:
        return {"avg": 0.0, "top": 0.0}
    
    # Get query embedding
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    qv = np.array(query_response.data[0].embedding)
    
    # Get chunk embeddings
    chunk_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    cv = np.array([data.embedding for data in chunk_response.data])
    
    # Calculate similarities
    sims = cosine_similarity([qv], cv)[0]
    return {"avg": float(np.mean(sims)), "top": float(np.max(sims))}


def render_messages(messages: List[Dict[str, str]], conversation_key: str) -> None:
    if not STREAMLIT_AVAILABLE:
        return
    tracker: Dict[str, str] = st.session_state.setdefault("audio_autoplay_tracker", {})
    last_played_id = tracker.get(conversation_key)

    audio_entries: List[tuple[int, str]] = []
    for idx, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("audio"):
            message_id = m.get("message_uid") or f"{conversation_key}-{idx}"
            audio_entries.append((idx, message_id))

    last_audio_idx = audio_entries[-1][0] if audio_entries else None
    last_audio_id = audio_entries[-1][1] if audio_entries else None

    for idx, msg in enumerate(messages):
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))

            if msg.get("role") != "assistant":
                continue

            audio_data = msg.get("audio")
            if audio_data:
                mime = msg.get("audio_format", "audio/mp3")
                try:
                    raw_bytes: bytes
                    if isinstance(audio_data, (bytes, bytearray)):
                        raw_bytes = bytes(audio_data)
                    else:
                        raw_bytes = base64.b64decode(audio_data)
                except Exception:
                    raw_bytes = b""

                if raw_bytes:
                    message_id = msg.get("message_uid") or f"{conversation_key}-{idx}"
                    should_autoplay = bool(
                        last_audio_idx is not None
                        and idx == last_audio_idx
                        and last_audio_id is not None
                        and message_id != last_played_id
                    )

                    if should_autoplay and last_audio_id is not None:
                        tracker[conversation_key] = last_audio_id

                    audio_dom_id = f"audio-player-{idx}-{uuid4().hex}"
                    button_dom_id = f"audio-button-{idx}-{uuid4().hex}"
                    encoded = base64.b64encode(raw_bytes).decode("utf-8")
                    autoplay_label = "Pause" if should_autoplay else "Resume"
                    auto_flag = "true" if should_autoplay else "false"

                    html = f"""
                        <style>
                        .tts-wrapper {{
                            display: flex;
                            flex-direction: column;
                            gap: 0.5rem;
                            margin: 0.35rem 0 0.1rem 0;
                        }}
                        .tts-button {{
                            align-self: flex-start;
                            background: linear-gradient(135deg, #6c5ce7, #0984e3);
                            color: white;
                            border: none;
                            border-radius: 999px;
                            padding: 0.4rem 1rem;
                            font-size: 0.875rem;
                            font-weight: 600;
                            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                            cursor: pointer;
                            transition: transform 0.15s ease, box-shadow 0.15s ease;
                        }}
                        .tts-button:hover {{
                            transform: translateY(-1px);
                            box-shadow: 0 6px 16px rgba(0,0,0,0.2);
                        }}
                        .tts-button:active {{
                            transform: translateY(0);
                            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                        }}
                        </style>
                        <div class=\"tts-wrapper\">
                            <audio id=\"{audio_dom_id}\" style=\"width:100%;\" preload=\"auto\">
                                <source src=\"data:{mime};base64,{encoded}\">
                            </audio>
                            <button id=\"{button_dom_id}\" class=\"tts-button\">{autoplay_label}</button>
                        </div>
                        <script>
                        (function(){{
                            const audio = document.getElementById('{audio_dom_id}');
                            const button = document.getElementById('{button_dom_id}');
                            if(!audio || !button){{return;}}

                            window.__activeAudio = window.__activeAudio || null;
                            window.__pauseOnNewRequest = window.__pauseOnNewRequest || function(){{
                                if(window.__activeAudio){{
                                    const prev = window.__activeAudio;
                                    prev.pause();
                                    if(prev.__controlButton){{
                                        prev.__controlButton.textContent = 'Resume';
                                    }}
                                    window.__activeAudio = null;
                                }}
                            }};

                            function updateButton(audioEl, label){{
                                if(audioEl && audioEl.__controlButton){{
                                    audioEl.__controlButton.textContent = label;
                                }}
                            }}

                            function setActive(audioEl){{
                                if(window.__activeAudio && window.__activeAudio !== audioEl){{
                                    updateButton(window.__activeAudio, 'Resume');
                                    window.__activeAudio.pause();
                                }}
                                window.__activeAudio = audioEl;
                                updateButton(audioEl, 'Pause');
                            }}

                            function playAudio(){{
                                window.__pauseOnNewRequest();
                                const promise = audio.play();
                                if(promise){{promise.catch(()=>{{}});}}
                                setActive(audio);
                            }}

                            function pauseAudio(){{
                                audio.pause();
                                if(window.__activeAudio === audio){{
                                    window.__activeAudio = null;
                                }}
                                updateButton(audio, 'Resume');
                            }}

                            audio.__controlButton = button;

                            if({auto_flag}){{
                                playAudio();
                            }} else {{
                                button.textContent = 'Resume';
                            }}

                            button.addEventListener('click', function(){{
                                if(audio.paused){{
                                    playAudio();
                                }} else {{
                                    pauseAudio();
                                }}
                            }});

                            audio.addEventListener('ended', pauseAudio);
                        }})();
                        </script>
                    """
                    components.html(html, height=120)

            metrics = msg.get("metrics")
            if metrics:
                _render_retrieval_metrics(metrics)