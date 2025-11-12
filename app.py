import pathlib
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from llm_client import LLMClient, load_models_config
from utils import ensure_state, render_messages, build_chat_messages
from prompt_template import DEFAULT_SYSTEM_PROMPT

BASE_DIR = pathlib.Path(__file__).parent
MODELS_PATH = BASE_DIR / "models.json"


def main():
    load_dotenv()
    st.set_page_config(page_title="Neil deGrasse Tyson Chatbot", page_icon="🔭", layout="centered")
    models = load_models_config(str(MODELS_PATH))
    ensure_state(models)

    st.title("Cosmic Conversations with Neil deGrasse Tyson")

    with st.sidebar:
        st.header("Model Type")
        finetuned = [m for m in models if (m.mode or "").lower() not in ("rag", "evaluation") and not m.key.endswith("-judge")]
        rag_models = [m for m in models if (m.mode or "").lower() == "rag"]
        model_category = st.radio("Select Category", ["Finetuned", "RAG"], index=0 if finetuned else 1)

        if model_category == "Finetuned":
            options = {m.name: m.key for m in finetuned}
        else:
            options = {m.name: m.key for m in rag_models}

        if not options:
            st.error("No models available in this category.")
            return

        chosen_name = st.selectbox("Model", list(options.keys()))
        chosen_key = options[chosen_name]
        st.divider()
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        if model_category == "Finetuned":
            max_new_tokens = st.slider("Max new tokens", 16, 4000, 256, 16)
        else:
            memory_length = st.slider("Conversation memory", 1, 10, 5, 1)
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=240)
        if st.button("Clear chat"):
            st.session_state.history[chosen_key] = []
            if model_category == "RAG":
                st.session_state.pop(f"rag_conv_{chosen_key}", None)

    selected = next((m for m in models if m.key == chosen_key), None)
    if not selected:
        st.error("Selected model not found.")
        return

    effective_system_prompt = system_prompt.rstrip()

    messages = st.session_state.history[selected.key]
    render_messages(messages, selected.key)

    user_input = st.chat_input("Ask Neil about the cosmos…")
    if user_input:
        st.write("<script>window.__pauseOnNewRequest && window.__pauseOnNewRequest();</script>", unsafe_allow_html=True)
        messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking…"):
            if (selected.mode or "").lower() == "rag":
                hist_key = f"rag_conv_{selected.key}"
                history = st.session_state.get(hist_key, [])
                client = LLMClient()
                audio_bytes = None
                audio_mime = None
                try:
                    text, metrics = client.rag_answer(
                        query=user_input,
                        history=history[-memory_length:] if 'memory_length' in locals() else history[-5:],
                        temperature=float(temperature),
                        model_id=selected.url,
                        system_prompt=effective_system_prompt,
                    )
                except Exception as e:
                    st.error(f"RAG generation failed: {e}")
                    return
                try:
                    audio_bytes, audio_fmt = client.synthesize_speech(text)
                    audio_mime = "audio/wav" if audio_fmt == "pcm" else f"audio/{audio_fmt}"
                except Exception as audio_error:
                    st.warning(f"Text generated, but TTS failed: {audio_error}")
                assistant_payload = {
                    "role": "assistant",
                    "content": text,
                    "metrics": metrics,
                    "message_uid": str(uuid4()),
                }
                if audio_bytes:
                    assistant_payload["audio"] = audio_bytes
                    assistant_payload["audio_format"] = audio_mime
                messages.append(assistant_payload)
                history.append({"user": user_input, "assistant": text})
                st.session_state[hist_key] = history
            else:
                chat_messages = build_chat_messages(messages)
                client = LLMClient()
                audio_bytes = None
                audio_mime = None
                try:
                    text = client.generate(
                        endpoint=selected,
                        prompt="",
                        parameters={
                            "temperature": float(temperature),
                            "max_new_tokens": int(max_new_tokens),
                        },
                        messages=chat_messages,
                        system_prompt=effective_system_prompt,
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    return
                try:
                    audio_bytes, audio_fmt = client.synthesize_speech(text)
                    audio_mime = "audio/wav" if audio_fmt == "pcm" else f"audio/{audio_fmt}"
                except Exception as audio_error:
                    st.warning(f"Text generated, but TTS failed: {audio_error}")
                assistant_payload = {
                    "role": "assistant",
                    "content": text,
                    "message_uid": str(uuid4()),
                }
                if audio_bytes:
                    assistant_payload["audio"] = audio_bytes
                    assistant_payload["audio_format"] = audio_mime
                messages.append(assistant_payload)
        st.rerun()

if __name__ == "__main__":
    main()
