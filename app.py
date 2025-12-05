import pathlib
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from services.llm_client import LLMClient, load_models_config
from services.rag import RagService
from services.speech import SpeechService
from utils import (
    ensure_state,
    render_messages,
    build_chat_messages,
    categorize_models,
    audio_payload_or_none,
    embed_query,
    retrieve_context,
    build_rag_prompt,
    evaluate_answer_alignment,
)
from prompt_template import DEFAULT_SYSTEM_PROMPT

BASE_DIR = pathlib.Path(__file__).parent
MODELS_PATH = BASE_DIR / "models.json"
def main():
    load_dotenv()
    st.set_page_config(page_title="Neil deGrasse Tyson Chatbot", page_icon="🔭", layout="centered")
    models = load_models_config(str(MODELS_PATH))
    ensure_state(models)

    speech_service = SpeechService()
    rag_service = RagService()
    llm_client = LLMClient()

    st.title("Cosmic Conversations with Neil deGrasse Tyson")

    finetuned, rag_models = categorize_models(models)

    with st.sidebar:
        st.header("Model Type")
        model_category = st.radio("Select Category", ["Finetuned", "RAG", "RAG+Finetuned"], index=0 if finetuned else 1)
        if model_category == "Finetuned":
            pool = finetuned
        elif model_category == "RAG":
            pool = rag_models
        else:  # RAG+Finetuned
            pool = finetuned
        options = {m.name: m.key for m in pool}
        chosen_name = st.selectbox("Model", list(options.keys()))
        chosen_key = options[chosen_name]
        # Use separate chat history for RAG+Finetuned so it starts fresh
        chat_key = (
            f"rf_{chosen_key}" if model_category == "RAG+Finetuned" else chosen_key
        )

        st.divider()
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        max_new_tokens = (
            st.slider("Max new tokens", 16, 4000, 256, 16)
            if model_category in ("Finetuned", "RAG+Finetuned")
            else None
        )
        memory_length = (
            st.slider("Conversation memory", 1, 10, 5, 1)
            if model_category in ("RAG", "RAG+Finetuned")
            else 0
        )

        if st.button("Clear chat"):
            st.session_state.history[chat_key] = []
            st.session_state.pop(f"rag_conv_{chosen_key}", None)
            st.session_state.pop(f"ragfinetuned_conv_{chosen_key}", None)

    selected = next((m for m in models if m.key == chosen_key), None)
    if not selected:
        st.error("Selected model not found.")
        return

    messages = st.session_state.history.setdefault(chat_key, [])
    render_messages(messages, chat_key)

    user_input = st.chat_input("Ask Neil about the cosmos…")
    if not user_input:
        return

    messages.append({"role": "user", "content": user_input})
    effective_system_prompt = DEFAULT_SYSTEM_PROMPT.rstrip()

    def append_audio(payload: dict, spoken_text: str):
        audio_bytes, audio_mime = audio_payload_or_none(spoken_text, speech_service)
        if audio_bytes:
            payload["audio"] = audio_bytes
            payload["audio_format"] = audio_mime

    with st.spinner("Thinking…"):
        if model_category == "RAG":
            # Standard RAG flow
            history_key = f"rag_conv_{selected.key}"
            history = st.session_state.get(history_key, [])
            try:
                text, metrics = rag_service.answer(
                    query=user_input,
                    history=history[-memory_length:] if memory_length else history[-5:],
                    temperature=float(temperature),
                    endpoint=selected,
                    system_prompt=effective_system_prompt,
                )
            except Exception as exc:
                st.error(f"RAG generation failed: {exc}")
                return

            assistant_payload = {
                "role": "assistant",
                "content": text,
                "metrics": metrics,
                "message_uid": str(uuid4()),
            }
            append_audio(assistant_payload, text)
            messages.append(assistant_payload)
            history.append({"user": user_input, "assistant": text})
            st.session_state[history_key] = history
        elif model_category == "RAG+Finetuned":
            # Retrieval augmented generation using finetuned LLMs
            # Retrieval pipeline setup
            rag_service._ensure()
            embed_model = rag_service._openai_client
            pc_index = rag_service._pinecone_index
            history_key = f"ragfinetuned_conv_{selected.key}"
            history = st.session_state.get(history_key, [])
            try:
                qv = embed_query(embed_model, user_input)
                chunks = retrieve_context(pc_index, qv)
                context_block = "\n\n".join(chunks)
                chat_messages = [
                    {
                        "role": "system",
                        "content": f"{effective_system_prompt}\n\nRelevant context:\n{context_block}",
                    },
                    {"role": "user", "content": user_input},
                ]
                gen_params = {"temperature": float(temperature)}
                if max_new_tokens:
                    gen_params["max_new_tokens"] = int(max_new_tokens)
                text = llm_client.generate(
                    endpoint=selected,
                    prompt="",
                    parameters=gen_params,
                    messages=chat_messages,
                    system_prompt=None,
                )
            except Exception as exc:
                st.error(f"RAG+Finetuned generation failed: {exc}")
                return

            # Calculate context alignment metrics
            metrics = evaluate_answer_alignment(embed_model, user_input, text, chunks)
            
            assistant_payload = {
                "role": "assistant",
                "content": text,
                "metrics": metrics,
                "message_uid": str(uuid4()),
            }
            append_audio(assistant_payload, text)
            messages.append(assistant_payload)
            history.append({"user": user_input, "assistant": text})
            st.session_state[history_key] = history
        else:
            chat_messages = build_chat_messages(messages)
            try:
                gen_params = {"temperature": float(temperature)}
                if max_new_tokens:
                    gen_params["max_new_tokens"] = int(max_new_tokens)
                text = llm_client.generate(
                    endpoint=selected,
                    prompt="",
                    parameters=gen_params,
                    messages=chat_messages,
                    system_prompt=effective_system_prompt,
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return

            assistant_payload = {
                "role": "assistant",
                "content": text,
                "message_uid": str(uuid4()),
            }
            append_audio(assistant_payload, text)
            messages.append(assistant_payload)

    st.rerun()

if __name__ == "__main__":
    main()
