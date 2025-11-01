import pathlib
from typing import List

import streamlit as st
from dotenv import load_dotenv
from llm_client import LLMClient, load_models_config, EndpointConfig
from utils import ensure_state, render_messages, build_prompt, build_chat_messages
from home import get_view, set_view, render_home
BASE_DIR = pathlib.Path(__file__).parent
MODELS_PATH = BASE_DIR / "models.json"
DEFAULT_SYSTEM_PROMPT = (
    "You are Neil deGrasse Tyson, an astrophysicist and science communicator. Speak with curiosity, clarity, and a hint of wit. "
    "Use everyday analogies for complex ideas.\n\n"
    "Constraints:\n\n"
    "- Be concise by default (under ~120 words) unless asked to elaborate.\n"
    "- For casual chats or pleasantries, reply in 1–2 sentences (under ~60 words).\n"
    "- Define jargon in plain English. Include units when citing numbers.\n"
    "- If uncertain or the info is outside your training, say “I’m not sure,” and suggest how to verify.\n"
    "- Never invent citations or quotes.\n"
    "- Stay respectful.\n"
    "- Safety: refuse harmful or disallowed requests.\n"
    "- When helpful, add one visual analogy (e.g., “Imagine the Milky Way as…”). Prioritize scientific accuracy over flair."
)

def render_finetuned_chat(models: List[EndpointConfig]):
    st.title("Neil deGrasse Tyson Chatbot 🔭")
    st.caption("Chat with your OpenAI-compatible models behind a simple UI. Select the model in the sidebar.")

    with st.sidebar:
        st.header("Navigation")
        if st.button("← Back to Home", use_container_width=True):
            set_view("home")

        st.header("Model")
        model_names = {m.name: m.key for m in models}
        selected_name = st.selectbox("Choose a model", list(model_names.keys()))
        selected_key = model_names[selected_name]
        selected = next(m for m in models if m.key == selected_key)

        st.divider()
        st.header("Behavior")
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=120)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        max_new_tokens = st.slider("Max new tokens", 16, 4000, 256, 16)

        if st.button("Clear chat"):
            st.session_state.history[selected.key] = []

    messages = st.session_state.history[selected.key]
    render_messages(messages)

    user_input = st.chat_input("Ask Neil about the cosmos…")

    if user_input:
        msg = {"role": "user", "content": user_input}
        messages.append(msg)
        with st.chat_message("user"):
            st.markdown(user_input)

        prompt = build_prompt(system_prompt, messages)
        chat_messages = build_chat_messages(messages)

        system_for_api = None
        if chat_messages and chat_messages[-1]["role"] == "user":
            chat_messages[-1]["content"] = f"{system_prompt}\n\n{chat_messages[-1]['content']}"

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                client = LLMClient()
                try:
                    text = client.generate(
                        endpoint=selected,
                        prompt=prompt,
                        parameters={
                            "temperature": float(temperature),
                            "max_new_tokens": int(max_new_tokens),
                        },
                        messages=chat_messages,
                        system_prompt=system_for_api,
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    return
                st.markdown(text)
        messages.append({"role": "assistant", "content": text})


def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot", page_icon="🔭", layout="centered")
    models = load_models_config(str(MODELS_PATH))
    ensure_state(models)
    view = get_view()
    if view == "finetuned":
        render_finetuned_chat(models)
    else:
        render_home()

if __name__ == "__main__":
    main()
