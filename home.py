import streamlit as st
from typing import List, Dict


def get_view() -> str:
    qp = st.query_params
    val = qp.get("view", st.session_state.get("view", "home"))
    if isinstance(val, list):
        return val[0] if val else "home"
    return val


def set_view(view: str):
    st.session_state.view = view
    st.query_params["view"] = view
    st.rerun()


def render_home():
    st.title("Choose Your Mode of Interaction with Neil")
    st.caption("Pick a mode to get started.")

    cards: List[Dict[str, object]] = [
        {
            "key": "rag",
            "title": "RAG + Prompt Engineering",
            "desc": "Retrieve relevant context from your documents, then generate an answer grounded in those sources.",
            "disabled": False,
        },
        {
            "key": "finetuned",
            "title": "Finetuned Models",
            "desc": "Chat with your fine‑tuned OpenAI models. Compare behaviors, test prompts, and evaluate outputs",
            "disabled": False,
        },
        {
            "key": "rag_plus",
            "title": "RAG + Finetuned",
            "desc": "Combine retrieval with your fine‑tuned model for grounded answers with your model's voice and task skills. Coming soon.",
            "disabled": True,
        },
    ]

    cols = st.columns(3, gap="small")

    CARD_HEIGHT = 300
    for i, card in enumerate(cards):
        with cols[i]:
            # Try to use a fixed-height container (Streamlit >= 1.36). If unavailable, fall back gracefully.
            try:
                card_box = st.container(height=CARD_HEIGHT, border=True)
            except TypeError:
                card_box = st.container(border=True)
            with card_box:
                st.subheader(str(card["title"]))
                st.caption(str(card["desc"]))
                clicked = st.button(
                    "Open",
                    key=f"open_{card['key']}",
                    disabled=bool(card["disabled"]),
                    use_container_width=True,
                    type="primary" if not card["disabled"] else "secondary",
                )
                if clicked:
                    if card["key"] == "finetuned":
                        set_view("finetuned")
                        return
                    elif card["key"] == "rag":
                        set_view("rag")
                        return
                    # For disabled cards we won't navigate, but guard anyway
                    st.toast("Coming soon", icon="⏳")
