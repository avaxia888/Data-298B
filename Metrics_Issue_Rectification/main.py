import streamlit as st
from llm_models import LLMService
from utils import rag_pipeline

# --------- Page Config ---------
st.set_page_config(page_title="Neil deGrasse Tyson AI", page_icon="🔭", layout="centered")

if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_history" not in st.session_state: st.session_state.conversation_history = []

st.title("Neil deGrasse Tyson AI")
st.markdown("Ask me about the cosmos, physics, and the wonders of science!")

# --------- Sidebar ---------
# Instantiate the service so we can expose available models in the sidebar
service = LLMService()

with st.sidebar:
    st.info("Neil deGrasse Tyson Chatbot")

    model_keys = list(service.model_registry.keys())
    if not model_keys:
        model_keys = [next(iter(service.model_registry.values()))]
    selected_model = st.selectbox("Model", options=model_keys, index=0)

    if st.button("Clear Chat"):
        st.session_state.messages, st.session_state.conversation_history = [], []
        st.rerun()

# Set default values for temperature and memory_length
temperature = 0.7
memory_length = 5

# --------- Load Models / LLM Service ---------
index = service.index
model = service.embed_model

# --------- Chat UI ---------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    history = st.session_state.conversation_history[-memory_length:]
    with st.chat_message("assistant"):
        response, metrics = rag_pipeline(service, prompt, history, temperature, model_id=selected_model)
        st.write(response)
        with st.expander("Retrieval Metrics"):
            st.write(metrics)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.conversation_history.append({"user": prompt, "assistant": response})
