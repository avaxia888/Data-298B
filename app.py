import pathlib
from typing import List

import streamlit as st
from dotenv import load_dotenv
from llm_client import LLMClient, load_models_config, EndpointConfig
from utils import ensure_state, render_messages, build_prompt, build_chat_messages
from home import get_view, set_view, render_home

# Import RAG modules conditionally
try:
    from RAG.llm_models import LLMService
    from RAG.utils import rag_pipeline
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"RAG modules not available: {e}")

# Import evaluation modules
try:
    from evaluation import LLMJudge
    EVAL_AVAILABLE = True
except ImportError as e:
    EVAL_AVAILABLE = False
    print(f"Evaluation modules not available: {e}")
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
        # Filter out evaluation-only models like GPT-5 judge
        chatbot_models = [m for m in models if not m.key.endswith("-judge") and "judge" not in m.key.lower()]
        model_names = {m.name: m.key for m in chatbot_models}
        selected_name = st.selectbox("Choose a model", list(model_names.keys()))
        selected_key = model_names[selected_name]
        selected = next(m for m in models if m.key == selected_key)

        st.divider()
        st.header("Behavior")
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=120)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        max_new_tokens = st.slider("Max new tokens", 16, 4000, 256, 16)
        
        st.divider()
        st.header("Evaluation")
        use_judge = st.checkbox("Enable GPT-5 Judge", value=False, help="Use GPT-5 to evaluate response quality")
        
        # Initialize judge if enabled
        judge = None
        if use_judge and EVAL_AVAILABLE:
            try:
                judge = LLMJudge(model_name="gpt-5")
            except Exception as e:
                st.error(f"Failed to initialize judge: {e}")
                use_judge = False

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

        # Prepend system prompt to every user message for OpenAI-mode endpoints;
        # for Hugging Face (Llama) we pass system separately to avoid template conflicts.
        if getattr(selected, "mode", "openai") == "openai" and system_prompt:
            for m in chat_messages:
                if m.get("role") == "user":
                    content = m.get("content", "")
                    if not content.lstrip().startswith(system_prompt[:16]):
                        m["content"] = f"{system_prompt}\n\n{content}"
            system_for_api = None
        else:
            system_for_api = system_prompt

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
                error_msg = str(e)
                if "403" in error_msg:
                    st.error(f"⚠️ HuggingFace endpoint access denied. The endpoint may be paused or require different permissions.")
                    st.info("💡 Try: 1) Check if the endpoint is active on HuggingFace, 2) Use the GPT-4o model instead, or 3) Try the RAG models")
                elif "401" in error_msg:
                    st.error(f"⚠️ Authentication failed. Please check your API keys in the .env file.")
                elif "404" in error_msg:
                    st.error("⚠️ Endpoint not found (404). The configured URL may be incorrect or missing the required path.")
                    st.info("💡 If you're using a Hugging Face Inference Endpoint, ensure it's running and the URL includes the proper route. For OpenAI-compatible mode, the path should be /v1/chat/completions. For HF text-generation-inference, /generate may be required.")
                else:
                    st.error(f"Generation failed: {e}")
                return
        # Evaluate response if judge is enabled
        eval_result = None
        if use_judge and judge and EVAL_AVAILABLE:
            try:
                eval_result = judge.evaluate_response(user_input, text, use_cache=True)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
        
        # Append assistant message with evaluation
        assistant_msg = {"role": "assistant", "content": text}
        if eval_result:
            assistant_msg["evaluation"] = {
                "scores": eval_result.scores,
                "overall_score": eval_result.overall_score,
                "strengths": eval_result.strengths,
                "weaknesses": eval_result.weaknesses,
                "suggestions": eval_result.suggestions
            }
        
        # Append assistant message once (avoid accidental duplicates across reruns)
        if not (messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == text):
            messages.append(assistant_msg)
        st.rerun()


def render_rag_chat():
    if not RAG_AVAILABLE:
        st.error("RAG modules are not properly installed. Please check the installation.")
        st.info("Try running: pip install sentence-transformers==2.2.2 pinecone-client boto3")
        return
        
    st.title("Neil deGrasse Tyson Chatbot 🔭 (RAG)")
    st.caption("Chat with AI models using Retrieval-Augmented Generation for grounded, accurate responses.")

    # Initialize RAG service
    service = LLMService()
    
    # Ensure session state for RAG models
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = {}
    
    # Initialize history for each RAG model
    for model_name in service.model_registry.keys():
        if model_name not in st.session_state.rag_history:
            st.session_state.rag_history[model_name] = []
    
    if "rag_conversation_history" not in st.session_state:
        st.session_state.rag_conversation_history = {}
        for model_name in service.model_registry.keys():
            st.session_state.rag_conversation_history[model_name] = []

    with st.sidebar:
        st.header("Navigation")
        if st.button("← Back to Home", use_container_width=True):
            set_view("home")

        st.header("Model")
        model_names = list(service.model_registry.keys())
        selected_model = st.selectbox("Choose a model", model_names)

        st.divider()
        st.header("Behavior")
        system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=120)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        max_new_tokens = st.slider("Max new tokens", 16, 4000, 256, 16)
        memory_length = st.slider("Conversation memory", 1, 10, 5, 1)
        
        st.divider()
        st.header("Evaluation")
        use_judge = st.checkbox("Enable GPT-5 Judge", value=False, help="Use GPT-5 to evaluate response quality")
        
        # Initialize judge if enabled
        judge = None
        if use_judge and EVAL_AVAILABLE:
            try:
                judge = LLMJudge(model_name="gpt-5")
            except Exception as e:
                st.error(f"Failed to initialize judge: {e}")
                use_judge = False

        if st.button("Clear chat"):
            st.session_state.rag_history[selected_model] = []
            st.session_state.rag_conversation_history[selected_model] = []

    messages = st.session_state.rag_history[selected_model]
    render_messages(messages)

    user_input = st.chat_input("Ask Neil about the cosmos…")

    if user_input:
        msg = {"role": "user", "content": user_input}
        messages.append(msg)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get conversation history for context
        history = st.session_state.rag_conversation_history[selected_model][-memory_length:]
        
        with st.spinner("Searching knowledge base and thinking…"):
            try:
                # Use RAG pipeline with selected model and system prompt
                response, metrics = rag_pipeline(
                    service, 
                    user_input, 
                    history, 
                    temperature, 
                    model_id=selected_model,
                    system_prompt=system_prompt,
                    evaluate_with_judge=use_judge,
                    judge=judge
                )
                
                # Update conversation history and messages first
                messages.append({"role": "assistant", "content": response, "metrics": metrics})
                st.session_state.rag_conversation_history[selected_model].append({
                    "user": user_input,
                    "assistant": response
                })
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return
        
        st.rerun()


def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot", page_icon="🔭", layout="centered")
    models = load_models_config(str(MODELS_PATH))
    ensure_state(models)
    view = get_view()
    if view == "finetuned":
        render_finetuned_chat(models)
    elif view == "rag":
        render_rag_chat()
    else:
        render_home()

if __name__ == "__main__":
    main()
