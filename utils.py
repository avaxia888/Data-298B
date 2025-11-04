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
            
            # Check if this is an assistant message
            if msg.get("role") == "assistant":
                # Display retrieval metrics if they exist (for RAG messages)
                if "metrics" in msg:
                    metrics = msg["metrics"]
                    
                    # Display retrieval metrics
                    with st.expander("📊 Retrieval Metrics", expanded=False):
                        st.write(f"**Average Similarity:** {metrics.get('avg', 0):.3f}")
                        st.write(f"**Top Similarity:** {metrics.get('top', 0):.3f}")
                    
                    # Display judge evaluation if available in metrics
                    if "judge_evaluation" in metrics:
                        eval_data = metrics["judge_evaluation"]
                        if "error" not in eval_data:
                            _render_evaluation(eval_data)
                
                # Display evaluation if it exists (for finetuned models)
                elif "evaluation" in msg:
                    eval_data = msg["evaluation"]
                    if "error" not in eval_data:
                        _render_evaluation(eval_data)

def _render_evaluation(eval_data: Dict[str, Any]) -> None:
    """Render the evaluation data in an expander."""
    with st.expander("⚖️ GPT-5 Evaluation", expanded=False):
        # Overall score
        overall = eval_data.get("overall_score", 0)
        st.metric("Overall Score", f"{overall:.2f}/10")
        
        # Individual scores
        if "scores" in eval_data:
            st.write("**Detailed Scores:**")
            for key, value in eval_data["scores"].items():
                formatted_key = key.replace("_", " ").title()
                st.write(f"• {formatted_key}: {value:.2f}/10")
        
        # Strengths and weaknesses
        if "strengths" in eval_data and eval_data["strengths"]:
            st.write("**Strengths:**")
            for strength in eval_data["strengths"]:
                st.write(f"✓ {strength}")
        
        if "weaknesses" in eval_data and eval_data["weaknesses"]:
            st.write("**Areas for Improvement:**")
            for weakness in eval_data["weaknesses"]:
                st.write(f"• {weakness}")
        
        if "suggestions" in eval_data and eval_data["suggestions"]:
            st.write(f"**Suggestions:** {eval_data['suggestions']}")


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
