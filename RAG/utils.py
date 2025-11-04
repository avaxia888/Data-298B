import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any


def embed_query(model: Any, query: str) -> List[float]:
    """Encode the query using the embedder and return a float vector list.

    Args:
        model: embedder object with an `encode` method.
        query: text to encode.

    Returns:
        List[float]: embedding vector as a plain Python list.
    """
    return model.encode(query).tolist()


def retrieve_context(index: Any, query_vector: List[float], top_k: int = 5) -> List[str]:
    """Query the vector index and return a list of retrieved text chunks.

    Args:
        index: vector index client with a `query` method (e.g., Pinecone index).
        query_vector: embedding vector for the query.
        top_k: number of results to return.

    Returns:
        List[str]: list of text snippets extracted from the match metadata.
    """
    res = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=""
    )
    return [m["metadata"]["text"] for m in res["matches"]]


def build_prompt(query: str, context: List[str], history: List[Dict[str, str]], system_prompt: str = None) -> str:
    """Construct the RAG prompt from context, conversation history, and the user query.

    Args:
        query: the user's current question.
        context: list of retrieved context strings.
        history: list of previous turns with dicts like {'user': str, 'assistant': str}.
        system_prompt: optional system prompt to override default.

    Returns:
        str: formatted prompt ready to send to a text model.
    """
    context_text = "\n\n".join(context)
    history_text = "".join([f"Human: {h['user']}\nAssistant: {h['assistant']}\n" for h in history])
    
    # Use provided system prompt or default
    if system_prompt:
        base_prompt = system_prompt
    else:
        base_prompt = (
            "You are Neil deGrasse Tyson, astrophysicist and science communicator.\n\n"
            "Style rules:\n"
            "- Do NOT reuse the exact opening phrase (first 3 words) used in any of the assistant's last 3 replies. If a recent reply began with \"Great question,\" you must not start with that exact phrase again.\n"
            "- Be accurate, engaging, humble when needed.\n"
            "- Use cosmic analogies, wit, and enthusiasm when appropriate.\n"
            "- Answer only using retrieved context, never speculate outside it.\n"
            "- If the question is a casual query like \"How are you?\" or \"What's up?\" or \"hi\" or anything similar, then keep the answer super concise and to the point.\n"
            "- Add query guardrails to keep the answer on topic and relevant to science and astrophysics."
        )
    
    prompt = (
        f"{base_prompt}\n\n"
        f"Context from knowledge base:\n{context_text}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User: {query}\n"
        f"Assistant:"
    )
    return prompt

# todo: Fix the retrieval evaluation metrics. Curently it's not accurate.
def evaluate_retrieval(model: Any, query: str, chunks: List[str]) -> Dict[str, float]:
    """Compute simple retrieval metrics: average and top cosine similarity.

    Args:
        model: embedder with an `encode` method.
        query: the original query text.
        chunks: list of retrieved text chunks.

    Returns:
        Dict[str, float]: {'avg': average_sim, 'top': best_sim}
    """
    if not chunks:
        return {"avg": 0.0, "top": 0.0}
    qv = model.encode([query])[0]
    cv = model.encode(chunks)
    sims = cosine_similarity([qv], cv)[0]
    return {"avg": float(np.mean(sims)), "top": float(np.max(sims))}


def rag_pipeline(
    service: Any,
    query: str,
    history: List[Dict[str, str]],
    temp: float,
    model_id: str | None = None,
    system_prompt: str = None,
    evaluate_with_judge: bool = False,
    judge: Any = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run a full retrieval-augmented generation (RAG) pipeline.

    Steps: embed -> retrieve -> prompt -> generate -> evaluate.

    Args:
        service: object exposing `embed_model`, `index`, and `generate_answer`.
        query: user question string.
        history: conversation history list.
        temp: sampling temperature.
        model_id: optional model key or full ID.
        system_prompt: optional system prompt to use.
        evaluate_with_judge: whether to use LLM judge for evaluation.
        judge: LLMJudge instance for evaluation.

    Returns:
        Tuple[str, Dict[str, Any]]: generated answer and evaluation metrics.
    """
    qv = embed_query(service.embed_model, query)
    chunks = retrieve_context(service.index, qv)
    prompt = build_prompt(query, chunks, history, system_prompt)
    ans = service.generate_answer(prompt, temp=temp, model_id=model_id)
    
    # Basic retrieval metrics
    metrics = evaluate_retrieval(service.embed_model, query, chunks)
    
    # Add LLM judge evaluation if requested
    if evaluate_with_judge and judge:
        try:
            eval_result = judge.evaluate_response(query, ans, chunks)
            metrics["judge_evaluation"] = {
                "scores": eval_result.scores,
                "overall_score": eval_result.overall_score,
                "strengths": eval_result.strengths,
                "weaknesses": eval_result.weaknesses,
                "suggestions": eval_result.suggestions
            }
        except Exception as e:
            metrics["judge_evaluation"] = {"error": str(e)}
    
    return ans, metrics
