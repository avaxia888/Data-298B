import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer

# Initialize classifier model for conversation type detection
_classifier_model = None

def get_classifier_model():
    """Get or initialize the classifier model for conversation type detection."""
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _classifier_model

def is_casual_conversation(query: str) -> bool:
    """Classify if a conversation is casual vs informational using semantic similarity.
    
    Args:
        query: the user's question/message
        
    Returns:
        bool: True if casual conversation, False if informational
    """
    # Define examples of casual vs informational conversations
    casual_examples = [
        "hi", "hello", "hey", "how are you", "what's up", "good morning", 
        "good evening", "how's it going", "nice to meet you", "thanks", 
        "thank you", "bye", "goodbye", "see you later", "have a good day",
        "how are you doing", "what's new", "how's your day", "take care"
    ]
    
    informational_examples = [
        "what is a black hole", "how do stars form", "explain quantum physics",
        "what is the universe made of", "how far is the nearest star",
        "what causes gravity", "explain relativity", "how big is the solar system",
        "what are exoplanets", "how do galaxies form", "what is dark matter",
        "explain the big bang theory", "how do telescopes work"
    ]
    
    classifier = get_classifier_model()
    
    # Get embeddings
    query_embedding = classifier.encode([query.lower().strip()])
    casual_embeddings = classifier.encode(casual_examples)
    informational_embeddings = classifier.encode(informational_examples)
    
    # Calculate similarities
    casual_similarities = cosine_similarity(query_embedding, casual_embeddings)[0]
    informational_similarities = cosine_similarity(query_embedding, informational_embeddings)[0]
    
    # Get max similarities
    max_casual_sim = np.max(casual_similarities)
    max_informational_sim = np.max(informational_similarities)
    
    # Additional heuristics for better classification
    query_lower = query.lower().strip()
    
    # Check for very short queries (likely casual)
    if len(query_lower.split()) <= 3:
        # Common casual patterns
        casual_patterns = ['hi', 'hello', 'hey', 'thanks', 'bye', 'how are you', 'what\'s up']
        if any(pattern in query_lower for pattern in casual_patterns):
            return True
    
    # If query is very short and doesn't contain science keywords, likely casual
    science_keywords = ['what', 'how', 'why', 'explain', 'universe', 'star', 'planet', 'physics', 
                       'space', 'galaxy', 'black hole', 'gravity', 'quantum', 'relativity']
    
    if len(query_lower.split()) <= 4 and not any(keyword in query_lower for keyword in science_keywords):
        return True
    
    # Use similarity threshold with bias toward informational
    # We want to be conservative - only classify as casual if we're quite confident
    threshold_difference = 0.1  # Casual must be significantly more similar
    
    return max_casual_sim > max_informational_sim + threshold_difference




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


def build_prompt(query: str, context: List[str], history: List[Dict[str, str]]) -> str:
    """Construct the RAG prompt from context, conversation history, and the user query.

    Args:
        query: the user's current question.
        context: list of retrieved context strings.
        history: list of previous turns with dicts like {'user': str, 'assistant': str}.

    Returns:
        str: formatted prompt ready to send to a text model.
    """
    context_text = "\n\n".join(context)
    history_text = "".join([f"Human: {h['user']}\nNeil: {h['assistant']}\n" for h in history])
    prompt = (
        f"You are Neil deGrasse Tyson, astrophysicist and science communicator.\n\n"
        f"Style rules:\n"
        f"- Do NOT reuse the exact opening phrase (first 3 words) used in any of the assistant's last 3 replies. If a recent reply began with \"Great question,\" you must not start with that exact phrase again.\n"
        f"- Be accurate, engaging, humble when needed.\n"
        f"- Use cosmic analogies, wit, and enthusiasm when appropriate.\n"
        f"- Answer only using retrieved context, never speculate outside it.\n"
        f"- If the question is a casual query like \"How are you?\" or \"What's up?\" or \"hi\" or anything similar, then keep the answer super concise and to the point.\n"
        f"- Keep the answer on topic and relevant to science and astrophysics.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User: {query}\n"
        f"Neil:"
    )
    return prompt

def evaluate_retrieval(model: Any, query: str, chunks: List[str]) -> Dict[str, float]:
    """Compute retrieval metrics: average and top cosine similarity.
    
    Returns 0 scores for casual conversations (greetings, pleasantries).
    Computes actual similarity metrics for informational queries.

    Args:
        model: embedder with an `encode` method.
        query: the original query text.
        chunks: list of retrieved text chunks.

    Returns:
        Dict[str, float]: metrics with classification info
    """
    # Check if this is a casual conversation
    if is_casual_conversation(query):
        return {
            "avg": 0.0, 
            "top": 0.0, 
            "is_casual": 1.0,
            "classification": "casual"
        }
    
    # For informational queries, compute actual metrics
    if not chunks:
        return {
            "avg": 0.0, 
            "top": 0.0, 
            "is_casual": 0.0,
            "classification": "informational"
        }
    
    qv = model.encode([query])[0]
    cv = model.encode(chunks)
    sims = cosine_similarity([qv], cv)[0]
    
    return {
        "avg": float(np.mean(sims)), 
        "top": float(np.max(sims)),
        "is_casual": 0.0,
        "classification": "informational"
    }


def rag_pipeline(
    service: Any,
    query: str,
    history: List[Dict[str, str]],
    temp: float,
    model_id: str | None = None,
) -> Tuple[str, Dict[str, float]]:
    """Run a full retrieval-augmented generation (RAG) pipeline.

    Handles different query types:
    - Casual conversations: Skips retrieval, uses minimal context
    - Informational queries: Full RAG pipeline

    Args:
        service: object exposing `embed_model`, `index`, and `generate_answer`.
        query: user question string.
        history: conversation history list.
        temp: sampling temperature.
        model_id: optional model key or full ID.

    Returns:
        Tuple[str, Dict[str, float]]: generated answer and retrieval metrics.
    """
    # Check if this is a casual conversation
    if is_casual_conversation(query):
        # For casual conversations, skip retrieval and use minimal context
        prompt = build_prompt(query, [], history)  # Empty context for casual chat
        ans = service.generate_answer(prompt, temp=temp, model_id=model_id)
        metrics = {
            "avg": 0.0, 
            "top": 0.0, 
            "is_casual": 1.0,
            "classification": "casual"
        }
        return ans, metrics
    
    # For informational queries, run full RAG pipeline
    qv = embed_query(service.embed_model, query)
    chunks = retrieve_context(service.index, qv)
    prompt = build_prompt(query, chunks, history)
    ans = service.generate_answer(prompt, temp=temp, model_id=model_id)
    metrics = evaluate_retrieval(service.embed_model, query, chunks)
    return ans, metrics
