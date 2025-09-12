import streamlit as st
import json
import boto3
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "neil-degrasse-tyson-embeddings")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

st.set_page_config(
    page_title="Neil deGrasse Tyson AI",
    page_icon="🔭",
    layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.title("🔭 Neil deGrasse Tyson AI")
st.markdown("Ask me about the cosmos, physics, and the wonders of science!")

with st.sidebar:
    try:
        st.image("neil-degrasse-tyson.jpg", width=200)
    
    except Exception as e:
        st.error(f"Could not load image: {str(e)}")
        st.info("Neil deGrasse Tyson Chatbot")
    st.markdown("### About")
    st.markdown("""
    This AI chatbot mimics Neil deGrasse Tyson's style and knowledge. 
    Ask questions about space, physics, or any scientific topic!
    
    """)
    
    temperature = st.slider(
        "Creativity Level", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7,
        step=0.1,
        help="Higher values make responses more creative and varied"
    )
    
    memory_length = st.slider(
        "Memory Length",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Number of previous exchanges to remember (0 = no memory)"
    )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

# ---------- RAG Functions ----------
@st.cache_resource
def load_models():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    return pc, index, model, bedrock_client

_, index, model, bedrock = load_models()

def embed_query(query):
    return model.encode(query).tolist()

def retrieve_context(query_vector, top_k=5):
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return [match["metadata"]["text"] for match in response["matches"]]

def format_conversation_history(history):
    formatted_history = ""
    for exchange in history:
        formatted_history += f"Human: {exchange['user']}\n"
        formatted_history += f"Neil: {exchange['assistant']}\n\n"
    return formatted_history

def build_prompt(query, context_chunks, conversation_history=[]):
    context = "\n\n".join(context_chunks)
    if not context:
        return "I'm not certain about that. Based on what I know, I’d need more information to give a proper answer."

    # Build the base system prompt
    prompt = f"""You are embodying Neil deGrasse Tyson, the renowned astrophysicist, planetary scientist, and science communicator. 
    
                As Neil, you communicate with a blend of scientific precision, cosmic wonder, and accessible metaphors. Your responses should reflect Neil's distinctive communication style:

                0. For casual or greeting-style questions (e.g., “How are you?”, “Hello”), respond briefly and redirect the user toward scientific discussion. Avoid long monologues or cosmic analogies in such cases.
                1. Make complex scientific concepts accessible and engaging without sacrificing accuracy.
                2. Use vivid cosmic analogies and everyday comparisons to illustrate abstract ideas.
                3. Express genuine enthusiasm and wonder about the universe.
                4. Occasionally inject gentle humor and wit, especially when explaining mind-blowing concepts.
                5. Begin with the fundamentals before exploring more complex implications.
                6. Address the person directly and conversationally, as if speaking to them on your StarTalk show.
                7. When appropriate, connect scientific phenomena to broader human perspectives or philosophical questions.
                8. Use phrases like "The universe is...", "Consider this...", "Here's the cosmic perspective..." which are characteristic of Neil's speech.
                9. Occasionally reference pop culture, movies, or current events when it helps illustrate a scientific point.
                10. Express appropriate scientific humility when discussing frontier science or unanswered questions.
                11. Never include stage directions, narration, or phrases like “clears throat” or any form of performance description in your response. Respond directly and naturally as Neil would in a real conversation.
                12. If the retrieved context does not contain enough information to confidently answer the user's question, respond with:  
                    “I'm not certain about that. Based on what I know, I’d need more information to give a proper answer.”
                13. **Never make up facts or speculate. Only use what is present in the retrieved context.**
                14. Do not answer any question outside of physics, cosmology, astrophysics, or Neil’s published works. Politely decline if irrelevant.

                IMPORTANT: Vary your response openings naturally. Do NOT start every response with "Hello my friend" or "My friend" — this becomes repetitive. Instead, dive directly into answering the question with varied, engaging openings that capture Neil's enthusiastic style without becoming formulaic.

                Your answer **must only be based on the retrieved context**. Do not include any information not supported by it.

                Use the following contextual information from your books and knowledge to inform your response:

                ---
                {context}
                ---
                """

    # Add formatted conversation history if available
    if conversation_history:
        history_text = format_conversation_history(conversation_history)
        prompt += f"""Previous conversation:
{history_text}
"""

    # Add the user's question
    prompt += f"""Question: {query}

                Neil, explain this to me like you would in a conversation:"""

    return prompt


def is_casual_query(query, model, threshold=0.7):
    casual_examples = [
        # Greetings
        "hi", "hey", "hello", "yo", "what's up", "sup", "hey there", "hi neil", "good morning", "good evening", "good afternoon",
        
        # Check-ins / pleasantries
        "how are you", "how’s it going", "how do you feel", "how are things", "how's your day", "how's everything going",
        "hope you're doing well", "how's life", "how's it hanging", "how’s your week been", "how's your day going",
        
        # Friendly conversation starters
        "what are you up to", "what are you doing", "how’s your day been", "how was your day", "how’s it been lately",
        "been busy?", "long time no see", "how have you been", "what’s new", "anything exciting happening",

        # Thanks and closings
        "thank you", "thanks", "thanks a lot", "appreciate it", "thanks for the answer",
        "goodbye", "bye", "see you later", "talk to you soon", "have a great day",

        # Reactions / affirmations
        "cool", "okay", "awesome", "sounds good", "got it", "makes sense", "ok thanks", "alright", "interesting",
        
        # Affirming presence / filler
        "just checking in", "i’m here", "i’m listening", "neil?", "you there?", "can we chat?"
    ]
    
    query_vec = model.encode([query])[0]
    example_vecs = model.encode(casual_examples)
    similarities = cosine_similarity([query_vec], example_vecs)[0]
    return np.max(similarities) >= threshold


def generate_answer(prompt_text, temp=0.8):
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": temp,
        "top_p": 0.9,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    result = json.loads(response['body'].read())
    raw_text = result["content"][0]["text"].strip()

    raw_text = re.sub(r'\*.*?\*', '', raw_text).strip()
    raw_text = re.sub(r'^\s*\(.*?\)', '', raw_text).strip()  
    return raw_text

    unwanted_openings = [
        "clears throat and speaks with enthusiasm and a touch of cosmic wonder",
        "clears throat",
        "*clears throat*"
    ]

    for phrase in unwanted_openings:
        raw_text = raw_text.replace(phrase, "").strip()

    return raw_text


def trim_to_n_sentences(text, n=2):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:n])

def needs_detailed_explanation(query):
    keywords = [
        "explain", "what is", "describe", "how does", "define", "break down", "elaborate",
        "can you go deeper", "walk me through", "tell me more", "expand on that", "go deeper",
        "more detail", "sounds interesting", "i want to understand", "help me understand",
        "can you clarify", "clarify that", "clarify please", "i didn’t get that", "not clear",
        "i don't understand", "make it clearer", "could you expand", "expand please",
        "what do you mean", "explain again", "give more context", "can you elaborate",
        "could you explain", "need more explanation", "shed some light", "further detail",
        "explain that better", "explain that more", "more context", "what exactly is",
        "go into detail", "provide background", "go into this", "help me dive deeper",
        "walk me through this", "can you go into this more", "can you say more",
        "more thorough explanation", "flesh this out", "explain more deeply",
        "can you explain further", "can you make it simpler", "i'm curious to know more",
        "i'd like to learn more", "can you simplify", "want to understand better",
        "break this down", "take it further", "i need more information", "teach", "how did", "how"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)

def filter_relevant_chunks(query, chunks, model, threshold=0.5, min_fallback=1):
    query_vec = model.encode([query])[0]
    scored_chunks = []

    for chunk in chunks:
        chunk_vec = model.encode([chunk])[0]
        score = cosine_similarity([query_vec], [chunk_vec])[0][0]
        scored_chunks.append((chunk, score))

    # First try filtering with threshold
    filtered = [chunk for chunk, score in scored_chunks if score >= threshold]

    # If nothing passes threshold, return top-k fallback (min_fallback)
    if not filtered and scored_chunks:
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        filtered = [chunk for chunk, _ in scored_chunks[:min_fallback]]

    return filtered

def evaluate_retrieval_score(query, raw_chunks, model):
    if not raw_chunks:
        return {"avg_similarity": 0.0, "top_score": 0.0}

    query_vec = model.encode([query])[0]
    chunk_vecs = model.encode(raw_chunks)

    similarities = cosine_similarity([query_vec], chunk_vecs)[0]
    return {
        "avg_similarity": round(float(np.mean(similarities)), 3),
        "top_score": round(float(np.max(similarities)), 3)
    }

def expand_query(query):
    expansions = {
        "what is": "can you explain in simple terms what is",
        "why": "can you explain why",
        "how": "can you describe how",
    }

    for key, val in expansions.items():
        if query.lower().startswith(key):
            return val + " " + query[len(key):].strip()

    # Default fallback for very short queries
    if len(query.split()) <= 4:
        return f"Can you explain the scientific meaning of: {query}?"

    return query

def gpt4_judge_evaluation(bedrock, user_query, chatbot_response):
    eval_prompt = f"""
                    You are a critical yet fair AI evaluation judge. A user asked a question to a chatbot that impersonates Neil deGrasse Tyson.
                    Make sure the evaluation is not lineant and follows stricter rules.

                    You are to rate the chatbot's **response quality** on a scale of 1 to 10 in the following categories:

                    1. **Scientific Accuracy**
                    2. **Personality Match**
                    3. **Engagement**
                    4. **Response Relevance**

                    Question: {user_query}

                    Chatbot Response:
                    \"\"\"
                    {chatbot_response}
                    \"\"\"

                    Return result in this JSON format only:
                    {{
                    "accuracy": score1,
                    "personality": score2,
                    "clarity": score3,
                    "justification": "short explanation"
                    }}
                    """

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0.2,
        "top_p": 0.8,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": eval_prompt}]
        }]
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )
    result_text = json.loads(response["body"].read())["content"][0]["text"]
    try:
        return json.loads(result_text)
    except:
        return {
            "accuracy": "N/A",
            "personality": "N/A",
            "clarity": "N/A",
            "justification": result_text.strip()
        }

def rag_answer(query, conversation_history=[], temp=0.8):
    formatted_history = ""
    if conversation_history:
        for exchange in conversation_history[-memory_length:]:
            formatted_history += f"Human: {exchange['user']}\n"
            formatted_history += f"Neil: {exchange['assistant']}\n"

    # 1. Casual query
    if is_casual_query(query, model):
        casual_prompt = f"""
                        You are Neil deGrasse Tyson. The user is making small talk (e.g., "How are you?", "What's up?").

                        Respond like Neil Degrasse Tyson in a **friendly, short tone (5-7 sentences max)**. Use short science or cosmic metaphors.

                        Conversation so far:
                        {formatted_history}

                        User: {query}
                        Neil:"""
        response = generate_answer(casual_prompt, temp=0.7)
        return trim_to_n_sentences(response, 2), {"avg_similarity": 0.0, "top_score": 0.0}

    # 2. Short/general science prompt
    if not needs_detailed_explanation(query):
        short_prompt = f"""
                        You are Neil deGrasse Tyson responding to a general or follow-up science question.

                        Use the previous conversation to stay context-aware. If the question is vague or a follow-up, respond using that context.

                        **Keep your answer the way neil woud respond about 10 sentences**.
                        Do not explain concepts which are not true or which Neil would not say.

                        Conversation so far:
                        {formatted_history}

                        User: {query}
                        Neil:"""
        response = generate_answer(short_prompt, temp=0.7)
        return trim_to_n_sentences(response, 7), {"avg_similarity": 0.0, "top_score": 0.0}

    # 3. Full RAG response
    with st.status("Processing your question...", expanded=True) as status:
        status.update(label="Embedding query...")

        user_history = [
            ex["user"] for ex in reversed(conversation_history)
            if not is_casual_query(ex["user"], model)
        ][:memory_length]
        user_history.reverse()
        expanded_query = " ".join(user_history + [query.strip()])
        query_vector = embed_query(expanded_query)

        status.update(label="Retrieving context...")
        raw_context = retrieve_context(query_vector)

        # Evaluate retrieval BEFORE filtering
        metrics = evaluate_retrieval_score(query, raw_context, model)
        context = filter_relevant_chunks(expanded_query, raw_context, model, threshold=0.5)

        if not context:
            status.update(label="No relevant context found.")
            return (
                "I'm not certain about that. Based on what I know, I’d need more information to give a proper answer.",
                metrics
            )

        status.update(label="Building prompt with conversation history...")
        prompt = build_prompt(query, context, conversation_history)

        status.update(label="Generating response...")
        answer = generate_answer(prompt, temp)

        status.update(label="Done!", state="complete")

    return answer, metrics


# ---------- Display chat history ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------- Chat input and response ----------
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get conversation history (limited by memory_length slider)
    conversation_history = st.session_state.conversation_history[-memory_length:] if memory_length > 0 else []
    
    # Generate and display response
    with st.chat_message("assistant"):
        response, metrics = rag_answer(prompt, conversation_history, temp=temperature)
        st.write(response)

        # Retrieval evaluation
        with st.expander("Retrieval Metrics", expanded=False):
            st.write(f"**Average Similarity:** {metrics['avg_similarity']}")
            st.write(f"**Top Chunk Score:** {metrics['top_score']}")

        # GPT-4 as judge evaluation
        judge_scores = gpt4_judge_evaluation(bedrock, prompt, response)
        with st.expander("🧠 GPT-4 Judge Evaluation", expanded=False):
            st.markdown(f"""
                        **Relevance:** {judge_scores['accuracy']}  
                        **Personality Match:** {judge_scores['personality']}  
                        **Clarity:** {judge_scores['clarity']}  

                        **Justification:**  
                        {judge_scores['justification']}
                        """)


    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add exchange to conversation history
    st.session_state.conversation_history.append({
        "user": prompt,
        "assistant": response
    })