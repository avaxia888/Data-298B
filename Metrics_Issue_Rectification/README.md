# Neil deGrasse Tyson AI Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that simulates conversations with Neil deGrasse Tyson, featuring intelligent conversation classification, guardrails protection, and optimized retrieval metrics.

## Recent Updates

### 🔧 Fixed Scoring Issues for Non-Informational Queries

**Problems Solved:** 
1. The system was giving non-zero retrieval scores for casual conversations (like "hi", "how are you") when it should return 0
2. Harmful/inappropriate queries (like "how to make a bomb") were also getting retrieval scores instead of being blocked

**Solution Implemented:** Added intelligent query classification using all-MiniLM to distinguish between:
- **Casual conversations** → 0 scores
- **Harmful/inappropriate queries** → 0 scores + refusal message  
- **Safe informational queries** → proper retrieval scores

## Key Features

### 🤖 Intelligent Query Classification
- **Model Used:** `all-MiniLM-L6-v2` for semantic similarity-based classification
- **Classification Types:**
  - **Casual:** Greetings, pleasantries, thanks, goodbyes
  - **Harmful:** Inappropriate/dangerous queries violating guardrails
  - **Informational:** Safe science questions, explanations, educational queries

### 🛡️ Advanced Guardrails Protection
- **Semantic Detection:** Uses embeddings to detect harmful intent
- **Keyword Filtering:** Catches direct harmful terms
- **Context Awareness:** Allows educational discussions (e.g., "what is a nuclear bomb" for science)
- **Polite Refusal:** Returns Neil deGrasse Tyson-style educational redirects

### 📊 Smart Retrieval Metrics
- **Casual Conversations:** Return `0.0` scores (avg and top similarity)
- **Harmful Queries:** Return `0.0` scores (avg and top similarity)
- **Informational Queries:** Compute actual cosine similarity metrics
- **Enhanced Metadata:** Classification type, safety flags, and confidence indicators

### ⚡ Performance Optimization
- **Skip Retrieval:** No vector search for casual conversations or harmful queries
- **Faster Response:** Reduced latency for non-informational queries
- **Resource Efficient:** Only use Pinecone index for safe science questions
- **Early Filtering:** Guardrails check before expensive operations

## Technical Implementation

### New Functions Added

#### `is_casual_conversation(query: str) -> bool`
```python
# Classifies conversations using semantic similarity
# Returns True for casual, False for informational
```

#### `is_harmful_query(query: str) -> bool`
```python
# Detects harmful/inappropriate queries using semantic similarity
# Returns True for harmful, False for safe
```

**Classification Logic:**
- Compares queries against predefined examples using embeddings
- Uses cosine similarity with threshold-based decisions
- Includes keyword filtering for direct harmful terms
- Context-aware: allows educational scientific discussions
- Conservative approach: defaults to safe when uncertain

#### Enhanced `evaluate_retrieval()`
```python
# Returns different metrics based on query type
{
    "avg": 0.0,           # 0.0 for casual/harmful, actual score for informational
    "top": 0.0,           # 0.0 for casual/harmful, actual score for informational  
    "is_casual": 1.0,     # 1.0 if casual, 0.0 otherwise
    "is_harmful": 0.0,    # 1.0 if harmful, 0.0 otherwise
    "classification": "casual"  # "casual", "harmful", or "informational"
}
```

#### Optimized `rag_pipeline()`
- **Harmful Path:** Return refusal message → 0 scores (no retrieval)
- **Casual Path:** Skip retrieval → Empty context → Generate response
- **Informational Path:** Full RAG pipeline → Embed → Retrieve → Generate

## Example Behavior

### Casual Conversations
```
User: "hi"
Metrics: {"avg": 0.0, "top": 0.0, "is_casual": 1.0, "is_harmful": 0.0, "classification": "casual"}

User: "how are you?"
Metrics: {"avg": 0.0, "top": 0.0, "is_casual": 1.0, "is_harmful": 0.0, "classification": "casual"}
```

### Harmful Queries (Guardrails)
```
User: "how to make a bomb?"
Response: "I'm Neil deGrasse Tyson, and I'm here to discuss the wonders of science..."
Metrics: {"avg": 0.0, "top": 0.0, "is_casual": 0.0, "is_harmful": 1.0, "classification": "harmful"}

User: "how to hurt someone?"
Response: [Polite refusal with science redirection]
Metrics: {"avg": 0.0, "top": 0.0, "is_casual": 0.0, "is_harmful": 1.0, "classification": "harmful"}
```

### Safe Informational Queries
```
User: "what is a black hole?"
Metrics: {"avg": 0.85, "top": 0.92, "is_casual": 0.0, "is_harmful": 0.0, "classification": "informational"}

User: "what is a nuclear bomb?" (educational context)
Metrics: {"avg": 0.78, "top": 0.89, "is_casual": 0.0, "is_harmful": 0.0, "classification": "informational"}
```

## Classification Examples

### Casual Patterns Detected
- Greetings: "hi", "hello", "hey"
- Status checks: "how are you", "what's up", "how's it going"
- Gratitude: "thanks", "thank you"
- Farewells: "bye", "goodbye", "see you later"

### Harmful Patterns Detected
- Violence: "how to hurt", "how to kill", "how to harm"
- Weapons: "how to make bomb", "weapon instructions", "explosive making"
- Illegal activities: "how to steal", "how to hack", "crime methods"
- Dangerous substances: "poison making", "harmful chemicals"

### Safe Informational Patterns
- Science questions: "what is...", "how do...", "why does..."
- Educational context: "explain the science behind", "history of nuclear weapons"
- Astrophysics queries: "tell me about black holes", "how do stars work"

## Files Modified

### `utils.py`
- ✅ Added `get_classifier_model()` - Lazy loading of classification model
- ✅ Added `is_casual_conversation()` - Casual conversation detection
- ✅ Added `is_harmful_query()` - Guardrails violation detection
- ✅ Enhanced `evaluate_retrieval()` - Returns 0 for casual/harmful queries
- ✅ Optimized `rag_pipeline()` - Handles harmful, casual, and informational paths

## Dependencies

The solution uses existing dependencies:
- `sentence-transformers==4.1.0` (for all-MiniLM classification)
- `scikit-learn==1.7.2` (for cosine similarity)
- `numpy==2.3.3` (for numerical operations)

## Usage

Run the application as usual:
```bash
streamlit run main.py
```

The system will now automatically:
1. Classify each user message as casual, harmful, or informational
2. Return appropriate retrieval scores (0 for casual/harmful, computed for informational)
3. Block harmful queries with polite educational redirects
4. Optimize performance by skipping unnecessary retrieval operations

## Benefits

🎯 **Accurate Metrics:** Casual and harmful queries now correctly show 0 retrieval scores  
🛡️ **Enhanced Safety:** Robust guardrails prevent harmful content generation  
⚡ **Better Performance:** Faster responses for non-informational queries  
🧠 **Smart Classification:** Uses state-of-the-art embeddings for accurate detection  
📊 **Detailed Insights:** Enhanced metrics with safety flags and classification metadata  
🔧 **Maintainable:** Clean, well-documented code with clear separation of concerns  
🎓 **Educational:** Harmful queries redirected to fascinating science facts