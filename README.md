# Neil deGrasse Tyson AI Chatbot System

A comprehensive AI chatbot system that mimics Neil deGrasse Tyson's conversational style, featuring fine-tuned models, RAG (Retrieval Augmented Generation), and advanced evaluation frameworks.

## Overview

This project implements a multi-faceted approach to creating an AI that can respond in Neil deGrasse Tyson's distinctive style:
- **Fine-tuned Models**: Custom-trained on Tyson's actual responses
- **RAG System**: Retrieves relevant context from Tyson's knowledge base
- **LLM Council Evaluation**: Multi-judge evaluation system for style assessment
- **Web Interface**: Interactive Gradio-based chat application

## Project Structure

```
Data-298B/
├── app.py                          # Main Gradio web application
├── llm_client.py                   # LLM client for model interactions
├── models.json                     # Model configurations
├── rag.py                          # RAG implementation with Pinecone
├── system_prompt.py                # Tyson system prompt
├── ground_truth_evaluation.json   # 16 Q&A pairs for evaluation
├── evaluation.py                   # Basic evaluation script
├── model_evaluation.py             # Model comparison evaluation
├── llm_council_evaluation.py      # LLM Council evaluation (Karpathy-style)
├── visualize_council_results.py   # Visualization for council results
└── requirements.txt                # Project dependencies
```

## Available Models

### Fine-tuned Models (Group A)
- **tyson-ft-gpt-4o-mini**: Fine-tuned GPT-4o-mini on Tyson responses
- **llama3-ft-neil**: Fine-tuned Llama 3 8B
- **qwen-2.5-7b-merged-neil**: Fine-tuned Qwen 2.5 7B

### Base + RAG Models (Group B)
- **rag-claude-3.5-haiku**: Claude 3.5 Haiku with RAG
- **rag-gpt-4o-mini**: GPT-4o-mini with RAG
- **rag-llama3-router**: Llama 3 8B with RAG
- **rag-qwen25-router**: Qwen 2.5 7B with RAG

### Fine-tuned + RAG Models (Group C)
- Combination of fine-tuning and RAG for optimal performance

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Anthropic API key (for Claude models)
- Pinecone API key (for RAG)
- HuggingFace token (for open models)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/avaxia888/Data-298B.git
cd Data-298B
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-evaluation.txt  # For evaluation tools
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Running the Chatbot

### Web Interface
```bash
python app.py
```
Access at `http://localhost:7860`

### Command Line
```python
from llm_client import LLMClient
from rag import RagService

# Initialize
client = LLMClient()
rag = RagService()

# Get response
question = "What is dark matter?"
context = rag.get_context(question)
response = client.get_model_response("rag-claude-3.5-haiku", question, context)
print(response)
```

## Evaluation Systems

### 1. Basic Evaluation
Evaluates models on 16 ground truth Q&A pairs:
```bash
python evaluation.py
```

### 2. Model Comparison
Compares multiple models with cosine similarity:
```bash
python model_evaluation.py
```

### 3. LLM Council Evaluation (Advanced)
Multi-judge evaluation system based on Karpathy's LLM Council:

```bash
python llm_council_evaluation.py
```

**Features:**
- 4 diverse judges: GPT-4o, Claude-Sonnet-4.5, Gemini-2.5-Pro, DeepSeek-V3
- Parallel execution (~16 minutes for full evaluation)
- Focus on Tyson style scoring (0-10 scale)
- Comprehensive visualizations

**Visualize Results:**
```bash
python visualize_council_results.py
```

Generates:
- Model comparison charts
- Judge agreement heatmaps
- Score distribution analysis
- Group performance comparisons


## Configuration

### models.json
Configure available models and their endpoints:
```json
{
  "key": "model-identifier",
  "name": "Display Name",
  "mode": "openai|rag|huggingface",
  "model": "model-name",
  "base_url": "api-endpoint"
}
```

### System Prompt
Tyson's personality is defined in `system_prompt.py`. The prompt emphasizes:
- Cosmic perspective and wonder
- Accessible explanations
- Conversational, engaging tone
- Pop culture references

## Testing

### Quick Test
```python
# Test a single model
python -c "from llm_client import LLMClient; client = LLMClient(); print(client.get_model_response('rag-claude-3.5-haiku', 'Who are you?'))"
```

### Evaluation Test (1 question)
Edit `llm_council_evaluation.py` line 477:
```python
for qa_idx, qa_pair in enumerate(self.ground_truth[:1]):  # Test with 1 question
```

## RAG System

The RAG system uses Pinecone vector database with OpenAI embeddings:

### Features
- **Embedding Model**: text-embedding-3-small (1536 dimensions)
- **Vector Database**: Pinecone (tyson-knowledge index)
- **Context Retrieval**: Top 3 most relevant chunks
- **Fallback**: Returns empty context if service unavailable

### Answer Alignment Metrics
The system now includes meaningful metrics to evaluate answer quality:

#### Two Key Metrics

1. **Query-Answer Alignment** (0.0 to 1.0)
   - Measures how well the answer addresses the user's question
   - Calculated using cosine similarity between query and answer embeddings
   - High (≥0.7): Answer directly addresses the question
   - Medium (0.5-0.7): Answer somewhat relevant to question
   - Low (<0.5): Answer diverges from the question

2. **Context-Answer Alignment** (0.0 to 1.0)
   - Measures how well the answer uses retrieved context
   - Calculated using cosine similarity between retrieved chunks and answer embeddings
   - High (≥0.7): Answer closely reflects retrieved context
   - Medium (0.5-0.7): Answer partially uses context
   - Low (<0.5): Answer ignores or diverges from context

#### How It Works
1. **Embedding Generation**: Query, answer, and context are converted to embeddings using OpenAI's text-embedding-3-small
2. **Cosine Similarity Calculation**: 
   - Query vs Answer: Shows relevance
   - Context vs Answer: Shows groundedness

#### Why This Matters
- **Model Differentiation**: Different models show different alignment patterns
- **Quality Assessment**: Higher query alignment = better relevance; Higher context alignment = better groundedness
- **Hallucination Detection**: Low context alignment may indicate hallucination

### Usage
```python
from rag import RagService

rag = RagService()
context = rag.get_context("What is dark energy?")
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is for educational purposes as part of Data 298B coursework.

## Acknowledgments

- Neil deGrasse Tyson for the inspiration and reference content
- Andrej Karpathy for the LLM Council evaluation concept
- OpenAI, Anthropic, HuggingFace for model APIs
- Pinecone for vector database services

## Contact

For questions or issues, please open a GitHub issue or contact the repository owner.

---

**Note**: This is an educational project demonstrating advanced NLP techniques including fine-tuning, RAG, and multi-model evaluation systems.