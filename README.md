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
├── services/
│   ├── llm_client.py               # Unified LLM client for model interactions
│   └── rag.py                      # RAG implementation with Pinecone
├── models.json                     # Model configurations
├── prompt_template.py              # Tyson system prompt
├── utils.py                        # Utility functions for embeddings & RAG
├── ground_truth_evaluation.json   # 16 Q&A pairs for evaluation
├── model_evaluation.py             # Quantitative evaluation (cosine similarity)
├── llm_council_evaluation.py      # Qualitative evaluation (LLM judges)
├── visualize_evaluation_scores.py  # Generate evaluation visualizations
├── results/                        # Evaluation results and visualizations
│   ├── evaluation_results.json     # Quantitative evaluation results
│   ├── evaluation_scores_bar_chart.png
│   ├── category_comparison.png
│   └── performance_vs_efficiency.png
├── evaluation_system_report.md     # Comprehensive evaluation documentation
└── requirements.txt                # Project dependencies
```

## Available Models

### Fine-tuned Models (Group A)
- **tyson-ft-gpt-4o-mini**: Fine-tuned GPT-4o-mini on Tyson responses
- **llama3-ft-neil**: Fine-tuned Llama 3 8B
- **qwen-2.5-7b-merged-neil**: Fine-tuned Qwen 2.5 7B
- **gemma-3-ndtv3**: Fine-tuned Gemma-2 9B

### Base + RAG Models (Group B)
- **rag-claude-3.5-haiku**: Claude 3.5 Haiku with RAG
- **rag-gpt-4o-mini**: GPT-4o-mini with RAG
- **rag-llama3-router**: Llama 3 8B with RAG
- **rag-qwen25-router**: Qwen 2.5 7B with RAG

### Fine-tuned + RAG Models (Group C)
- Combination of fine-tuning and RAG for optimal performance
- Same models as Group A but augmented with RAG context retrieval

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Anthropic API key (for Claude models)
- Google API key (for Gemini models)
- DeepSeek API key (for DeepSeek judge)
- Pinecone API key (for RAG)
- HuggingFace token (for open models)
- AWS credentials (optional, for Bedrock models)

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
from services.llm_client import LLMClient, load_models_config
from services.rag import RagService

# Initialize
client = LLMClient()
rag = RagService()
models = load_models_config("models.json")

# Get response with RAG
question = "What is dark matter?"
model = next(m for m in models if m.key == "rag-claude-3.5-haiku")
response, context = rag.answer(question, [], 0.7, model)
print(response)
```

## Evaluation Systems

### 1. Quantitative Evaluation (model_evaluation.py)
Evaluates models using cosine similarity between embeddings:
```bash
python model_evaluation.py
```

**Features:**
- Tests 12 models across 3 configurations (base+RAG, fine-tuned, fine-tuned+RAG)
- Uses OpenAI text-embedding-3-small for embeddings
- Measures semantic similarity (0-1 scale)
- Response time tracking
- Results saved to `results/evaluation_results.json`

### 2. Qualitative Evaluation (llm_council_evaluation.py)
Multi-judge evaluation system based on Karpathy's LLM Council:

```bash
python llm_council_evaluation.py
```

**Features:**
- 4 diverse judges: GPT-4o, Claude-Sonnet-4.5, Gemini-2.0-Flash, DeepSeek-V3
- Optimized to use pre-generated answers from `results/evaluation_results.json`
- Fast parallel execution (~20-25 minutes for full evaluation)
- Focus on Tyson style scoring (0-10 scale)
- Scoring criteria:
  - Vocabulary (cosmic terminology): 3 points
  - Enthusiasm and wonder: 2 points
  - Educational storytelling: 2 points
  - Humor and accessibility: 2 points
  - Signature phrases: 1 point

**Performance Optimization:**
- Pre-generated answers reduce evaluation from 3+ hours to ~22 minutes
- Parallel judge evaluation using ThreadPoolExecutor
- 10x performance improvement over sequential approach

### 3. Visualizations
Generate comprehensive evaluation charts:
```bash
python visualize_evaluation_scores.py
```

Generates:
- Model performance bar charts
- Category comparison (base+RAG vs fine-tuned vs fine-tuned+RAG)
- Performance vs efficiency scatter plots
- Saved to `results/` directory


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
Tyson's personality is defined in `prompt_template.py`. The prompt emphasizes:
- Cosmic perspective and wonder
- Accessible explanations
- Conversational, engaging tone
- Pop culture references
- Educational storytelling approach

## Testing

### Quick Test
```python
# Test a single model
from services.llm_client import LLMClient, load_models_config
from services.rag import RagService

client = LLMClient()
rag = RagService()
models = load_models_config("models.json")
model = next(m for m in models if m.key == "rag-claude-3.5-haiku")
response, _ = rag.answer("Who are you?", [], 0.7, model)
print(response)
```

### Evaluation Test (Limited Questions)
To test with fewer questions, modify the ground truth loop:
```python
# In model_evaluation.py or llm_council_evaluation.py
for qa_pair in self.ground_truth[:1]:  # Test with 1 question
```

## RAG System

The RAG system uses Pinecone vector database with OpenAI embeddings:

### Features
- **Embedding Model**: text-embedding-3-small (1536 dimensions)
- **Vector Database**: Pinecone (tyson-embeddings-openai-1536 index)
- **Context Retrieval**: Top 5 most relevant chunks (configurable)
- **Query Rewriting**: Optional query optimization for better retrieval
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
from services.rag import RagService
from services.llm_client import load_models_config

rag = RagService()
models = load_models_config("models.json")
model = next(m for m in models if m.key == "rag-claude-3.5-haiku")

# Get response with context
response, context = rag.answer(
    query="What is dark energy?",
    history=[],
    temperature=0.7,
    endpoint=model
)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is for educational purposes as part of Data 298B coursework.

## Key Improvements & Features

### Evaluation System
- **Dual Evaluation Approach**: Quantitative (cosine similarity) and qualitative (LLM judges)
- **Performance Optimization**: 10x speedup through parallel processing and pre-generated answers
- **Comprehensive Metrics**: Response time, semantic similarity, and style scoring

### Model Categories
- **Group A**: Fine-tuned models capture Tyson's personality
- **Group B**: Base models with RAG for factual accuracy
- **Group C**: Fine-tuned + RAG combining personality and knowledge

### Technical Highlights
- **Unified LLM Client**: Single interface for multiple model providers
- **Robust Error Handling**: Retry logic and fallback mechanisms
- **Visualization Suite**: Automatic generation of performance charts
- **Modular Architecture**: Clean separation of services and utilities

## Acknowledgments

- Neil deGrasse Tyson for the inspiration and reference content
- Andrej Karpathy for the LLM Council evaluation concept
- OpenAI, Anthropic, Google, HuggingFace, DeepSeek for model APIs
- Pinecone for vector database services

## Contact

For questions or issues, please open a GitHub issue or contact the repository owner.

---

**Note**: This is an educational project demonstrating advanced NLP techniques including fine-tuning, RAG, and multi-model evaluation systems.