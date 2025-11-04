# Neil deGrasse Tyson Chatbot 🔭

A sophisticated chatbot that emulates Neil deGrasse Tyson's distinctive communication style using both finetuned models and Retrieval-Augmented Generation (RAG). The system features an LLM-as-judge evaluation framework for continuous quality assessment.

## Features

### 🎯 Dual Approach Architecture
- **Finetuned Models**: LoRA-adapted models that excel at capturing Neil's personality and communication style
- **RAG Models**: Retrieval-augmented generation for factually grounded, accurate responses

### 🤖 Supported Models

#### Finetuned Models
- **Llama-3 8B**: LoRA finetuned on Neil deGrasse Tyson transcripts (HuggingFace endpoint)
- **Mistral 7B**: LoRA finetuned variant with sliding window attention (HuggingFace endpoint)
- **GPT-4o Mini**: OpenAI's efficient multimodal model, finetuned for style

#### RAG Models
- **Claude 3 Haiku**: Fast, efficient model via AWS Bedrock
- **Mistral 7B**: Long-context processing with sliding window attention via AWS Bedrock

### ⚖️ LLM-as-Judge Evaluation
- Automated quality assessment using GPT-5 (with GPT-4 fallback)
- Weighted scoring system:
  - Style Authenticity: 40%
  - Scientific Accuracy: 30%
  - Relevance: 20%
  - Clarity & Engagement: 10%

### 🎨 User Interface
- Clean Streamlit interface with model selection
- Real-time response streaming
- Evaluation metrics display
- Conversation history management
- Customizable system prompts and parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/avaxia888/Data-298B.git
cd Data-298B
```

2. Create a virtual environment:
```bash
python -m venv data_298b
source data_298b/bin/activate  # On Windows: data_298b\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY (for GPT-4o Mini and evaluation)
# - AWS_ACCESS_KEY_ID (for Bedrock)
# - AWS_SECRET_ACCESS_KEY (for Bedrock)
# - AWS_DEFAULT_REGION (for Bedrock)
# - PINECONE_API_KEY (for RAG vector database)
# - PINECONE_INDEX (your Pinecone index name)
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Navigate to `http://localhost:8501` in your browser

3. Choose between:
   - **Finetuned Models**: For personality-rich responses
   - **RAG Models**: For factually grounded information

4. Enable the GPT-5 Judge for real-time quality evaluation

## Project Structure

```
Data-298B/
├── app.py                 # Main application file
├── home.py               # Home page interface
├── llm_client.py         # LLM client for finetuned models
├── models.json           # Model configurations
├── evaluation/           # LLM-as-judge evaluation system
│   ├── llm_judge.py     # Judge implementation
│   ├── metrics.py       # Scoring and metrics
│   └── prompts.py       # Evaluation prompts
├── RAG/                  # RAG implementation
│   ├── llm_models.py    # Bedrock integration
│   └── utils.py         # RAG pipeline utilities
├── utils.py             # Shared utilities
└── requirements.txt     # Python dependencies
```

## Key Findings

### Performance Characteristics
- **LoRA Finetuning**: Excels at personality capture and style authenticity
- **RAG**: Superior factual accuracy and real-time information retrieval
- **Hybrid Approach**: Leverages strengths of both methods

### Technical Highlights
- Parameter-efficient finetuning using LoRA
- Semantic search with Pinecone vector database
- Multi-provider LLM integration (OpenAI, AWS Bedrock, HuggingFace)
- Automated evaluation with structured scoring

## Configuration

### Model Endpoints
Configure model endpoints in `models.json`:
- HuggingFace endpoints for finetuned models
- AWS Bedrock for RAG models
- OpenAI for GPT-4o Mini

### System Prompts
Default Neil deGrasse Tyson persona prompt can be customized in the sidebar

### Evaluation Weights
Adjust evaluation criteria weights in `evaluation/metrics.py`

## Requirements
- Python 3.9+
- 8GB RAM minimum
- Stable internet connection for API access
- Valid API keys for OpenAI, AWS, Pinecone, and HuggingFace

## License
Academic project for SJSU Data 298B

## Acknowledgments
- San Jose State University
- OpenAI, Anthropic, AWS, HuggingFace for model access
- Neil deGrasse Tyson for inspiration