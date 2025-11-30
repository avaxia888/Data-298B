# Neil deGrasse Tyson Chatbot with RAG Integration

This application combines fine-tuned models with Retrieval-Augmented Generation (RAG) to create an interactive chatbot that embodies Neil deGrasse Tyson's communication style.

## Features

### Three Interaction Modes:
1. **RAG + Prompt Engineering**: Uses Claude 3 Haiku or Mistral 7B with vector retrieval from Pinecone
2. **Finetuned Models**: Direct interaction with fine-tuned GPT-4o, Llama-3, or Mistral models
3. **RAG + Finetuned** (Coming Soon): Combines both approaches

### Key Capabilities:
- 🔍 Vector-based retrieval from knowledge base
- 💬 Maintains conversation context
- 📊 Displays retrieval quality metrics
- 🎛️ Adjustable temperature and token limits
- 🧠 Separate chat histories for each model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/avaxia888/Data-298B.git
cd Data-298B
git checkout rag-integration
```

2. Create virtual environment:
```bash
python -m venv data_298b
source data_298b/bin/activate  # On Windows: data_298b\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file with your API credentials:
```env
# For finetuned models
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_API_TOKEN=your_hf_token
HF_TOKEN=your_hf_token

# For RAG models
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=neil-degrasse-tyson-embeddings
```

## Usage

Run the application:
```bash
streamlit run app.py
```

Navigate to http://localhost:8501 in your browser.

## Project Structure

```
├── app.py                 # Main application file
├── home.py               # Home page with mode selection
├── llm_client.py         # Client for finetuned models
├── utils.py              # Utility functions
├── RAG/
│   ├── llm_models.py     # RAG model implementations
│   ├── utils.py          # RAG-specific utilities
│   └── main.py           # Standalone RAG app
├── models.json           # Finetuned model configurations
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables (not in git)
```

## Models Available

### RAG Models (AWS Bedrock):
- **Claude 3 Haiku**: Fast, efficient responses
- **Mistral 7B**: Open-source alternative

### Finetuned Models:
- **GPT-4o-mini**: OpenAI fine-tuned model
- **Llama-3 8B**: Meta's model with LoRA fine-tuning
- **Mistral 7B**: Mistral with LoRA fine-tuning

## Technologies Used

- **Frontend**: Streamlit
- **Vector Database**: Pinecone
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM Providers**: AWS Bedrock, OpenAI, HuggingFace
- **Language Models**: Claude, Mistral, GPT-4o, Llama-3

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of SJSU Data 298B coursework.
