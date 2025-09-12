# Neil deGrasse Tyson AI Chatbot

An AI-powered chatbot that emulates the personality and communication style of Neil deGrasse Tyson, making science education more engaging and accessible.

## 🎯 Project Overview

### Background
This project aims to transform science education by creating an AI chatbot that speaks in the engaging, humorous, and accessible style of Neil deGrasse Tyson. By combining his personality traits with scientific knowledge, the chatbot makes learning science feel more like entertainment than formal education.

### Problem Statement
Traditional science education methods often fail to maintain engagement and can seem boring or intimidating to many learners. This project addresses the problem of low engagement in science education by providing an interactive, personality-driven learning experience.

### Solution Approach
- **Data Collection**: Gather Tyson's personality traits from tweets, books, Wikipedia, and quotations
- **Knowledge Base**: Store text chunks in DynamoDB and vector embeddings in Pinecone
- **RAG Pipeline**: Retrieve relevant context and generate responses in Tyson's style using LLMs
- **Infrastructure**: Deploy on AWS for scalability and reliability

## 🚀 Features

- Interactive conversational interface via Streamlit
- Personality-driven responses mimicking Neil deGrasse Tyson's communication style
- Vector-based semantic search for relevant content retrieval
- Support for various data sources (tweets, books, quotes, Wikipedia)
- Scalable cloud-based architecture

## 📋 Prerequisites

- Python 3.8+
- AWS Account with appropriate permissions
- Pinecone account and API key
- Git

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/avaxia888/Data-298B.git
   cd Data-298B
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your credentials:
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_INDEX`: Your Pinecone index name
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `AWS_REGION`: AWS region (default: us-east-1)

## 📁 Project Structure

```
.
├── app.py                    # Main Streamlit application
├── book_chunking.py          # Process and chunk book content
├── quote_chunking.py         # Process and chunk quotes
├── twitter_chunking.py       # Process and chunk tweets
├── wikipedia.py              # Process Wikipedia content
├── Tyson_Embedder.ipynb     # Jupyter notebook for creating embeddings
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
└── .gitignore               # Git ignore file
```

## 🔧 Module Descriptions

### `app.py`
Main application file that:
- Provides the Streamlit web interface
- Handles user queries
- Performs semantic search using Pinecone
- Generates responses using the RAG pipeline
- Maintains conversation context

### `book_chunking.py`
- Processes Neil deGrasse Tyson's book content
- Splits text into manageable chunks
- Prepares data for embedding generation

### `quote_chunking.py`
- Extracts and processes famous quotes
- Structures quote data with metadata
- Maintains attribution and context

### `twitter_chunking.py`
- Fetches and processes tweets
- Handles tweet-specific formatting
- Preserves engagement metrics and timestamps

### `wikipedia.py`
- Extracts relevant Wikipedia content
- Processes biographical and scientific information
- Maintains source references

### `Tyson_Embedder.ipynb`
- Generates vector embeddings using sentence transformers
- Uploads embeddings to Pinecone
- Provides data pipeline visualization

## 💡 Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

3. **Start chatting**
   Ask questions about science, space, or any topic you're curious about!

## 🏗️ Architecture

```
User Query → Streamlit UI → Embedding Generation
                                    ↓
                             Pinecone Search
                                    ↓
                           Context Retrieval
                                    ↓
                          LLM Response Generation
                                    ↓
                          Tyson-style Response
```

## 🔐 Security

- Never commit `.env` files with actual credentials
- Use environment variables for all sensitive information
- Rotate credentials regularly
- Follow AWS IAM best practices for minimal permissions

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is for educational purposes as part of Data 298B coursework.

## ⚠️ Important Notes

- **Rotate any exposed credentials immediately**
- Keep your `.env` file secure and never commit it
- Monitor AWS usage to avoid unexpected charges
- Ensure Pinecone index is properly configured before running

## 🙏 Acknowledgments

- Neil deGrasse Tyson for inspiring scientific communication
- AWS for cloud infrastructure
- Pinecone for vector database services
- Streamlit for the web framework