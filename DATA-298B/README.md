# Retrieval-Augmented Generation (RAG) Pipeline

Local semantic search over `data.json` powered by ChromaDB + Sentence-Transformers and optional answer generation with AWS Bedrock (Claude Sonnet).

---

## 1. Setup

```bash
# install user-level deps (no sudo)
python3 -m pip install --upgrade pip
python3 -m pip install --user -r requirements.txt
```

### Environment variables
Create a `.env` file (or edit the one provided) with your Bedrock credentials:

```ini
AWS_ACCESS_KEY_ID=…
AWS_SECRET_ACCESS_KEY=…
AWS_DEFAULT_REGION=us-east-1  # or your region
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0  # default
```
If you omit the keys the pipeline still works in **retrieval-only** mode.

---

## 2. Run interactive demo

```bash
python3 rag_pipeline.py
```
First launch downloads the MiniLM embedding model (~80 MB), embeds & indexes the data (746 docs) into `chroma_db/`, then opens a REPL:

```
RAG ready. Type your questions (or 'quit').
> Tell me about E.T. being a plant
…
```

---

## 3. Programmatic usage

```python
from rag_pipeline import RAGPipeline
rag = RAGPipeline()              # builds/loads index automatically

print(rag.answer("Who is Jake Roper?"))        # string response
hits = rag.retrieve("aliens", k=5)             # raw nearest docs w/ distances
```

---

## 4. Project structure

```
├── data.json           # conversation dataset
├── rag_pipeline.py     # minimal end-to-end pipeline (this is what you run)
├── rag_bedrock.py      # full class version (alternative, heavier)
├── requirements.txt    # pinned deps (numpy<2, transformers<4.31, …)
├── .env.example        # creds template
└── chroma_db/          # vector store created on first run
```

---

## 5. Troubleshooting

| Issue                                     | Fix                                                         |
|-------------------------------------------|-------------------------------------------------------------|
| `np.float_ removed` error                 | `python3 -m pip install --user "numpy<2.0"`                |
| `cached_download` import error            | `python3 -m pip install --user "huggingface_hub<0.20"`     |
| Bedrock permission / quota errors         | Check IAM permissions, or set `use_bedrock=False`          |
| Telemetry warnings from Chroma            | Safe to ignore                                              |

---

## 6. Evaluation / scores (optional)
Use the `distance` returned by `rag.retrieve()` for similarity or build recall@k metrics over a labeled set.

```python
def recall_at_k(pairs, k=3):
    hits = 0
    for q, gold_id in pairs:
        if any(d['id'] == gold_id for d in rag.retrieve(q, k)):
            hits += 1
    return hits / len(pairs)
```

---

**Enjoy exploring your conversational data with RAG!**
