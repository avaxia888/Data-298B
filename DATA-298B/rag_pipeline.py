import json
import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import boto3

# -----------------------------------------------------------------------------
# Load environment variables (.env file with AWS credentials, etc.)
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Helper: Bedrock client wrapper (returns None if credentials unavailable)
# -----------------------------------------------------------------------------

def get_bedrock_client():
    """Return an AWS Bedrock runtime client if credentials are present."""
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        return boto3.client("bedrock-runtime", region_name=region)
    # Fallback to AWS CLI/instance credentials
    try:
        return boto3.client("bedrock-runtime")
    except Exception:
        return None

# -----------------------------------------------------------------------------
# RAG Pipeline
# -----------------------------------------------------------------------------

class RAGPipeline:
    """Minimal Retrieval-Augmented Generation pipeline using Chroma & Bedrock."""

    def __init__(
        self,
        data_path: str = "data.json",
        chroma_dir: str = "./chroma_db",
        embed_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_collection",
    ) -> None:
        self.data_path = data_path
        self.embed_model_name = embed_model
        self.bedrock_model_id = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
        )

        # Embeddings
        self.embed_model = SentenceTransformer(embed_model)

        # Vector store (persistent)
        self.chroma = chromadb.PersistentClient(path=chroma_dir)
        try:
            self.col = self.chroma.get_collection(collection_name)
        except Exception:
            self.col = self.chroma.create_collection(collection_name, metadata={"hnsw:space": "cosine"})

        # Ingest if empty
        if self.col.count() == 0:
            self._index_documents()

        # Bedrock client (may be None)
        self.bedrock = get_bedrock_client()

    # ---------------------------- Indexing ----------------------------------

    def _load_data(self) -> List[Dict[str, str]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        docs = []
        for idx, item in enumerate(raw):
            conv = item.get("conversations", [])
            q = next((x["value"] for x in conv if x["from"] == "human"), None)
            a = next((x["value"] for x in conv if x["from"] == "gpt"), None)
            if q and a:
                docs.append({"id": f"doc_{idx}", "text": f"Q: {q}\nA: {a}", "q": q, "a": a})
        return docs

    def _index_documents(self):
        print("[RAG] Indexing documents into Chroma…")
        docs = self._load_data()
        embeds = self.embed_model.encode([d["text"] for d in docs], show_progress_bar=True)
        self.col.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            embeddings=[e.tolist() for e in embeds],
            metadatas=[{"q": d["q"], "a": d["a"]} for d in docs],
        )
        print(f"[RAG] Indexed {len(docs)} docs.")

    # ---------------------------- Retrieval ----------------------------------

    def retrieve(self, query: str, k: int = 3):
        q_emb = self.embed_model.encode([query])
        res = self.col.query(query_embeddings=q_emb.tolist(), n_results=k)
        docs = []
        if res["documents"]:
            for i in range(len(res["documents"][0])):
                docs.append({
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "q": res["metadatas"][0][i]["q"],
                    "a": res["metadatas"][0][i]["a"],
                })
        return docs

    # ---------------------------- Generation ---------------------------------

    def answer(self, query: str, k: int = 3, use_bedrock: bool = True) -> str:
        docs = self.retrieve(query, k)
        if not docs:
            return "No relevant information found."

        context = "\n\n".join([d["text"] for d in docs])

        if use_bedrock and self.bedrock is not None:
            prompt = (
                "Human: Using the following context, answer the question. "
                "If the context is not helpful say so.\n\n" +
                f"Context:\n{context}\n\nQuestion: {query}\n\nAssistant:"
            )
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": prompt}],
            })
            try:
                resp = self.bedrock.invoke_model(modelId=self.bedrock_model_id, body=body)
                data = json.loads(resp["body"].read())
                return data["content"][0]["text"].strip()
            except Exception as e:
                print(f"[Bedrock error] {e}. Falling back to retrieval-only response.")

        # Simple retrieval answer (first doc's answer)
        return docs[0]["a"]

# -----------------------------------------------------------------------------
# CLI demo when run directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rag = RAGPipeline()
    print("RAG ready. Type your questions (or 'quit').")
    while True:
        try:
            q = input("\n> ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            print("\n" + rag.answer(q))
        except KeyboardInterrupt:
            break
