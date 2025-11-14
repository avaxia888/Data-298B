from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
from pinecone import Pinecone as PC
from sentence_transformers import SentenceTransformer
from utils import (
    build_rag_prompt,
    embed_query,
    evaluate_retrieval,
    extract_bedrock_text,
    get_env,
    retrieve_context,
    sanitize_output,
)


class RagService:
    def __init__(self, *, timeout: float = 60.0):
        self.timeout = timeout
        self._embed_model: Optional[SentenceTransformer] = None
        self._pinecone_index = None
        self._bedrock = None

    def _ensure(self) -> None:
        if self._embed_model is None:
            self._embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=False)

        if self._pinecone_index is None:
            api_key = get_env(["PINECONE_API_KEY"], required=True)
            index_name = get_env(["PINECONE_INDEX"], default="neil-degrasse-tyson-embeddings")
            self._pinecone_index = PC(api_key=api_key).Index(index_name)

        if self._bedrock is None:
            region = get_env(["AWS_REGION", "AWS_DEFAULT_REGION"], default="us-east-1")
            kwargs = {"region_name": region}
            if aws_key := os.getenv("AWS_ACCESS_KEY_ID"):
                if aws_secret := os.getenv("AWS_SECRET_ACCESS_KEY"):
                    kwargs.update({"aws_access_key_id": aws_key, "aws_secret_access_key": aws_secret})
            self._bedrock = boto3.client("bedrock-runtime", **kwargs)

    def answer(
        self,
        *,
        query: str,
        history: List[Dict[str, str]],
        temperature: float,
        model_id: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._ensure()
        qv = embed_query(self._embed_model, query)
        chunks = retrieve_context(self._pinecone_index, qv)
        prompt = build_rag_prompt(query, chunks, history, system_prompt, include_system=False)

        lower = (model_id or "").lower()
        if "anthropic" in lower:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 800,
                "temperature": temperature,
                "top_p": 0.9,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            }
        elif "mistral" in lower or lower.startswith("mistral."):
            payload = {"prompt": f"<s>[INST] {prompt} [/INST]", "max_tokens": 800, "temperature": temperature}
        else:
            payload = {"input": prompt, "temperature": temperature, "max_tokens": 800}

        res = self._bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(res["body"].read())
        raw_text = extract_bedrock_text(body)
        text = sanitize_output(raw_text)
        metrics = evaluate_retrieval(self._embed_model, query, chunks)
        return text, metrics
