from __future__ import annotations

import json
import os
import math
from dataclasses import replace as _dc_replace
from typing import Any, Dict, List, Optional, Tuple

import boto3
import httpx
from pinecone import Pinecone as PC
from openai import OpenAI

from utils import (
    build_rag_prompt,
    embed_query,
    evaluate_retrieval,
    extract_bedrock_text,
    get_env,
    retrieve_context,
    sanitize_output,
)
from prompt_template import DEFAULT_SYSTEM_PROMPT
from services.llm_client import LLMClient, EndpointConfig as _EP


class RagService:
    def __init__(self):
        self._openai_client: Optional[OpenAI] = None
        self._pinecone_index = None
        self._bedrock = None

    def _ensure(self) -> None:
        # Initialize OpenAI client for text-embedding-3-small
        if self._openai_client is None:
            api_key = get_env(["OPENAI_API_KEY"], required=True)
            self._openai_client = OpenAI(api_key=api_key)

        if self._pinecone_index is None:
            api_key = get_env(["PINECONE_API_KEY"], required=True)
            index_name = get_env(["PINECONE_INDEX"], default="tyson-embeddings-openai-1536")
            self._pinecone_index = PC(api_key=api_key).Index(index_name)

        if self._bedrock is None:
            region = get_env(["AWS_REGION", "AWS_DEFAULT_REGION"], default="us-east-1")
            kwargs = {"region_name": region}

            # Use explicit credentials if provided; boto3 automatically uses session tokens if present
            if aws_key := os.getenv("AWS_ACCESS_KEY_ID"):
                if aws_secret := os.getenv("AWS_SECRET_ACCESS_KEY"):
                    kwargs.update({"aws_access_key_id": aws_key, "aws_secret_access_key": aws_secret})

            self._bedrock = boto3.client("bedrock-runtime", **kwargs)


    # Bedrock helper wrapper
    def _invoke_bedrock(self, model_id: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        # Helper for standard Bedrock invocation; returns parsed JSON body
        res = self._bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        return json.loads(res["body"].read())

    # Internal Enhancements

    def _rewrite_query(self, model_id: str, query: str) -> str:
        # Internal rewriting to improve retrieval signal; user-facing prompt unchanged
        rewrite_prompt = (
            "Rewrite this user query into a short, retrieval-optimized query. "
            "Keep it concise (1 sentence) and add key technical terms, synonyms, and specific nouns.\n\n"
            f"User query: {query}\n\nRewritten:"
        )
        payload = {"input": rewrite_prompt, "temperature": 0.0, "max_tokens": 64}
        body = self._invoke_bedrock(model_id, payload)
        text = extract_bedrock_text(body)
        return (text or query).strip()

    def _score_candidates_with_bedrock(self, model_id: str, query: str, candidates: List[str]) -> List[float]:
        # Rerank candidates using Bedrock to produce 0.0–1.0 relevance scores
        scores = []
        for chunk_start in range(0, len(candidates), 5):
            batch = candidates[chunk_start : chunk_start + 5]
            items_text = "\n\n".join(f"---\nText:\n{c}\n---" for c in batch)

            scoring_prompt = (
                "Rate the relevance of each 'Text' to the Query on a 0.0-1.0 scale where 1.0 is perfectly relevant. "
                "Return only a JSON array of numbers in the same order as the texts.\n\n"
                f"Query: {query}\n\n{items_text}\n\nReturn JSON array:"
            )

            payload = {"input": scoring_prompt, "temperature": 0.0, "max_tokens": 128}
            body = self._invoke_bedrock(model_id, payload)
            scored_text = extract_bedrock_text(body)

            # Parse JSON array of scores
            batch_scores = []
            try:
                parsed = json.loads(scored_text)
                if isinstance(parsed, list):
                    batch_scores = [float(min(max(0.0, float(s)), 1.0)) for s in parsed]
            except Exception:
                # Fallback: attempt to extract floats
                import re
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", scored_text)
                batch_scores = [min(max(0.0, float(n)), 1.0) for n in nums[: len(batch)]]

            # If extraction failed, default mid-score
            if len(batch_scores) != len(batch):
                batch_scores = [0.5] * len(batch)

            scores.extend(batch_scores)

        return scores

    def _compress_chunk(self, model_id: str, chunk: str) -> str:
        # Summarize chunky text into a short factual paragraph to reduce token load
        summary_prompt = (
            "Summarize the following text into a concise factual paragraph (1-2 sentences) that preserves the "
            "key facts and numbers. Do NOT add new information.\n\n"
            f"Text:\n{chunk}\n\nSummary:"
        )
        payload = {"input": summary_prompt, "temperature": 0.0, "max_tokens": 120}
        body = self._invoke_bedrock(model_id, payload)
        return (extract_bedrock_text(body) or "").strip()

    def _validate_grounding(self, model_id: str, answer: str, contexts: List[str]) -> Optional[str]:
        # Validate that the generated answer is grounded in retrieved context
        combined = "\n\n".join(contexts[:6])
        validation_prompt = (
            "You will be given an Answer and the Retrieved Contexts. "
            "If the Answer is fully supported by the contexts, return the single line: SUPPORTED. "
            "If the Answer contains claims not supported by the contexts, return a brief grounded rewrite "
            "that strictly uses only information from the contexts. "
            "If contexts are insufficient, respond with: NOT_ENOUGH_CONTEXT.\n\n"
            f"Contexts:\n{combined}\n\nAnswer:\n{answer}\n\nRespond:"
        )

        payload = {"input": validation_prompt, "temperature": 0.0, "max_tokens": 180}
        body = self._invoke_bedrock(model_id, payload)
        resp = (extract_bedrock_text(body) or "").strip()

        if not resp:
            return None
        if resp.upper().startswith("SUPPORTED"):
            return None
        if resp.upper().startswith("NOT_ENOUGH_CONTEXT"):
            return "NOT_ENOUGH_CONTEXT"

        return resp  # grounded rewrite

    # Main public RAG entry point
    def answer(
        self,
        *,
        query: str,
        history: List[Dict[str, str]],
        temperature: float,
        endpoint: EndpointConfig,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:

        self._ensure()

        # Determine the underlying Bedrock model identifier from the selected endpoint
        # For Bedrock-based RAG models we use the "url" field to store the modelId, otherwise fall back to "model".
        model_id = endpoint.url or endpoint.model or ""

        # Model used for internal enhancements (rewrite/rerank/etc.)
        bedrock_for_aux = model_id

        # Internal query rewrite to improve retrieval
        try:
            rewritten_query = self._rewrite_query(bedrock_for_aux, query)
        except Exception:
            rewritten_query = query

        # Embed using OpenAI text-embedding-3-small model
        qv = embed_query(self._openai_client, rewritten_query)

        # If the selected RAG endpoint is NOT a Bedrock model, switch to a simplified flow that
        # still performs retrieval but uses the generic LLMClient for generation.
        # Bedrock model identifiers do not start with http(s); HuggingFace/OpenAI endpoints do.
        if not model_id or model_id.startswith("http"):

            # Basic retrieval (no Bedrock-based reranking/compression)
            raw_matches = retrieve_context(self._pinecone_index, qv, top_k=4)
            prompt_ctx = build_rag_prompt(
                query,
                raw_matches,
                history,
                system_prompt or DEFAULT_SYSTEM_PROMPT,
                include_system=True,
            )

            # Choose appropriate mode for the fallback endpoint
            # Any endpoint that defines a base_url should use the OpenAI-compatible flow
            # (router.huggingface.co implements the OpenAI Chat Completions API). When
            # base_url is absent we assume the old-style HF inference endpoint.
            _mode = "openai" if endpoint.base_url else "huggingface"
            tmp_endpoint: _EP = _dc_replace(endpoint, mode=_mode)
            llm_client = LLMClient()
            gen_params = {"temperature": temperature}
            text = llm_client.generate(
                endpoint=tmp_endpoint,
                prompt="",  # we provide full prompt via messages to preserve persona
                parameters=gen_params,
                messages=[{"role": "user", "content": prompt_ctx}],
                system_prompt=None,
            )
            metrics = evaluate_retrieval(self._openai_client, query, raw_matches)
            return text, metrics

        # Retrieve candidates (larger set for improved reranking)
        top_k = 12
        raw_matches = retrieve_context(self._pinecone_index, qv, top_k=top_k)

        if not raw_matches:
            # No context found — fallback to standard generation
            prompt = build_rag_prompt(query, [], history, system_prompt or DEFAULT_SYSTEM_PROMPT, include_system=False)
            payload = {"input": prompt, "temperature": temperature, "max_tokens": 800}
            body = self._invoke_bedrock(model_id, payload)
            raw_text = extract_bedrock_text(body)
            text = sanitize_output(raw_text)
            metrics = {"avg": 0.0, "top": 0.0}
            return text, metrics

        # Rerank passages using Bedrock scoring
        try:
            scores = self._score_candidates_with_bedrock(bedrock_for_aux, rewritten_query, raw_matches)
            paired = sorted(zip(raw_matches, scores), key=lambda x: x[1], reverse=True)
        except Exception:
            paired = [(m, 0.5) for m in raw_matches]

        # Select top passages
        final_k = 4
        top_passages = [p for p, _ in paired[:final_k]]

        # Compress passages to reduce token load but retain factual content
        compressed = []
        for p in top_passages:
            try:
                compressed.append(self._compress_chunk(bedrock_for_aux, p) or p)
            except Exception:
                compressed.append(p)

        # Build full RAG prompt (includes system persona)
        prompt = build_rag_prompt(
            query,
            compressed,
            history,
            system_prompt or DEFAULT_SYSTEM_PROMPT,
            include_system=True,
        )

        # Generate answer using chosen model
        lower = model_id.lower() if model_id else ""
        if "anthropic" in lower:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 800,
                "temperature": temperature,
                "top_p": 0.9,
                "system": system_prompt or DEFAULT_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            }
        # Removed Mistral-specific payload logic
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

        # Validate grounding to prevent hallucination
        try:
            grounding = self._validate_grounding(bedrock_for_aux, text, top_passages)
            if grounding == "NOT_ENOUGH_CONTEXT":
                text = (
                    "I don't have enough reliable context in the knowledge base to fully support that claim. "
                    "Try rephrasing or ask a related question."
                )
            elif grounding:
                text = grounding
        except Exception:
            pass

        # Compute retrieval metrics using OpenAI embeddings
        metrics = evaluate_retrieval(self._openai_client, query, top_passages)

        return text, metrics
