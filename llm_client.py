from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import boto3
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PC
from utils import llama3_chat_template, embed_query, retrieve_context, build_rag_prompt, evaluate_retrieval


@dataclass
class EndpointConfig:
    key: str
    name: str
    url: str
    mode: str = "openai"
    model: Optional[str] = None
    base_url: Optional[str] = None


def load_models_config(path: str) -> List[EndpointConfig]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        EndpointConfig(
            key=item["key"],
            name=item.get("name", item["key"]),
            url=item["url"],
            mode=item.get("mode", "openai"),
            model=item.get("model"),
            base_url=item.get("base_url"),
        )
        for item in data
    ]


class LLMClient:
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        # Lazy RAG components
        self._embed_model: Optional[SentenceTransformer] = None
        self._pinecone_index = None
        self._bedrock = None

    # ---------- helpers ----------
    @staticmethod
    def _get_env(names: List[str], default: Optional[str] = None, required: bool = False) -> Optional[str]:
        for n in names:
            val = os.getenv(n)
            if val:
                return val
        if required and default is None:
            raise RuntimeError(f"Missing required environment variable(s): {', '.join(names)}")
        return default

    @staticmethod
    def _extract_bedrock_text(body: Any) -> str:
        """Extract text from Bedrock API responses."""
        if not isinstance(body, dict):
            return str(body)
        
        # Anthropic content list
        if content := body.get("content"):
            if isinstance(content, list) and content:
                if text := content[0].get("text"):
                    return text.strip()
        
        # Mistral/other outputs list
        if outputs := body.get("outputs"):
            if isinstance(outputs, list) and outputs:
                if text := outputs[0].get("text"):
                    return text.strip()
        
        # Generic fallbacks
        if text := body.get("generated_text"):
            return text.strip()
        
        return json.dumps(body)

    @staticmethod
    def _extract_hf_text(data: Any) -> str:
        """Extract text from HuggingFace responses."""
        if isinstance(data, list) and data:
            return str(data[0].get("generated_text", "")).strip()
        if isinstance(data, dict):
            return str(data.get("generated_text", "")).strip()
        return json.dumps(data)

    def generate(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using the configured endpoint.

        Supports two modes:
        - openai: OpenAI-compatible Chat Completions API
        - huggingface: Hugging Face Inference API (string inputs)
        """
        mode = (endpoint.mode or "openai").lower()
        if mode == "huggingface":
            return self._generate_huggingface(endpoint, prompt, parameters, messages=messages, system_prompt=system_prompt)
        # Default to OpenAI-compatible
        return self._generate_openai(endpoint, prompt, parameters, messages=messages, system_prompt=system_prompt)

    # ---------- Bedrock + RAG ----------
    def _ensure_rag_setup(self):
        if self._embed_model is None:
            self._embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=False)

        if self._pinecone_index is None:
            api_key = self._get_env(["PINECONE_API_KEY"], required=True)
            index_name = self._get_env(["PINECONE_INDEX"], default="neil-degrasse-tyson-embeddings")
            self._pinecone_index = PC(api_key=api_key).Index(index_name)

        if self._bedrock is None:
            region = self._get_env(["AWS_REGION", "AWS_DEFAULT_REGION"], default="us-east-1")
            kwargs = {"region_name": region}
            if aws_key := os.getenv("AWS_ACCESS_KEY_ID"):
                if aws_secret := os.getenv("AWS_SECRET_ACCESS_KEY"):
                    kwargs.update({"aws_access_key_id": aws_key, "aws_secret_access_key": aws_secret})
            self._bedrock = boto3.client("bedrock-runtime", **kwargs)

    def rag_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        *,
        temperature: float,
        model_id: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate an answer using Retrieval-Augmented Generation via Bedrock.

        model_id: full Bedrock model id (e.g., anthropic.claude-3-haiku-20240307-v1:0)
        """
        self._ensure_rag_setup()
        qv = embed_query(self._embed_model, query)
        chunks = retrieve_context(self._pinecone_index, qv)
        # Build user content only (system prompt passed separately when supported)
        prompt = build_rag_prompt(query, chunks, history, system_prompt, include_system=False)

        # Build payload by model family
        lower = (model_id or "").lower()
        if "anthropic" in lower:
            # Use system field to increase adherence & forbid stage directions
            system_field = (system_prompt or "") + "\n\nDo NOT start replies with stage directions, narration, or action statements (e.g., *clears throat*, *speaks with*). Respond directly."
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 800,
                "temperature": temperature,
                "top_p": 0.9,
                "system": system_field,
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
        raw_text = self._extract_bedrock_text(body)
        text = self._sanitize_output(raw_text)
        metrics = evaluate_retrieval(self._embed_model, query, chunks)
        return text, metrics

    def _generate_huggingface(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        api_key = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("Missing API key: set HUGGINGFACE_API_TOKEN or HF_TOKEN")

        temp_msgs = messages or ([{"role": "user", "content": prompt}] if prompt else None)
        inputs = llama3_chat_template(system_prompt, temp_msgs) or prompt or system_prompt or ""
        
        params = {"return_full_text": False}
        if parameters:
            if "temperature" in parameters:
                params["temperature"] = parameters["temperature"]
            if "max_new_tokens" in parameters:
                params["max_new_tokens"] = parameters["max_new_tokens"]

        payload = {"inputs": inputs, "parameters": params}
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(endpoint.url, headers=headers, json=payload)
                resp.raise_for_status()
                return self._sanitize_output(self._extract_hf_text(resp.json()))
            except httpx.HTTPStatusError as e:
                if e.response and e.response.status_code == 404:
                    # Try /generate endpoint for TGI
                    alt_url = endpoint.url.rstrip("/") + "/generate"
                    resp = client.post(alt_url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return self._sanitize_output(self._extract_hf_text(resp.json()))
                raise

    def _generate_openai(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        is_url = endpoint.url.startswith("https://")
        model_id = endpoint.model if is_url else endpoint.url
        base_url = endpoint.url if is_url else endpoint.base_url
        
        if not model_id:
            raise RuntimeError("OpenAI mode requires a model id")

        # Determine API key based on base URL
        if not base_url or "api.openai.com" in base_url:
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        
        if not api_key:
            raise RuntimeError("Missing API key: set OPENAI_API_KEY or HUGGINGFACE_API_TOKEN")

        # Build messages
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages or [{"role": "user", "content": prompt}])
        
        payload = {"model": model_id, "messages": chat_messages}
        
        # Add parameters
        if parameters:
            if "temperature" in parameters:
                payload["temperature"] = parameters["temperature"]
            if "max_new_tokens" in parameters:
                payload["max_tokens"] = parameters["max_new_tokens"]
            for key in ("top_p", "frequency_penalty", "presence_penalty", "stop"):
                if key in parameters:
                    payload[key] = parameters[key]

        # Determine URL
        url = base_url or "https://api.openai.com/v1/chat/completions"
        if "huggingface.cloud" in url and not url.endswith(("/v1/chat/completions", "/chat/completions")):
            url = url.rstrip("/") + "/v1/chat/completions"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = e.response.text if e.response else str(e)
                code = e.response.status_code if e.response else "unknown"
                raise RuntimeError(f"OpenAI error ({code}): {detail}") from e
            
            data = resp.json()
            if isinstance(data, dict) and (choices := data.get("choices")):
                if content := choices[0].get("message", {}).get("content"):
                    return self._sanitize_output(content)
            return self._sanitize_output(json.dumps(data))

    # ---------- output sanitization ----------
    @staticmethod
    def _sanitize_output(text: str) -> str:
        """Remove leading stage directions or role-play actions (e.g., *clears throat*)."""
        # Pattern: leading asterisk-block or bracket parentheses describing action
        patterns = [
            r"^(?:\s*\*[^\n]*?\*\s*)+",  # *clears throat* or multiple
            r"^(?:\s*\([^)]{0,80}\)\s*)",  # (speaks with an energetic tone)
        ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        # Also remove any leftover leading quotes or whitespace artifacts
        cleaned = cleaned.lstrip("\n ")
        return cleaned
