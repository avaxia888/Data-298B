from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
import httpx
from pinecone import Pinecone as PC
from sentence_transformers import SentenceTransformer
from utils import (
    build_rag_prompt,
    embed_query,
    evaluate_retrieval,
    extract_bedrock_text,
    extract_hf_text,
    get_env,
    llama3_chat_template,
    retrieve_context,
    sanitize_output,
)


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
        self._embed_model: Optional[SentenceTransformer] = None
        self._pinecone_index = None
        self._bedrock = None


    def generate(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        mode = (endpoint.mode or "openai").lower()
        if mode == "huggingface":
            return self._generate_huggingface(endpoint, prompt, parameters, messages=messages, system_prompt=system_prompt)
        return self._generate_openai(endpoint, prompt, parameters, messages=messages, system_prompt=system_prompt)

    def _ensure_rag_setup(self):
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

    def rag_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        *,
        temperature: float,
        model_id: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        self._ensure_rag_setup()
        qv = embed_query(self._embed_model, query)
        chunks = retrieve_context(self._pinecone_index, qv)
        prompt = build_rag_prompt(query, chunks, history, system_prompt, include_system=False)
        lower = (model_id or "").lower()
        if "anthropic" in lower:
            system_field = system_prompt
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
        raw_text = extract_bedrock_text(body)
        text = sanitize_output(raw_text)
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
                return sanitize_output(extract_hf_text(resp.json()))
            except httpx.HTTPStatusError as e:
                if e.response and e.response.status_code == 404:
                    # Try /generate endpoint for TGI
                    alt_url = endpoint.url.rstrip("/") + "/generate"
                    resp = client.post(alt_url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return sanitize_output(extract_hf_text(resp.json()))
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

        if not base_url or "api.openai.com" in base_url:
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = os.getenv("HUGGINGFACE_API_TOKEN")
        
        if not api_key:
            raise RuntimeError("Missing API key: set OPENAI_API_KEY or HUGGINGFACE_API_TOKEN")

        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages or [{"role": "user", "content": prompt}])
        
        payload = {"model": model_id, "messages": chat_messages}
        
        if parameters:
            if "temperature" in parameters:
                payload["temperature"] = parameters["temperature"]
            if "max_new_tokens" in parameters:
                payload["max_tokens"] = parameters["max_new_tokens"]
            for key in ("top_p", "frequency_penalty", "presence_penalty", "stop"):
                if key in parameters:
                    payload[key] = parameters[key]

        url = base_url or "https://api.openai.com/v1/chat/completions"
        if "huggingface.cloud" in url and not url.endswith(("/v1/chat/completions", "/chat/completions")):
            url = url.rstrip("/") + "/v1/chat/completions"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and (choices := data.get("choices")):
                if content := choices[0].get("message", {}).get("content"):
                    return sanitize_output(content)
            return sanitize_output(json.dumps(data))

