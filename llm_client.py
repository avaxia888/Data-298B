from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


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
    out: List[EndpointConfig] = []
    for item in data:
        out.append(
            EndpointConfig(
                key=item["key"],
                name=item.get("name", item["key"]),
                url=item["url"],
                mode=item.get("mode", "openai"),
                model=item.get("model"),
                base_url=item.get("base_url"),
            )
        )
    return out


class LLMClient:
    def __init__(self, token: Optional[str] = None, timeout: float = 60.0):
        self.timeout = timeout

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

    # --- Internal helpers ---
    def _generate_huggingface(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Prefer standard env var, but allow HF_TOKEN as a fallback
        api_key = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("Missing API key: set HUGGINGFACE_API_TOKEN (or HF_TOKEN).")

        inputs = prompt or system_prompt or ""
        hf_payload: Dict[str, Any] = {"inputs": inputs, "parameters": {}}

        if parameters:
            p: Dict[str, Any] = {}
            if "temperature" in parameters:
                p["temperature"] = parameters["temperature"]
            if "max_new_tokens" in parameters:
                p["max_new_tokens"] = parameters["max_new_tokens"]
            if "top_p" in parameters:
                p["top_p"] = parameters["top_p"]
            if "stop" in parameters:
                p["stop_sequences"] = parameters["stop"]
            hf_payload["parameters"] = p

        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(
                    endpoint.url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=hf_payload,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = e.response.text if e.response is not None else str(e)
                code = e.response.status_code if e.response is not None else "unknown"
                raise RuntimeError(f"Hugging Face error ({code}): {detail}") from e
            data = resp.json()

        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return str(data[0]["generated_text"]).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()
        return json.dumps(data)

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
            raise RuntimeError("OpenAI mode requires a model id (set 'url' or 'model' in models.json).")

        api_key = os.getenv("OPENAI_API_KEY" if (not base_url or "api.openai.com" in (base_url or "")) else "HUGGINGFACE_API_TOKEN")
        if not api_key:
            raise RuntimeError("Missing API key: set OPENAI_API_KEY or HUGGINGFACE_API_TOKEN.")

        chat_messages: List[Dict[str, str]] = ([] if not system_prompt else [{"role": "system", "content": system_prompt}]) + (
            messages or [{"role": "user", "content": prompt}]
        )
        oa_payload: Dict[str, Any] = {"model": model_id, "messages": chat_messages}
        if parameters:
            mapped = {k: parameters[k] for k in ("top_p", "frequency_penalty", "presence_penalty", "stop") if k in parameters}
            if "temperature" in parameters:
                mapped["temperature"] = parameters["temperature"]
            if "max_new_tokens" in parameters:
                mapped["max_tokens"] = parameters["max_new_tokens"]
            oa_payload.update(mapped)

        url = base_url or "https://api.openai.com/v1/chat/completions"

        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(
                    url,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    json=oa_payload,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = e.response.text if e.response is not None else str(e)
                code = e.response.status_code if e.response is not None else "unknown"
                raise RuntimeError(f"OpenAI error ({code}): {detail}") from e
            data = resp.json()

        choices = data.get("choices") if isinstance(data, dict) else None
        if choices:
            content = choices[0].get("message", {}).get("content")
            if isinstance(content, str):
                return content
        return json.dumps(data)
