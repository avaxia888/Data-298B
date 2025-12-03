from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from utils import llama3_chat_template, extract_hf_text, sanitize_output, truncate_to_complete_sentence


@dataclass
class EndpointConfig:
    key: str
    name: str
    url: Optional[str] = None
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
            url=item.get("url"),
            mode=item.get("mode", "openai"),
            model=item.get("model"),
            base_url=item.get("base_url"),
        )
        for item in data
    ]


class LLMClient:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.default_openai_base = "https://api.openai.com/v1/chat/completions"
        self.openai_base_headers: Dict[str, str] = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.openai_key}"} if self.openai_key else {})}
        self.hf_base_headers: Dict[str, str] = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {})}

    
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
        # Qwen should use Hugging Face flow, not OpenAI
        if endpoint.key == "qwen-2.5-7b-merged-neil":
            return self._generate_huggingface(endpoint, prompt, parameters, messages=messages, system_prompt=system_prompt)
        if mode == "huggingface":
            return self._generate_huggingface(endpoint, prompt, parameters, messages=messages, system_prompt=system_prompt)
        if mode == "openai":
            return self._generate_openai(endpoint, prompt, parameters, messages=messages, system_prompt=system_prompt)
    
    
    def _generate_huggingface(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not endpoint.url:
            raise RuntimeError("Hugging Face endpoints must define a 'url' in models.json")
        if not self.hf_token:
            raise RuntimeError("Missing API key: set HUGGINGFACE_API_TOKEN or HF_TOKEN")

        temp_msgs = messages or ([{"role": "user", "content": prompt}] if prompt else None)
        inputs = llama3_chat_template(system_prompt, temp_msgs) or prompt or system_prompt or ""
        params = {"return_full_text": False, "repetition_penalty": 1.15}
        if parameters:
            if "temperature" in parameters:
                params["temperature"] = parameters["temperature"]
            if "max_new_tokens" in parameters:
                params["max_new_tokens"] = parameters["max_new_tokens"]
            if "top_p" in parameters:
                params["top_p"] = parameters["top_p"]
            if "repetition_penalty" in parameters:
                params["repetition_penalty"] = parameters["repetition_penalty"]

        payload = {"inputs": inputs, "parameters": params}
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(endpoint.url, headers=dict(self.hf_base_headers), json=payload)
            resp.raise_for_status()
            return truncate_to_complete_sentence(sanitize_output(extract_hf_text(resp.json())))

    def _generate_openai(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        model_id = endpoint.model
        base_url = endpoint.base_url
        if not model_id:
            raise RuntimeError(f"models.json missing 'model' for openai endpoint: {endpoint.key}")
        if not base_url:
            raise RuntimeError(f"models.json missing 'base_url' for openai endpoint: {endpoint.key}")
        # Require an auth token: OPENAI_API_KEY for OpenAI hosts, or HF_TOKEN for
        # Hugging Face router and other HF-hosted chat-completions endpoints.
        if not self.openai_key and not ("huggingface" in base_url and self.hf_token):
            raise RuntimeError("Missing API key: set OPENAI_API_KEY or HF_TOKEN")

        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages or [{"role": "user", "content": prompt}])

        payload = {"model": model_id, "messages": chat_messages, "frequency_penalty": 0.5}

        if parameters:
            if "temperature" in parameters:
                payload["temperature"] = parameters["temperature"]
            if "max_new_tokens" in parameters:
                payload["max_tokens"] = parameters["max_new_tokens"]
            for key in ("top_p", "frequency_penalty", "presence_penalty", "stop"):
                if key in parameters:
                    payload[key] = parameters[key]

        url = base_url

        if not parameters:
            host = url
            if endpoint.key == "gemma-3-ndtv3":
                payload["max_tokens"] = payload.get("max_tokens", 512)

        with httpx.Client(timeout=60.0) as client:
            headers: Dict[str, str] = {"Content-Type": "application/json"}
            host = url
            # Determine which auth token header to send based on the host. Any Hugging Face
            # inference endpoints (including the generic router.huggingface.co) should use the
            # HF token, otherwise default to the OpenAI API key.
            if "huggingface" in host:
                if self.hf_token:
                    headers["Authorization"] = f"Bearer {self.hf_token}"
            else:
                if self.openai_key:
                    headers["Authorization"] = f"Bearer {self.openai_key}"

            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and (choices := data.get("choices")):
                if content := choices[0].get("message", {}).get("content"):
                    return truncate_to_complete_sentence(sanitize_output(content))
            return truncate_to_complete_sentence(sanitize_output(json.dumps(data)))
