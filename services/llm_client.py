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

    def _generate_huggingface(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        from utils import llama3_chat_template, extract_hf_text, sanitize_output  # lazy import to avoid cycles

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
            resp = client.post(endpoint.url, headers=headers, json=payload)
            if resp.status_code == 404:
                alt_url = endpoint.url.rstrip("/") + "/generate"
                resp = client.post(alt_url, headers=headers, json=payload)
            resp.raise_for_status()
            return sanitize_output(extract_hf_text(resp.json()))

    def _generate_openai(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        from utils import sanitize_output  # lazy

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
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            org_id = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
            if org_id:
                headers["OpenAI-Organization"] = org_id
            project_id = os.getenv("OPENAI_PROJECT_ID")
            if project_id:
                headers["OpenAI-Project"] = project_id
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and (choices := data.get("choices")):
                if content := choices[0].get("message", {}).get("content"):
                    return sanitize_output(content)
            return sanitize_output(json.dumps(data))
