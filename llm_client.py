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
