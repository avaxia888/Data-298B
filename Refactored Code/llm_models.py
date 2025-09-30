import os
import json
import re
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

class LLMService:
    """Service that manages embeddings, index access and model invocation.

    Responsibilities:
        - Initialize embedding model, Pinecone index and Bedrock client.
        - Provide a method to generate model answers with default parameters.
        - Maintain a list of available models.

    Attributes:
        index: Pinecone Index instance
        embed_model: SentenceTransformer instance
        bedrock: boto3 Bedrock runtime client
    """

    # We will be adding more models here
    MODEL_REGISTRY = {
        "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        # Mistral 7B instruct model
        "Mistral 7B": "mistral.mistral-7b-instruct-v0:2",
    }

    def __init__(self):
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX = os.getenv("PINECONE_INDEX", "neil-degrasse-tyson-embeddings")
        AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        self.model_registry = dict(self.MODEL_REGISTRY)

        # default model id used when callers don't specify one: pick the
        # first registered model value (safe fallback when keys don't match)
        self.default_model_id = next(iter(self.model_registry.values())) if self.model_registry else None

    def generate_answer(self, prompt, temp=0.7, max_tokens=800, model_id: str | None = None):
        """Generate a model completion.

        Params:
            prompt (str): user prompt
            temp (float): sampling temperature (default 0.7)
            max_tokens (int): max tokens to generate (default 800)
            model_id (str|None): friendly key or full model id

        Returns:
            str: cleaned generated text
        """
        model_id = model_id or self.default_model_id
        model_id = self._get_model(model_id)

        mid = model_id.lower() if isinstance(model_id, str) else ""
        if "anthropic" in mid:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temp,
                "top_p": 0.9,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            }
        elif "mistral" in mid or mid.startswith("mistral."):
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            payload = {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temperature": temp,
            }
        else:
            payload = {"input": prompt, "temperature": temp, "max_tokens": max_tokens}

        # Build a single payload per detected model family and invoke once.
        res = self.bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        body = json.loads(res["body"].read())

        # Response parsing: prefer Mistral native shape (outputs[0]['text']),
        # then try other common shapes (Anthropic, choices, generated_text).
        text = None

        # 1) Mistral/Bedrock native: outputs[0]['text']
        if isinstance(body, dict) and "outputs" in body and isinstance(body["outputs"], list):
            out0 = body["outputs"][0]
            if isinstance(out0, dict) and "text" in out0:
                text = out0["text"].strip()

        # 2) Anthropic-style
        if not text and isinstance(body, dict) and "content" in body:
            content = body.get("content")
            if isinstance(content, list) and len(content) and isinstance(content[0], dict):
                text_field = content[0].get("text")
                if isinstance(text_field, str):
                    text = text_field.strip()

        # 3) other common shapes
        if not text and isinstance(body, dict):
            if "generated_text" in body and isinstance(body["generated_text"], str):
                text = body["generated_text"].strip()
            elif "output" in body:
                out = body["output"]
                if isinstance(out, list):
                    text = "\n".join([str(x) for x in out])
                else:
                    text = str(out).strip()
            elif "choices" in body and isinstance(body["choices"], list):
                first_choice = body["choices"][0]
                if isinstance(first_choice, dict) and "text" in first_choice:
                    text = str(first_choice["text"]).strip()
                elif isinstance(first_choice, dict) and "message" in first_choice and isinstance(first_choice["message"], dict):
                    candidate = first_choice["message"].get("content")
                    if isinstance(candidate, str):
                        text = candidate.strip()

        # Final fallback: stringify body
        if not text:
            text = str(body)

        # remove emphasis markup and return
        return re.sub(r"\*.*?\*", "", text)

    # --- Registry helpers ---
    def _get_model(self, key_or_id: str) -> str:
        """Lookup a model key and return the full model id.

        Params:
            key_or_id (str): key or full model id

        Returns:
            str: resolved full model id (default claude_haiku if not specified by user)
        """
        if not key_or_id:
            return self.default_model_id

        # If the caller passed a full model id that already matches one of
        # the registered values, return it directly.
        if key_or_id in self.model_registry.values():
            return key_or_id

        # Case-insensitive match against friendly keys (allow underscores or spaces)
        key_lower = key_or_id.lower()
        for friendly, full_id in self.model_registry.items():
            if friendly.lower() == key_lower or friendly.replace(" ", "_").lower() == key_lower:
                return full_id

        # If it looks like a full model identifier (heuristic: contains a dot and a colon)
        if ('.' in key_or_id and ':' in key_or_id) or '/' in key_or_id:
            return key_or_id

        # Fallback to the default model id
        return self.default_model_id
