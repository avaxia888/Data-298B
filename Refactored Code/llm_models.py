import os
import json
import re
import boto3
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
        
        # default model id used when callers don't specify one
        self.default_model_id = (
            self.model_registry.get("claude_haiku")
            if "claude_haiku" in self.model_registry
            else next(iter(self.model_registry.values()))
        )

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
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
        res = self.bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(res["body"].read())
        text = body["content"][0]["text"].strip()
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
        return self.model_registry.get(key_or_id, self.default_model_id)
