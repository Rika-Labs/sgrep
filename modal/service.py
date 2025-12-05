"""
Modal.dev service for sgrep embedding and reranking offload.

This service provides:
- /embed - Embedding endpoint using Qwen3-Embedding-8B
- /rerank - Reranking endpoint using Qwen3-Reranker-8B
- /health - Health check endpoint

Configuration via environment variables:
- GPU_TIER: "budget" (T4), "balanced" (A10G), or "high" (L40S, default)
- SGREP_EMBED_MODEL: Embedding model (default: Qwen/Qwen3-Embedding-8B)
- SGREP_RERANK_MODEL: Reranker model (default: Qwen/Qwen3-Reranker-8B)

Deploy with: modal deploy modal/service.py
"""

import modal
import os
from pydantic import BaseModel
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Tuple

GPU_TIERS = {
    "budget": "T4",        # ~$0.25/hr, slower
    "balanced": "A10G",    # ~$0.45/hr, good balance
    "high": "L40S",        # ~$1.10/hr, best performance (default)
}
GPU_TIER = os.environ.get("GPU_TIER", "high")
GPU_CONFIG = GPU_TIERS.get(GPU_TIER, "L40S")

EMBED_MODEL = os.environ.get("SGREP_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
RERANK_MODEL = os.environ.get("SGREP_RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")

MINUTES = 60

app = modal.App("sgrep-offload")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8",
        "transformers",
        "torch",
        "fastapi[standard]",
        "pydantic",
        "sentence-transformers",
    )
)


class EmbedRequest(BaseModel):
    texts: List[str]
    dimension: int = 4096


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 10


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str


auth_scheme = HTTPBearer()


def verify_token(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Verify the Bearer token against the stored secret."""
    expected = os.environ.get("SGREP_API_TOKEN")
    if not expected or token.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    scaledown_window=5 * MINUTES,
    secrets=[modal.Secret.from_name("sgrep-auth")],
)
class Embedder:
    @modal.enter()
    def load_model(self):
        from vllm import LLM

        print(f"Loading embedding model: {EMBED_MODEL}")
        self.model = LLM(model=EMBED_MODEL, task="embed", trust_remote_code=True)
        print("Embedding model loaded successfully")

    @modal.method()
    def embed(self, texts: List[str], dimension: int = 4096) -> List[List[float]]:
        outputs = self.model.embed(texts)
        # Truncate to requested dimension and convert to list
        return [list(out.outputs.embedding[:dimension]) for out in outputs]


@app.function(image=image, secrets=[modal.Secret.from_name("sgrep-auth")])
@modal.fastapi_endpoint(method="POST")
def embed(request: EmbedRequest, _: None = Depends(verify_token)) -> EmbedResponse:
    """Embed texts using Qwen3-Embedding-8B."""
    embedder = Embedder()
    embeddings = embedder.embed.remote(request.texts, request.dimension)
    return EmbedResponse(
        embeddings=embeddings,
        model=EMBED_MODEL,
        dimension=request.dimension,
    )


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    scaledown_window=5 * MINUTES,
    secrets=[modal.Secret.from_name("sgrep-auth")],
)
class Reranker:
    @modal.enter()
    def load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        print(f"Loading reranker model: {RERANK_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            RERANK_MODEL, trust_remote_code=True
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                RERANK_MODEL, trust_remote_code=True, torch_dtype=torch.float16
            )
            .cuda()
            .eval()
        )
        print("Reranker model loaded successfully")

    @modal.method()
    def rerank(
        self, query: str, documents: List[str], top_k: int = 10
    ) -> List[Tuple[int, float]]:
        import torch

        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=8192,
        ).to("cuda")

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().tolist()

        # Handle case where scores is a single float
        if isinstance(scores, float):
            scores = [scores]

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


@app.function(image=image, secrets=[modal.Secret.from_name("sgrep-auth")])
@modal.fastapi_endpoint(method="POST")
def rerank(request: RerankRequest, _: None = Depends(verify_token)) -> RerankResponse:
    """Rerank documents by relevance to query using Qwen3-Reranker-8B."""
    reranker = Reranker()
    results = reranker.rerank.remote(request.query, request.documents, request.top_k)
    return RerankResponse(
        results=[RerankResult(index=idx, score=score) for idx, score in results],
        model=RERANK_MODEL,
    )


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint - returns service status and GPU tier."""
    return {"status": "ok", "gpu_tier": GPU_TIER, "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL}
