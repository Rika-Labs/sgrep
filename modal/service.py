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
MAX_MODEL_LEN = int(os.environ.get("SGREP_MODAL_MAX_MODEL_LEN", "2048"))

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
    dimension: int = 384  # Matches local embedder for compatibility


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


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    scaledown_window=5 * MINUTES,
)
class Embedder:
    @modal.enter()
    def load_model(self):
        from vllm import LLM

        print(f"Loading embedding model: {EMBED_MODEL}")
        self.model = LLM(
            model=EMBED_MODEL,
            task="embed",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,  # Use more GPU memory for larger batches
            max_model_len=MAX_MODEL_LEN,  # Guarded below with tokenizer truncation
            enforce_eager=True,           # Faster for embeddings (no CUDA graphs overhead)
        )
        try:
            self.tokenizer = self.model.get_tokenizer()
        except Exception:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                EMBED_MODEL, trust_remote_code=True
            )
        self.max_tokens = max(1, MAX_MODEL_LEN - 8)
        print("Embedding model loaded successfully")

    @modal.method()
    def embed(self, texts: List[str], dimension: int = 384) -> List[List[float]]:
        """Embed texts and truncate to requested dimension.

        Args:
            texts: List of texts to embed (max 1000 per request)
            dimension: Must be 384 to match local embedder for compatibility
        """
        if not texts:
            return []
        if len(texts) > 1000:
            raise ValueError(f"Too many texts: {len(texts)} > 1000 max")
        if dimension != 384:
            raise ValueError(f"Dimension must be 384 for local/remote compatibility, got {dimension}")

        truncated = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > self.max_tokens:
                tokens = tokens[: self.max_tokens]
                text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            truncated.append(text)

        outputs = self.model.embed(truncated)
        return [list(out.outputs.embedding[:dimension]) for out in outputs]


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def embed(request: EmbedRequest) -> EmbedResponse:
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
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: Documents to rerank (max 500)
            top_k: Number of top results to return
        """
        import torch

        if not documents:
            return []
        if len(documents) > 500:
            raise ValueError(f"Too many documents: {len(documents)} > 500 max")
        if not query.strip():
            raise ValueError("Query cannot be empty")

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


@app.function(image=image)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def rerank(request: RerankRequest) -> RerankResponse:
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
