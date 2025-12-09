import modal
import os
import socket
import subprocess
import time
from pydantic import BaseModel
from typing import List

# Model configuration
MODELS = {
    "mxbai": {
        "id": "mixedbread-ai/mxbai-embed-xsmall-v1",
        "native_dim": 384,
    },
    "jina": {
        "id": "jinaai/jina-embeddings-v2-base-code",
        "native_dim": 768,
    },
}

MODEL_KEY = os.getenv("SGREP_MODAL_MODEL", "mxbai")
if MODEL_KEY not in MODELS:
    MODEL_KEY = "mxbai"
MODEL_CONFIG = MODELS[MODEL_KEY]
MODEL_ID = MODEL_CONFIG["id"]
NATIVE_DIM = MODEL_CONFIG["native_dim"]

GPU = "A10G"
PORT = 8000
MAX_CONCURRENT = 64
MINUTES = 60
STARTUP_TIMEOUT = 300
MAX_BATCH_TOKENS = 65536
SERVER_READY_TIMEOUT = 180

TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:86-1.8"

app = modal.App("sgrep-offload")

health_image = modal.Image.debian_slim().pip_install("fastapi")

def spawn_server() -> subprocess.Popen:
    print(f"Starting TEI server with model {MODEL_ID}...")
    start_time = time.time()
    process = subprocess.Popen(
        [
            "text-embeddings-router",
            "--model-id", MODEL_ID,
            "--port", str(PORT),
            "--max-batch-tokens", str(MAX_BATCH_TOKENS),
            "--max-client-batch-size", "1024",
            "--auto-truncate",
        ],
    )
    attempt = 0
    while True:
        elapsed = time.time() - start_time
        attempt += 1
        if elapsed > SERVER_READY_TIMEOUT:
            process.terminate()
            raise RuntimeError(f"TEI server failed to start within {SERVER_READY_TIMEOUT}s")
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            print(f"TEI server ready after {elapsed:.1f}s ({attempt} attempts)")
            return process
        except (socket.timeout, ConnectionRefusedError):
            if process.poll() is not None:
                raise RuntimeError(f"TEI server exited with code {process.returncode}")
            if attempt % 30 == 0:
                print(f"Waiting for TEI server... {elapsed:.0f}s elapsed")

def download_model():
    spawn_server().terminate()

def download_model_hf():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, cache_dir="/data")

tei_image = (
    modal.Image.from_registry(TEI_IMAGE, add_python="3.11")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("httpx", "pydantic", "fastapi[standard]", "huggingface_hub")
    .run_function(download_model_hf)
    .run_function(download_model, gpu=GPU)
)


class EmbedRequest(BaseModel):
    texts: List[str]
    dimension: int = 384


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int


@app.cls(gpu=GPU, image=tei_image, scaledown_window=5 * MINUTES, max_containers=10, startup_timeout=STARTUP_TIMEOUT)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class Embedder:
    @modal.enter()
    def setup_server(self):
        import httpx
        self.process = spawn_server()
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=120.0)

    @modal.exit()
    def teardown_server(self):
        self.process.terminate()

    @modal.method()
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if len(texts) > 1000:
            raise ValueError(f"Too many texts: {len(texts)} > 1000 max")
        resp = self.client.post("/embed", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()


@app.function(image=tei_image, startup_timeout=STARTUP_TIMEOUT, timeout=600)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def embed(request: EmbedRequest) -> EmbedResponse:
    embedder = Embedder()
    embeddings = embedder.embed.remote(request.texts)
    dim = request.dimension
    if NATIVE_DIM > dim:
        embeddings = [emb[:dim] for emb in embeddings]
    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_ID,
        dimension=dim,
    )


@app.function(image=health_image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_key": MODEL_KEY,
        "native_dim": NATIVE_DIM,
        "gpu": GPU,
    }
