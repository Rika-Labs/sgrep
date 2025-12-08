import modal
import os
import socket
import subprocess
from pydantic import BaseModel
from typing import List

MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
GPU_CONFIG = os.environ.get("SGREP_MODAL_GPU", "T4")
PORT = 8000
MAX_CONCURRENT = int(os.environ.get("SGREP_MODAL_MAX_CONCURRENT", "64"))
MINUTES = 60

TEI_VERSION = "1.8"
TEI_IMAGE_VARIANTS = {
    "T4": f"ghcr.io/huggingface/text-embeddings-inference:turing-{TEI_VERSION}",
    "L4": f"ghcr.io/huggingface/text-embeddings-inference:89-{TEI_VERSION}",
    "L40S": f"ghcr.io/huggingface/text-embeddings-inference:89-{TEI_VERSION}",
    "A10G": f"ghcr.io/huggingface/text-embeddings-inference:86-{TEI_VERSION}",
    "A40": f"ghcr.io/huggingface/text-embeddings-inference:86-{TEI_VERSION}",
    "A100": f"ghcr.io/huggingface/text-embeddings-inference:{TEI_VERSION}",
    "A100-40GB": f"ghcr.io/huggingface/text-embeddings-inference:{TEI_VERSION}",
    "A100-80GB": f"ghcr.io/huggingface/text-embeddings-inference:{TEI_VERSION}",
    "H100": f"ghcr.io/huggingface/text-embeddings-inference:hopper-{TEI_VERSION}",
}
TEI_IMAGE = TEI_IMAGE_VARIANTS.get(GPU_CONFIG, TEI_IMAGE_VARIANTS["T4"])

app = modal.App("sgrep-offload")

health_image = modal.Image.debian_slim().pip_install("fastapi")

MAX_BATCH_TOKENS = int(os.environ.get("SGREP_MODAL_MAX_BATCH_TOKENS", "65536"))

def spawn_server() -> subprocess.Popen:
    process = subprocess.Popen(
        [
            "text-embeddings-router",
            "--model-id", MODEL_ID,
            "--port", str(PORT),
            "--max-batch-tokens", str(MAX_BATCH_TOKENS),
            "--auto-truncate",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    while True:
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            return process
        except (socket.timeout, ConnectionRefusedError):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"TEI server exited with code {process.returncode}\n"
                    f"stdout: {stdout.decode()}\n"
                    f"stderr: {stderr.decode()}"
                )

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
    .run_function(download_model, gpu=GPU_CONFIG)
)


class EmbedRequest(BaseModel):
    texts: List[str]
    dimension: int = 768


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int


@app.cls(gpu=GPU_CONFIG, image=tei_image, scaledown_window=5 * MINUTES, max_containers=10)
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


@app.function(image=tei_image)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def embed(request: EmbedRequest) -> EmbedResponse:
    embedder = Embedder()
    embeddings = embedder.embed.remote(request.texts)
    dim = request.dimension
    if dim != 768:
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
        "gpu": GPU_CONFIG,
        "tei_image": TEI_IMAGE,
        "max_concurrent": MAX_CONCURRENT,
    }
