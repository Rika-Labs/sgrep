import pytest


class TestEmbedRequest:
    def test_embed_request_validates_dimension(self):
        from service import EmbedRequest
        request = EmbedRequest(texts=["test"], dimension=768)
        assert request.dimension == 768

    def test_embed_request_default_dimension(self):
        from service import EmbedRequest
        request = EmbedRequest(texts=["test"])
        assert request.dimension == 768

    def test_embed_request_max_texts_limit(self):
        from service import EmbedRequest
        texts = ["text"] * 1000
        request = EmbedRequest(texts=texts, dimension=768)
        assert len(request.texts) == 1000


class TestEmbedResponse:
    def test_embed_response_structure(self):
        from service import EmbedResponse
        embeddings = [[0.1] * 768]
        response = EmbedResponse(
            embeddings=embeddings,
            model="jinaai/jina-embeddings-v2-base-code",
            dimension=768
        )
        assert len(response.embeddings) == 1
        assert response.model == "jinaai/jina-embeddings-v2-base-code"
        assert response.dimension == 768


class TestModelConfig:
    def test_model_id_is_jina(self):
        from service import MODEL_ID
        assert MODEL_ID == "jinaai/jina-embeddings-v2-base-code"

    def test_default_gpu_is_t4(self):
        import os
        original = os.environ.get("SGREP_MODAL_GPU")
        try:
            if "SGREP_MODAL_GPU" in os.environ:
                del os.environ["SGREP_MODAL_GPU"]
            import importlib
            import service
            importlib.reload(service)
            assert service.GPU_CONFIG == "T4"
        finally:
            if original:
                os.environ["SGREP_MODAL_GPU"] = original


class TestConcurrencySettings:
    def test_max_concurrent_default(self):
        import os
        original = os.environ.get("SGREP_MODAL_MAX_CONCURRENT")
        try:
            if "SGREP_MODAL_MAX_CONCURRENT" in os.environ:
                del os.environ["SGREP_MODAL_MAX_CONCURRENT"]
            import importlib
            import service
            importlib.reload(service)
            assert service.MAX_CONCURRENT == 64
        finally:
            if original:
                os.environ["SGREP_MODAL_MAX_CONCURRENT"] = original

    def test_max_concurrent_env_override(self):
        import os
        original = os.environ.get("SGREP_MODAL_MAX_CONCURRENT")
        try:
            os.environ["SGREP_MODAL_MAX_CONCURRENT"] = "128"
            import importlib
            import service
            importlib.reload(service)
            assert service.MAX_CONCURRENT == 128
        finally:
            if original:
                os.environ["SGREP_MODAL_MAX_CONCURRENT"] = original
            elif "SGREP_MODAL_MAX_CONCURRENT" in os.environ:
                del os.environ["SGREP_MODAL_MAX_CONCURRENT"]


class TestSpawnServer:
    def test_spawn_server_function_exists(self):
        from service import spawn_server
        assert callable(spawn_server)

    def test_download_model_function_exists(self):
        from service import download_model
        assert callable(download_model)


class TestTEIImageSelection:
    def test_t4_uses_turing_image(self):
        from service import TEI_IMAGE_VARIANTS
        assert "turing" in TEI_IMAGE_VARIANTS["T4"]

    def test_a10g_uses_86_image(self):
        from service import TEI_IMAGE_VARIANTS
        assert "86-" in TEI_IMAGE_VARIANTS["A10G"]

    def test_a100_uses_default_image(self):
        from service import TEI_IMAGE_VARIANTS
        assert "turing" not in TEI_IMAGE_VARIANTS["A100"]
        assert "hopper" not in TEI_IMAGE_VARIANTS["A100"]
        assert "86-" not in TEI_IMAGE_VARIANTS["A100"]
        assert "89-" not in TEI_IMAGE_VARIANTS["A100"]

    def test_h100_uses_hopper_image(self):
        from service import TEI_IMAGE_VARIANTS
        assert "hopper" in TEI_IMAGE_VARIANTS["H100"]

    def test_l4_uses_89_image(self):
        from service import TEI_IMAGE_VARIANTS
        assert "89-" in TEI_IMAGE_VARIANTS["L4"]

    def test_default_tei_image_matches_gpu_config(self):
        from service import TEI_IMAGE, GPU_CONFIG, TEI_IMAGE_VARIANTS
        assert TEI_IMAGE == TEI_IMAGE_VARIANTS.get(GPU_CONFIG, TEI_IMAGE_VARIANTS["T4"])

    def test_tei_version_is_1_8(self):
        from service import TEI_VERSION
        assert TEI_VERSION == "1.8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
