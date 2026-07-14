import os

from pnlc_agentclinic.embedding.embedder import OpenAICompatibleEmbedder
from pnlc_agentclinic.embedding.huggingface_embedder import HuggingFaceEmbedder


def build_embedder(model_backends_cfg):
    backend_type = model_backends_cfg.backend_type

    if backend_type == "openai_compatible":
        api_key = os.environ.get(model_backends_cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {model_backends_cfg.api_key_env} in your environment before running this.")
        return OpenAICompatibleEmbedder(
            base_url=model_backends_cfg.base_url,
            api_key=api_key,
            model_name=model_backends_cfg.model_name,
        )

    if backend_type == "huggingface":
        return HuggingFaceEmbedder(
            model_name=model_backends_cfg.model_name,
            device=model_backends_cfg.get("device", "cpu"),
        )

    raise ValueError(f"Unknown backend_type '{backend_type}'. Expected 'openai_compatible' or 'huggingface'.")
