import os

from pnlc_agentclinic.llm_backends.openai_compatible import OpenAICompatibleBackend
from pnlc_agentclinic.llm_backends.huggingface_backend import HuggingFaceBackend


def build_generation_backend(model_backends_cfg):
    backend_type = model_backends_cfg.backend_type

    if backend_type == "openai_compatible":
        api_key = os.environ.get(model_backends_cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {model_backends_cfg.api_key_env} in your environment before running this.")
        return OpenAICompatibleBackend(
            base_url=model_backends_cfg.base_url,
            api_key=api_key,
            model_name=model_backends_cfg.model_name,
        )

    if backend_type == "huggingface":
        return HuggingFaceBackend(
            model_name=model_backends_cfg.model_name,
            device=model_backends_cfg.get("device", "cpu"),
            max_tokens=model_backends_cfg.get("max_tokens", 200),
            temperature=model_backends_cfg.get("temperature", 0.7),
        )

    raise ValueError(f"Unknown backend_type '{backend_type}'. Expected 'openai_compatible' or 'huggingface'.")
