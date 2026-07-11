import os
import sys
from pathlib import Path

from pnlc_agentclinic.llm_backends.openai_compatible import OpenAICompatibleBackend

AGENTCLINIC_PATH = Path(__file__).resolve().parents[3] / "external" / "AgentClinic"

_backend_cache = {}


def get_backend_for_model(model_str: str) -> OpenAICompatibleBackend:
    if model_str in _backend_cache:
        return _backend_cache[model_str]

    known_models = {
        "qwen2.5-72b": dict(
            base_url="http://api.openai.ukrc.huawei.com:4000/v1",
            model_name="qwen2.5-72b-instruct",
            api_key=os.environ.get("QWEN_API_KEY"),
        ),
    }

    if model_str not in known_models:
        raise ValueError(
            f"No backend config for model '{model_str}'. "
            f"Known models: {list(known_models.keys())}"
        )

    backend = OpenAICompatibleBackend(**known_models[model_str])
    _backend_cache[model_str] = backend
    return backend


def patched_query_model(
    model_str,
    prompt,
    system_prompt,
    tries=30,
    timeout=20.0,
    image_requested=False,
    scene=None,
    max_prompt_len=2**14,
    clip_prompt=False,
):
    backend = get_backend_for_model(model_str)
    return backend.generate(prompt, system_prompt=system_prompt)


def install_patch():
    sys.path.insert(0, str(AGENTCLINIC_PATH))
    import agentclinic
    agentclinic.query_model = patched_query_model
    return agentclinic