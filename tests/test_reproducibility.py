import random
from types import SimpleNamespace

import numpy as np
import torch

from pnlc_agentclinic.llm_backends.huggingface_backend import (
    HuggingFaceBackend,
)
from pnlc_agentclinic.llm_backends.openai_compatible import (
    OpenAICompatibleBackend,
)
from pnlc_agentclinic.reproducibility import seed_everything


class RecordingCompletions:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content="response")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def build_openai_backend(seed):
    backend = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
    backend.model_name = "test-model"
    backend.max_tokens = 20
    backend.temperature = 0.7
    backend.seed = seed
    completions = RecordingCompletions()
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return backend, completions


def test_openai_backend_sends_configured_seed():
    backend, completions = build_openai_backend(seed=42)

    assert backend.generate("hello") == "response"
    assert completions.request["seed"] == 42


def test_openai_backend_applies_explicit_seed_offset():
    backend, completions = build_openai_backend(seed=42)

    backend.generate("same prompt", seed_offset=3)

    assert completions.request["seed"] == 45


def test_openai_backend_omits_disabled_seed():
    backend, completions = build_openai_backend(seed=None)

    backend.generate("hello")
    assert "seed" not in completions.request


def test_huggingface_backend_builds_distinct_repeatable_candidate_generators():
    backend = HuggingFaceBackend.__new__(HuggingFaceBackend)
    backend.device = "cpu"
    backend.seed = 17

    first = torch.rand(3, generator=backend._generation_generator(seed_offset=0))
    second = torch.rand(3, generator=backend._generation_generator(seed_offset=1))
    replay = torch.rand(3, generator=backend._generation_generator(seed_offset=0))

    assert not torch.equal(first, second)
    assert torch.equal(first, replay)


def test_seed_everything_repeats_local_rng_streams():
    seed_everything(9)
    first = (random.random(), np.random.random(), torch.rand(1))

    seed_everything(9)
    second = (random.random(), np.random.random(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
