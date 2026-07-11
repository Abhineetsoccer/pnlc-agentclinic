"""
Adapter between AgentClinic's vendored source (external/AgentClinic/agentclinic.py,
untouched) and our own OpenAICompatibleBackend.

Usage, from a run script:

    from pnlc_agentclinic.env.agentclinic_adapter import install_patch, AGENTCLINIC_PATH
    import os

    agentclinic = install_patch()
    os.chdir(AGENTCLINIC_PATH)   # required: ScenarioLoaderMedQA opens its .jsonl
                                  # with a bare relative path, so the working
                                  # directory must be external/AgentClinic/ itself.
    agentclinic.main(
        api_key=None, replicate_api_key=None, inf_type="llm",
        doctor_bias="None", patient_bias="None",
        doctor_llm="qwen2.5-72b", patient_llm="qwen2.5-72b",
        measurement_llm="qwen2.5-72b", moderator_llm="qwen2.5-72b",
        num_scenarios=2, dataset="MedQA", img_request=False,
        total_inferences=20, anthropic_api_key=None,
    )
"""

import os
import sys
from pathlib import Path

from pnlc_agentclinic.llm_backends.openai_compatible import OpenAICompatibleBackend

AGENTCLINIC_PATH = Path(__file__).resolve().parents[3] / "external" / "AgentClinic"

_backend_cache = {}


def get_backend_for_model(model_str: str) -> OpenAICompatibleBackend:
    """Resolve a model name string (as passed to --doctor_llm etc.) to a
    ready-to-use backend, building it once and reusing it on repeat calls."""
    if model_str in _backend_cache:
        return _backend_cache[model_str]

    known_models = {
        "qwen2.5-72b": dict(
            base_url="http://api.openai.ukrc.huawei.com:4000/v1",
            model_name="qwen2.5-72b-instruct",
            api_key=os.environ.get("QWEN_API_KEY"),
        ),
        # Add more company-hosted models here as you need them, e.g.:
        # "qwen3-30b-a3b": dict(base_url=..., model_name=..., api_key=os.environ.get("QWEN3_API_KEY")),
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
    """
    Drop-in replacement for agentclinic.query_model. Signature matches the
    original exactly (including unused args like tries/timeout/image_requested)
    so every existing call site in agentclinic.py keeps working unmodified.
    """
    backend = get_backend_for_model(model_str)
    return backend.generate(prompt, system_prompt=system_prompt)


import json

_results_log = []
_original_compare_results = None


def patched_compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    """
    Wraps the original compare_results: same behavior, same return value,
    but also records (diagnosis, correct answer, moderator's raw response)
    to _results_log as a side effect. Call order == scenario order, since
    compare_results is called at most once per completed scenario in
    agentclinic.main()'s loop.

    Caveat: if a scenario's doctor never emits "DIAGNOSIS READY" within its
    turn budget, compare_results is never called for it -- so len(_results_log)
    can be less than num_scenarios if that happens. Worth checking for.
    """
    answer = _original_compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe)
    _results_log.append({
        "scenario_index": len(_results_log),
        "doctor_diagnosis_text": diagnosis,
        "correct_diagnosis": correct_diagnosis,
        "moderator_raw_answer": answer,
        "correct": answer == "yes",
    })
    return answer


def save_results_log(path):
    with open(path, "w") as f:
        json.dump(_results_log, f, indent=2)
    return path


def get_results_log():
    return _results_log


def install_patch():
    """
    Makes external/AgentClinic importable, then reassigns agentclinic.query_model
    to our patched version. Every class in agentclinic.py calls query_model as a
    bare name resolved from the module's own namespace at call time, so this one
    reassignment redirects all of them -- PatientAgent, DoctorAgent,
    MeasurementAgent, and compare_results -- without editing agentclinic.py itself.
    Also patches compare_results to log structured results as a side effect.
    Returns the agentclinic module so the caller can use it directly.
    """
    global _original_compare_results
    sys.path.insert(0, str(AGENTCLINIC_PATH))
    import agentclinic  # vendored module, unmodified on disk
    agentclinic.query_model = patched_query_model
    _original_compare_results = agentclinic.compare_results
    agentclinic.compare_results = patched_compare_results
    return agentclinic