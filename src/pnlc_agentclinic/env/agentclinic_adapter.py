import os
import sys
import json
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


_results_log = []
_original_compare_results = None


def patched_compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
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


_trajectory_log = []
_current_scenario_index = -1
_original_doctor_reset = None
_original_inference_doctor = None


def patched_doctor_reset(self):
    global _current_scenario_index
    _current_scenario_index += 1
    return _original_doctor_reset(self)


def patched_inference_doctor(self, question, image_requested=False):
    state = self.agent_hist
    answer = _original_inference_doctor(self, question, image_requested=image_requested)
    _trajectory_log.append({
        "scenario_index": _current_scenario_index,
        "turn_index": self.infs,
        "state": state,
        "incoming_message": question,
        "doctor_action": answer,
    })
    return answer


def save_trajectory_log(path):
    with open(path, "w") as f:
        json.dump(_trajectory_log, f, indent=2)
    return path


def get_trajectory_log():
    return _trajectory_log


def install_patch():
    global _original_compare_results, _original_doctor_reset, _original_inference_doctor
    sys.path.insert(0, str(AGENTCLINIC_PATH))
    import agentclinic
    agentclinic.query_model = patched_query_model
    _original_compare_results = agentclinic.compare_results
    agentclinic.compare_results = patched_compare_results
    _original_doctor_reset = agentclinic.DoctorAgent.reset
    agentclinic.DoctorAgent.reset = patched_doctor_reset
    _original_inference_doctor = agentclinic.DoctorAgent.inference_doctor
    agentclinic.DoctorAgent.inference_doctor = patched_inference_doctor
    return agentclinic