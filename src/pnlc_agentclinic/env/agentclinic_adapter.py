import os
import re
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
            base_url="",
            model_name="qwen2.5-72b-instruct",
            api_key=os.environ.get("QWEN_API_KEY"),
            max_tokens=400,
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
_original_doctor_system_prompt = None

THOUGHT_ACTION_INSTRUCTION = (
    "\n\nIMPORTANT: Structure every response in exactly this format:\n"
    "THOUGHT: <your private clinical reasoning about what to ask, test, or conclude next>\n"
    "ACTION: <your actual dialogue line to the patient, or a test request formatted as "
    "REQUEST TEST: [test], or your diagnosis formatted as DIAGNOSIS READY: [diagnosis]>"
)


def patched_doctor_reset(self):
    global _current_scenario_index
    _current_scenario_index += 1
    return _original_doctor_reset(self)


def patched_doctor_system_prompt(self):
    return _original_doctor_system_prompt(self) + THOUGHT_ACTION_INSTRUCTION


def parse_thought_action(raw_text):
    thought_match = re.search(r"THOUGHT:\s*(.*?)(?=ACTION:|$)", raw_text, re.DOTALL)
    action_match = re.search(r"ACTION:\s*(.*)", raw_text, re.DOTALL)
    if thought_match and action_match:
        return thought_match.group(1).strip(), action_match.group(1).strip(), True
    return "", raw_text.strip(), False


def patched_inference_doctor(self, question, image_requested=False):
    if self.infs >= self.MAX_INFS:
        return "Maximum inferences reached"

    state = self.agent_hist
    raw_answer = patched_query_model(
        self.backend,
        "\nHere is a history of your dialogue: " + self.agent_hist +
        "\n Here was the patient response: " + question +
        "Now please continue your dialogue\nDoctor: ",
        self.system_prompt(),
        image_requested=image_requested,
        scene=self.scenario,
    )
    thought, action, parsed_ok = parse_thought_action(raw_answer)

    self.agent_hist += question + "\n\n" + action + "\n\n"
    self.infs += 1

    _trajectory_log.append({
        "scenario_index": _current_scenario_index,
        "turn_index": self.infs,
        "state": state,
        "incoming_message": question,
        "thought": thought,
        "doctor_action": action,
        "raw_model_output": raw_answer,
        "parsed_ok": parsed_ok,
    })
    return action


def save_trajectory_log(path):
    with open(path, "w") as f:
        json.dump(_trajectory_log, f, indent=2)
    return path


def get_trajectory_log():
    return _trajectory_log


def get_thought_action_compliance_rate():
    if not _trajectory_log:
        return None
    return sum(t["parsed_ok"] for t in _trajectory_log) / len(_trajectory_log)


def install_patch():
    global _original_compare_results, _original_doctor_reset, _original_doctor_system_prompt
    sys.path.insert(0, str(AGENTCLINIC_PATH))
    import agentclinic
    agentclinic.query_model = patched_query_model
    _original_compare_results = agentclinic.compare_results
    agentclinic.compare_results = patched_compare_results
    _original_doctor_reset = agentclinic.DoctorAgent.reset
    agentclinic.DoctorAgent.reset = patched_doctor_reset
    _original_doctor_system_prompt = agentclinic.DoctorAgent.system_prompt
    agentclinic.DoctorAgent.system_prompt = patched_doctor_system_prompt
    agentclinic.DoctorAgent.inference_doctor = patched_inference_doctor
    return agentclinic