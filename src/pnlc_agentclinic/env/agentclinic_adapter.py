import re
import sys
import json
from pathlib import Path

AGENTCLINIC_PATH = Path(__file__).resolve().parents[3] / "external" / "AgentClinic"

_backend_cache = {}
_doctor_planner = None


def register_backend(model_str: str, backend):
    """Register an already-built backend (OpenAICompatibleBackend or HuggingFaceBackend)
    under `model_str`. Callers (hydra entrypoint scripts) build the backend via
    `pnlc_agentclinic.llm_backends.factory.build_generation_backend(cfg.model_backends)`,
    keeping any endpoint/secret/model choice out of this module.
    """
    _backend_cache[model_str] = backend
    return backend


def get_backend_for_model(model_str: str):
    if model_str not in _backend_cache:
        raise ValueError(
            f"No backend registered for model '{model_str}'. "
            f"Call register_backend(...) with hydra-resolved config before running the simulation. "
            f"Registered models: {list(_backend_cache.keys())}"
        )
    return _backend_cache[model_str]


def register_doctor_planner(planner):
    """Enable PNLC refinement for doctor turns in the patched AgentClinic loop."""
    global _doctor_planner
    _doctor_planner = planner
    return planner


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


def patched_compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    raw_answer = patched_query_model(
        moderator_llm,
        "REFERENCE DIAGNOSIS:\n"
        f"{correct_diagnosis}\n\n"
        "DOCTOR DIAGNOSIS:\n"
        f"{diagnosis}\n\n"
        "Does the doctor give the same diagnosis as the reference?",
        (
            "You are an independent clinical diagnosis evaluator. Mark Yes only when "
            "the doctor's final diagnosis is the reference diagnosis or an established "
            "medical synonym. Mark No for a broader disease family, an unspecified "
            "lesion, a related mechanism, an incomplete syndrome, a differential with "
            "multiple alternatives, or a different diagnosis. Ignore treatment and "
            "explanatory text. Respond with exactly Yes or No."
        ),
    )
    match = re.search(r"\b(yes|no)\b", raw_answer, flags=re.IGNORECASE)
    answer = match.group(1).lower() if match else "invalid"
    _results_log.append({
        "scenario_index": len(_results_log),
        "doctor_diagnosis_text": diagnosis,
        "correct_diagnosis": correct_diagnosis,
        "moderator_raw_answer": raw_answer,
        "moderator_normalized_answer": answer,
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


def normalize_diagnosis_action(action):
    match = re.search(
        r"DIAGNOSIS\s+READY\s*:\s*(.+)",
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match or not match.group(1).strip():
        return None
    return "DIAGNOSIS READY: " + match.group(1).strip()


def force_final_diagnosis(self, question):
    raw_answer = patched_query_model(
        self.backend,
        "\nHere is a history of your dialogue: " + self.agent_hist +
        "\nHere was the latest patient response or test result: " + question +
        "\nFINAL DIAGNOSIS REQUIRED. Use the evidence already available and commit "
        "to one most likely diagnosis. Do not ask a question or request a test. "
        "Return exactly:\n"
        "THOUGHT: <brief private diagnostic reasoning>\n"
        "ACTION: DIAGNOSIS READY: <single most likely diagnosis>",
        self.system_prompt(),
        scene=self.scenario,
    )
    forced_thought, forced_action, forced_parsed_ok = parse_thought_action(raw_answer)
    candidate = forced_action if forced_parsed_ok else raw_answer.strip()
    normalized = normalize_diagnosis_action(candidate)
    if normalized is None:
        candidate = re.sub(
            r"^(?:ACTION\s*:|DIAGNOSIS\s*:)\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        if not candidate:
            raise ValueError("The model returned an empty forced final diagnosis.")
        normalized = "DIAGNOSIS READY: " + candidate
    return forced_thought, normalized, raw_answer


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
    must_diagnose = self.infs == self.MAX_INFS - 1
    critic_record = None
    critic_error = None
    forced_diagnosis_used = False
    forced_diagnosis_raw_output = None
    if _doctor_planner is not None and parsed_ok:
        try:
            plan = _doctor_planner.plan(
                dialogue_state=state,
                incoming_message=question,
                initial_thought=thought,
                initial_action=action,
                doctor_system_prompt=self.system_prompt(),
                clinical_objective=str(self.presentation),
                must_diagnose=must_diagnose,
            )
            thought = plan.refined_thought
            action = plan.action
            critic_record = plan.to_dict()
        except Exception as error:
            critic_error = f"{type(error).__name__}: {error}"
            print(
                "PNLC planner failed; using the doctor's original action. "
                f"Reason: {critic_error}"
            )
    elif _doctor_planner is not None:
        critic_error = (
            "Initial doctor response did not contain both THOUGHT and ACTION labels."
        )

    if _doctor_planner is not None and must_diagnose:
        normalized_diagnosis = normalize_diagnosis_action(action)
        if normalized_diagnosis is not None:
            action = normalized_diagnosis
        else:
            forced_thought, action, forced_diagnosis_raw_output = (
                force_final_diagnosis(self, question)
            )
            if forced_thought:
                thought = forced_thought
            forced_diagnosis_used = True

    self.agent_hist += question + "\n\n" + action + "\n\n"
    self.infs += 1

    trajectory_record = {
        "scenario_index": _current_scenario_index,
        "turn_index": self.infs,
        "state": state,
        "incoming_message": question,
        "thought": thought,
        "doctor_action": action,
        "raw_model_output": raw_answer,
        "parsed_ok": parsed_ok,
    }
    if _doctor_planner is not None:
        trajectory_record.update({
            "critic_used": critic_record is not None,
            "critic": critic_record,
            "critic_error": critic_error,
            "forced_diagnosis_used": forced_diagnosis_used,
            "forced_diagnosis_raw_output": forced_diagnosis_raw_output,
        })
    _trajectory_log.append(trajectory_record)
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
    global _original_doctor_reset, _original_doctor_system_prompt
    sys.path.insert(0, str(AGENTCLINIC_PATH))
    import agentclinic
    agentclinic.query_model = patched_query_model
    agentclinic.compare_results = patched_compare_results
    _original_doctor_reset = agentclinic.DoctorAgent.reset
    agentclinic.DoctorAgent.reset = patched_doctor_reset
    _original_doctor_system_prompt = agentclinic.DoctorAgent.system_prompt
    agentclinic.DoctorAgent.system_prompt = patched_doctor_system_prompt
    agentclinic.DoctorAgent.inference_doctor = patched_inference_doctor
    return agentclinic
