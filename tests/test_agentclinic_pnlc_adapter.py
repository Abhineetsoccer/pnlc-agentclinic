from pnlc_agentclinic.env.agentclinic_adapter import (
    get_trajectory_log,
    patched_inference_doctor,
    register_backend,
    register_doctor_planner,
)


class InitialResponseBackend:
    def generate(self, prompt, system_prompt=""):
        if "FINAL DIAGNOSIS REQUIRED" in prompt:
            return (
                "THOUGHT: The fatigability pattern supports myasthenia gravis.\n"
                "ACTION: Myasthenia gravis"
            )
        return (
            "THOUGHT: Ask about a discriminating symptom.\n"
            "ACTION: Is the weakness worse after activity?"
        )


class FakePlan:
    refined_thought = "Ask about fatigability before diagnosing."
    action = "Does the weakness improve after resting?"

    def to_dict(self):
        return {
            "refined_thought": self.refined_thought,
            "action": self.action,
        }


class FakeFinalPlan(FakePlan):
    action = "diagnosis ready: Myasthenia gravis"


class FakePlanner:
    last_must_diagnose = None

    def plan(self, **kwargs):
        assert kwargs["initial_thought"] == "Ask about a discriminating symptom."
        self.last_must_diagnose = kwargs["must_diagnose"]
        return FakeFinalPlan() if kwargs["must_diagnose"] else FakePlan()


class FailingPlanner:
    def plan(self, **kwargs):
        raise RuntimeError("critic unavailable")


class FakeDoctor:
    infs = 0
    MAX_INFS = 20
    agent_hist = ""
    backend = "fake-backend"
    scenario = None
    presentation = "Assess the cause of fatigable weakness."

    @staticmethod
    def system_prompt():
        return "You are a doctor."


def test_adapter_executes_and_logs_the_refined_action():
    register_backend("fake-backend", InitialResponseBackend())
    planner = register_doctor_planner(FakePlanner())
    doctor = FakeDoctor()

    action = patched_inference_doctor(
        doctor,
        "The weakness is worse in the evening.",
    )
    record = get_trajectory_log()[-1]

    assert action == "Does the weakness improve after resting?"
    assert record["thought"] == "Ask about fatigability before diagnosing."
    assert record["doctor_action"] == action
    assert record["critic_used"] is True
    assert record["critic_error"] is None
    assert planner.last_must_diagnose is False

    register_doctor_planner(None)


def test_adapter_marks_the_last_available_turn_as_diagnosis_required():
    register_backend("fake-backend", InitialResponseBackend())
    planner = register_doctor_planner(FakePlanner())
    doctor = FakeDoctor()
    doctor.infs = doctor.MAX_INFS - 1

    action = patched_inference_doctor(
        doctor,
        "This is the final question. Please provide a diagnosis.",
    )

    assert planner.last_must_diagnose is True
    assert action == "DIAGNOSIS READY: Myasthenia gravis"
    register_doctor_planner(None)


def test_adapter_forces_marker_if_planner_fails_on_final_turn():
    register_backend("fake-backend", InitialResponseBackend())
    register_doctor_planner(FailingPlanner())
    doctor = FakeDoctor()
    doctor.infs = doctor.MAX_INFS - 1

    action = patched_inference_doctor(
        doctor,
        "This is the final question. Please provide a diagnosis.",
    )
    record = get_trajectory_log()[-1]

    assert action == "DIAGNOSIS READY: Myasthenia gravis"
    assert record["forced_diagnosis_used"] is True
    assert "critic unavailable" in record["critic_error"]
    register_doctor_planner(None)
