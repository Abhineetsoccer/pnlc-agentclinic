from pnlc_agentclinic.env.agentclinic_adapter import (
    get_trajectory_log,
    patched_inference_doctor,
    register_backend,
    register_doctor_planner,
)


class InitialResponseBackend:
    def generate(self, prompt, system_prompt=""):
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


class FakePlanner:
    def plan(self, **kwargs):
        assert kwargs["initial_thought"] == "Ask about a discriminating symptom."
        return FakePlan()


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
    register_doctor_planner(FakePlanner())
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

    register_doctor_planner(None)
