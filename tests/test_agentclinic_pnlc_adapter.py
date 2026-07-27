from pnlc_agentclinic.env.agentclinic_adapter import (
    begin_scenario_log,
    finalize_results_log,
    get_trajectory_log,
    get_results_log,
    patched_compare_results,
    patched_inference_doctor,
    register_backend,
    register_doctor_planner,
    reset_run_logs,
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


class PunctuatedJudgeBackend:
    def generate(self, prompt, system_prompt=""):
        assert "broader disease family" in system_prompt
        return "Yes."


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


def test_moderator_normalizes_a_clear_yes_or_no_response():
    register_backend("independent-judge", PunctuatedJudgeBackend())

    answer = patched_compare_results(
        "DIAGNOSIS READY: Myasthenia gravis",
        "Myasthenia gravis",
        "independent-judge",
        None,
    )
    record = get_results_log()[-1]

    assert answer == "yes"
    assert record["moderator_raw_answer"] == "Yes."
    assert record["moderator_normalized_answer"] == "yes"
    assert record["correct"] is True


def test_result_log_preserves_source_ids_for_no_diagnosis_scenarios():
    reset_run_logs()
    register_backend("independent-judge", PunctuatedJudgeBackend())

    begin_scenario_log("Benign Paroxysmal Positional Vertigo")
    begin_scenario_log("Schizotypal personality disorder")
    patched_compare_results(
        "DIAGNOSIS READY: Schizotypal personality disorder",
        "Schizotypal personality disorder",
        "independent-judge",
        None,
    )
    begin_scenario_log("Bowen's disease")

    results = finalize_results_log(expected_scenarios=3)

    assert [record["scenario_index"] for record in results] == [0, 1, 2]
    assert results[0]["moderator_normalized_answer"] == "no_diagnosis"
    assert results[0]["correct_diagnosis"] == (
        "Benign Paroxysmal Positional Vertigo"
    )
    assert results[0]["reached_diagnosis"] is False
    assert results[1]["doctor_diagnosis_text"] == (
        "DIAGNOSIS READY: Schizotypal personality disorder"
    )
    assert results[1]["reached_diagnosis"] is True
    assert results[2]["moderator_normalized_answer"] == "no_diagnosis"
    assert results[2]["correct_diagnosis"] == "Bowen's disease"
    assert results[2]["reached_diagnosis"] is False

    reset_run_logs()
