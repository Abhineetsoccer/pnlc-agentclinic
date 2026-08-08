import pytest
import torch

from pnlc_agentclinic.planning.pnlc_planner import NaturalLanguageCriticPlanner


class FakeBackend:
    def __init__(self):
        self.positive_count = 0
        self.negative_count = 0
        self.action_count = 0
        self.goal_seed_offsets = []

    def generate(self, prompt, system_prompt="", seed_offset=0):
        if "harmful or unproductive future" in prompt:
            self.goal_seed_offsets.append(seed_offset)
            self.negative_count += 1
            return f"The doctor misses important finding {self.negative_count}."
        if "plausible productive future" in prompt:
            self.goal_seed_offsets.append(seed_offset)
            self.positive_count += 1
            return f"The patient reveals discriminating symptom {self.positive_count}."
        if "critic trained on prior clinical trajectories" in prompt:
            return "THOUGHT: Ask a focused discriminating question before diagnosing."
        if "produce the doctor's next environment action" in prompt:
            self.action_count += 1
            if "previous response did not contain" in prompt:
                return "ACTION: DIAGNOSIS READY: Myasthenia gravis"
            return "ACTION: Can you tell me when the weakness is worst?"
        raise AssertionError(f"Unexpected prompt: {prompt}")


class DuplicateGoalBackend:
    def __init__(self, always_duplicate=False):
        self.always_duplicate = always_duplicate
        self.positive_calls = 0
        self.seed_offsets = []

    def generate(self, prompt, system_prompt="", seed_offset=0):
        self.seed_offsets.append(seed_offset)
        if "plausible productive future" in prompt:
            self.positive_calls += 1
            if self.positive_calls == 1 or self.always_duplicate:
                return "The patient reveals ptosis."
            if self.positive_calls == 2:
                return "  THE patient   reveals ptosis.  "
            return "The patient reveals fatigability."
        return "The doctor anchors on an unsupported diagnosis."


class FakeSummarizer:
    def summarize(self, state):
        return "The patient has intermittent weakness."


class FakeEmbedder:
    def __init__(self, dimension=3):
        self.dimension = dimension

    def embed_batch(self, texts):
        base = [0.1] * self.dimension
        return [base for _ in texts]


class FakeCritic(torch.nn.Module):
    state_dim = 3
    thought_dim = 3

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def score(self, state, thought, goal):
        return torch.tensor([0.8, 0.7, 0.4, 0.3], device=state.device)


def build_planner(embedder=None):
    return NaturalLanguageCriticPlanner(
        generation_backend=FakeBackend(),
        summarizer=FakeSummarizer(),
        embedder=embedder or FakeEmbedder(),
        critic=FakeCritic(),
    )


def test_planner_refines_thought_and_generates_action():
    planner = build_planner()
    result = planner.plan(
        dialogue_state="history",
        incoming_message="It is worse in the evening.",
        initial_thought="Consider neuromuscular causes.",
        initial_action="Do you have weakness?",
        doctor_system_prompt="You are a doctor.",
    )

    assert result.refined_thought == (
        "Ask a focused discriminating question before diagnosing."
    )
    assert result.action == "Can you tell me when the weakness is worst?"
    assert len(result.rounds) == 1
    assert [item.polarity for item in result.rounds[0].assessments] == [
        "positive",
        "positive",
        "negative",
        "negative",
    ]
    assert result.rounds[0].assessments[0].raw_score == pytest.approx(0.8)
    assert planner.generation_backend.goal_seed_offsets == [0, 1, 2, 3]


def test_final_turn_retries_until_the_diagnosis_marker_is_present():
    planner = build_planner()
    result = planner.plan(
        dialogue_state="history",
        incoming_message="This is the final question. Please provide a diagnosis.",
        initial_thought="Commit to the most likely neuromuscular diagnosis.",
        initial_action="DIAGNOSIS READY: Myasthenia gravis",
        doctor_system_prompt="You are a doctor.",
        must_diagnose=True,
    )

    assert result.action == "DIAGNOSIS READY: Myasthenia gravis"
    assert result.must_diagnose is True
    assert result.diagnosis_retry_used is True
    assert planner.generation_backend.action_count == 2


def test_duplicate_goals_are_resampled_with_new_seed_offsets():
    planner = NaturalLanguageCriticPlanner.__new__(NaturalLanguageCriticPlanner)
    planner.generation_backend = DuplicateGoalBackend()
    planner.positive_goals = 2
    planner.negative_goals = 1

    goals = planner._generate_goals(
        "state",
        "message",
        "thought",
        "objective",
    )

    assert [goal for polarity, goal in goals if polarity == "positive"] == [
        "The patient reveals ptosis.",
        "The patient reveals fatigability.",
    ]
    assert planner.generation_backend.seed_offsets == [0, 1, 2, 3]


def test_persistent_duplicate_goals_fail_instead_of_silently_scoring_copies():
    planner = NaturalLanguageCriticPlanner.__new__(NaturalLanguageCriticPlanner)
    planner.generation_backend = DuplicateGoalBackend(always_duplicate=True)
    planner.positive_goals = 2
    planner.negative_goals = 1

    with pytest.raises(ValueError, match="duplicate positive goals"):
        planner._generate_goals(
            "state",
            "message",
            "thought",
            "objective",
        )


def test_planner_rejects_an_incompatible_embedder():
    with pytest.raises(ValueError, match="same embedding model"):
        build_planner(embedder=FakeEmbedder(dimension=4)).plan(
            dialogue_state="history",
            incoming_message="message",
            initial_thought="thought",
            initial_action="action",
            doctor_system_prompt="system",
        )
