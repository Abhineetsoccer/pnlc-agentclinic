import pytest
import torch

from pnlc_agentclinic.planning.pnlc_planner import NaturalLanguageCriticPlanner


class FakeBackend:
    def __init__(self):
        self.positive_count = 0
        self.negative_count = 0

    def generate(self, prompt, system_prompt=""):
        if "harmful or unproductive future" in prompt:
            self.negative_count += 1
            return f"The doctor misses important finding {self.negative_count}."
        if "plausible productive future" in prompt:
            self.positive_count += 1
            return f"The patient reveals discriminating symptom {self.positive_count}."
        if "critic trained on prior clinical trajectories" in prompt:
            return "THOUGHT: Ask a focused discriminating question before diagnosing."
        if "produce the doctor's next environment action" in prompt:
            return "ACTION: Can you tell me when the weakness is worst?"
        raise AssertionError(f"Unexpected prompt: {prompt}")


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
    result = build_planner().plan(
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


def test_planner_rejects_an_incompatible_embedder():
    with pytest.raises(ValueError, match="same embedding model"):
        build_planner(embedder=FakeEmbedder(dimension=4)).plan(
            dialogue_state="history",
            incoming_message="message",
            initial_thought="thought",
            initial_action="action",
            doctor_system_prompt="system",
        )
