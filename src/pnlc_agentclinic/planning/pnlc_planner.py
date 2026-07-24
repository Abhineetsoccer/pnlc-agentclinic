import re
from dataclasses import asdict, dataclass

import torch


CLINICAL_CONTEXT = (
    "A doctor is evaluating a patient in a clinic. The doctor must discover the "
    "condition by asking questions, performing examinations, and requesting tests. "
    "The doctor has a limited number of interactions before committing to a diagnosis."
)


@dataclass
class GoalAssessment:
    polarity: str
    goal: str
    raw_score: float
    displayed_probability: float


@dataclass
class RefinementRound:
    input_thought: str
    assessments: list[GoalAssessment]
    output_thought: str


@dataclass
class PlanningResult:
    state_summary: str
    clinical_objective: str
    must_diagnose: bool
    diagnosis_retry_used: bool
    initial_thought: str
    initial_action: str
    refined_thought: str
    action: str
    rounds: list[RefinementRound]

    def to_dict(self):
        return asdict(self)


def _strip_label(text: str, label: str) -> str:
    text = text.strip()
    match = re.search(
        rf"(?:^|\n)\s*{re.escape(label)}\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        text = match.group(1).strip()
    return text


def _parse_thought(text: str) -> str:
    text = _strip_label(text, "THOUGHT")
    text = re.split(r"\n\s*ACTION\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip()


def _parse_action(text: str) -> str:
    action_match = re.search(
        r"(?:^|\n)\s*ACTION\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if action_match:
        return action_match.group(1).strip()
    if re.search(r"(?:^|\n)\s*THOUGHT\s*:", text, flags=re.IGNORECASE):
        return ""
    return text.strip()


def _normalize_diagnosis_marker(action: str) -> str:
    match = re.search(
        r"DIAGNOSIS\s+READY\s*:\s*(.+)",
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    diagnosis = match.group(1).strip()
    if not diagnosis:
        return ""
    return f"DIAGNOSIS READY: {diagnosis}"


class NaturalLanguageCriticPlanner:
    """PNLC inference loop specialized to AgentClinic dialogue."""

    def __init__(
        self,
        generation_backend,
        summarizer,
        embedder,
        critic,
        positive_goals: int = 2,
        negative_goals: int = 2,
        refinement_rounds: int = 1,
    ):
        if positive_goals < 1 or negative_goals < 1:
            raise ValueError("PNLC requires at least one positive and one negative goal.")
        if refinement_rounds < 1:
            raise ValueError("PNLC requires at least one refinement round.")

        self.generation_backend = generation_backend
        self.summarizer = summarizer
        self.embedder = embedder
        self.critic = critic
        self.positive_goals = positive_goals
        self.negative_goals = negative_goals
        self.refinement_rounds = refinement_rounds
        self.critic.eval()

    @property
    def device(self):
        return next(self.critic.parameters()).device

    def _goal_prompt(
        self,
        state_summary: str,
        incoming_message: str,
        thought: str,
        polarity: str,
        clinical_objective: str,
    ) -> str:
        if polarity == "positive":
            outcome_instruction = (
                "Describe one plausible productive future: for example, the patient or "
                "a test reveals diagnostically useful information, or the doctor reaches "
                "a well-supported diagnosis."
            )
        else:
            outcome_instruction = (
                "Describe one plausible harmful or unproductive future: for example, "
                "important information remains hidden, an unnecessary test is pursued, "
                "or the doctor becomes confident in an incorrect diagnosis."
            )

        return (
            f"{CLINICAL_CONTEXT}\n\n"
            "Doctor's clinical objective:\n"
            f"{clinical_objective or 'Determine the most likely diagnosis.'}\n\n"
            "Current dialogue summary:\n"
            f"{state_summary or 'No previous dialogue has occurred.'}\n\n"
            "Latest patient message or test result:\n"
            f"{incoming_message or 'No new information.'}\n\n"
            "Doctor's proposed thought:\n"
            f"{thought}\n\n"
            f"{outcome_instruction}\n"
            "Write only a concise future clinical-state summary. Do not include a "
            "label, probability, recommendation, or explanation."
        )

    def _generate_goals(
        self,
        state_summary: str,
        incoming_message: str,
        thought: str,
        clinical_objective: str,
    ) -> list[tuple[str, str]]:
        goals = []
        for polarity, count in (
            ("positive", self.positive_goals),
            ("negative", self.negative_goals),
        ):
            for _ in range(count):
                raw_goal = self.generation_backend.generate(
                    self._goal_prompt(
                        state_summary,
                        incoming_message,
                        thought,
                        polarity,
                        clinical_objective,
                    )
                )
                goal = _strip_label(raw_goal, "GOAL")
                if not goal:
                    raise ValueError(f"The generator returned an empty {polarity} goal.")
                goals.append((polarity, goal))
        return goals

    def _embed_and_score(
        self,
        state_summary: str,
        thought: str,
        goals: list[tuple[str, str]],
    ) -> list[GoalAssessment]:
        texts = [state_summary, thought, *(goal for _, goal in goals)]
        vectors = self.embedder.embed_batch(texts)
        state_vector = torch.as_tensor(
            vectors[0], dtype=torch.float32, device=self.device
        )
        thought_vector = torch.as_tensor(
            vectors[1], dtype=torch.float32, device=self.device
        )
        goal_vectors = torch.as_tensor(
            vectors[2:], dtype=torch.float32, device=self.device
        )

        if state_vector.ndim != 1 or state_vector.shape[0] != self.critic.state_dim:
            raise ValueError(
                "Runtime state embedding dimension "
                f"{tuple(state_vector.shape)} does not match critic state dimension "
                f"{self.critic.state_dim}. Use the same embedding model used to build "
                "the critic-training dataset."
            )
        if (
            thought_vector.ndim != 1
            or thought_vector.shape[0] != self.critic.thought_dim
        ):
            raise ValueError(
                "Runtime thought embedding dimension "
                f"{tuple(thought_vector.shape)} does not match critic thought dimension "
                f"{self.critic.thought_dim}. Use the same embedding model used to build "
                "the critic-training dataset."
            )
        if goal_vectors.ndim != 2 or goal_vectors.shape[1] != self.critic.state_dim:
            raise ValueError(
                "Runtime goal embeddings do not match the critic state dimension."
            )

        state_batch = state_vector.unsqueeze(0).expand(len(goals), -1)
        thought_batch = thought_vector.unsqueeze(0).expand(len(goals), -1)
        with torch.no_grad():
            scores = self.critic.score(state_batch, thought_batch, goal_vectors)
        if not torch.isfinite(scores).all():
            raise ValueError("The critic produced a NaN or infinite goal score.")

        assessments = []
        for (polarity, goal), score in zip(goals, scores.tolist()):
            assessments.append(
                GoalAssessment(
                    polarity=polarity,
                    goal=goal,
                    raw_score=float(score),
                    displayed_probability=min(1.0, max(0.0, float(score))),
                )
            )
        return assessments

    @staticmethod
    def _natural_language_value(assessments: list[GoalAssessment]) -> str:
        lines = []
        for assessment in assessments:
            probability = round(100 * assessment.displayed_probability)
            lines.append(
                f"- {assessment.polarity.upper()} future "
                f"(critic-estimated likelihood {probability}%): {assessment.goal}"
            )
        return "\n".join(lines)

    def _refine_thought(
        self,
        state_summary: str,
        incoming_message: str,
        thought: str,
        assessments: list[GoalAssessment],
        clinical_objective: str,
        must_diagnose: bool,
    ) -> str:
        if must_diagnose:
            decision_instruction = (
                "This is the final doctor turn. The refined thought must commit to "
                "one most likely diagnosis using the evidence already available. Do "
                "not propose another question, examination, or test."
            )
        else:
            decision_instruction = (
                "Decide whether the current thought is still the best next strategy. "
                "If not, diagnose the likely failure and replace it with a better "
                "clinical reasoning strategy that mitigates that failure. Preserve "
                "the current thought if the feedback does not justify a change."
            )

        prompt = (
            f"{CLINICAL_CONTEXT}\n\n"
            "Doctor's clinical objective:\n"
            f"{clinical_objective or 'Determine the most likely diagnosis.'}\n\n"
            "Current dialogue summary:\n"
            f"{state_summary or 'No previous dialogue has occurred.'}\n\n"
            "Latest patient message or test result:\n"
            f"{incoming_message or 'No new information.'}\n\n"
            "Current proposed clinical thought:\n"
            f"{thought}\n\n"
            "A critic trained on prior clinical trajectories assessed possible "
            "outcomes of this thought:\n"
            f"{self._natural_language_value(assessments)}\n\n"
            f"{decision_instruction}\n"
            "Return exactly one concise private reasoning step in this format:\n"
            "THOUGHT: <refined clinical reasoning>"
        )
        raw_refinement = self.generation_backend.generate(prompt)
        refined = _parse_thought(raw_refinement)
        if not refined:
            raise ValueError("The generator returned an empty refined thought.")
        return refined

    def _generate_action(
        self,
        state_summary: str,
        incoming_message: str,
        thought: str,
        doctor_system_prompt: str,
        clinical_objective: str,
        must_diagnose: bool,
    ) -> tuple[str, bool]:
        context = (
            "Use the following private clinical reasoning to produce the doctor's "
            "next environment action. Do not reveal or discuss the private reasoning.\n\n"
            "Doctor's clinical objective:\n"
            f"{clinical_objective or 'Determine the most likely diagnosis.'}\n\n"
            "Current dialogue summary:\n"
            f"{state_summary or 'No previous dialogue has occurred.'}\n\n"
            "Latest patient message or test result:\n"
            f"{incoming_message or 'No new information.'}\n\n"
            "Private clinical reasoning:\n"
            f"{thought}\n\n"
        )
        if must_diagnose:
            action_instruction = (
                "FINAL TURN: You must commit to exactly one most likely diagnosis. "
                "Do not ask another question or request another test. Return exactly:\n"
                "ACTION: DIAGNOSIS READY: <single most likely diagnosis>"
            )
        else:
            action_instruction = (
                "Return exactly one of the following, prefixed with ACTION:\n"
                "ACTION: <a 1-3 sentence question or response to the patient>\n"
                "ACTION: REQUEST TEST: [test]\n"
                "ACTION: DIAGNOSIS READY: [diagnosis]"
            )

        raw_action = self.generation_backend.generate(
            context + action_instruction,
            system_prompt=doctor_system_prompt,
        )
        action = _parse_action(raw_action)
        if not action:
            raise ValueError("The generator returned an empty action.")
        if not must_diagnose:
            return action, False

        normalized = _normalize_diagnosis_marker(action)
        if normalized:
            return normalized, False

        retry_prompt = (
            context
            + "Your previous response did not contain the required final-diagnosis "
            "marker. Do not provide reasoning, questions, tests, alternatives, or "
            "explanation. Return exactly:\n"
            "ACTION: DIAGNOSIS READY: <single most likely diagnosis>"
        )
        retry_raw = self.generation_backend.generate(
            retry_prompt,
            system_prompt=doctor_system_prompt,
        )
        retry_action = _parse_action(retry_raw)
        normalized = _normalize_diagnosis_marker(retry_action)
        if normalized:
            return normalized, True
        if not retry_action:
            raise ValueError("The generator returned an empty final diagnosis.")

        # The retry was diagnosis-only, so preserve its content while guaranteeing
        # the exact marker AgentClinic uses to terminate and score a scenario.
        return f"DIAGNOSIS READY: {retry_action}", True

    def plan(
        self,
        dialogue_state: str,
        incoming_message: str,
        initial_thought: str,
        initial_action: str,
        doctor_system_prompt: str,
        clinical_objective: str = "",
        must_diagnose: bool = False,
    ) -> PlanningResult:
        """Refine an initial thought once or more, then realize it as an action."""
        state_summary = self.summarizer.summarize(dialogue_state)
        thought = initial_thought
        rounds = []

        for _ in range(self.refinement_rounds):
            goals = self._generate_goals(
                state_summary,
                incoming_message,
                thought,
                clinical_objective,
            )
            assessments = self._embed_and_score(state_summary, thought, goals)
            refined_thought = self._refine_thought(
                state_summary,
                incoming_message,
                thought,
                assessments,
                clinical_objective,
                must_diagnose,
            )
            rounds.append(
                RefinementRound(
                    input_thought=thought,
                    assessments=assessments,
                    output_thought=refined_thought,
                )
            )
            thought = refined_thought

        action, diagnosis_retry_used = self._generate_action(
            state_summary,
            incoming_message,
            thought,
            doctor_system_prompt,
            clinical_objective,
            must_diagnose,
        )
        return PlanningResult(
            state_summary=state_summary,
            clinical_objective=clinical_objective,
            must_diagnose=must_diagnose,
            diagnosis_retry_used=diagnosis_retry_used,
            initial_thought=initial_thought,
            initial_action=initial_action,
            refined_thought=thought,
            action=action,
            rounds=rounds,
        )
