from pnlc_agentclinic.llm_backends.factory import build_generation_backend


BACKGROUND = (
    "The following is some background information about the setting. A doctor is "
    "evaluating a patient in a clinic. The doctor does not know the patient's condition "
    "and must discover it by asking questions, performing examinations, and requesting "
    "medical tests. The doctor has a limited number of interactions before they must "
    "commit to a diagnosis. The patient knows their own symptoms and history but does "
    "not know their diagnosis, and only reveals information when asked."
)

FRAMING = (
    "The following is a conversation between a Doctor and a Patient. The Doctor is "
    "trying to determine the Patient's correct diagnosis."
)

INSTRUCTION = (
    "Please summarize the dialogue. Try to keep all the useful information, including "
    "the diagnostic strategies employed by the Doctor and how the Patient responded, "
    "along with any test results obtained."
)


class StateSummarizer:
    def __init__(self, backend, few_shot_examples=""):
        self.backend = backend
        self.few_shot_examples = few_shot_examples

    @classmethod
    def from_config(cls, model_backends_cfg, few_shot_examples=""):
        return cls(build_generation_backend(model_backends_cfg), few_shot_examples)

    def build_prompt(self, dialogue):
        parts = [BACKGROUND, FRAMING, dialogue, INSTRUCTION]
        if self.few_shot_examples:
            parts.append("Here are examples below:\n" + self.few_shot_examples)
        return "\n\n".join(parts)

    def summarize(self, state_text):
        if not state_text or not state_text.strip():
            return ""
        return self.backend.generate(self.build_prompt(state_text))

    def summarize_batch(self, state_texts):
        return [self.summarize(s) for s in state_texts]