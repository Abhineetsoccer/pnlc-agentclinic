import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceBackend:
    """Local text-generation backend: downloads/loads weights via `transformers`
    (respects HF_HOME for the cache location) and runs inference in-process --
    no base_url or api_key involved.
    """

    def __init__(
        self,
        model_name,
        device="cpu",
        max_tokens=200,
        temperature=0.7,
        seed=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = None if seed is None else int(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def _seed_for_offset(self, seed_offset=0):
        if self.seed is None:
            return None
        return int(self.seed) + int(seed_offset)

    def _generation_generator(self, seed_offset=0):
        generation_seed = self._seed_for_offset(seed_offset)
        if generation_seed is None or str(self.device).startswith("mps"):
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(generation_seed)
        return generator

    def generate(self, prompt, system_prompt="", seed_offset=0):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.tokenizer.chat_template:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        else:
            text = (system_prompt + "\n\n" if system_prompt else "") + prompt
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        generation_seed = self._seed_for_offset(seed_offset)
        generator = self._generation_generator(seed_offset)
        if generator is not None:
            generation_kwargs["generator"] = generator
        elif generation_seed is not None:
            # PyTorch does not currently expose an MPS Generator constructor.
            torch.manual_seed(generation_seed)
            torch.mps.manual_seed(generation_seed)

        output = self.model.generate(input_ids, **generation_kwargs)
        generated_tokens = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
