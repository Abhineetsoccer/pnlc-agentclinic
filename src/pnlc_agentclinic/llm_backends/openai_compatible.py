from openai import OpenAI


class OpenAICompatibleBackend:

    def __init__(self, base_url, api_key, model_name, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 200)
        self.temperature = kwargs.get("temperature", 0.7)
        self.seed = kwargs.get("seed")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _seed_for_offset(self, seed_offset=0):
        if self.seed is None:
            return None
        return int(self.seed) + int(seed_offset)

    def generate(self, prompt, system_prompt="", seed_offset=0):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        request = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        request_seed = self._seed_for_offset(seed_offset)
        if request_seed is not None:
            request["seed"] = request_seed

        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content
