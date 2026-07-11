from openai import OpenAI

class OpenAICompatibleBackend:

    def __init__(self, base_url, api_key, model_name, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", 200)
        self.temperature = kwargs.get("temperature", 0.7)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt, system_prompt=""):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content