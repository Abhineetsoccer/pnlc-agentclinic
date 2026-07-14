from openai import OpenAI


class OpenAICompatibleEmbedder:
    def __init__(self, base_url, api_key, model_name, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, text):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]