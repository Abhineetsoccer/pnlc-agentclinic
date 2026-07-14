from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    """Local embedding backend: downloads/loads weights via `sentence-transformers`
    (respects HF_HOME for the cache location) and embeds in-process -- no base_url
    or api_key involved.
    """

    def __init__(self, model_name, device="cpu", **kwargs):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, text):
        return self.model.encode(text).tolist()

    def embed_batch(self, texts):
        return self.model.encode(texts).tolist()
