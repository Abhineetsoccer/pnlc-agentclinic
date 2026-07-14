import os
from pnlc_agentclinic.summarize_embed.embedder import OpenAICompatibleEmbedder

embedder = OpenAICompatibleEmbedder(
    base_url="",
    api_key=os.environ["QWEN_API_KEY"],
    model_name="qwen3-embed",
)

vec = embedder.embed("The patient reports a persistent cough and mild fever.")
print(f"Single embedding: type={type(vec)}, length={len(vec)}")
print(f"First 5 values: {vec[:5]}")

batch = embedder.embed_batch([
    "Patient has chest pain.",
    "Doctor requests a chest X-ray.",
])
print(f"\nBatch: {len(batch)} embeddings, each length {len(batch[0])}")