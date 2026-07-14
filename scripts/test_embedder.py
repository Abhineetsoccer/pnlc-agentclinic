import os

import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.embedding.embedder import OpenAICompatibleEmbedder


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    api_key = os.environ.get(cfg.model_backends.api_key_env)
    embedder = OpenAICompatibleEmbedder(
        base_url=cfg.model_backends.base_url,
        api_key=api_key,
        model_name=cfg.model_backends.model_name,
    )

    vec = embedder.embed("The patient reports a persistent cough and mild fever.")
    print(f"Single embedding: type={type(vec)}, length={len(vec)}")
    print(f"First 5 values: {vec[:5]}")

    batch = embedder.embed_batch([
        "Patient has chest pain.",
        "Doctor requests a chest X-ray.",
    ])
    print(f"\nBatch: {len(batch)} embeddings, each length {len(batch[0])}")


if __name__ == "__main__":
    main()
