import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.embedding.factory import build_embedder


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    embedder = build_embedder(cfg.embedding)

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
