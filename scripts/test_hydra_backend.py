import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.llm_backends.factory import build_generation_backend


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    backend = build_generation_backend(cfg.model_backends)
    response = backend.generate("Say hello in one short sentence.")
    print(response)


if __name__ == "__main__":
    main()
