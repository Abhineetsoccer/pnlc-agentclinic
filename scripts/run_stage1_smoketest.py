import os

import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.env.agentclinic_adapter import (
    install_patch,
    register_backend,
    AGENTCLINIC_PATH,
)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_name = cfg.model_backends.name
    api_key = os.environ.get(cfg.model_backends.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {cfg.model_backends.api_key_env} in your environment before running this.")

    register_backend(
        model_name,
        base_url=cfg.model_backends.base_url,
        api_key=api_key,
        model_name=cfg.model_backends.model_name,
    )

    agentclinic = install_patch()
    os.chdir(AGENTCLINIC_PATH)

    agentclinic.main(
        api_key=None,
        replicate_api_key=None,
        inf_type="llm",
        doctor_bias="None",
        patient_bias="None",
        doctor_llm=model_name,
        patient_llm=model_name,
        measurement_llm=model_name,
        moderator_llm=model_name,
        num_scenarios=2,
        dataset="MedQA",
        img_request=False,
        total_inferences=20,
        anthropic_api_key=None,
    )


if __name__ == "__main__":
    main()