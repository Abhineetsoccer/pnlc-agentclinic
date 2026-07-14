import os

import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.env.agentclinic_adapter import (
    install_patch,
    register_backend,
    AGENTCLINIC_PATH,
)
from pnlc_agentclinic.llm_backends.factory import build_generation_backend


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_name = cfg.model_backends.name
    register_backend(model_name, build_generation_backend(cfg.model_backends))

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