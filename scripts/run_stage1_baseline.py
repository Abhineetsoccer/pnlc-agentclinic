import os
import time

import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.env.agentclinic_adapter import (
    install_patch,
    register_backend,
    AGENTCLINIC_PATH,
    save_results_log,
    get_results_log,
    save_trajectory_log,
    get_trajectory_log,
    get_thought_action_compliance_rate,
)
from pnlc_agentclinic.llm_backends.factory import build_generation_backend

NUM_SCENARIOS = 30


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_name = cfg.model_backends.name
    register_backend(model_name, build_generation_backend(cfg.model_backends))
    moderator_name = model_name
    if cfg.moderator.separate:
        moderator_name = cfg.moderator.name
        register_backend(
            moderator_name,
            build_generation_backend(cfg.moderator),
        )

    REPO_ROOT = AGENTCLINIC_PATH.parent.parent
    LOGS_DIR = REPO_ROOT / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    run_id = int(time.time())
    results_path = LOGS_DIR / f"stage1_baseline_{run_id}.json"
    trajectories_path = LOGS_DIR / f"stage1_trajectories_{run_id}.json"

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
        moderator_llm=moderator_name,
        num_scenarios=NUM_SCENARIOS,
        dataset="MedQA",
        img_request=False,
        total_inferences=20,
        anthropic_api_key=None,
    )

    results = get_results_log()
    save_results_log(str(results_path))

    trajectories = get_trajectory_log()
    save_trajectory_log(str(trajectories_path))

    num_correct = sum(r["correct"] for r in results)
    print(f"\nModerator backend: {moderator_name}")
    if results:
        print(f"\n{num_correct}/{len(results)} correct ({100 * num_correct / len(results):.1f}%)")
    else:
        print("\nNo scenarios reached a diagnosis -- check the transcript above for what went wrong.")
    print(f"Saved {len(results)} structured results to {results_path}")
    print(f"Saved {len(trajectories)} trajectory turns to {trajectories_path}")

    compliance = get_thought_action_compliance_rate()
    if compliance is not None:
        print(f"THOUGHT/ACTION format compliance: {100 * compliance:.1f}% of turns")


if __name__ == "__main__":
    main()
