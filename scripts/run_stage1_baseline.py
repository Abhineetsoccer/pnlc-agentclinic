import os
import time

import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.env.agentclinic_adapter import (
    install_patch,
    register_backend,
    AGENTCLINIC_PATH,
    save_results_log,
    save_trajectory_log,
    get_trajectory_log,
    get_thought_action_compliance_rate,
    finalize_results_log,
    reset_run_logs,
)
from pnlc_agentclinic.llm_backends.factory import build_generation_backend
from pnlc_agentclinic.reproducibility import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    num_scenarios = int(cfg.num_scenarios)
    if num_scenarios < 1:
        raise ValueError("num_scenarios must be at least 1.")
    seed = seed_everything(cfg.get("seed"))

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

    reset_run_logs()
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
        num_scenarios=num_scenarios,
        dataset="MedQA",
        img_request=False,
        total_inferences=20,
        anthropic_api_key=None,
    )

    results = finalize_results_log(expected_scenarios=num_scenarios)
    for result in results:
        result["run_seed"] = seed
    save_results_log(str(results_path), expected_scenarios=num_scenarios)

    trajectories = get_trajectory_log()
    for turn in trajectories:
        turn["run_seed"] = seed
    save_trajectory_log(str(trajectories_path))

    num_correct = sum(r["correct"] for r in results)
    print(f"\nRun seed: {seed}")
    print(f"\nModerator backend: {moderator_name}")
    num_diagnosed = sum(result["reached_diagnosis"] for result in results)
    print(
        f"\n{num_correct}/{len(results)} correct "
        f"({100 * num_correct / len(results):.1f}%); "
        f"{num_diagnosed}/{len(results)} reached a diagnosis"
    )
    print(f"Saved {len(results)} structured results to {results_path}")
    print(f"Saved {len(trajectories)} trajectory turns to {trajectories_path}")

    compliance = get_thought_action_compliance_rate()
    if compliance is not None:
        print(f"THOUGHT/ACTION format compliance: {100 * compliance:.1f}% of turns")


if __name__ == "__main__":
    main()
