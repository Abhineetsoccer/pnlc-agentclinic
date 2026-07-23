"""Run PNLC-assisted clinical consultations in AgentClinic."""

import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from pnlc_agentclinic.embedding.factory import build_embedder
from pnlc_agentclinic.env.agentclinic_adapter import (
    AGENTCLINIC_PATH,
    get_results_log,
    get_thought_action_compliance_rate,
    get_trajectory_log,
    install_patch,
    register_backend,
    register_doctor_planner,
    save_results_log,
    save_trajectory_log,
)
from pnlc_agentclinic.llm_backends.factory import build_generation_backend
from pnlc_agentclinic.planning import NaturalLanguageCriticPlanner
from pnlc_agentclinic.summarization.summarizer import StateSummarizer
from pnlc_agentclinic.value_learning.iql_critic import load_critic_checkpoint


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_name = cfg.model_backends.name
    generation_backend = build_generation_backend(cfg.model_backends)
    register_backend(model_name, generation_backend)

    checkpoint_value = os.path.expanduser(str(cfg.critic.checkpoint))
    checkpoint_path = Path(hydra.utils.to_absolute_path(checkpoint_value))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Critic checkpoint not found: {checkpoint_path}")

    critic_device = resolve_device(str(cfg.critic.device))
    critic = load_critic_checkpoint(str(checkpoint_path), device=critic_device)
    embedder = build_embedder(cfg.embedding)
    planner = NaturalLanguageCriticPlanner(
        generation_backend=generation_backend,
        summarizer=StateSummarizer(generation_backend),
        embedder=embedder,
        critic=critic,
        positive_goals=int(cfg.critic.positive_goals),
        negative_goals=int(cfg.critic.negative_goals),
        refinement_rounds=int(cfg.critic.refinement_rounds),
    )
    register_doctor_planner(planner)

    repo_root = AGENTCLINIC_PATH.parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    run_id = int(time.time())
    results_path = logs_dir / f"stage1_pnlc_results_{run_id}.json"
    trajectories_path = logs_dir / f"stage1_pnlc_trajectories_{run_id}.json"

    print(f"Loaded critic: {checkpoint_path}")
    print(f"Critic device: {critic_device}")
    print(
        "PNLC loop: "
        f"{cfg.critic.positive_goals} positive + "
        f"{cfg.critic.negative_goals} negative goals, "
        f"{cfg.critic.refinement_rounds} refinement round(s)"
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
        num_scenarios=int(cfg.critic.num_scenarios),
        dataset="MedQA",
        img_request=False,
        total_inferences=int(cfg.critic.total_inferences),
        anthropic_api_key=None,
    )

    results = get_results_log()
    trajectories = get_trajectory_log()
    save_results_log(str(results_path))
    save_trajectory_log(str(trajectories_path))

    num_correct = sum(result["correct"] for result in results)
    critic_turns = sum(turn.get("critic_used", False) for turn in trajectories)
    fallback_turns = sum(bool(turn.get("critic_error")) for turn in trajectories)
    if results:
        print(
            f"\n{num_correct}/{len(results)} correct "
            f"({100 * num_correct / len(results):.1f}%)"
        )
    else:
        print("\nNo scenarios reached a diagnosis.")
    print(
        f"Critic used on {critic_turns}/{len(trajectories)} turns; "
        f"{fallback_turns} planner fallbacks"
    )
    print(f"Saved structured results to {results_path}")
    print(f"Saved PNLC trajectories to {trajectories_path}")

    compliance = get_thought_action_compliance_rate()
    if compliance is not None:
        print(f"Initial THOUGHT/ACTION compliance: {100 * compliance:.1f}%")


if __name__ == "__main__":
    main()
