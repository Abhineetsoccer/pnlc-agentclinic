import os
import time
from pnlc_agentclinic.env.agentclinic_adapter import (
    install_patch,
    AGENTCLINIC_PATH,
    save_results_log,
    get_results_log,
    save_trajectory_log,
    get_trajectory_log,
)

if not os.environ.get("QWEN_API_KEY"):
    raise RuntimeError("Set QWEN_API_KEY in your environment before running this.")

NUM_SCENARIOS = 30

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
    doctor_llm="qwen2.5-72b",
    patient_llm="qwen2.5-72b",
    measurement_llm="qwen2.5-72b",
    moderator_llm="qwen2.5-72b",
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
if results:
    print(f"\n{num_correct}/{len(results)} correct ({100 * num_correct / len(results):.1f}%)")
else:
    print("\nNo scenarios reached a diagnosis -- check the transcript above for what went wrong.")
print(f"Saved {len(results)} structured results to {results_path}")
print(f"Saved {len(trajectories)} trajectory turns to {trajectories_path}")