import os
import time
from pnlc_agentclinic.env.agentclinic_adapter import (
    install_patch,
    AGENTCLINIC_PATH,
    save_results_log,
    get_results_log,
)

if not os.environ.get("QWEN_API_KEY"):
    raise RuntimeError("Set QWEN_API_KEY in your environment before running this.")

NUM_SCENARIOS = 10

REPO_ROOT = AGENTCLINIC_PATH.parent.parent  # external/AgentClinic -> external -> repo root
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
output_path = LOGS_DIR / f"stage1_baseline_{int(time.time())}.json"  # absolute, computed before chdir

agentclinic = install_patch()
os.chdir(AGENTCLINIC_PATH)  # required: ScenarioLoaderMedQA opens a bare relative path

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
save_results_log(str(output_path))  # absolute path, works regardless of current cwd

num_correct = sum(r["correct"] for r in results)
if results:
    print(f"\n{num_correct}/{len(results)} correct ({100 * num_correct / len(results):.1f}%)")
else:
    print("\nNo scenarios reached a diagnosis -- check the transcript above for what went wrong.")
print(f"Saved {len(results)} structured results to {output_path}")