import os
from pnlc_agentclinic.env.agentclinic_adapter import install_patch, AGENTCLINIC_PATH

if not os.environ.get("QWEN_API_KEY"):
    raise RuntimeError("Set QWEN_API_KEY in your environment before running this.")

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
    num_scenarios=2,
    dataset="MedQA",
    img_request=False,
    total_inferences=20,
    anthropic_api_key=None,
)