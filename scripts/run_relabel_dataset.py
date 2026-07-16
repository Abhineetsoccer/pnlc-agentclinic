import json
import time
from pathlib import Path

import numpy as np

from pnlc_agentclinic.value_learning.her_relabel import relabel_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

GOALS_PER_TURN = 4
SEED = 0


def main():
    input_path = sorted(LOGS_DIR.glob("stage1_embedded_*.jsonl"))[-1]
    print(f"Loading embedded turns from: {input_path}")

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"{len(records)} embedded turns loaded")

    dataset = relabel_dataset(records, goals_per_turn=GOALS_PER_TURN, seed=SEED)

    num_tuples = len(dataset["reward"])
    reward_fraction = dataset["reward"].mean() if num_tuples else 0.0
    print(f"\n{num_tuples} (s, thought, s', g, r) tuples produced")
    print(f"Reward fraction (goal == current state): {reward_fraction:.3f}")

    output_path = LOGS_DIR / f"stage1_relabeled_{int(time.time())}.npz"
    np.savez(output_path, **dataset)
    print(f"Saved relabeled dataset to {output_path}")


if __name__ == "__main__":
    main()
