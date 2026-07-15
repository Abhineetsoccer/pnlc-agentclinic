import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import hydra
from omegaconf import DictConfig

from pnlc_agentclinic.summarization.summarizer import StateSummarizer
from pnlc_agentclinic.embedding.factory import build_embedder
from pnlc_agentclinic.data.schema import TrajectoryField


CONCURRENCY = 8


def process_turn(turn, summarizer, embedder):
    summary = summarizer.summarize(turn[TrajectoryField.STATE])
    state_embedding = embedder.embed(summary)
    thought_embedding = embedder.embed(turn[TrajectoryField.THOUGHT])
    return {
        TrajectoryField.SCENARIO_INDEX: turn[TrajectoryField.SCENARIO_INDEX],
        TrajectoryField.TURN_INDEX: turn[TrajectoryField.TURN_INDEX],
        "state_summary": summary,
        "state_embedding": state_embedding,
        "thought": turn[TrajectoryField.THOUGHT],
        "thought_embedding": thought_embedding,
    }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    repo_root = Path(hydra.utils.get_original_cwd())
    logs_dir = repo_root / "logs"

    input_path = sorted(logs_dir.glob("stage1_trajectories_*.json"))[-1]
    print(f"Loading trajectories from: {input_path}")

    with open(input_path) as f:
        turns = json.load(f)

    clean_turns = [t for t in turns if t[TrajectoryField.PARSED_OK]]
    print(f"{len(clean_turns)} clean turns (of {len(turns)} total)")

    summarizer = StateSummarizer.from_config(cfg.model_backends)
    embedder = build_embedder(cfg.embedding)

    output_path = logs_dir / f"stage1_embedded_{int(time.time())}.jsonl"
    print(f"Writing to: {output_path}")

    completed = 0
    with open(output_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = [ex.submit(process_turn, t, summarizer, embedder) for t in clean_turns]
            for fut in futures:
                try:
                    record = fut.result()
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    completed += 1
                    if completed % 25 == 0:
                        print(f"  {completed}/{len(clean_turns)} done")
                except Exception as e:
                    print(f"  FAILED a turn: {e}")

    print(f"\nDone. {completed}/{len(clean_turns)} turns embedded -> {output_path}")


if __name__ == "__main__":
    main()