import random
from collections import defaultdict

import numpy as np


def group_into_trajectories(records):
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[r["scenario_index"]].append(r)
    for scenario_index in by_scenario:
        by_scenario[scenario_index].sort(key=lambda r: r["turn_index"])
    return by_scenario


def relabel_trajectory(turns, goals_per_turn, rng):
    samples = []
    T = len(turns)
    for i in range(T - 1):
        goal_indices = list(range(i, T))
        for _ in range(goals_per_turn):
            goal_idx = rng.choice(goal_indices)
            samples.append({
                "state": turns[i]["state_embedding"],
                "thought": turns[i]["thought_embedding"],
                "next_state": turns[i + 1]["state_embedding"],
                "goal": turns[goal_idx]["state_embedding"],
                "reward": 1.0 if goal_idx == i else 0.0,
                "scenario_index": turns[i]["scenario_index"],
                "turn_index": turns[i]["turn_index"],
                "goal_turn_index": turns[goal_idx]["turn_index"],
            })
    return samples


def relabel_dataset(records, goals_per_turn=4, seed=0):
    rng = random.Random(seed)
    by_scenario = group_into_trajectories(records)

    all_samples = []
    for scenario_index in sorted(by_scenario):
        all_samples.extend(
            relabel_trajectory(by_scenario[scenario_index], goals_per_turn, rng)
        )

    return {
        "state": np.array([s["state"] for s in all_samples], dtype=np.float32),
        "thought": np.array([s["thought"] for s in all_samples], dtype=np.float32),
        "next_state": np.array([s["next_state"] for s in all_samples], dtype=np.float32),
        "goal": np.array([s["goal"] for s in all_samples], dtype=np.float32),
        "reward": np.array([s["reward"] for s in all_samples], dtype=np.float32),
        "scenario_index": np.array([s["scenario_index"] for s in all_samples], dtype=np.int32),
        "turn_index": np.array([s["turn_index"] for s in all_samples], dtype=np.int32),
        "goal_turn_index": np.array([s["goal_turn_index"] for s in all_samples], dtype=np.int32),
    }
