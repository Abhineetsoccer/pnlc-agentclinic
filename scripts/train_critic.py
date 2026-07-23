import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pnlc_agentclinic.value_learning.iql_critic import (
    GoalConditionedIQLCritic,
    expectile_loss,
)


REQUIRED_FIELDS = ("state", "thought", "next_state", "goal", "reward")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a goal-conditioned IQL critic on a relabeled trajectory dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Relabeled .npz dataset. Defaults to the newest logs/stage1_relabeled_*.npz.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--expectile", type=float, default=0.8)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument(
        "--target-update-rate",
        type=float,
        default=0.005,
        help="Polyak update rate for the target value network.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
    )
    parser.add_argument(
        "--no-normalize-inputs",
        action="store_true",
        help="Disable feature standardization learned from the training split.",
    )
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path):
    with np.load(path) as data:
        missing = [field for field in REQUIRED_FIELDS if field not in data]
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {missing}")

        arrays = {
            field: np.asarray(data[field], dtype=np.float32)
            for field in REQUIRED_FIELDS
        }
        arrays["done"] = (
            np.asarray(data["done"], dtype=np.float32)
            if "done" in data
            else np.zeros_like(arrays["reward"], dtype=np.float32)
        )
        arrays["scenario_index"] = (
            np.asarray(data["scenario_index"])
            if "scenario_index" in data
            else np.arange(len(arrays["reward"]))
        )

    sample_count = len(arrays["reward"])
    if sample_count < 2:
        raise ValueError("At least two relabeled samples are required.")
    if arrays["state"].ndim != 2 or arrays["thought"].ndim != 2:
        raise ValueError("'state' and 'thought' must be two-dimensional arrays.")
    if arrays["next_state"].shape != arrays["state"].shape:
        raise ValueError("'next_state' must have the same shape as 'state'.")
    if arrays["goal"].shape != arrays["state"].shape:
        raise ValueError("'goal' must have the same shape as 'state'.")
    for name, array in arrays.items():
        if len(array) != sample_count:
            raise ValueError(
                f"Array '{name}' has {len(array)} rows; expected {sample_count}."
            )
    for name in REQUIRED_FIELDS:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Array '{name}' contains NaN or infinite values.")
    return arrays


def split_indices(scenario_index, validation_fraction, seed):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    scenarios = np.unique(scenario_index)
    if len(scenarios) >= 2:
        rng.shuffle(scenarios)
        validation_count = min(
            len(scenarios) - 1,
            max(1, round(len(scenarios) * validation_fraction)),
        )
        validation_scenarios = scenarios[:validation_count]
        validation_mask = np.isin(scenario_index, validation_scenarios)
        return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)

    indices = rng.permutation(len(scenario_index))
    validation_count = max(1, round(len(indices) * validation_fraction))
    return indices[validation_count:], indices[:validation_count]


def make_loader(arrays, indices, batch_size, shuffle):
    tensors = [
        torch.from_numpy(arrays[name][indices])
        for name in ("state", "thought", "next_state", "goal", "reward", "done")
    ]
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=min(batch_size, len(indices)),
        shuffle=shuffle,
    )


def compute_normalization(arrays, train_indices):
    state_values = np.concatenate(
        (
            arrays["state"][train_indices],
            arrays["next_state"][train_indices],
            arrays["goal"][train_indices],
        ),
        axis=0,
    )
    thought_values = arrays["thought"][train_indices]
    return (
        torch.from_numpy(state_values.mean(axis=0).astype(np.float32)),
        torch.from_numpy(state_values.std(axis=0).astype(np.float32)),
        torch.from_numpy(thought_values.mean(axis=0).astype(np.float32)),
        torch.from_numpy(thought_values.std(axis=0).astype(np.float32)),
    )


def batch_losses(model, batch, expectile, discount):
    state, thought, next_state, goal, reward, done = batch
    with torch.no_grad():
        q_target = reward + discount * (1.0 - done) * model.target_value(
            next_state, goal
        )

    q_prediction = model.q(state, thought, goal)
    q_loss = torch.nn.functional.mse_loss(q_prediction, q_target)

    value_prediction = model.value(state, goal)
    value_loss = expectile_loss(q_prediction.detach() - value_prediction, expectile)
    return q_loss, value_loss, q_prediction, q_target


@torch.no_grad()
def evaluate(model, loader, expectile, discount, device):
    model.eval()
    totals = {"q_loss": 0.0, "value_loss": 0.0, "q_mae": 0.0}
    sample_count = 0
    for batch in loader:
        batch = tuple(value.to(device) for value in batch)
        q_loss, value_loss, q_prediction, q_target = batch_losses(
            model, batch, expectile, discount
        )
        count = len(batch[0])
        totals["q_loss"] += q_loss.item() * count
        totals["value_loss"] += value_loss.item() * count
        totals["q_mae"] += (q_prediction - q_target).abs().mean().item() * count
        sample_count += count
    return {key: value / sample_count for key, value in totals.items()}


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not args.hidden_dims or any(size < 1 for size in args.hidden_dims):
        raise ValueError("--hidden-dims must contain positive integers.")
    if not 0.5 < args.expectile < 1.0:
        raise ValueError("--expectile must be greater than 0.5 and less than 1.")
    if not 0.0 <= args.discount <= 1.0:
        raise ValueError("--discount must be between 0 and 1.")
    if not 0.0 < args.target_update_rate <= 1.0:
        raise ValueError("--target-update-rate must be greater than 0 and at most 1.")

    repo_root = Path(__file__).resolve().parent.parent
    if args.input is None:
        candidates = sorted((repo_root / "logs").glob("stage1_relabeled_*.npz"))
        if not candidates:
            raise FileNotFoundError(
                "No relabeled dataset found. Pass its path with --input or run "
                "scripts/run_relabel_dataset.py first."
            )
        input_path = candidates[-1]
    else:
        input_path = args.input.expanduser().resolve()

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else repo_root / "models" / f"iql_critic_{int(time.time())}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    arrays = load_dataset(input_path)
    train_indices, validation_indices = split_indices(
        arrays["scenario_index"], args.validation_fraction, args.seed
    )
    train_loader = make_loader(
        arrays, train_indices, args.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        arrays, validation_indices, args.batch_size, shuffle=False
    )

    model = GoalConditionedIQLCritic(
        state_dim=arrays["state"].shape[1],
        thought_dim=arrays["thought"].shape[1],
        hidden_dims=tuple(args.hidden_dims),
        normalize_inputs=not args.no_normalize_inputs,
    )
    if model.normalize_inputs:
        model.set_normalization_stats(
            *compute_normalization(arrays, train_indices)
        )
    model.to(device)

    q_optimizer = torch.optim.Adam(
        model.q_network.parameters(), lr=args.learning_rate
    )
    value_optimizer = torch.optim.Adam(
        model.value_network.parameters(), lr=args.learning_rate
    )

    print(f"Dataset: {input_path}")
    print(
        f"Samples: {len(train_indices)} train / "
        f"{len(validation_indices)} validation"
    )
    print(
        f"Dimensions: state={model.state_dim}, thought={model.thought_dim}; "
        f"device={device}"
    )

    history = []
    best_validation_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = tuple(value.to(device) for value in batch)
            q_loss, value_loss, _, _ = batch_losses(
                model, batch, args.expectile, args.discount
            )

            q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            q_optimizer.step()

            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            value_optimizer.step()

            model.update_target(args.target_update_rate)

        train_metrics = evaluate(
            model, train_loader, args.expectile, args.discount, device
        )
        validation_metrics = evaluate(
            model, validation_loader, args.expectile, args.discount, device
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)

        validation_loss = (
            validation_metrics["q_loss"] + validation_metrics["value_loss"]
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:>3}/{args.epochs}: "
                f"train Q={train_metrics['q_loss']:.5f}, "
                f"V={train_metrics['value_loss']:.5f}; "
                f"validation Q={validation_metrics['q_loss']:.5f}, "
                f"V={validation_metrics['value_loss']:.5f}"
            )

    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "state_dim": model.state_dim,
            "thought_dim": model.thought_dim,
            "hidden_dims": list(model.hidden_dims),
            "normalize_inputs": model.normalize_inputs,
        },
        "training_config": {
            "expectile": args.expectile,
            "discount": args.discount,
            "target_update_rate": args.target_update_rate,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "source_dataset": str(input_path),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "reward_fraction": float(arrays["reward"].mean()),
        "best_validation_loss": best_validation_loss,
    }
    torch.save(checkpoint, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(history, indent=2))
    print(f"Saved critic checkpoint: {output_path}")
    print(f"Saved training metrics: {metrics_path}")


if __name__ == "__main__":
    main()
