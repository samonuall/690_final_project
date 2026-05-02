import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
# TODO: normalize reward components since the numbers are kind of different magnitudes
# Will need to say what fitness value is maximum, and its more about comparing trends

def _load_logs(log_dir):
    """Return list of (label, log_dict) for every training_log.json found under log_dir."""
    paths = glob.glob(os.path.join(log_dir, "**", "training_log.json"), recursive=True)
    results = []
    for path in sorted(paths):
        label = os.path.relpath(os.path.dirname(path), log_dir)
        with open(path) as f:
            results.append((label, json.load(f)))
    return results


def plot_training(log_dir):
    """
    Reads all training_log.json files under log_dir and produces:
      1. fitness_over_training.png  — mean eval fitness per checkpoint, one line per run
      2. reward_over_training.png   — mean eval reward per checkpoint, one line per run
      3. fitness_vs_reward.png      — mean eval fitness vs mean eval reward per checkpoint
      4. ppo_losses.png             — policy/value/entropy loss and explained variance
      5. lava_train_vs_test.png     — bar chart (only if train + test subdirs both exist)

    Plots saved to {log_dir}/plots/.
    """
    logs = _load_logs(log_dir)
    if not logs:
        print(f"No training_log.json files found under {log_dir}")
        return

    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: Mean eval fitness over training steps
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for label, log in logs:
        steps = [e["step"] for e in log["entries"]]
        mean_fitness = [np.mean([ep["fitness"] for ep in e["episodes"]]) for e in log["entries"]]
        ax.plot(steps, mean_fitness, label=label)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode fitness (eval)")
    ax.set_title("Fitness over training")
    ax.legend()
    fig.savefig(os.path.join(plots_dir, "fitness_over_training.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Mean eval reward over training steps
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for label, log in logs:
        steps = [e["step"] for e in log["entries"]]
        mean_reward = [
            np.mean([ep["reward_components"].get("total", 0.0) for ep in e["episodes"]])
            for e in log["entries"]
        ]
        ax.plot(steps, mean_reward, label=label)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode reward (eval)")
    ax.set_title("Reward over training")
    ax.legend()
    fig.savefig(os.path.join(plots_dir, "reward_over_training.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 3: Eval fitness vs eval reward total over training steps (standardized)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for label, log in logs:
        steps = [e["step"] for e in log["entries"]]
        mean_fitness = [np.mean([ep["fitness"] for ep in e["episodes"]]) for e in log["entries"]]
        mean_reward = [
            np.mean([ep["reward_components"].get("total", 0.0) for ep in e["episodes"]])
            for e in log["entries"]
        ]
        fitness_std = np.std(mean_fitness)
        reward_std = np.std(mean_reward)
        fitness_series = (np.array(mean_fitness) - np.mean(mean_fitness)) / (fitness_std if fitness_std > 0 else 1.0)
        reward_series = (np.array(mean_reward) - np.mean(mean_reward)) / (reward_std if reward_std > 0 else 1.0)
        ax.plot(steps, fitness_series, label=f"{label} fitness (std)")
        ax.plot(steps, reward_series, linestyle="--", label=f"{label} reward total (std)")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Standardized mean episode value (eval)")
    ax.set_title("Fitness vs reward total over training (standardized)")
    ax.legend()
    fig.savefig(os.path.join(plots_dir, "fitness_vs_reward.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 4: PPO loss components over training steps
    # ------------------------------------------------------------------
    has_ppo_stats = any(
        e.get("ppo_stats")
        for _, log in logs
        for e in log["entries"]
    )
    if has_ppo_stats:
        stat_keys = ["policy_gradient_loss", "value_loss", "entropy_loss", "explained_variance"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        titles = {
            "policy_gradient_loss": "Policy Gradient Loss",
            "value_loss": "Value Loss",
            "entropy_loss": "Entropy Loss",
            "explained_variance": "Explained Variance",
        }
        for ax, key in zip(axes.flat, stat_keys):
            for label, log in logs:
                steps = [e["step"] for e in log["entries"] if e.get("ppo_stats", {}).get(key) is not None]
                vals = [e["ppo_stats"][key] for e in log["entries"] if e.get("ppo_stats", {}).get(key) is not None]
                ax.plot(steps, vals, label=label)
            ax.set_title(titles[key])
            ax.set_xlabel("Training steps")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "ppo_losses.png"), bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 5: Lava train vs test bar chart
    # Expects separate logs labelled 'train' and 'test' (or containing those strings)
    # ------------------------------------------------------------------
    train_logs = [(l, d) for l, d in logs if "train" in l]
    test_logs = [(l, d) for l, d in logs if "test" in l]

    if train_logs and test_logs:
        fig, ax = plt.subplots()
        labels, means, sems = [], [], []
        for label, log in train_logs + test_logs:
            all_fitness = [ep["fitness"] for e in log["entries"] for ep in e["episodes"]]
            labels.append(label)
            means.append(np.mean(all_fitness))
            sems.append(np.std(all_fitness) / max(np.sqrt(len(all_fitness)), 1))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Mean episode fitness")
        ax.set_title("Lava: train vs test performance")
        fig.savefig(os.path.join(plots_dir, "lava_train_vs_test.png"), bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved to {plots_dir}")
