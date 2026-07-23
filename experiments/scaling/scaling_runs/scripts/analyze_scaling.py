#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.io_utils import atomic_write_json
from scaling.metrics import read_jsonl


RUN_MATRIX = [
    ("m13", "low", "m13_low_120m_seed42", 12_913_920, 120_000_000),
    ("m13", "medium", "m13_medium_230m_seed42", 12_913_920, 230_000_000),
    ("m13", "high", "m13_high_460m_seed42", 12_913_920, 460_000_000),
    ("m30", "low", "m30_low_50m_seed42", 29_990_784, 50_000_000),
    ("m30", "medium", "m30_medium_100m_seed42", 29_990_784, 100_000_000),
    ("m30", "high", "m30_high_200m_seed42", 29_990_784, 200_000_000),
    ("m60", "low", "m60_low_25m_seed42", 60_439_040, 25_000_000),
    ("m60", "medium", "m60_medium_50m_seed42", 60_439_040, 50_000_000),
    ("m60", "high", "m60_high_100m_seed42", 60_439_040, 100_000_000),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def final_general_validation_loss(run_dir: Path) -> float | None:
    records = read_jsonl(run_dir / "metrics.jsonl")
    final = [
        record
        for record in records
        if record.get("type") == "validation"
        and record.get("split") == "general_final"
    ]
    if final:
        return float(final[-1]["loss"])
    periodic = [
        record
        for record in records
        if record.get("type") == "validation"
        and str(record.get("split", "")).startswith("general")
    ]
    return float(periodic[-1]["loss"]) if periodic else None


def collect_results(runs_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for family, tier, run_name, parameters, target_tokens in RUN_MATRIX:
        run_dir = runs_dir / run_name
        if not run_dir.is_dir() or not (run_dir / "summary.json").exists():
            missing.append(run_name)
            continue
        summary = read_json(run_dir / "summary.json")
        loss = final_general_validation_loss(run_dir)
        if summary.get("status") != "completed" or loss is None:
            missing.append(run_name)
            continue
        rows.append(
            {
                "family": family,
                "tier": tier,
                "run_name": run_name,
                "parameters": parameters,
                "target_tokens": target_tokens,
                "tokens_seen": int(summary.get("tokens_seen", target_tokens)),
                "validation_loss": loss,
                "training_seconds": float(summary.get("training_seconds", 0.0)),
                "estimated_flops": 6 * parameters * target_tokens,
            }
        )
    return rows, missing


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def loss_model(theta: np.ndarray, n_millions: np.ndarray, d_millions: np.ndarray) -> np.ndarray:
    irreducible, log_a, alpha, log_b, beta = theta
    return (
        irreducible
        + np.exp(log_a) * np.power(n_millions, -alpha)
        + np.exp(log_b) * np.power(d_millions, -beta)
    )


def fit_law(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = np.array([row["parameters"] for row in rows], dtype=np.float64) / 1e6
    d = np.array([row["target_tokens"] for row in rows], dtype=np.float64) / 1e6
    observed = np.array([row["validation_loss"] for row in rows], dtype=np.float64)
    upper_e = max(1e-6, float(observed.min()) - 1e-6)
    initial = np.array([0.9 * upper_e, 0.0, 0.3, 0.0, 0.3])
    lower = np.array([0.0, -20.0, 0.01, -20.0, 0.01])
    upper = np.array([upper_e, 20.0, 2.0, 20.0, 2.0])
    result = least_squares(
        lambda theta: loss_model(theta, n, d) - observed,
        x0=initial,
        bounds=(lower, upper),
        max_nfev=100_000,
    )
    predicted = loss_model(result.x, n, d)
    residuals = observed - predicted
    denominator = float(np.square(observed - observed.mean()).sum())
    r_squared = 1.0 - float(np.square(residuals).sum()) / denominator if denominator else 0.0
    irreducible, log_a, alpha, log_b, beta = result.x
    a = float(np.exp(log_a))
    b = float(np.exp(log_b))

    target_n = 200.0
    try:
        compute_million_squared = (
            target_n ** (alpha + beta) * beta * b / (alpha * a)
        ) ** (1.0 / beta)
        predicted_tokens_millions = compute_million_squared / target_n
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        predicted_tokens_millions = math.nan

    return {
        "success": bool(result.success),
        "message": result.message,
        "irreducible_loss_E": float(irreducible),
        "A": a,
        "alpha": float(alpha),
        "B": b,
        "beta": float(beta),
        "r_squared": r_squared,
        "rmse": float(np.sqrt(np.square(residuals).mean())),
        "predicted_validation_loss": predicted.tolist(),
        "residuals": residuals.tolist(),
        "indicative_compute_optimal_tokens_for_200m": (
            None
            if not math.isfinite(predicted_tokens_millions)
            else float(predicted_tokens_millions * 1e6)
        ),
        "warning": (
            "The 200M estimate is an extrapolation from nine one-seed pilot runs. "
            "Treat it as a budget-selection signal, not a universal law."
        ),
    }


def make_plots(rows: list[dict[str, Any]], fit: dict[str, Any], output_dir: Path) -> None:
    colors = {"low": "#6aa7ff", "medium": "#31d7c4", "high": "#f2bd5a"}
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5))
    for tier in ("low", "medium", "high"):
        tier_rows = sorted(
            (row for row in rows if row["tier"] == tier),
            key=lambda row: row["parameters"],
        )
        axis.plot(
            [row["parameters"] / 1e6 for row in tier_rows],
            [row["validation_loss"] for row in tier_rows],
            marker="o",
            label=tier,
            color=colors[tier],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Parameters (millions)")
    axis.set_ylabel("Final held-out loss")
    axis.set_title("IsoFLOP validation loss")
    axis.grid(alpha=0.2)
    axis.legend(title="Compute tier")
    fig.tight_layout()
    fig.savefig(output_dir / "isoflop_validation_loss.png", dpi=180)
    plt.close(fig)

    observed = np.array([row["validation_loss"] for row in rows])
    predicted = np.array(fit["predicted_validation_loss"])
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(observed, predicted, color="#31d7c4")
    bounds = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
    axis.plot(bounds, bounds, linestyle="--", color="#8e99aa")
    for row, x, y in zip(rows, observed, predicted):
        axis.annotate(f"{row['family']}-{row['tier']}", (x, y), fontsize=8)
    axis.set_xlabel("Observed loss")
    axis.set_ylabel("Fitted loss")
    axis.set_title("Scaling-law fit")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "observed_vs_fitted_loss.png", dpi=180)
    plt.close(fig)


def write_report(
    rows: list[dict[str, Any]], fit: dict[str, Any], output_path: Path
) -> None:
    winners: dict[str, dict[str, Any]] = {}
    for tier in ("low", "medium", "high"):
        tier_rows = [row for row in rows if row["tier"] == tier]
        winners[tier] = min(tier_rows, key=lambda row: row["validation_loss"])

    estimate = fit["indicative_compute_optimal_tokens_for_200m"]
    estimate_text = "unstable" if estimate is None else f"{estimate / 1e9:.3f}B tokens"
    lines = [
        "# Scaling-law pilot report",
        "",
        f"Fit RMSE: **{fit['rmse']:.5f}**  ",
        f"Fit R²: **{fit['r_squared']:.5f}**  ",
        f"Indicative compute-optimal tokens for a 200M model: **{estimate_text}**",
        "",
        "## IsoFLOP winners",
        "",
        "| Tier | Best run | Parameters | Tokens | Validation loss |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for tier, row in winners.items():
        lines.append(
            f"| {tier} | {row['run_name']} | {row['parameters']:,} | "
            f"{row['target_tokens']:,} | {row['validation_loss']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fitted form",
            "",
            "`L(N,D) = E + A/N^alpha + B/D^beta`, where N and D are measured in millions.",
            "",
            f"- E = {fit['irreducible_loss_E']:.8f}",
            f"- A = {fit['A']:.8f}",
            f"- alpha = {fit['alpha']:.8f}",
            f"- B = {fit['B']:.8f}",
            f"- beta = {fit['beta']:.8f}",
            "",
            "## Interpretation limit",
            "",
            fit["warning"],
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the nine scaling-law runs")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a partial CSV even when some runs are incomplete",
    )
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows, missing = collect_results(runs_dir)
    write_csv(rows, output_dir / "scaling_results.csv")

    if missing:
        message = "Missing or incomplete runs: " + ", ".join(missing)
        if not args.allow_partial:
            raise RuntimeError(message)
        print(message)
        print(f"Wrote partial results for {len(rows)} run(s).")
        return

    fit = fit_law(rows)
    atomic_write_json(output_dir / "scaling_fit.json", fit)
    make_plots(rows, fit, output_dir)
    write_report(rows, fit, output_dir / "scaling_report.md")
    print(f"Scaling analysis written to {output_dir}")


if __name__ == "__main__":
    main()

