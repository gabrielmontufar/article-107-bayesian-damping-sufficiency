"""Reproducible synthetic validation for MIIUM30-62.

The base model is a set of independent free-decay SDOF records with known
modal frequency and unknown damping. Each record has its own nuisance
amplitude and phase, while damping is shared across records. For each candidate
damping value, amplitude and phase are obtained by conditional least squares.
The posterior over damping is then computed by deterministic grid quadrature
with an explicit truncated-normal prior. Calling the records "independent
windows" avoids conflating raw time samples with independent information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def logsumexp(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return vmax + float(np.log(np.sum(np.exp(values - vmax))))


def wilson_interval(successes: int, trials: int, z_value: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    center = (proportion + z_value**2 / (2.0 * trials)) / denominator
    half = z_value * np.sqrt(
        proportion * (1.0 - proportion) / trials + z_value**2 / (4.0 * trials**2)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def quantile_from_grid(grid: np.ndarray, weights: np.ndarray, probability: float) -> float:
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return float(np.interp(probability, cdf, grid))


def simulate_windows(
    n_windows: int,
    window_cycles: int,
    f_n_hz: float,
    zeta_true: float,
    amplitude: float,
    phase_rad: float,
    sample_rate_hz: float,
    noise_sd: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(round(window_cycles * sample_rate_hz / f_n_hz))
    time_s = np.arange(n_samples, dtype=float) / sample_rate_hz
    omega_n = 2.0 * np.pi * f_n_hz
    omega_d = omega_n * np.sqrt(1.0 - zeta_true**2)
    clean = amplitude * np.exp(-zeta_true * omega_n * time_s) * np.sin(
        omega_d * time_s + phase_rad
    )
    observed = np.broadcast_to(clean, (n_windows, n_samples)).copy()
    observed += rng.normal(0.0, noise_sd, size=observed.shape)
    return time_s, observed


def posterior_zeta(
    time_s: np.ndarray,
    observed_windows: np.ndarray,
    f_n_hz: float,
    noise_sd: float,
    zeta_grid: np.ndarray,
    prior_mu: float,
    prior_sd: float,
) -> dict[str, float]:
    """Compute a conditional Gaussian likelihood over the damping grid."""
    omega_n = 2.0 * np.pi * f_n_hz
    omega_d = omega_n * np.sqrt(1.0 - zeta_grid**2)
    envelope = np.exp(-zeta_grid[:, None] * omega_n * time_s[None, :])
    x_sin = envelope * np.sin(omega_d[:, None] * time_s[None, :])
    x_cos = envelope * np.cos(omega_d[:, None] * time_s[None, :])

    xx00 = np.sum(x_sin * x_sin, axis=1)
    xx01 = np.sum(x_sin * x_cos, axis=1)
    xx11 = np.sum(x_cos * x_cos, axis=1)
    xy0 = (observed_windows @ x_sin.T).T
    xy1 = (observed_windows @ x_cos.T).T
    yy = np.sum(observed_windows * observed_windows, axis=1)

    determinant = xx00 * xx11 - xx01 * xx01
    determinant = np.maximum(determinant, np.finfo(float).tiny)
    beta_sin = (xy0 * xx11[:, None] - xy1 * xx01[:, None]) / determinant[:, None]
    beta_cos = (xy1 * xx00[:, None] - xy0 * xx01[:, None]) / determinant[:, None]
    sse = yy[None, :] - beta_sin * xy0 - beta_cos * xy1
    sse = np.maximum(sse, np.finfo(float).tiny)

    # Use the conditional/profile Gaussian likelihood for the posterior and
    # comparator. The nuisance amplitudes/phases are fitted conditionally;
    # this avoids silently mixing a marginal-likelihood penalty with an MLE.
    profile_log_likelihood = -0.5 * np.sum(sse, axis=1) / (noise_sd**2)
    log_likelihood = profile_log_likelihood
    log_prior = -0.5 * ((zeta_grid - prior_mu) / prior_sd) ** 2 - np.log(prior_sd)
    log_posterior = log_likelihood + log_prior
    normalizer = logsumexp(log_posterior)
    weights = np.exp(log_posterior - normalizer)
    weights /= np.sum(weights)

    mean = float(np.sum(weights * zeta_grid))
    variance = float(np.sum(weights * (zeta_grid - mean) ** 2))
    q025 = quantile_from_grid(zeta_grid, weights, 0.025)
    q975 = quantile_from_grid(zeta_grid, weights, 0.975)
    mle_index = int(np.argmax(profile_log_likelihood))
    mle = float(zeta_grid[mle_index])
    # A prior-free profile-likelihood interval is retained as a fair comparator.
    # The cutoff is the usual one-parameter 95% likelihood-ratio threshold.
    profile_cutoff = float(np.max(profile_log_likelihood) - 0.5 * 3.841459)
    profile_mask = profile_log_likelihood >= profile_cutoff
    profile_indices = np.flatnonzero(profile_mask)
    if len(profile_indices) == 0:
        profile_q025 = float(zeta_grid[mle_index])
        profile_q975 = float(zeta_grid[mle_index])
    else:
        profile_q025 = float(zeta_grid[profile_indices[0]])
        profile_q975 = float(zeta_grid[profile_indices[-1]])
    return {
        "zeta_mean": mean,
        "zeta_sd": float(np.sqrt(max(variance, 0.0))),
        "zeta_q025": q025,
        "zeta_q975": q975,
        "zeta_mle": mle,
        "profile_q025": profile_q025,
        "profile_q975": profile_q975,
        "weights": weights,
    }


def decision_metrics(posterior: dict[str, float], epsilon_r: float, z_value: float) -> dict[str, float | bool]:
    zeta_mean = max(float(posterior["zeta_mean"]), 1e-12)
    zeta_sd = float(posterior["zeta_sd"])
    sensitivity = 1.0  # R(zeta)=1/(2*zeta) at resonance.
    coefficient_of_variation = zeta_sd / zeta_mean
    lambda_zeta = z_value * sensitivity * coefficient_of_variation / epsilon_r

    response_at_mean = 1.0 / (2.0 * zeta_mean)
    response_q025 = 1.0 / (2.0 * max(float(posterior["zeta_q975"]), 1e-12))
    response_q975 = 1.0 / (2.0 * max(float(posterior["zeta_q025"]), 1e-12))
    nonlinear_response_half_width = (response_q975 - response_q025) / (2.0 * response_at_mean)
    credible_width = (float(posterior["zeta_q975"]) - float(posterior["zeta_q025"])) / (2.0 * zeta_mean)
    profile_q025 = float(posterior["profile_q025"])
    profile_q975 = float(posterior["profile_q975"])
    profile_response_q025 = 1.0 / (2.0 * max(profile_q975, 1e-12))
    profile_response_q975 = 1.0 / (2.0 * max(profile_q025, 1e-12))
    response_at_mle = 1.0 / (2.0 * max(float(posterior["zeta_mle"]), 1e-12))
    profile_half_width = (profile_response_q975 - profile_response_q025) / (2.0 * response_at_mle)
    lambda_profile = z_value * (profile_q975 - profile_q025) / (4.0 * zeta_mean * epsilon_r)

    return {
        "sensitivity": sensitivity,
        "cv_zeta": coefficient_of_variation,
        "lambda_zeta": lambda_zeta,
        "response_half_width_nonlinear": nonlinear_response_half_width,
        "zeta_credible_half_width": credible_width,
        "profile_response_half_width": profile_half_width,
        "lambda_profile": lambda_profile,
        "accept_lambda": bool(lambda_zeta <= 1.0),
        "accept_nonlinear": bool(nonlinear_response_half_width <= epsilon_r),
        "accept_profile": bool(profile_half_width <= epsilon_r),
    }


def truth_metrics(
    posterior: dict[str, float], zeta_true: float, epsilon_r: float,
    accepted: bool, profile_accepted: bool
) -> dict[str, float | bool]:
    response_true = 1.0 / (2.0 * zeta_true)
    response_estimated = 1.0 / (2.0 * max(float(posterior["zeta_mean"]), 1e-12))
    response_mle = 1.0 / (2.0 * max(float(posterior["zeta_mle"]), 1e-12))
    response_bias_mle = abs(response_mle - response_true) / response_true
    response_bias_relative = abs(response_estimated - response_true) / response_true
    return {
        "response_bias_relative": response_bias_relative,
        "response_bias_mle_relative": response_bias_mle,
        "credible_coverage": bool(
            float(posterior["zeta_q025"]) <= zeta_true <= float(posterior["zeta_q975"])
        ),
        "zeta_error_sq": (float(posterior["zeta_mean"]) - zeta_true) ** 2,
        "false_accept": bool(accepted and response_bias_relative > epsilon_r),
        "false_reject": bool((not accepted) and response_bias_relative <= epsilon_r),
        "profile_false_accept": bool(profile_accepted and response_bias_mle > epsilon_r),
        "profile_false_reject": bool((not profile_accepted) and response_bias_mle <= epsilon_r),
    }


def run(config: dict, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    zeta_grid = np.linspace(config["zeta_grid_min"], config["zeta_grid_max"], config["zeta_grid_points"])
    rows: list[dict] = []
    priors = config["priors"]
    for prior_index, prior in enumerate(priors):
        for window_index, n_windows in enumerate(config["n_windows"]):
            for replicate in range(config["replicates"]):
                seed = (
                    int(config["seed"])
                    + 1000 * window_index
                    + replicate
                )
                rng = np.random.default_rng(seed)
                time_s, observed_windows = simulate_windows(
                    n_windows=n_windows,
                    window_cycles=config["window_cycles"],
                    f_n_hz=config["f_n_hz"],
                    zeta_true=config["zeta_true"],
                    amplitude=config["amplitude"],
                    phase_rad=config["phase_rad"],
                    sample_rate_hz=config["sample_rate_hz"],
                    noise_sd=config["noise_sd"],
                    rng=rng,
                )
                posterior = posterior_zeta(
                    time_s=time_s,
                    observed_windows=observed_windows,
                    f_n_hz=config["f_n_hz"],
                    noise_sd=config["noise_sd"],
                    zeta_grid=zeta_grid,
                    prior_mu=prior["mu"],
                    prior_sd=prior["sd"],
                )
                metrics = decision_metrics(posterior, config["epsilon_r"], config["z_value"])
                truth = truth_metrics(
                    posterior,
                    config["zeta_true"],
                    config["epsilon_r"],
                    bool(metrics["accept_lambda"]),
                    bool(metrics["accept_profile"]),
                )
                rows.append(
                    {
                        "prior": prior["name"],
                        "n_windows": n_windows,
                        "replicate": replicate,
                        "seed": seed,
                        "window_cycles": config["window_cycles"],
                        "n_samples_per_window": len(time_s),
                        "n_samples_total": int(observed_windows.size),
                        "zeta_true": config["zeta_true"],
                        "f_n_hz": config["f_n_hz"],
                        "noise_sd": config["noise_sd"],
                        "prior_mu": prior["mu"],
                        "prior_sd": prior["sd"],
                        **{key: value for key, value in posterior.items() if key != "weights"},
                        **metrics,
                        **truth,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "synthetic_results.csv", index=False)
    aggregate = (
        result.groupby(["prior", "n_windows"], as_index=False)
        .agg(
            replicates=("replicate", "count"),
            zeta_mean_mean=("zeta_mean", "mean"),
            zeta_mean_sd=("zeta_mean", "std"),
            zeta_sd_mean=("zeta_sd", "mean"),
            lambda_mean=("lambda_zeta", "mean"),
            lambda_sd=("lambda_zeta", "std"),
            response_bias_mean=("response_bias_relative", "mean"),
            response_bias_mle_mean=("response_bias_mle_relative", "mean"),
            zeta_rmse=("zeta_error_sq", lambda values: float(np.sqrt(np.mean(values)))),
            coverage_rate=("credible_coverage", "mean"),
            accept_rate=("accept_lambda", "mean"),
            false_accept_rate=("false_accept", "mean"),
            false_reject_rate=("false_reject", "mean"),
            profile_accept_rate=("accept_profile", "mean"),
            profile_false_accept_rate=("profile_false_accept", "mean"),
            profile_false_reject_rate=("profile_false_reject", "mean"),
        )
    )
    for rate_column in ["accept_lambda", "false_accept", "false_reject", "credible_coverage", "accept_profile", "profile_false_accept", "profile_false_reject"]:
        low_values = []
        high_values = []
        for _, aggregate_row in aggregate.iterrows():
            subset = result[
                (result["prior"] == aggregate_row["prior"])
                & (result["n_windows"] == aggregate_row["n_windows"])
            ]
            low, high = wilson_interval(
                int(subset[rate_column].sum()), len(subset), config["z_value"]
            )
            low_values.append(low)
            high_values.append(high)
        aggregate[f"{rate_column}_ci_low"] = low_values
        aggregate[f"{rate_column}_ci_high"] = high_values
    aggregate.to_csv(output_dir / "synthetic_aggregate.csv", index=False)
    make_epsilon_sensitivity(result, output_dir, config)
    summary = {
        "model": "SDOF free decay; conditional amplitude/phase least squares; zeta grid quadrature",
        "config": config,
        "outputs": {
            "rows": int(len(result)),
            "result_csv": "synthetic_results.csv",
            "aggregate_csv": "synthetic_aggregate.csv",
        },
    }
    (output_dir / "synthetic_config_and_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def make_epsilon_sensitivity(result: pd.DataFrame, output_dir: Path, config: dict) -> None:
    """Re-evaluate the same posterior draws at 5, 10 and 20 percent tolerance."""
    rows: list[dict] = []
    for epsilon in [0.05, 0.10, 0.20]:
        working = result.copy()
        working["lambda_epsilon"] = config["z_value"] * working["sensitivity"] * working["cv_zeta"] / epsilon
        working["accept_epsilon"] = working["lambda_epsilon"] <= 1.0
        working["false_accept_epsilon"] = working["accept_epsilon"] & (working["response_bias_relative"] > epsilon)
        working["false_reject_epsilon"] = (~working["accept_epsilon"]) & (working["response_bias_relative"] <= epsilon)
        summary = (
            working.groupby(["prior", "n_windows"], as_index=False)
            .agg(
                epsilon_r=("lambda_epsilon", lambda _: epsilon),
                lambda_mean=("lambda_epsilon", "mean"),
                accept_rate=("accept_epsilon", "mean"),
                false_accept_rate=("false_accept_epsilon", "mean"),
                false_reject_rate=("false_reject_epsilon", "mean"),
            )
        )
        rows.extend(summary.to_dict("records"))
    pd.DataFrame(rows).to_csv(output_dir / "epsilon_sensitivity.csv", index=False)


def make_plot(result: pd.DataFrame, output_dir: Path) -> None:
    grouped = result.groupby(["prior", "n_windows"], as_index=False)["lambda_zeta"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    for prior, subset in grouped.groupby("prior"):
        ax.plot(subset["n_windows"], subset["lambda_zeta"], marker="o", label=prior)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Λζ = 1")
    ax.set_xscale("log")
    ax.set_xlabel("Registros independientes")
    ax.set_ylabel("Media de Λζ")
    ax.set_title("Suficiencia sintética: sensibilidad al prior y al tamaño del registro")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_lambda_summary.png", dpi=300)
    plt.close(fig)


def default_config() -> dict:
    return {
        "seed": 20260728,
        "replicates": 25,
        "n_windows": [20, 50, 100, 300],
        "window_cycles": 5,
        "f_n_hz": 1.0,
        "zeta_true": 0.03,
        "amplitude": 1.0,
        "phase_rad": 0.35,
        "sample_rate_hz": 50.0,
        "noise_sd": 0.50,
        "epsilon_r": 0.10,
        "z_value": 1.96,
        "zeta_grid_min": 0.002,
        "zeta_grid_max": 0.12,
        "zeta_grid_points": 240,
        "priors": [
            {"name": "debil", "mu": 0.03, "sd": 0.03},
            {"name": "informativo", "mu": 0.03, "sd": 0.01},
            {"name": "sesgado", "mu": 0.045, "sd": 0.01},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args()
    config = default_config()
    if args.replicates is not None:
        config["replicates"] = args.replicates
    result = run(config, args.output_dir)
    make_plot(result, args.output_dir)
    print(result.groupby(["prior", "n_windows"])[["zeta_mean", "zeta_sd", "lambda_zeta", "accept_lambda"]].mean())


if __name__ == "__main__":
    main()
