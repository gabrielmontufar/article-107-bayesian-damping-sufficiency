"""Exploratory damping re-estimation from the public Mendeley bridge records.

The records are operational/traffic acceleration histories, not free decays.
This script therefore reports half-power bandwidth estimates around resolved
spectral peaks, with a segment bootstrap. It is a pilot and is not treated as
direct validation of the Bayesian damping criterion unless peaks and crossings
are resolved reproducibly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_quickdaq(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path, skiprows=10)
    return raw[:, 0], raw[:, 1:]


def segment_psd(values: np.ndarray, sample_rate: float, nfft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    step = nfft // 2
    if len(values) < nfft:
        raise ValueError("record is shorter than the PSD window")
    starts = np.arange(0, len(values) - nfft + 1, step)
    window = np.hanning(nfft)
    scale = sample_rate * np.sum(window**2)
    spectra = []
    for start in starts:
        segment = values[start : start + nfft]
        segment = segment - np.mean(segment)
        spectrum = np.fft.rfft(segment * window)
        psd = (np.abs(spectrum) ** 2) / scale
        psd[1:-1] *= 2.0
        spectra.append(psd)
    return np.fft.rfftfreq(nfft, 1.0 / sample_rate), np.asarray(spectra)


def crossing_frequency(freq: np.ndarray, curve: np.ndarray, index: int, direction: int, level: float) -> float | None:
    j = index
    while 0 < j < len(curve) - 1:
        nxt = j + direction
        if curve[nxt] <= level:
            f0, f1 = freq[j], freq[nxt]
            y0, y1 = curve[j], curve[nxt]
            if y1 == y0:
                return float(f1)
            return float(f0 + (level - y0) * (f1 - f0) / (y1 - y0))
        j = nxt
    return None


def find_peaks(freq: np.ndarray, curve: np.ndarray, fmin: float, fmax: float, min_separation_hz: float = 0.25) -> list[int]:
    mask = (freq >= fmin) & (freq <= fmax)
    indices = np.flatnonzero(mask)
    candidates = [int(i) for i in indices[1:-1] if curve[i] > curve[i - 1] and curve[i] >= curve[i + 1]]
    candidates.sort(key=lambda i: float(curve[i]), reverse=True)
    selected: list[int] = []
    for i in candidates:
        if all(abs(freq[i] - freq[j]) >= min_separation_hz for j in selected):
            selected.append(i)
        if len(selected) >= 8:
            break
    return sorted(selected)


def half_power(curve: np.ndarray, freq: np.ndarray, peak_index: int) -> tuple[float, float, float] | None:
    peak = float(curve[peak_index])
    if not np.isfinite(peak) or peak <= 0:
        return None
    level = peak / 2.0
    left = crossing_frequency(freq, curve, peak_index, -1, level)
    right = crossing_frequency(freq, curve, peak_index, 1, level)
    if left is None or right is None or right <= left:
        return None
    fn = float(freq[peak_index])
    zeta = (right - left) / (2.0 * fn)
    return fn, zeta, right - left


def bootstrap_estimate(segments: np.ndarray, freq: np.ndarray, peak_index: int, rng: np.random.Generator, reps: int = 200) -> np.ndarray:
    estimates: list[float] = []
    for _ in range(reps):
        sample = segments[rng.integers(0, len(segments), len(segments))]
        curve = np.mean(sample, axis=0)
        estimate = half_power(curve, freq, peak_index)
        if estimate is not None:
            estimates.append(estimate[1])
    return np.asarray(estimates, dtype=float)


def run(input_dir: Path, output_dir: Path, bootstrap_reps: int = 200) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rng = np.random.default_rng(20260728)
    for path in sorted(input_dir.glob("test*.txt")):
        time_s, channels = read_quickdaq(path)
        sample_rate = 1.0 / float(np.median(np.diff(time_s)))
        for channel in range(channels.shape[1]):
            freq, segments = segment_psd(channels[:, channel], sample_rate)
            mean_psd = np.mean(segments, axis=0)
            peak_indices = find_peaks(freq, mean_psd, 0.5, min(20.0, sample_rate / 2.0 - 0.1))
            for peak_index in peak_indices:
                base = half_power(mean_psd, freq, peak_index)
                if base is None:
                    continue
                fn, zeta, bandwidth = base
                boot = bootstrap_estimate(segments, freq, peak_index, rng, bootstrap_reps)
                if len(boot) >= 20:
                    q025, q975 = np.quantile(boot, [0.025, 0.975])
                else:
                    q025 = q975 = np.nan
                rows.append({
                    "test": path.stem,
                    "channel": channel + 1,
                    "sample_rate_hz": sample_rate,
                    "n_samples": len(time_s),
                    "n_segments": len(segments),
                    "frequency_hz": fn,
                    "half_power_bandwidth_hz": bandwidth,
                    "zeta_half_power": zeta,
                    "zeta_bootstrap_mean": float(np.mean(boot)) if len(boot) else np.nan,
                    "zeta_bootstrap_sd": float(np.std(boot, ddof=1)) if len(boot) > 1 else np.nan,
                    "zeta_bootstrap_q025": q025,
                    "zeta_bootstrap_q975": q975,
                    "bootstrap_valid_reps": len(boot),
                })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "real_acceleration_half_power.csv", index=False)
    if len(result):
        result["frequency_cluster_hz"] = (result["frequency_hz"] / 0.25).round() * 0.25
        mode_summary = (
            result.groupby("frequency_cluster_hz", as_index=False)
            .agg(
                estimates=("zeta_half_power", "size"),
                tests=("test", "nunique"),
                channels=("channel", "nunique"),
                frequency_median_hz=("frequency_hz", "median"),
                zeta_median=("zeta_bootstrap_mean", "median"),
                zeta_q025_median=("zeta_bootstrap_q025", "median"),
                zeta_q975_median=("zeta_bootstrap_q975", "median"),
            )
            .query("tests >= 6")
            .sort_values("frequency_median_hz")
        )
        mode_summary.to_csv(output_dir / "real_acceleration_mode_summary.csv", index=False)
    summary = {
        "source": "Mendeley Data, Bridge vibration monitoring dataset, v2, DOI 10.17632/d3by55pjh7.2",
        "method": "Hann-windowed averaged periodograms (2048 samples, 50% overlap); half-power bandwidth; bootstrap resampling of PSD segments",
        "interpretation": "exploratory operational modal damping estimates; not direct free-decay validation",
        "rows": int(len(result)),
        "stable_frequency_rule": "frequency bins of 0.25 Hz with estimates present in at least 6 of 8 tests",
        "tests": sorted(result["test"].unique().tolist()) if len(result) else [],
    }
    (output_dir / "real_acceleration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if len(result):
        first = sorted(result["test"].unique())[0]
        path = input_dir / f"{first}.txt"
        _, channels = read_quickdaq(path)
        freq, segments = segment_psd(channels[:, 0], 200.0)
        curve = np.mean(segments, axis=0)
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
        ax.semilogy(freq, curve, color="#1f4e79", linewidth=0.8)
        ax.set_xlim(0.5, 20)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD (g²/Hz)")
        ax.set_title(f"Espectro operacional: {first}, canal 1")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "real_acceleration_psd_test1_channel1.png", dpi=300)
        plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    args = parser.parse_args()
    result = run(args.input_dir, args.output_dir, args.bootstrap_reps)
    print(result.groupby("test")["zeta_half_power"].count())
    print(result.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
