"""Static figures for the README, reports and the review dashboard."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg

PRIMARY = "#1F4E79"
ACCENT = "#C55A11"
NEUTRAL = "#7F7F7F"
PALETTE = ["#1F4E79", "#2E75B6", "#9DC3E6", "#C55A11", "#F4B183", "#7F7F7F"]

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _save(fig, name):
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURE_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def label_integrity_chart(integrity):
    data = integrity.sort_values("records")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(data["issue"], data["records"], color=PRIMARY)
    for y, (count, share) in enumerate(zip(data["records"], data["share_pct"])):
        ax.text(count, y, f"  {count:,} ({share:.1f}%)", va="center", fontsize=8)
    ax.set_title("Data integrity issues found in the capture export")
    ax.set_xlabel("Records affected")
    ax.set_xlim(0, data["records"].max() * 1.3)
    return _save(fig, "01_data_integrity_issues")


def decision_mix_chart(summary):
    data = summary.reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(data["decision"], data["captures"], color=[ACCENT, PRIMARY, "#2E75B6", NEUTRAL, "#9DC3E6"])
    for bar, share in zip(bars, data["share_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{share:.1f}%", ha="center", va="bottom")
    ax.set_title("Rules layer decision mix across all captures")
    ax.set_ylabel("Captures")
    return _save(fig, "02_decision_mix")


def reconciliation_chart(df):
    counts = df["reconciliation"].value_counts()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [PRIMARY if label.startswith("Agree") else ACCENT for label in counts.index]
    ax.barh(counts.index, counts.values, color=colors)
    for y, value in enumerate(counts.values):
        ax.text(value, y, f"  {value:,} ({value / counts.sum() * 100:.1f}%)", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_title("Rules layer versus recorded violation label")
    ax.set_xlabel("Captures")
    return _save(fig, "03_label_reconciliation")


def corridor_value_chart(corridors):
    data = corridors.sort_values("fine_value_usd")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(data["corridor_name"], data["fine_value_usd"] / 1000, color=PRIMARY)
    ax.set_title("Fine value at stake by corridor (rules layer)")
    ax.set_xlabel("Fine value (USD thousands)")
    return _save(fig, "04_corridor_fine_value")


def hotspot_heatmap(matrix):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Speeding rate (%)")
    ax.set_title("Speeding hotspots by corridor and time band")
    return _save(fig, "05_speeding_hotspots")


def latency_chart(df):
    models = sorted(df["model_name"].unique())
    models = [m for m in models if m != "Unknown"]
    data = [df.loc[df["model_name"].eq(m), "processing_latency_ms"].dropna() for m in models]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=models, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#9DC3E6"}, medianprops={"color": PRIMARY})
    ax.axhline(cfg.LATENCY_SLA_MS, color=ACCENT, linestyle=":", linewidth=2, label=f"SLA {cfg.LATENCY_SLA_MS:.0f} ms")
    ax.set_ylim(bottom=0)
    ax.set_title("Processing latency by model")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    return _save(fig, "06_latency_by_model")


def ocr_condition_heatmap(matrix):
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Mean OCR confidence")
    ax.set_title("OCR confidence by lighting and weather")
    return _save(fig, "07_ocr_conditions")


def threshold_chart(sweep):
    data = sweep[sweep["detection_threshold"].eq(cfg.EVIDENCE_GATES["detection_confidence_min"])]
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(data["ocr_threshold"], data["auto_issue_pct"], marker="o", color=PRIMARY, label="Auto issue share of candidates")
    ax1.set_xlabel("OCR confidence threshold")
    ax1.set_ylabel("Auto issue share (%)", color=PRIMARY)
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(data["ocr_threshold"], data["review_hours_per_10k_captures"], marker="s", color=ACCENT,
             label="Review hours per 10k captures")
    ax2.set_ylabel("Review hours per 10k captures", color=ACCENT)
    ax2.set_ylim(bottom=0)
    ax2.grid(False)
    ax1.axvline(cfg.EVIDENCE_GATES["ocr_confidence_min"], color=NEUTRAL, linestyle=":")
    ax1.set_title("Automation versus review workload (detection threshold 0.70)")
    return _save(fig, "08_threshold_tradeoff")


def monthly_trend_chart(trend):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(trend))
    ax.plot(x, trend["captures"], color=NEUTRAL, label="Captures")
    ax.plot(x, trend["rules_violations"], color=PRIMARY, label="Rules layer violations")
    ax.plot(x, trend["recorded_violations"], color=ACCENT, label="Recorded violations")
    step = max(1, len(trend) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(trend["observation_month"].dt.strftime("%b %Y").iloc[::step], rotation=35, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_title("Monthly capture and violation volume")
    ax.legend()
    return _save(fig, "09_monthly_trend")


def separability_chart(separability):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = separability["signal"].str.replace("_", " ").str.replace("ocr", "OCR").str.capitalize()
    ax.bar(labels, separability["roc_auc"], color=PRIMARY)
    ax.axhline(0.5, color=ACCENT, linestyle=":", linewidth=2, label="No information (0.5)")
    ax.axhline(0.7, color=NEUTRAL, linestyle=":", linewidth=1.5, label="Usable triage (0.7)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC for false positives")
    ax.set_title("Do quality scores identify false positive detections")
    ax.legend()
    return _save(fig, "10_false_positive_separability")


def model_benchmark_chart(scorecard):
    data = scorecard.sort_values("model_name")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("p95_ms", "p95 latency (ms)"),
        ("false_positive_pct", "False positive rate (%)"),
        ("evidence_gate_pass_pct", "Evidence gate pass (%)"),
    ]
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(data["model_name"], data[column], color=PALETTE[: len(data)])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(bottom=0)
    fig.suptitle("Model benchmark", fontweight="bold")
    return _save(fig, "11_model_benchmark")
