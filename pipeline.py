"""End to end orchestration: processing, rules, analytics, privacy and outputs."""

import logging
import time
from operator import sub

from . import benchmarking, cleaning, dashboard, enforcement_analysis, figures
from . import config as cfg
from . import modeling, perception_analysis, privacy, reporting, rules_engine

LOGGER = logging.getLogger("alpr_analytics")


def run(raw_path=None, run_model=True):
    """Execute the full pipeline and return a results bundle."""
    started = time.time()
    for directory in (cfg.PROCESSED_DIR, cfg.FIGURE_DIR, cfg.REPORT_DIR, cfg.DASHBOARD_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # Stage 1: processing
    LOGGER.info("Loading raw export")
    raw = cleaning.load_raw(raw_path)
    raw_profile = cleaning.profile_raw(raw)
    clean_df, log = cleaning.clean(raw)
    LOGGER.info("Processing complete: %s analysis ready captures", f"{len(clean_df):,}")

    # Stage 2: rules layer
    decided = rules_engine.apply_rules(clean_df)
    decision_summary = rules_engine.decision_summary(decided)

    # Stage 3: privacy controls applied before anything is exported
    safe = privacy.apply_privacy(decided)
    retention = privacy.retention_summary(safe)

    # Stage 4: analytics
    LOGGER.info("Running perception, enforcement and benchmark analytics")
    scorecard = perception_analysis.model_scorecard(safe)
    conditions = perception_analysis.condition_effects(safe)
    ocr_matrix = perception_analysis.ocr_condition_matrix(safe)
    separability = perception_analysis.false_positive_separability(safe)
    sweep = perception_analysis.threshold_sweep(safe)
    perception_tests = perception_analysis.perception_tests(safe)

    corridors = enforcement_analysis.corridor_summary(safe)
    hotspot = enforcement_analysis.hotspot_matrix(safe)
    segment_tables, segment_tests = enforcement_analysis.segment_rate_tables(safe)
    trend = enforcement_analysis.monthly_trend(safe)
    sensitivity = enforcement_analysis.limit_source_sensitivity(safe)
    integrity = enforcement_analysis.label_integrity(safe)
    appeals = enforcement_analysis.citation_and_appeals(safe)
    workload = enforcement_analysis.workload_and_value(safe)

    benchmark = benchmarking.benchmark_scorecard(safe)
    latency_drivers = benchmarking.latency_drivers(safe)
    overall = benchmarking.overall_benchmark(safe)

    if run_model:
        LOGGER.info("Evaluating false positive triage models")
        model_results, model_summary = modeling.evaluate(safe)
    else:
        model_results, model_summary = None, {}

    reconciliation = (
        safe["reconciliation"].value_counts().rename_axis("reconciliation").reset_index(name="captures")
    )
    reconciliation["share_pct"] = (reconciliation["captures"] / len(safe) * 100).round(2)

    # Stage 5: outputs
    LOGGER.info("Writing processed data, reports, figures and dashboard")
    reporting.write_csv(safe, cfg.PROCESSED_DIR / "alpr_captures_processed.csv")
    reporting.write_csv(log.to_frame(), cfg.PROCESSED_DIR / "data_quality_log.csv")

    tables = {
        "decision_summary": decision_summary.reset_index(),
        "reconciliation": reconciliation,
        "label_integrity": integrity,
        "model_scorecard": scorecard,
        "condition_effects": conditions,
        "ocr_condition_matrix": ocr_matrix.reset_index(),
        "false_positive_separability": separability,
        "threshold_sweep": sweep,
        "benchmark_scorecard": benchmark,
        "latency_drivers": latency_drivers,
        "corridor_summary": corridors,
        "speeding_hotspot_matrix": hotspot.reset_index(),
        "monthly_trend": trend,
        "retention_summary": retention,
    }
    for factor, table in segment_tables.items():
        tables[f"violation_rate_by_{factor}"] = table
    if model_results is not None:
        tables["false_positive_model_comparison"] = model_results
    for name, table in tables.items():
        reporting.write_csv(table, cfg.REPORT_DIR / f"{name}.csv")

    metrics = {
        "overall_benchmark": overall,
        "workload_and_value": workload,
        "limit_source_sensitivity": sensitivity,
        "citations_and_appeals": appeals,
        "perception_tests": perception_tests,
        "segment_tests": segment_tests,
        "false_positive_model": model_summary,
        "records": {
            "raw_rows": int(len(raw)),
            "processed_captures": int(len(safe)),
            "captures_with_integrity_flags": int((safe["data_issue_count"] > 0).sum()),
        },
    }
    reporting.write_json(metrics, cfg.REPORT_DIR / "metrics.json")

    figure_paths = [
        figures.label_integrity_chart(integrity),
        figures.decision_mix_chart(decision_summary),
        figures.reconciliation_chart(safe),
        figures.corridor_value_chart(corridors),
        figures.hotspot_heatmap(hotspot),
        figures.latency_chart(safe),
        figures.ocr_condition_heatmap(ocr_matrix),
        figures.threshold_chart(sweep),
        figures.monthly_trend_chart(trend),
        figures.separability_chart(separability),
        figures.model_benchmark_chart(benchmark),
    ]

    dashboard.build_dashboard(
        decisions=safe,
        corridor=corridors,
        hotspot=hotspot,
        benchmark=benchmark[["model_name", "captures", "p50_ms", "p95_ms", "p99_ms", "sla_compliance_pct",
                             "false_positive_pct", "evidence_gate_pass_pct", "rules_label_agreement_pct",
                             "composite_rank"]],
        integrity=integrity,
        figure_paths=[figure_paths[i] for i in (1, 2, 4, 5, 7)],
    )

    results = {
        "clean": safe,
        "metrics": metrics,
        "decision_summary": decision_summary,
        "reconciliation": reconciliation,
        "label_integrity": integrity,
        "model_scorecard": scorecard,
        "separability": separability,
        "threshold_sweep": sweep,
        "benchmark": benchmark,
        "model_results": model_results,
        "corridor_summary": corridors,
        "retention_summary": retention,
    }
    reporting.data_quality_report(raw_profile, log, safe, cfg.REPORT_DIR / "data_quality_report.md")
    if model_results is not None:
        reporting.analytics_report(results, cfg.REPORT_DIR / "analytics_report.md")

    LOGGER.info("Pipeline finished in %.1f seconds", sub(time.time(), started))
    return results
