"""Command line entry point.

Usage
    python run_pipeline.py                  run on data/raw/alpr_traffic_raw.xlsx
    python run_pipeline.py path/to/export   run on another export with the same schema
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alpr_analytics import pipeline  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    results = pipeline.run(raw_path=raw_path)
    metrics = results["metrics"]
    print()
    print("Captures processed      ", f"{metrics['records']['processed_captures']:,}")
    print("Priority alerts         ", f"{metrics['workload_and_value']['priority_alerts']:,}")
    print("Auto issue citations    ", f"{metrics['workload_and_value']['auto_issue_cases']:,}")
    print("Manual review cases     ", f"{metrics['workload_and_value']['manual_review_cases']:,}")
    print("Latency p95 (ms)        ", metrics["overall_benchmark"]["latency_p95_ms"])
    print("Outputs written to outputs/ and data/processed/")


if __name__ == "__main__":
    main()
