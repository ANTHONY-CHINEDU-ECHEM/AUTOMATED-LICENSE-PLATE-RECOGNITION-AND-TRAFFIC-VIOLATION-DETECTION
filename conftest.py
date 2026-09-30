"""Shared fixtures: path setup and a builder for small synthetic raw exports."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# chr(45) reproduces the separator character used by the raw export.
SEP = chr(45)

BASE_ROW = {
    "record_id": f"TRF{SEP}0000001",
    "intersection_id": f"INT{SEP}0001",
    "corridor_name": "Downtown Loop",
    "observation_date": "2024/03/15",
    "time_of_day": f"07:00{SEP}09:00",
    "day_of_week": "Friday",
    "district_zone": f"Zone 1 {SEP} Downtown",
    "jurisdiction": "City DOT",
    "road_type": "Arterial",
    "speed_limit_mph": "35",
    "weather_condition": "Clear",
    "sensor_camera_id": f"CAM{SEP}0001",
    "vehicle_count": "400",
    "avg_speed_mph": "30",
    "capture_id": f"ALP{SEP}0000001",
    "capture_timestamp": "2024/03/15",
    "camera_id": f"CAM{SEP}0001",
    "plate_number_raw": f"PLT{SEP}00001",
    "plate_state": "OH",
    "ocr_confidence_score": "0.95",
    "vehicle_make": "Ford",
    "vehicle_color": "Black",
    "vehicle_type": "Sedan",
    "speed_detected_mph": "50",
    "posted_speed_limit_mph": "35",
    "violation_type": "Speeding",
    "violation_flag": "Yes",
    "citation_issued_flag": "Y",
    "citation_amount_usd": "200",
    "plate_match_database_flag": "TRUE",
    "stolen_vehicle_alert_flag": "0",
    "amber_alert_match_flag": "no",
    "model_name": "YOLOv8",
    "detection_confidence": "0.90",
    "false_positive_flag": "N",
    "manual_review_flag": "false",
    "reviewer_id": None,
    "image_quality_score": "80",
    "weather_at_capture": "Clear",
    "lighting_condition": "Day",
    "lane_number": "2",
    "toll_gantry_id": None,
    "processing_latency_ms": "110",
    "appeal_filed_flag": "No",
    "appeal_outcome": "Not Applicable",
}


def make_raw(overrides):
    """Build a raw export frame; overrides is a list of dicts, one per row."""
    rows = []
    for i, change in enumerate(overrides, start=1):
        row = dict(BASE_ROW)
        number = change.pop("_n", i)
        row["record_id"] = f"TRF{SEP}{number:07d}"
        row["intersection_id"] = f"INT{SEP}{number:04d}"
        row["sensor_camera_id"] = f"CAM{SEP}{number:04d}"
        row["camera_id"] = f"CAM{SEP}{number:04d}"
        row["capture_id"] = f"ALP{SEP}{number:07d}"
        row["plate_number_raw"] = f"PLT{SEP}{number:05d}"
        row.update(change)
        rows.append(row)
    return pd.DataFrame(rows).astype(object)


@pytest.fixture
def raw_builder():
    return make_raw
