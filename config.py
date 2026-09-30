"""Central configuration for the ALPR and traffic violation analytics pipeline.

Every threshold, tolerance, vocabulary and policy parameter used anywhere in the
pipeline lives in this module so that enforcement policy can be reviewed and
changed in one place without touching processing logic.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "alpr_traffic_raw.xlsx"
RAW_SHEET_NAME = "Dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
DASHBOARD_DIR = OUTPUT_DIR / "dashboard"

RANDOM_STATE = 42

# Column groups
ID_COLUMNS = {
    "record_id": "TRF",
    "intersection_id": "INT",
    "sensor_camera_id": "CAM",
    "camera_id": "CAM",
    "capture_id": "ALP",
    "plate_number_raw": "PLT",
    "reviewer_id": "REV",
    "toll_gantry_id": "TG",
}
ID_PAD_WIDTH = 7
PRIMARY_KEY = "capture_id"

BOOLEAN_COLUMNS = [
    "violation_flag",
    "citation_issued_flag",
    "plate_match_database_flag",
    "stolen_vehicle_alert_flag",
    "amber_alert_match_flag",
    "false_positive_flag",
    "manual_review_flag",
    "appeal_filed_flag",
]
TRUE_TOKENS = {"yes", "y", "true", "1", "t"}
FALSE_TOKENS = {"no", "n", "false", "0", "f"}

DATE_COLUMNS = ["observation_date", "capture_timestamp"]

NUMERIC_COLUMNS = [
    "speed_limit_mph",
    "vehicle_count",
    "avg_speed_mph",
    "ocr_confidence_score",
    "speed_detected_mph",
    "posted_speed_limit_mph",
    "citation_amount_usd",
    "detection_confidence",
    "image_quality_score",
    "lane_number",
    "processing_latency_ms",
]

# Canonical vocabularies. Keys are produced by lower casing a value and removing
# every character that is not a letter or digit, so that casing, whitespace,
# underscores, trailing punctuation and separators all collapse to one key.
CANONICAL_VOCAB = {
    "corridor_name": [
        "5th St Corridor", "Downtown Loop", "Harbor Bridge Approach",
        "Highway 9 Interchange", "Industrial Pkwy", "Lakeshore Dr",
        "Main St & Broadway", "North Ring Rd", "Riverside Ave", "University Blvd",
    ],
    "day_of_week": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "district_zone": [
        "Zone 1 Downtown", "Zone 2 North", "Zone 3 South",
        "Zone 4 East Industrial", "Zone 5 West Residential",
    ],
    "jurisdiction": ["City DOT", "County DOT", "State DOT", "Transit Authority"],
    "road_type": ["Arterial", "Collector", "Freeway", "Highway Ramp", "Local"],
    "weather_condition": ["Clear", "Fog", "Overcast", "Rain", "Snow", "Wind"],
    "weather_at_capture": ["Clear", "Fog", "Overcast", "Rain", "Snow", "Wind"],
    "vehicle_color": ["Black", "Blue", "Gray", "Red", "Silver", "White"],
    "vehicle_type": ["Bus", "Motorcycle", "Sedan", "SUV", "Truck", "Van"],
    "vehicle_make": [
        "BMW", "Chevrolet", "Ford", "Honda", "Hyundai", "Jeep", "Kia",
        "Nissan", "Tesla", "Toyota",
    ],
    "model_name": ["EasyOCR", "Faster RCNN", "YOLOv5", "YOLOv8"],
    "lighting_condition": ["Dawn", "Day", "Dusk", "Night"],
    "violation_type": [
        "Expired Registration", "No Violation", "Red Light", "Speeding", "Toll Evasion",
    ],
    "appeal_outcome": ["Not Applicable", "Overturned", "Pending", "Upheld"],
}

# Models by pipeline role. The capture log records one model per capture: the
# detector for detector captures, the recogniser for OCR led captures.
MODEL_ROLE = {
    "YOLOv5": "Detector",
    "YOLOv8": "Detector",
    "Faster RCNN": "Detector",
    "EasyOCR": "Recogniser",
}

# Time band ordering (band start hour to label)
TIME_BAND_ORDER = [
    "00:00 to 04:00", "04:00 to 07:00", "07:00 to 09:00", "09:00 to 12:00",
    "12:00 to 15:00", "15:00 to 18:00", "18:00 to 21:00", "21:00 to 24:00",
]
PEAK_BANDS = {"07:00 to 09:00", "15:00 to 18:00"}

# Ambiguous slash dates (both parts twelve or lower) are resolved month first,
# matching the agency export convention shown in the data dictionary.
AMBIGUOUS_SLASH_DATE_ORDER = "month_first"

# Plausibility bounds for measured quantities. Values outside the bounds are
# treated as sensor faults: preserved in a raw column, nulled in the analytic
# column, and flagged.
PLAUSIBILITY_BOUNDS = {
    "speed_detected_mph": (0.0, 120.0),
    "avg_speed_mph": (0.0, 100.0),
    "vehicle_count": (0.0, 1500.0),
    "processing_latency_ms": (0.0, 1000.0),
    "image_quality_score": (0.0, 100.0),
    "ocr_confidence_score": (0.0, 1.0),
    "detection_confidence": (0.0, 1.0),
    "citation_amount_usd": (0.0, 1000.0),
}
ROBUST_Z_THRESHOLD = 6.0

# Violation rules layer
SPEED_TOLERANCE_MPH = 5.0          # enforcement tolerance above the posted limit
SPEED_TIERS = [                     # (label, minimum mph over the limit)
    ("Minor", 5.0),
    ("Major", 10.0),
    ("Severe", 20.0),
]
FINE_SCHEDULE_USD = {
    "Speeding Minor": 100.0,
    "Speeding Major": 200.0,
    "Speeding Severe": 400.0,
    "Red Light": 250.0,
    "Expired Registration": 75.0,
    "Toll Evasion": 50.0,
}
RED_LIGHT_AMBER_GRACE_S = 0.3       # used when signal phase data is supplied

# Evidence gates that a capture must pass before any automated issuance
EVIDENCE_GATES = {
    "ocr_confidence_min": 0.80,
    "detection_confidence_min": 0.70,
    "image_quality_min": 40.0,
}

# Decision labels
DECISION_PRIORITY_ALERT = "Priority Alert"
DECISION_AUTO_ISSUE = "Auto Issue"
DECISION_MANUAL_REVIEW = "Manual Review"
DECISION_REJECT = "Reject"
DECISION_NO_ACTION = "No Action"

# Benchmarking
LATENCY_SLA_MS = 200.0
BOOTSTRAP_ITERATIONS = 2000

# Privacy and retention
PSEUDONYM_KEY_ENV = "ALPR_PSEUDONYM_KEY"
DEVELOPMENT_PSEUDONYM_KEY = "development_only_key_rotate_before_production"
RETENTION_DAYS_NO_HIT = 30          # captures with no violation and no alert
RETENTION_DAYS_REJECTED = 90        # rejected evidence kept for audit
RETENTION_DAYS_CASE = 1095          # evidence supporting a citation or alert
PLATE_VISIBLE_CHARS = 2

# Review effort used to size the manual review queue
REVIEW_MINUTES_PER_CASE = 3.0
REVIEWER_HOURS_PER_SHIFT = 7.5

# Threshold sweep grid for evidence gate tuning
OCR_THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DETECTION_THRESHOLD_GRID = [0.50, 0.60, 0.70, 0.80, 0.90]
