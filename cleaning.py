"""Data processing layer.

Transforms the raw roadside capture export into an analysis ready, auditable
table. Every transformation is logged to a QualityLog so that the processing
can be reproduced, reviewed and defended, which matters when the downstream
output supports enforcement decisions.

Guiding principles
1. Never invent enforcement evidence. Measured quantities (speed, confidence,
   image quality) are never imputed; invalid readings are nulled and flagged.
2. Preserve provenance. Repairs are flagged per row so any record can be traced
   back to what the raw export contained.
3. Separate descriptive fields from evidential fields. Descriptive categoricals
   may take an explicit Unknown level; evidential fields may not.
"""

import re

import numpy as np
import pandas as pd

from . import config as cfg


class QualityLog:
    """Collects one entry per processing action for the data quality report."""

    def __init__(self):
        self.entries = []

    def add(self, stage, column, issue, rows_affected, action):
        self.entries.append(
            {
                "stage": stage,
                "column": column,
                "issue": issue,
                "rows_affected": int(rows_affected),
                "action": action,
            }
        )

    def to_frame(self):
        return pd.DataFrame(self.entries)


# Loading and profiling

def load_raw(path=None, sheet_name=None):
    """Read the raw export exactly as delivered, with every column as text."""
    path = path or cfg.RAW_DATA_PATH
    sheet_name = sheet_name or cfg.RAW_SHEET_NAME
    return pd.read_excel(path, sheet_name=sheet_name, dtype=str)


def profile_raw(raw):
    """Column level profile of the raw export before any processing."""
    rows = []
    for column in raw.columns:
        series = raw[column]
        rows.append(
            {
                "column": column,
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean()) * 100, 2),
                "distinct_raw_values": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


# Primitive normalisers

def canonical_key(value):
    """Collapse casing, whitespace, underscores and punctuation to one key."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = re.sub(r"[\W_]+", "", str(value).lower())
    return key or None


def normalise_categorical(series, vocabulary):
    """Map messy text variants onto a canonical vocabulary.

    Returns the canonical series and the count of values that were changed.
    Values that do not match the vocabulary are title cased and kept so that
    they surface in the quality report instead of being silently discarded.
    """
    lookup = {canonical_key(label): label for label in vocabulary}

    def _map(value):
        key = canonical_key(value)
        if key is None:
            return np.nan
        if key in lookup:
            return lookup[key]
        return str(value).strip().rstrip(".").replace("_", " ").title()

    cleaned = series.map(_map)
    changed = (series.notna() & (series.astype("string") != cleaned.astype("string"))).sum()
    return cleaned, int(changed)


def to_boolean(series):
    """Map mixed boolean tokens (Yes, Y, TRUE, 1, no, 0 and so on) to pandas boolean."""
    tokens = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[tokens.isin(cfg.TRUE_TOKENS)] = True
    result[tokens.isin(cfg.FALSE_TOKENS)] = False
    return result


def canonical_id(series, prefix, width=cfg.ID_PAD_WIDTH):
    """Standardise identifiers to PREFIX_0000000 with consistent zero padding.

    The raw export mixes padded and unpadded numbers (INT 0002 versus INT 10379)
    which breaks joins against master data. The numeric part is the join key.
    """
    digits = series.astype("string").str.extract(r"(\d+)")[0]
    numbers = pd.to_numeric(digits, errors="coerce")
    padded = numbers.map(lambda n: np.nan if pd.isna(n) else f"{prefix}_{int(n):0{width}d}")
    return padded, numbers


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_mixed_dates(series):
    """Parse the four date layouts present in the export.

    Layouts: year first with any separator, day with abbreviated month name,
    and two digit slash dates that may be day first or month first. Slash
    dates are resolved by magnitude where possible (a part above twelve must
    be the day). When both parts are twelve or lower the configured default
    order is applied and the row is flagged as ambiguous.

    Returns a DataFrame with the parsed date, the detected layout and the
    ambiguity flag.
    """
    text = series.astype("string").str.strip()
    year = pd.Series(np.nan, index=series.index)
    month = pd.Series(np.nan, index=series.index)
    day = pd.Series(np.nan, index=series.index)
    layout = pd.Series(pd.NA, index=series.index, dtype="string")
    ambiguous = pd.Series(False, index=series.index)

    iso = text.str.extract(r"^(\d{4})\D(\d{1,2})\D(\d{1,2})$")
    is_iso = iso[0].notna()
    year[is_iso] = iso.loc[is_iso, 0].astype(float)
    month[is_iso] = iso.loc[is_iso, 1].astype(float)
    day[is_iso] = iso.loc[is_iso, 2].astype(float)
    layout[is_iso] = "Year first"

    named = text.str.extract(r"^(\d{1,2})\W([^\W\d_]{3,9})\W(\d{4})$")
    is_named = named[0].notna()
    day[is_named] = named.loc[is_named, 0].astype(float)
    month[is_named] = named.loc[is_named, 1].str.lower().str[:3].map(MONTHS).astype(float)
    year[is_named] = named.loc[is_named, 2].astype(float)
    layout[is_named] = "Day with month name"

    slash = text.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
    is_slash = slash[0].notna()
    first = slash[0].astype(float)
    second = slash[1].astype(float)
    day_first = is_slash & (first > 12)
    month_first = is_slash & (second > 12)
    unresolved = is_slash & ~day_first & ~month_first

    day[day_first] = first[day_first]
    month[day_first] = second[day_first]
    layout[day_first] = "Slash day first"

    day[month_first] = second[month_first]
    month[month_first] = first[month_first]
    layout[month_first] = "Slash month first"

    if cfg.AMBIGUOUS_SLASH_DATE_ORDER == "month_first":
        day[unresolved] = second[unresolved]
        month[unresolved] = first[unresolved]
    else:
        day[unresolved] = first[unresolved]
        month[unresolved] = second[unresolved]
    layout[unresolved] = "Slash ambiguous"
    ambiguous[unresolved] = True
    year[is_slash] = slash.loc[is_slash, 2].astype(float)

    parsed = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": day}), errors="coerce"
    )

    # Window check: an ambiguous date whose default reading falls outside the
    # window established by unambiguous records, while the alternative reading
    # falls inside it, is resolved to the alternative reading.
    certain = parsed[~ambiguous & parsed.notna()]
    if unresolved.any() and len(certain):
        window_start, window_end = certain.min(), certain.max()
        alternative = pd.to_datetime(
            pd.DataFrame({"year": year, "month": day, "day": month}), errors="coerce"
        )
        outside = (parsed < window_start) | (parsed > window_end)
        alt_inside = (alternative >= window_start) & (alternative <= window_end)
        swap = unresolved & outside & alt_inside
        parsed[swap] = alternative[swap]
        layout[swap] = "Slash ambiguous resolved by window"
    return pd.DataFrame({"date": parsed, "layout": layout, "ambiguous": ambiguous})


def parse_time_band(series):
    """Normalise time bands such as ' 07:00 to 09:00.' into a canonical label."""
    parts = series.astype("string").str.extract(r"(\d{1,2}):(\d{2})\D+(\d{1,2}):(\d{2})")
    valid = parts[0].notna()
    label = pd.Series(np.nan, index=series.index, dtype="object")
    start_hour = pd.Series(np.nan, index=series.index)
    label[valid] = (
        parts.loc[valid, 0].astype(int).map("{:02d}".format) + ":" + parts.loc[valid, 1]
        + " to "
        + parts.loc[valid, 2].astype(int).map("{:02d}".format) + ":" + parts.loc[valid, 3]
    )
    start_hour[valid] = parts.loc[valid, 0].astype(float)
    return label, start_hour


def robust_z(series):
    """Median absolute deviation based z score (robust to the outliers it detects)."""
    values = series.astype(float)
    median = values.median()
    mad = values.sub(median).abs().median() * 1.4826
    if not mad or np.isnan(mad):
        return pd.Series(0.0, index=series.index)
    return values.sub(median).abs() / mad


# Main processing routine

def clean(raw, log=None):
    """Run the full processing sequence and return (clean_frame, quality_log)."""
    log = log or QualityLog()
    df = raw.copy()
    log.add("Load", "all", "Raw export loaded", len(df), "Every column read as text")

    # 1. Structural issues
    empty = df.isna().all(axis=1)
    df = df.loc[~empty].copy()
    log.add("Structure", "all", "Fully empty rows", empty.sum(), "Removed")

    duplicated = df.duplicated(keep="first")
    df = df.loc[~duplicated].copy()
    log.add("Structure", "all", "Exact duplicate rows", duplicated.sum(), "Removed, first kept")

    # 2. Identifiers
    numbers = {}
    for column, prefix in cfg.ID_COLUMNS.items():
        padded, nums = canonical_id(df[column], prefix)
        changed = (df[column].notna() & (df[column] != padded)).sum()
        df[column] = padded
        numbers[column] = nums
        log.add("Identifiers", column, "Inconsistent prefix separator and zero padding",
                changed, f"Standardised to {prefix}_ plus {cfg.ID_PAD_WIDTH} digits")

    capture_number = numbers["capture_id"]
    linked = ["intersection_id", "sensor_camera_id", "camera_id", "plate_number_raw"]
    link_ok = pd.concat([numbers[c].eq(capture_number) for c in linked], axis=1).all(axis=1)
    log.add("Identifiers", "capture_id", "Capture key linkage to intersection, camera and plate",
            (~link_ok).sum(), "Verified; capture_id adopted as primary key")

    record_mismatch = numbers["record_id"].ne(capture_number)
    df["record_id_repaired"] = record_mismatch.values
    repaired = capture_number.map(
        lambda n: f"{cfg.ID_COLUMNS['record_id']}_{int(n):0{cfg.ID_PAD_WIDTH}d}"
    )
    df.loc[record_mismatch, "record_id"] = repaired[record_mismatch]
    log.add("Identifiers", "record_id",
            "Record identifier collides with another capture and breaks the key linkage",
            record_mismatch.sum(), "Rebuilt from capture number and flagged")

    if df[cfg.PRIMARY_KEY].duplicated().any():
        raise ValueError("Primary key is not unique after processing")

    # 3. Categorical text
    for column, vocabulary in cfg.CANONICAL_VOCAB.items():
        df[column], changed = normalise_categorical(df[column], vocabulary)
        log.add("Categorical text", column, "Casing, whitespace, underscore and punctuation variants",
                changed, "Mapped to canonical vocabulary")

    df["plate_state"] = df["plate_state"].astype("string").str.strip().str.upper()

    raw_band = df["time_of_day"].astype("string")
    untidy_band = raw_band.str.strip().ne(raw_band) | raw_band.str.endswith(".")
    df["time_of_day"], df["band_start_hour"] = parse_time_band(df["time_of_day"])
    log.add("Categorical text", "time_of_day", "Padded and punctuated time bands",
            untidy_band.fillna(False).sum(), "Parsed to canonical band label and start hour")

    # 4. Booleans
    for column in cfg.BOOLEAN_COLUMNS:
        raw_tokens = df[column].astype("string").str.strip().str.lower()
        variants = raw_tokens.nunique()
        df[column] = to_boolean(df[column])
        log.add("Booleans", column, f"{variants} distinct truth tokens", df[column].notna().sum(),
                "Mapped to a single boolean type")

    # 5. Dates
    for column in cfg.DATE_COLUMNS:
        parsed = parse_mixed_dates(df[column])
        failed = df[column].notna() & parsed["date"].isna()
        df[f"{column}_layout"] = parsed["layout"]
        df[f"{column}_ambiguous"] = parsed["ambiguous"].values
        df[column] = parsed["date"]
        for layout_name, count in parsed["layout"].value_counts().items():
            log.add("Dates", column, f"Layout detected: {layout_name}", count, "Parsed")
        log.add("Dates", column, "Slash dates with both parts twelve or lower",
                parsed["ambiguous"].sum(),
                "Resolved against the observed date window, otherwise month first; flagged")
        log.add("Dates", column, "Unparseable date text", failed.sum(), "Set to null")
        log.add("Dates", column, "Missing date", df[column].isna().sum(),
                "Left null; excluded from time trend analysis only")

    derived_weekday = df["observation_date"].dt.day_name()
    weekday_conflict = df["day_of_week"].notna() & derived_weekday.notna() & df["day_of_week"].ne(derived_weekday)
    df["day_of_week_recorded"] = df["day_of_week"]
    df["day_of_week"] = derived_weekday.fillna(df["day_of_week"])
    df["weekday_conflict_flag"] = weekday_conflict.values
    log.add("Dates", "day_of_week", "Recorded weekday contradicts the observation date",
            weekday_conflict.sum(), "Replaced by weekday derived from date; original retained")

    df["observation_month"] = df["observation_date"].dt.to_period("M").dt.to_timestamp()
    df["observation_year"] = df["observation_date"].dt.year

    # 6. Numerics
    for column in cfg.NUMERIC_COLUMNS:
        before = df[column].notna()
        df[column] = pd.to_numeric(df[column], errors="coerce")
        coerced = (before & df[column].isna()).sum()
        if coerced:
            log.add("Numerics", column, "Non numeric text", coerced, "Set to null")

    for column, (low, high) in cfg.PLAUSIBILITY_BOUNDS.items():
        invalid = df[column].notna() & ((df[column] < low) | (df[column] > high))
        df[f"{column}_invalid_flag"] = invalid.values
        df.loc[invalid, column] = np.nan
        log.add("Outliers", column, f"Outside plausible range {low:g} to {high:g}",
                invalid.sum(), "Nulled (sensor fault) and flagged")
        extreme = robust_z(df[column]) > cfg.ROBUST_Z_THRESHOLD
        if extreme.any():
            df.loc[extreme, f"{column}_invalid_flag"] = True
            df.loc[extreme, column] = np.nan
            log.add("Outliers", column,
                    f"Robust z score above {cfg.ROBUST_Z_THRESHOLD:g} within plausible range",
                    extreme.sum(), "Nulled and flagged")

    df["lane_number"] = df["lane_number"].round().astype("Int64")

    # 7. Cross field consistency
    limit_conflict = (
        df["speed_limit_mph"].notna() & df["posted_speed_limit_mph"].notna()
        & df["speed_limit_mph"].ne(df["posted_speed_limit_mph"])
    )
    df["speed_limit_conflict_flag"] = limit_conflict.values
    df["legal_speed_limit_mph"] = df["posted_speed_limit_mph"].fillna(df["speed_limit_mph"])
    log.add("Consistency", "posted_speed_limit_mph",
            "Segment limit disagrees with the limit posted at the capture point",
            limit_conflict.sum(), "Capture point limit adopted as the legal reference")

    type_says = df["violation_type"].ne("No Violation") & df["violation_type"].notna()
    flag_conflict = df["violation_flag"].notna() & df["violation_type"].notna() & (
        df["violation_flag"].astype(bool) != type_says
    )
    df["violation_label_conflict_flag"] = flag_conflict.values
    log.add("Consistency", "violation_flag", "Violation flag contradicts violation type",
            flag_conflict.sum(), "Flagged; rules layer rederives the decision")

    cited_without_violation = df["citation_issued_flag"].fillna(False).astype(bool) & ~type_says
    df["citation_without_violation_flag"] = cited_without_violation.values
    log.add("Consistency", "citation_issued_flag", "Citation recorded against a No Violation capture",
            cited_without_violation.sum(), "Flagged for audit")

    cited = df["citation_issued_flag"].fillna(False).astype(bool)
    orphan_amount = df["citation_amount_usd"].notna() & ~cited
    df["orphan_citation_amount_flag"] = orphan_amount.values
    df["citation_amount_recorded_usd"] = df["citation_amount_usd"]
    df.loc[orphan_amount, "citation_amount_usd"] = np.nan
    log.add("Consistency", "citation_amount_usd", "Amount recorded where no citation was issued",
            orphan_amount.sum(), "Removed from citation amount; original retained")
    missing_amount = cited & df["citation_amount_usd"].isna()
    df["citation_amount_missing_flag"] = missing_amount.values
    log.add("Consistency", "citation_amount_usd", "Citation issued with no amount",
            missing_amount.sum(), "Flagged; not imputed")

    filed = df["appeal_filed_flag"].fillna(False).astype(bool)
    has_outcome = df["appeal_outcome"].notna() & df["appeal_outcome"].ne("Not Applicable")
    outcome_without_appeal = has_outcome & ~filed
    df["appeal_outcome_recorded"] = df["appeal_outcome"]
    df.loc[outcome_without_appeal, "appeal_outcome"] = "Not Applicable"
    df["appeal_outcome_orphan_flag"] = outcome_without_appeal.values
    log.add("Consistency", "appeal_outcome", "Appeal outcome recorded with no appeal filed",
            outcome_without_appeal.sum(), "Reset to Not Applicable; original retained")
    filed_no_outcome = filed & ~has_outcome
    df.loc[filed_no_outcome, "appeal_outcome"] = "Not Recorded"
    log.add("Consistency", "appeal_outcome", "Appeal filed with no outcome",
            filed_no_outcome.sum(), "Set to Not Recorded")
    appeal_without_citation = filed & ~cited
    df["appeal_without_citation_flag"] = appeal_without_citation.values
    log.add("Consistency", "appeal_filed_flag", "Appeal filed against an uncited capture",
            appeal_without_citation.sum(), "Flagged for audit")

    reviewed = df["reviewer_id"].notna()
    df["reviewer_assigned_flag"] = reviewed.values
    review_conflict = reviewed & ~df["manual_review_flag"].fillna(False).astype(bool)
    df["review_assignment_conflict_flag"] = review_conflict.values
    log.add("Consistency", "reviewer_id", "Reviewer assigned but record not flagged for review",
            review_conflict.sum(), "Flagged")

    df["toll_gantry_present"] = df["toll_gantry_id"].notna()
    toll_without_gantry = df["violation_type"].eq("Toll Evasion") & ~df["toll_gantry_present"]
    df["toll_evasion_without_gantry_flag"] = toll_without_gantry.values
    log.add("Consistency", "toll_gantry_id", "Toll evasion recorded where no toll gantry exists",
            toll_without_gantry.sum(), "Flagged; cannot be enforced as recorded")

    weather_diff = (
        df["weather_condition"].notna() & df["weather_at_capture"].notna()
        & df["weather_condition"].ne(df["weather_at_capture"])
    )
    df["weather_source_conflict_flag"] = weather_diff.values
    log.add("Consistency", "weather_at_capture", "Segment weather differs from capture weather",
            weather_diff.sum(), "Capture weather used for image analysis")

    capture_gap = df["capture_timestamp"].sub(df["observation_date"]).dt.days.abs()
    df["capture_observation_gap_days"] = capture_gap
    log.add("Consistency", "capture_timestamp", "Capture date more than 30 days from observation date",
            (capture_gap > 30).sum(), "Flagged; observation date used for trend analysis")

    # 8. Missing values in descriptive fields
    descriptive = [
        "corridor_name", "district_zone", "jurisdiction", "road_type", "weather_condition",
        "weather_at_capture", "vehicle_color", "vehicle_type", "vehicle_make", "model_name",
        "lighting_condition", "time_of_day", "plate_state",
    ]
    for column in descriptive:
        missing = df[column].isna()
        df[column] = df[column].astype("object").where(~missing, "Unknown")
        log.add("Missing values", column, "Missing descriptive value", missing.sum(),
                "Explicit Unknown level")

    for column in ["violation_type"]:
        missing = df[column].isna()
        df[column] = df[column].where(~missing, "Unrecorded")
        log.add("Missing values", column, "Missing violation type", missing.sum(),
                "Explicit Unrecorded level; routed to review")

    for column in ["ocr_confidence_score", "detection_confidence", "image_quality_score",
                   "speed_detected_mph", "processing_latency_ms"]:
        log.add("Missing values", column, "Missing or invalid evidential measurement",
                df[column].isna().sum(), "Left null; never imputed")

    df["model_role"] = df["model_name"].map(cfg.MODEL_ROLE).fillna("Unknown")
    df["peak_period_flag"] = df["time_of_day"].isin(cfg.PEAK_BANDS)
    issue_pattern = re.compile(r"(conflict|invalid|orphan|repaired|missing|without)_flag$")
    issue_flags = [c for c in df.columns if issue_pattern.search(c)]
    df["data_issue_count"] = df[issue_flags].astype(bool).sum(axis=1)

    df = df.reset_index(drop=True)
    log.add("Output", "all", "Analysis ready records", len(df), "Written to data/processed")
    return df, log
