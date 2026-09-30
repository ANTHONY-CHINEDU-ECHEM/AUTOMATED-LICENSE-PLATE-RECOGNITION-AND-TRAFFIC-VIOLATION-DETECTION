import pandas as pd

from alpr_analytics import cleaning, config as cfg, rules_engine


def _decide(raw_builder, rows):
    clean, _ = cleaning.clean(raw_builder(rows))
    return rules_engine.apply_rules(clean).set_index("capture_id")


def test_speed_tiers_and_tolerance(raw_builder):
    out = _decide(raw_builder, [
        {"speed_detected_mph": "39"},   # within tolerance
        {"speed_detected_mph": "43"},   # minor
        {"speed_detected_mph": "50"},   # major
        {"speed_detected_mph": "60"},   # severe
    ])
    assert out["speed_tier"].tolist() == ["None", "Minor", "Major", "Severe"]
    assert out["rules_fine_usd"].tolist() == [0, 100, 200, 400]


def test_decision_paths(raw_builder):
    out = _decide(raw_builder, [
        {},                                                  # auto issue
        {"ocr_confidence_score": "0.50"},                    # weak evidence, review
        {"false_positive_flag": "Yes"},                      # reject
        {"stolen_vehicle_alert_flag": "1", "speed_detected_mph": "30"},  # alert
        {"speed_detected_mph": "30", "violation_type": "No Violation"},  # nothing
        {"speed_detected_mph": "30", "violation_type": "Red Light"},     # red light, no phase data
    ])
    assert out["decision"].tolist() == [
        cfg.DECISION_AUTO_ISSUE, cfg.DECISION_MANUAL_REVIEW, cfg.DECISION_REJECT,
        cfg.DECISION_PRIORITY_ALERT, cfg.DECISION_NO_ACTION, cfg.DECISION_MANUAL_REVIEW,
    ]
    assert "Signal phase data unavailable" in out["decision_reasons"].iloc[5]


def test_red_light_with_signal_phase_can_auto_issue(raw_builder):
    clean, _ = cleaning.clean(raw_builder([
        {"speed_detected_mph": "30", "violation_type": "Red Light"},
        {"speed_detected_mph": "30", "violation_type": "Red Light"},
    ]))
    clean[rules_engine.SIGNAL_PHASE_COLUMN] = [1.5, 0.1]
    out = rules_engine.apply_rules(clean)
    assert out["decision"].tolist() == [cfg.DECISION_AUTO_ISSUE, cfg.DECISION_NO_ACTION]


def test_toll_evasion_requires_gantry(raw_builder):
    out = _decide(raw_builder, [
        {"speed_detected_mph": "30", "violation_type": "Toll Evasion", "toll_gantry_id": "TG_1"},
        {"speed_detected_mph": "30", "violation_type": "Toll Evasion"},
    ])
    assert out["rules_violation_type"].tolist() == ["Toll Evasion", "No Violation"]


def test_reconciliation_labels():
    recorded = pd.Series(["No Violation", "Speeding", "Speeding", "Red Light", "No Violation"])
    rules = pd.Series(["No Violation", "Speeding", "No Violation", "Speeding", "Speeding"])
    assert rules_engine.reconcile(recorded, rules).tolist() == [
        "Agree no violation", "Agree same violation", "Recorded but unsupported",
        "Different violation type", "Supported but not recorded",
    ]
