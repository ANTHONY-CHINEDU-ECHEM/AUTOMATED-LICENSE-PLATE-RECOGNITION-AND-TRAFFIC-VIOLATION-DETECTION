import numpy as np
import pandas as pd

from alpr_analytics import config as cfg, privacy


def test_pseudonym_is_deterministic_and_keyed():
    a = privacy.pseudonymise("PLT_0000001", key=b"alpha")
    assert a == privacy.pseudonymise(" plt_0000001 ", key=b"alpha")
    assert a != privacy.pseudonymise("PLT_0000001", key=b"beta")
    assert "0000001" not in a


def test_mask_shows_only_tail():
    assert privacy.mask_plate("ABC1234") == "*****34"
    assert privacy.mask_plate("7") == "7"


def test_pixelation_changes_only_the_box():
    image = (np.arange(60 * 80 * 3) % 251).reshape(60, 80, 3).astype(np.uint8)
    out = privacy.pixelate_regions(image, [(10, 10, 40, 30)], block=8)
    assert not np.array_equal(out[10:30, 10:40], image[10:30, 10:40])
    assert np.array_equal(out[40:, 50:], image[40:, 50:])


def test_retention_classes():
    frame = pd.DataFrame({
        "decision": [cfg.DECISION_NO_ACTION, cfg.DECISION_REJECT, cfg.DECISION_AUTO_ISSUE],
        "citation_issued_flag": pd.array([False, False, True], dtype="boolean"),
        "capture_timestamp": pd.to_datetime(["2024/01/01", "2024/01/01", "2024/01/01"]),
        "observation_date": pd.to_datetime(["2024/01/01"] * 3),
    })
    out = privacy.retention_schedule(frame, reference_date="2024/06/01")
    assert out["retention_class"].tolist() == ["No hit", "Rejected evidence", "Case evidence"]
    assert out["purge_due"].tolist() == [True, True, False]
