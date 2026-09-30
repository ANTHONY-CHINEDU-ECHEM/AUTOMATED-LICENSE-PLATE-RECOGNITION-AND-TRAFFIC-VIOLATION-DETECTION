import numpy as np
import pandas as pd

from alpr_analytics import cleaning
from conftest import SEP


def test_boolean_tokens_are_unified():
    series = pd.Series(["Yes", "y", "TRUE", "1", "no", "N", "false", "0", None, "maybe"])
    result = cleaning.to_boolean(series)
    assert result.iloc[:4].tolist() == [True] * 4
    assert result.iloc[4:8].tolist() == [False] * 4
    assert result.iloc[8:].isna().all()


def test_categorical_variants_collapse_to_vocabulary():
    raw = pd.Series([f"Zone 1 {SEP} Downtown", "zone 1 downtown.", f"Zone_1_{SEP}_Downtown", "  ZONE 1 DOWNTOWN "])
    cleaned, changed = cleaning.normalise_categorical(raw, ["Zone 1 Downtown"])
    assert cleaned.unique().tolist() == ["Zone 1 Downtown"]
    assert changed == 4


def test_identifier_padding_is_standardised():
    padded, numbers = cleaning.canonical_id(pd.Series([f"INT{SEP}0002", f"INT{SEP}10379"]), "INT")
    assert padded.tolist() == ["INT_0000002", "INT_0010379"]
    assert numbers.tolist() == [2, 10379]


def test_mixed_date_layouts_and_ambiguity():
    values = pd.Series([
        f"2024{SEP}03{SEP}15", "2024/03/15", f"15{SEP}Mar{SEP}2024", "15/03/2024", "03/15/2024", "03/04/2024",
    ])
    parsed = cleaning.parse_mixed_dates(values)
    expected = pd.Timestamp(2024, 3, 15)
    assert (parsed["date"].iloc[:5] == expected).all()
    assert parsed["ambiguous"].tolist() == [False] * 5 + [True]
    assert parsed["date"].iloc[5] == pd.Timestamp(2024, 3, 4)


def test_ambiguous_date_outside_window_takes_alternative_reading():
    values = pd.Series(["2022/06/01", "2022/12/31", "01/06/2022"])
    parsed = cleaning.parse_mixed_dates(values)
    assert parsed["date"].iloc[2] == pd.Timestamp(2022, 6, 1)
    assert parsed["layout"].iloc[2] == "Slash ambiguous resolved by window"


def test_time_band_parsing():
    label, start = cleaning.parse_time_band(pd.Series([f" 07:00{SEP}09:00 ", f"21:00{SEP}24:00."]))
    assert label.tolist() == ["07:00 to 09:00", "21:00 to 24:00"]
    assert start.tolist() == [7.0, 21.0]


def test_clean_end_to_end(raw_builder):
    raw = raw_builder([
        {},
        {},                                   # exact duplicate of row one after numbering
        {"_n": 3, "speed_detected_mph": "250"},
        {"_n": 4, "citation_issued_flag": "No", "citation_amount_usd": "150"},
    ])
    raw.iloc[1] = raw.iloc[0]
    raw.loc[3, "record_id"] = raw.loc[0, "record_id"]        # identifier collision
    empty = pd.DataFrame([{c: np.nan for c in raw.columns}])
    raw = pd.concat([raw, empty], ignore_index=True)

    clean, log = cleaning.clean(raw)
    assert len(clean) == 3
    assert clean["capture_id"].is_unique
    assert clean.loc[clean["capture_id"].eq("ALP_0000004"), "record_id"].item() == "TRF_0000004"
    assert clean["record_id_repaired"].sum() == 1
    speed_row = clean[clean["capture_id"].eq("ALP_0000003")].iloc[0]
    assert np.isnan(speed_row["speed_detected_mph"])
    assert bool(speed_row["speed_detected_mph_invalid_flag"])
    orphan = clean[clean["capture_id"].eq("ALP_0000004")].iloc[0]
    assert np.isnan(orphan["citation_amount_usd"])
    assert orphan["citation_amount_recorded_usd"] == 150
    assert clean["district_zone"].eq("Zone 1 Downtown").all()
    assert not log.to_frame().empty
