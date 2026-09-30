from alpr_analytics import vision


def test_pattern_correction_fixes_confusable_characters():
    assert vision.correct_by_pattern("A8C 1Z3O", "LLLDDDD") == "ABC1230"
    assert vision.correct_by_pattern("ABC", "LLLDDDD") is None


def test_frame_fusion_uses_confidence_weighted_votes():
    text, agreement = vision.fuse_frame_reads([("ABC123", 0.9), ("A8C123", 0.4), ("ABC123", 0.8)])
    assert text == "ABC123"
    assert 0.9 < agreement <= 1.0
