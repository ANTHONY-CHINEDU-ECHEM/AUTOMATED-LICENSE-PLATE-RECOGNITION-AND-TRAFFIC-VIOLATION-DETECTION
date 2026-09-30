"""Repository style audit: no hyphen, en dash or em dash characters anywhere.

Covers source code, tests, documentation and every generated output (CSV,
JSON, Markdown and the Excel dashboard). The untouched raw export in data/raw
is excluded because it is preserved exactly as delivered.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {chr(45): "hyphen", chr(8211): "en dash", chr(8212): "em dash"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".cfg", ".toml", ".ini", ""}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "raw"}


def _text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith("."):
            yield path


def _violations(text):
    return sorted({name for char, name in FORBIDDEN.items() if char in text})


@pytest.mark.parametrize("path", list(_text_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_text_file_is_clean(path):
    found = _violations(path.read_text(encoding="utf8", errors="ignore"))
    assert not found, f"{path.relative_to(ROOT)} contains: {', '.join(found)}"


def test_file_and_folder_names_are_clean():
    offenders = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                 if not SKIP_PARTS.intersection(p.relative_to(ROOT).parts) and _violations(p.name)]
    assert not offenders, offenders


def test_dashboard_cells_are_clean():
    openpyxl = pytest.importorskip("openpyxl")
    path = ROOT / "outputs" / "dashboard" / "violation_review_dashboard.xlsx"
    if not path.exists():
        pytest.skip("Dashboard not generated yet; run run_pipeline.py first")
    workbook = openpyxl.load_workbook(path, read_only=True)
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows(values_only=True):
            for value in row:
                if isinstance(value, str):
                    assert not _violations(value), f"{sheet.title}: {value}"
