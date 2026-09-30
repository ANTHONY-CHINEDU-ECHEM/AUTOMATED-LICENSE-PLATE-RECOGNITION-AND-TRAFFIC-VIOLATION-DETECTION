"""Privacy preserving handling of plate data and capture imagery.

Controls implemented
1. Keyed pseudonymisation. Plates are replaced by an HMAC SHA256 token. The
   same plate always maps to the same token (so repeat offending can still be
   analysed) but the token cannot be reversed without the secret key, and a
   dictionary attack over the plate space is not possible without that key.
2. Display masking. Review screens show only the last characters of a plate.
3. Image redaction. Plate and occupant regions in frames are pixelated before
   any image leaves the evidence store.
4. Retention. Every capture receives a retention class and a purge date; non
   hit captures are purged quickly, evidence is kept only while a case is live.
5. Data minimisation. Analytic exports drop the raw plate string entirely.
"""

import hashlib
import hmac
import os
from operator import sub

import numpy as np
import pandas as pd

from . import config as cfg


def _key():
    return os.environ.get(cfg.PSEUDONYM_KEY_ENV, cfg.DEVELOPMENT_PSEUDONYM_KEY).encode("utf8")


def pseudonymise(value, key=None):
    """Deterministic keyed token for a plate value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    key = key or _key()
    digest = hmac.new(key, str(value).strip().upper().encode("utf8"), hashlib.sha256).hexdigest()
    return "PX" + digest[:16].upper()


def mask_plate(value, visible=cfg.PLATE_VISIBLE_CHARS):
    """Show only the final characters of a plate, for example *******12."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value)
    hidden = sub(len(text), min(visible, len(text)))
    return "*" * hidden + text[hidden:]


def pixelate_regions(image, boxes, block=12):
    """Pixelate rectangular regions of an image array in place and return it.

    image  : numpy array of shape (height, width) or (height, width, channels)
    boxes  : iterable of (x1, y1, x2, y2) pixel boxes, for example plate and
             windscreen detections from the perception stage
    block  : pixel block size; larger blocks give stronger redaction
    """
    out = np.array(image, copy=True)
    height, width = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((int(max(0, x1)), int(min(width, x2))))
        y1, y2 = sorted((int(max(0, y1)), int(min(height, y2))))
        for top in range(y1, y2, block):
            for left in range(x1, x2, block):
                bottom = min(top + block, y2)
                right = min(left + block, x2)
                patch = out[top:bottom, left:right]
                if patch.size:
                    out[top:bottom, left:right] = patch.mean(axis=(0, 1)).astype(out.dtype)
    return out


def retention_schedule(df, reference_date=None):
    """Assign a retention class, purge date and purge status to every capture."""
    reference_date = pd.Timestamp(reference_date) if reference_date is not None else df["capture_timestamp"].max()
    is_case = df["decision"].isin([cfg.DECISION_AUTO_ISSUE, cfg.DECISION_MANUAL_REVIEW, cfg.DECISION_PRIORITY_ALERT]) | (
        df["citation_issued_flag"].fillna(False).astype(bool)
    )
    is_rejected = df["decision"].eq(cfg.DECISION_REJECT) & ~is_case

    retention_class = np.select([is_case, is_rejected], ["Case evidence", "Rejected evidence"], default="No hit")
    days = np.select(
        [is_case, is_rejected],
        [cfg.RETENTION_DAYS_CASE, cfg.RETENTION_DAYS_REJECTED],
        default=cfg.RETENTION_DAYS_NO_HIT,
    )
    anchor = df["capture_timestamp"].fillna(df["observation_date"])
    purge_date = anchor + pd.to_timedelta(days, unit="D")
    return pd.DataFrame(
        {
            "retention_class": retention_class,
            "retention_days": days,
            "purge_date": purge_date,
            "purge_due": purge_date <= reference_date,
        },
        index=df.index,
    )


def apply_privacy(df, reference_date=None):
    """Pseudonymise, mask, attach retention and drop the raw plate string."""
    out = df.copy()
    out["plate_token"] = out["plate_number_raw"].map(pseudonymise)
    out["plate_masked"] = out["plate_number_raw"].map(mask_plate)
    out = pd.concat([out, retention_schedule(out, reference_date)], axis=1)
    return out.drop(columns=["plate_number_raw"])


def retention_summary(df):
    """Records per retention class and how many are already due for purge."""
    return (
        df.groupby("retention_class")
        .agg(captures=("capture_id", "count"), purge_due=("purge_due", "sum"), retention_days=("retention_days", "first"))
        .reset_index()
    )
