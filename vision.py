"""Two stage perception interface: vehicle detection, plate localisation and OCR.

This module is the integration point for live camera frames. The analytics in
this repository run on the capture log that this stage produces; the classes
below define how frames become capture records so that the same rules layer,
privacy controls and benchmarks apply unchanged in production.

Stage 1  A YOLO style detector finds vehicles, then a plate detector (a second
         YOLO head fine tuned on plate crops) localises the plate inside each
         vehicle box.
Stage 2  OCR reads the plate crop. Reads are cleaned with position aware
         character correction and fused across consecutive frames, which is
         the main defence against glare, motion blur and low light.

Heavy dependencies (ultralytics, easyocr, opencv) are imported lazily so the
analytics pipeline and tests run without them. Install them from
requirements_vision.txt when connecting real cameras.
"""

import re
from collections import Counter, defaultdict

LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T"}


def normalise_plate_text(text):
    """Upper case and keep only letters and digits."""
    return re.sub(r"[\W_]+", "", str(text)).upper()


def correct_by_pattern(text, pattern):
    """Fix common OCR confusions using a plate layout pattern.

    pattern uses L for a letter position, D for a digit position and ? for
    either, for example LLLDDDD for a three letter four digit plate. Returns
    None when the length does not match the pattern.
    """
    text = normalise_plate_text(text)
    if len(text) != len(pattern):
        return None
    fixed = []
    for char, slot in zip(text, pattern.upper()):
        if slot == "D" and not char.isdigit():
            char = LETTER_TO_DIGIT.get(char, char)
        elif slot == "L" and char.isdigit():
            char = DIGIT_TO_LETTER.get(char, char)
        fixed.append(char)
    return "".join(fixed)


def fuse_frame_reads(reads):
    """Fuse OCR reads of one vehicle across frames by confidence weighted voting.

    reads : iterable of (text, confidence) pairs from consecutive frames.
    Returns (fused_text, agreement) where agreement is the weighted share of
    votes behind the winning character, averaged over positions.
    """
    cleaned = [(normalise_plate_text(t), float(c)) for t, c in reads if t]
    if not cleaned:
        return None, 0.0
    length_votes = Counter()
    for text, conf in cleaned:
        length_votes[len(text)] += conf
    length = max(length_votes, key=length_votes.get)
    same_length = [(t, c) for t, c in cleaned if len(t) == length]

    fused, agreement = [], []
    for position in range(length):
        votes = defaultdict(float)
        for text, conf in same_length:
            votes[text[position]] += conf
        winner = max(votes, key=votes.get)
        fused.append(winner)
        agreement.append(votes[winner] / sum(votes.values()))
    return "".join(fused), round(sum(agreement) / len(agreement), 4)


class TwoStagePlateReader:
    """Vehicle detector, plate localiser and OCR reader for single frames.

    vehicle_weights : path to detector weights for vehicles
    plate_weights   : path to detector weights fine tuned on plates
    plate_pattern   : optional layout pattern for position aware correction
    """

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

    def __init__(self, vehicle_weights, plate_weights, plate_pattern=None, languages=("en",)):
        from ultralytics import YOLO  # lazy import
        import easyocr  # lazy import

        self.vehicle_model = YOLO(vehicle_weights)
        self.plate_model = YOLO(plate_weights)
        self.reader = easyocr.Reader(list(languages), gpu=False)
        self.plate_pattern = plate_pattern

    def read(self, frame, vehicle_conf=0.5, plate_conf=0.5):
        """Return one record per plate found in the frame."""
        records = []
        vehicles = self.vehicle_model(frame, conf=vehicle_conf, verbose=False)[0]
        for box in vehicles.boxes:
            label = vehicles.names[int(box.cls)]
            if label not in self.VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            plates = self.plate_model(crop, conf=plate_conf, verbose=False)[0]
            for plate_box in plates.boxes:
                px1, py1, px2, py2 = (int(v) for v in plate_box.xyxy[0].tolist())
                plate_crop = crop[py1:py2, px1:px2]
                text, ocr_conf = self._ocr(plate_crop)
                records.append(
                    {
                        "vehicle_type": label,
                        "detection_confidence": float(box.conf),
                        "plate_confidence": float(plate_box.conf),
                        "plate_text": text,
                        "ocr_confidence_score": ocr_conf,
                        "vehicle_box": (x1, y1, x2, y2),
                        "plate_box": (x1 + px1, y1 + py1, x1 + px2, y1 + py2),
                    }
                )
        return records

    def _ocr(self, plate_crop):
        results = self.reader.readtext(plate_crop, detail=1)
        if not results:
            return None, 0.0
        text = "".join(r[1] for r in results)
        confidence = sum(r[2] for r in results) / len(results)
        text = normalise_plate_text(text)
        if self.plate_pattern:
            text = correct_by_pattern(text, self.plate_pattern) or text
        return text, round(float(confidence), 4)
