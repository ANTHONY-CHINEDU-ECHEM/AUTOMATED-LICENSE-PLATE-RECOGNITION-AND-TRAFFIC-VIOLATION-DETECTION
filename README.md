# AUTOMATED LICENSE PLATE RECOGNITION AND TRAFFIC VIOLATION DETECTION MODEL

An industrial analytics pipeline that turns a raw roadside camera export into auditable enforcement decisions. It processes and repairs the capture log, runs a deterministic violation rules layer over the evidence, applies privacy controls before anything is exported, benchmarks the perception models on speed and accuracy, and delivers a prioritised review workbook for enforcement staff.

## Project Brief

A traffic enforcement programme wants to run an Automated License Plate Recognition (ALPR) pipeline across its roadside cameras. Each camera captures passing vehicles, a detector locates the vehicle and its plate, an OCR model reads the plate, and the read is cross referenced against a vehicle registry and against speed, red light, registration and toll rules. The output of that chain is not a dashboard curiosity: it is a legal notice sent to a member of the public, and in the case of a stolen vehicle or Amber Alert hit, a dispatch to an officer. Every decision therefore has to be explainable, reproducible and defensible on appeal.

The programme had already been running and had produced an export of 15,394 capture records across ten corridors, five district zones and four perception models (YOLOv5, YOLOv8, Faster RCNN and EasyOCR) between June 2022 and December 2025. The export was known to be messy. Before the authority committed to automating citations, it needed to know three things: whether the data it had been collecting could be trusted as evidence, whether its perception models were good enough to issue citations without a human in the loop, and how much reviewer effort a responsible rollout would actually require.

This project answers those questions end to end. The raw export is processed first, with every repair logged and every original value preserved, because an enforcement dataset that has been silently cleaned cannot be defended. The core analytics then rebuild each enforcement decision from measured evidence rather than trusting the recorded labels, reconcile those decisions against what the programme actually did, and quantify the trade off between automation and review workload. The result is a set of findings that are uncomfortable in places, most notably that the recorded enforcement labels cannot be reproduced from the evidence, but that is precisely the finding an authority needs before it automates anything.

A note on scope: the export is a capture log, not imagery. The perception stage is therefore evaluated through the confidence, quality, latency and reviewer false positive labels it logged. The frame level two stage reader (vehicle detection, plate localisation, OCR with position aware correction and multi frame fusion) is implemented in `src/alpr_analytics/vision.py` as the production integration point, so that live frames feed the same rules layer, privacy controls and benchmarks used here.

## Headline Findings

1. **The recorded enforcement labels cannot be reproduced from the evidence.** When every capture is rescored by the rules layer, only 63.1% agree with the recorded label. In 3,407 captures (22.3%) the measured speed supports a violation that was never recorded, worth an estimated 795,400 USD under the configured fine schedule, and in 1,776 captures (11.6%) a recorded violation has no evidential support. Of the 2,959 citations on record, 1,786 (60.4%) sit on captures where the evidence does not support any violation.
2. **The legal speed limit itself is uncertain.** The segment speed limit and the limit posted at the capture point disagree on 85.3% of captures. Recorded speeding is supported by measured speed in only 32.8% of cases, and that figure barely moves (33.0%) when the alternative limit field is used, which means the labels were not generated from either limit.
3. **Confidence scores carry no information about false positives.** Detection confidence, OCR confidence, image quality and a combined evidence strength score all separate false positive detections from genuine ones with a ROC AUC between 0.500 and 0.504, which is indistinguishable from a coin toss. A cross validated triage model built on every available feature reached AUC 0.496 (permutation test p value 0.58), so no false positive filter can be deployed on the current signals.
4. **OCR confidence does not respond to lighting or weather.** Mean OCR confidence is flat across dawn, day, dusk and night (Kruskal Wallis p value 0.87) and across weather (p value 0.08). A calibrated OCR model reading real plates degrades at night and in rain; this one does not, which indicates that the logged confidence fields are not behaving like calibrated model outputs and should not be used as evidence until they are validated.
5. **Automation is small and review is large under a defensible policy.** With evidence gates of OCR confidence 0.80, detection confidence 0.70, image quality 40 and a registry match, only 6.9% of captures pass. The rules layer auto issues 332 citations (72,075 USD) and routes 5,391 cases (1,204,200 USD) to review, about 270 reviewer hours or 36 shifts, equal to 177 reviewer hours per 10,000 captures.
6. **Red light enforcement cannot be automated yet.** The export contains no signal phase timing, so no red light event can prove the vehicle entered after the light turned red. All 805 evidential red light candidates route to review until the signal controller feed is connected; the rules layer activates the timing check automatically once that column is present.
7. **Processing speed misses the service level.** Median latency is 119.8 ms but p95 is 219.0 ms (95% bootstrap interval 216.9 to 220.5 ms), so 9.2% of captures breach the 200 ms SLA. Latency does not differ by model, lighting, weather, vehicle type, road type or time band, which points to infrastructure (queueing, network, storage) rather than model choice as the lever.
8. **Retention policy is not being enforced.** Of the 7,480 captures with no violation and no hotlist hit, 7,276 (97.3%) are already past a 30 day retention window and should have been purged. Holding non hit plate reads is the single largest privacy exposure in the programme.
9. **Enforcement demand is spread evenly.** Violation rates vary only between 37.5% and 40.4% by corridor, and differences by zone, road type, jurisdiction, time band and vehicle type are not statistically significant at the 5% level. Localised hotspots exist, such as 5th St Corridor between 00:00 and 04:00 (42.8% speeding) and University Blvd between 04:00 and 07:00 (40%), but camera allocation should be driven by harm and volume rather than by violation rate.

## Recommendations

1. **Do not automate citations on the historical labels.** Freeze automated issuance, audit the 1,786 citations without evidential support, and treat the recorded violation field as untrusted until the labelling process is rebuilt around the rules layer in this repository.
2. **Fix the speed limit reference.** Establish a single authoritative limit per camera location from the traffic regulation order register and join it by camera identifier; the capture point limit is used in the meantime.
3. **Recalibrate and validate the perception confidence outputs.** Commission a labelled validation set stratified by lighting and weather, fit calibration (temperature or isotonic scaling) per model, and rerun the separability analysis. Automation thresholds should only be tuned once confidence predicts correctness.
4. **Connect the signal controller feed** so that red light cases can be proven and moved out of manual review.
5. **Size the review team to the policy.** At current gates the programme needs about 177 reviewer hours per 10,000 captures. Relaxing the OCR gate from 0.80 to 0.60 would double automation (5.7% to 11.1% of candidates) but, with uninformative confidence scores, would do so without any measured gain in precision, so the gate should stay at 0.80 until calibration is proven.
6. **Enforce retention now.** Purge non hit captures after 30 days and rejected evidence after 90 days using the retention classes the pipeline attaches to every record.
7. **Target latency at the platform, not the model.** Profile queueing and I/O on the p95 tail; model selection alone will not bring the pipeline inside the 200 ms SLA.

## Approach Roadmap

<table>
<thead><tr><th>Step</th><th>Roadmap item</th><th>Implementation</th></tr></thead>
<tbody>
<tr><td>0</td><td>Thorough data processing</td><td><code>cleaning.py</code>: identifier standardisation and key repair, canonical vocabularies, boolean unification, four layout date parsing with window based disambiguation, plausibility and robust outlier screening, twelve cross field consistency checks, provenance flags</td></tr>
<tr><td>1</td><td>Two stage detection pipeline</td><td><code>vision.py</code> frame reader (vehicle detector, plate localiser); <code>perception_analysis.py</code> model scorecard and false positive rates with Wilson intervals</td></tr>
<tr><td>2</td><td>OCR under varied lighting and angles</td><td>Position aware character correction and confidence weighted multi frame fusion in <code>vision.py</code>; condition effect tests, OCR condition matrix, separability and threshold sweep in <code>perception_analysis.py</code></td></tr>
<tr><td>3</td><td>Violation classification rules layer</td><td><code>rules_engine.py</code>: speed tolerance and tiers, red light phase check, registration and toll validity checks, evidence gates, hotlist alerts, decision reason codes, reconciliation</td></tr>
<tr><td>4</td><td>Privacy preserving handling</td><td><code>privacy.py</code>: keyed HMAC SHA256 plate tokens, display masking, image region pixelation, retention classes and purge dates, raw plate removed from all exports</td></tr>
<tr><td>5</td><td>Benchmark accuracy and speed</td><td><code>benchmarking.py</code>: latency percentiles with bootstrap intervals, SLA compliance, throughput, accuracy proxies and composite model ranking; <code>modeling.py</code> triage model feasibility</td></tr>
<tr><td>6</td><td>Violation review dashboard</td><td><code>dashboard.py</code>: Excel workbook with live KPI tiles, prioritised review queue with reviewer input columns, priority alerts, auto issue batch, hotspots, benchmark and integrity sheets</td></tr>
</tbody>
</table>

## Data Processing

The data dictionary warned of missing values, mixed date formats, inconsistent casing and whitespace, duplicates and outliers. Profiling found all of these and a number of deeper structural and logical problems. Each is handled explicitly and logged to `data/processed/data_quality_log.csv`, with a narrative report in `outputs/reports/data_quality_report.md`.

<table>
<thead><tr><th>Issue</th><th>Scale</th><th>Treatment</th></tr></thead>
<tbody>
<tr><td>Fully empty rows</td><td>22</td><td>Removed</td></tr>
<tr><td>Exact duplicate rows</td><td>122</td><td>Removed, first occurrence kept</td></tr>
<tr><td>Identifier formats with inconsistent zero padding</td><td>All identifier columns</td><td>Standardised to PREFIX_0000000 so joins to master data work</td></tr>
<tr><td>Record identifiers colliding across different captures</td><td>92</td><td>Capture identifier verified as the true key (it links intersection, camera and plate on every row); record identifier rebuilt and flagged</td></tr>
<tr><td>Categorical variants (case, padding, underscores, trailing periods)</td><td>Up to 35 variants per field</td><td>Mapped to canonical vocabularies through a normalised key</td></tr>
<tr><td>Boolean fields using eight different truth tokens</td><td>8 columns</td><td>Unified to a nullable boolean</td></tr>
<tr><td>Four date layouts, including slash dates that may be day first or month first</td><td>Both date columns</td><td>Resolved by magnitude where possible; 2,452 ambiguous observation dates checked against the window set by unambiguous records (98 corrected) and otherwise read month first, with a per row flag</td></tr>
<tr><td>Recorded weekday contradicting the date</td><td>12,938</td><td>Weekday derived from the date; original retained</td></tr>
<tr><td>Physically impossible measurements (negative or extreme speed, counts)</td><td>440 values</td><td>Nulled as sensor faults and flagged; never imputed</td></tr>
<tr><td>Two conflicting speed limit fields</td><td>13,001</td><td>Capture point limit adopted as the legal reference; conflict flagged</td></tr>
<tr><td>Citation amounts with no citation, appeals with no citation, outcomes with no appeal</td><td>3,664 / 618 / 1,467</td><td>Separated from valid values; originals retained in companion columns</td></tr>
<tr><td>Toll evasion recorded where no toll gantry exists</td><td>375</td><td>Flagged; excluded from enforceable candidates</td></tr>
<tr><td>Missing descriptive values</td><td>Up to 305 per field</td><td>Explicit Unknown level</td></tr>
<tr><td>Missing evidential values (confidence, quality, speed)</td><td>Up to 604 per field</td><td>Left null and routed to review; evidence is never invented</td></tr>
</tbody>
</table>

The output is 15,250 unique captures keyed on capture identifier, each carrying its repairs as flags. Three rules governed every decision: evidential measurements are never imputed, because an imputed speed cannot support a citation; descriptive gaps are made explicit rather than guessed; and no repair is invisible.

![Data integrity issues](outputs/figures/01_data_integrity_issues.png)

## Core Analytics

### Rules layer decisions

Every capture receives one of five decisions with machine readable reasons. Hotlist hits are dispatched regardless of other evidence; candidate violations marked as false positives are rejected; candidates passing every evidence gate are auto issued; everything else plausible goes to review.

<table>
<thead><tr><th>Decision</th><th>Captures</th><th>Share</th><th>Fine value (USD)</th></tr></thead>
<tbody>
<tr><td>Priority Alert</td><td>36</td><td>0.24%</td><td>2,775</td></tr>
<tr><td>Auto Issue</td><td>332</td><td>2.18%</td><td>72,075</td></tr>
<tr><td>Manual Review</td><td>5,391</td><td>35.35%</td><td>1,204,200</td></tr>
<tr><td>Reject</td><td>227</td><td>1.49%</td><td>49,525</td></tr>
<tr><td>No Action</td><td>9,264</td><td>60.75%</td><td>0</td></tr>
</tbody>
</table>

Among review cases, low OCR confidence is the most common blocking reason (76.6% of cases), followed by low detection confidence (60.8%) and low image quality (43.9%).

![Decision mix](outputs/figures/02_decision_mix.png)

![Label reconciliation](outputs/figures/03_label_reconciliation.png)

### Perception stage and threshold tuning

<table>
<thead><tr><th>Model</th><th>Captures</th><th>Mean detection confidence</th><th>Mean OCR confidence</th><th>All gates pass</th><th>False positive rate (95% interval)</th></tr></thead>
<tbody>
<tr><td>EasyOCR</td><td>3,832</td><td>0.652</td><td>0.653</td><td>6.84%</td><td>3.16% (2.65 to 3.76)</td></tr>
<tr><td>Faster RCNN</td><td>3,801</td><td>0.653</td><td>0.653</td><td>7.08%</td><td>3.68% (3.13 to 4.33)</td></tr>
<tr><td>YOLOv5</td><td>3,820</td><td>0.649</td><td>0.646</td><td>6.60%</td><td>4.16% (3.57 to 4.84)</td></tr>
<tr><td>YOLOv8</td><td>3,797</td><td>0.647</td><td>0.652</td><td>6.90%</td><td>4.21% (3.62 to 4.90)</td></tr>
</tbody>
</table>

False positive rates differ by model only at the margin of significance (chi square p value 0.055, Cramer V 0.022). The threshold sweep shows the automation and workload trade off: at a detection threshold of 0.70, lowering the OCR gate from 0.80 to 0.60 lifts automation from 5.7% to 11.1% of candidates while the false positive share of accepted cases stays between 2% and 3.5%, close to the 3.8% base rate, confirming that the confidence scores are not buying precision.

![Threshold trade off](outputs/figures/08_threshold_tradeoff.png)

![False positive separability](outputs/figures/10_false_positive_separability.png)

![OCR conditions](outputs/figures/07_ocr_conditions.png)

### Benchmark

<table>
<thead><tr><th>Model</th><th>p50 (ms)</th><th>p95 (ms)</th><th>p99 (ms)</th><th>SLA compliance</th><th>Frames per second per stream</th><th>Rules label agreement</th><th>Composite rank</th></tr></thead>
<tbody>
<tr><td>Faster RCNN</td><td>120.9</td><td>218.9</td><td>264.2</td><td>90.93%</td><td>8.23</td><td>63.17%</td><td>1</td></tr>
<tr><td>EasyOCR</td><td>119.9</td><td>219.1</td><td>251.1</td><td>90.79%</td><td>8.33</td><td>63.47%</td><td>2</td></tr>
<tr><td>YOLOv5</td><td>120.1</td><td>217.1</td><td>259.2</td><td>90.77%</td><td>8.24</td><td>63.69%</td><td>3</td></tr>
<tr><td>YOLOv8</td><td>118.1</td><td>220.2</td><td>264.1</td><td>90.92%</td><td>8.35</td><td>62.13%</td><td>4</td></tr>
</tbody>
</table>

The composite ranking is reported for completeness, but the bootstrap intervals on p95 latency overlap for every model and none of the latency drivers tested is significant. On this evidence no model should be retired on performance grounds; the decision should be made on licensing, hardware and maintainability once confidence outputs are calibrated.

![Latency by model](outputs/figures/06_latency_by_model.png)

### Corridors and hotspots

![Speeding hotspots](outputs/figures/05_speeding_hotspots.png)

![Corridor fine value](outputs/figures/04_corridor_fine_value.png)

![Monthly trend](outputs/figures/09_monthly_trend.png)

### Privacy and retention

Plates are replaced by a keyed HMAC SHA256 token (the key is read from the `ALPR_PSEUDONYM_KEY` environment variable and must be rotated from the development default before production), shown to reviewers only as a masked tail, and removed from every export. Image redaction pixelates plate and occupant regions supplied by the detector. Each capture receives a retention class: 30 days for no hit captures, 90 days for rejected evidence, and three years for case evidence.

<table>
<thead><tr><th>Retention class</th><th>Captures</th><th>Retention (days)</th><th>Already due for purge</th></tr></thead>
<tbody>
<tr><td>No hit</td><td>7,480</td><td>30</td><td>7,276</td></tr>
<tr><td>Rejected evidence</td><td>175</td><td>90</td><td>162</td></tr>
<tr><td>Case evidence</td><td>7,595</td><td>1,095</td><td>12</td></tr>
</tbody>
</table>

### Citations and appeals

Appeals were filed on 161 of 2,959 citations (5.4%), but only 13 appeals have a recorded decision, of which 4 were overturned. That sample is too small to model and is reported descriptively only; appeal outcomes should be captured systematically because they are the strongest external check on evidence quality.

## Violation Review Dashboard

`outputs/dashboard/violation_review_dashboard.xlsx` is the enforcement staff deliverable.

<table>
<thead><tr><th>Sheet</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td>Overview</td><td>Live KPI tiles driven by formulas over the Decisions sheet, usage guidance and policy assumptions</td></tr>
<tr><td>Review Queue</td><td>5,391 cases ranked by priority score (fine value and evidence strength) with reason codes, a Reviewer Decision dropdown (Confirm, Dismiss, Escalate) and a notes column</td></tr>
<tr><td>Priority Alerts</td><td>Stolen vehicle and Amber Alert hotlist hits for immediate action</td></tr>
<tr><td>Auto Issue Batch</td><td>Citations that passed every evidence gate</td></tr>
<tr><td>Corridor Hotspots</td><td>Corridor enforcement profile and a colour scaled corridor by time band speeding matrix</td></tr>
<tr><td>Model Benchmark</td><td>Speed and accuracy scorecard per model</td></tr>
<tr><td>Data Integrity</td><td>Integrity issues found in the source export</td></tr>
<tr><td>Decisions</td><td>Every capture with its decision, using privacy safe fields only</td></tr>
<tr><td>Charts</td><td>Key figures</td></tr>
</tbody>
</table>

## Repository Structure

```
alpr_traffic_violation_analytics/
    run_pipeline.py                  entry point
    requirements.txt                 analytics dependencies
    requirements_vision.txt          optional dependencies for live frames
    data/
        raw/alpr_traffic_raw.xlsx    source export, preserved as delivered
        processed/                   processed captures and quality log
    src/alpr_analytics/
        config.py                    every threshold, vocabulary and policy parameter
        cleaning.py                  data processing and quality log
        rules_engine.py              violation rules layer and reconciliation
        perception_analysis.py       detection and OCR analysis, threshold sweep
        benchmarking.py              latency and accuracy benchmark
        modeling.py                  false positive triage feasibility study
        enforcement_analysis.py      hotspots, integrity, citations, workload
        privacy.py                   pseudonymisation, masking, redaction, retention
        vision.py                    two stage frame reader and OCR post processing
        stats_utils.py               Wilson intervals, chi square, Kruskal Wallis, bootstrap
        figures.py                   charts
        dashboard.py                 Excel review workbook
        reporting.py                 CSV, JSON and Markdown writers
        pipeline.py                  orchestration
    tests/                           unit tests and repository style audit
    outputs/
        figures/                     charts used in this README
        reports/                     analysis tables, metrics.json, generated reports
        dashboard/                   violation review workbook
```

## How to Run

Python 3.10 or later is required.

```
pip install pandas numpy scipy scikit_learn matplotlib openpyxl pillow pytest
python run_pipeline.py
pytest tests
```

The pipeline takes about half a minute and regenerates everything under `data/processed` and `outputs`. To run on a newer export with the same schema, pass its path: `python run_pipeline.py path/to/export.xlsx`. Enforcement policy (tolerance, tiers, fines, evidence gates, SLA, retention) is changed in `src/alpr_analytics/config.py` without touching processing logic. For live cameras, install the packages in `requirements_vision.txt` and use `TwoStagePlateReader` in `vision.py` to produce capture records in the same schema.

The test suite covers the processing primitives, an end to end cleaning run on synthetic records, every rules layer decision path, the privacy controls and the OCR post processing, and includes a repository style audit.

## Design Decisions and Limitations

1. **Evidence over labels.** The rules layer derives violations from measurements and treats recorded labels as something to audit, not a target to learn. Training a classifier on labels that contradict the evidence would automate the contradictions.
2. **Conservative gating.** A capture is only auto issued when every evidence gate passes and a type specific validity check holds. Missing evidence routes to review, never to issuance.
3. **Honest modelling.** The triage model is reported as not deployable because it does not beat a prior only baseline. Publishing a weak model with a confident narrative would be the more dangerous outcome.
4. **Capture log scope.** No imagery was supplied, so detection and OCR quality are assessed through logged outputs and reviewer labels. The frame reader is provided as the integration point but its accuracy on this programme's cameras is untested.
5. **Fine schedule.** Fine values are configurable placeholders used to express value at stake; they should be replaced with the jurisdiction's schedule.
6. **Ambiguous dates.** 2,354 observation dates remain genuinely ambiguous after the window check and are read month first. They affect monthly trends only, never an enforcement decision, and are flagged per row.

## Skills Demonstrated

Python (pandas, NumPy, SciPy, scikit learn, Matplotlib, openpyxl), industrial data processing with full provenance, rules engine design with explainable reason codes, OCR post processing and multi frame fusion, object detection pipeline design (YOLO style detection and plate localisation), statistical testing (Wilson intervals, chi square, Kruskal Wallis, bootstrap, permutation testing), cross validated model evaluation, privacy by design (keyed pseudonymisation, masking, redaction, retention), pipeline benchmarking against a service level, and operational dashboard delivery for enforcement staff.
