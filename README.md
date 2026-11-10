# ppe-reporting-profile

## What this is

This package accompanies the paper "What Does a Single Error Rate Hide? Missed Violations and Human Review Workload in AI-Based PPE Monitoring on Construction Sites" by Roy Uzoma Lan (Western Kentucky University), submitted for publication. It holds the evaluation code, the class mappings, the fold assignments, the duplicate-exclusion lists, the matching settings, the saved summary results, and the detections for the two datasets whose licences allow redistribution of derived files (Construction Site Safety and Hard Hat Workers). The paper's contribution is an operational reporting profile that reports alert errors, missed violations and human review workload jointly when a PPE monitor is commissioned, with a three-band policy that reports, refers or suppresses each detection.

## Contents

| Folder or file | What it holds |
|---|---|
| `routing/` | The evaluation code: matching, calibration, report and hide thresholds, cross-validated experiments, person-level and budget analyses |
| `tests/` | Unit tests of the core functions |
| `analysis/` | The held-out hide-threshold analysis and the report-pool composition analysis, with their saved JSON outputs |
| `data/class_map.json` | The five datasets, their class mappings to the unified classes and the classes each one labels |
| `data/matching_settings.json` | Calibration targets, matching rule per dataset, containment parameters, fold rule and duplicate-exclusion settings |
| `data/folds.json` | The fold of each of the 700 source photos |
| `data/duplicates/` | One CSV per dataset comparison of near-duplicate photos, and `summary.json` with the counts |
| `data/manifests/` | File name, fold (source only) and image size of every photo in the released detection sets |
| `data/environment.json`, `data/yolox_source.json` | Training environment and the YOLOX source revision |
| `detections/css_v30/` | Detections of each of the five fold models on every source photo, for YOLOX Nano, S, M and L |
| `detections/hard_hat_workers/` | Detections of each fold model on every Hard Hat Workers photo, for the same four sizes |
| `results/` | Saved summary results of every analysis in the paper (JSON) |
| `tools/` | The build, export, scan, reproduction and fresh-environment test scripts |

Each detection file `yolox_<size>_fold<k>.npz` holds, in the order of the manifest, the photo file names, the image height and width, the fold (source only), the ground-truth boxes and classes, and the detection boxes, scores and classes. Ragged per-photo arrays are stored concatenated with `gt_offsets` and `det_offsets`, so that the rows of photo i are `offsets[i]` to `offsets[i + 1]`. Boxes are x1, y1, x2, y2 in pixels; classes index the unified classes `helmet`, `head`, `vest` and `person`.

## Datasets

No image is redistributed. Each dataset must be obtained from its provider under its own terms.

| Dataset | Role | Where to download | Licence as stated by the provider | Licence page read |
|---|---|---|---|---|
| Construction Site Safety, Roboflow Universe Projects (export v30, 700 photos) | Source | https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety | CC BY 4.0 | 9 November 2026 |
| Hard Hat Workers, Northeastern University, China | Target | Harvard Dataverse doi:10.7910/DVN/7CBGOS, or https://public.roboflow.com/object-detection/hard-hat-workers | CC0 1.0 | 9 November 2026 |
| CHV (Color Helmet and Vest), Wang et al. 2021 | Target | https://github.com/ZijianWang-ZW/PPE_detection | No formal licence; free use with citation of the repository | 9 November 2026 |
| SHWD (Safety Helmet Wearing Dataset), web-collected portion | Target | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset | MIT at repository level | 9 November 2026 |
| SH17, Ahmad and Rahimi 2024 | Target, outside construction | https://github.com/ahmadmughees/SH17dataset | CC BY-NC-SA 4.0 | 9 November 2026 |

`LICENSE-DATA.md` records what each page stated.

## Method settings

All values below are read from the code and written to `data/class_map.json` and `data/matching_settings.json`.

- Unified classes: `helmet` (hard hat worn), `head` (bare head, the violation class), `vest` and `person`. Each dataset is mapped to these classes, and a detector is scored only on the classes that dataset labels.
- Matching: greedy matching in descending score order within a class. CHV and the source use IoU at or above 0.5. Hard Hat Workers and SHWD box the whole head while the detector boxes the hat, so a detection matches when at least 70 percent of it lies inside the ground-truth box and the IoU is at least 0.2. SH17 uses this containment rule for bare heads and IoU 0.5 for the other classes.
- Folds: five folds, assigned by `int(md5(original file stem), 16) % 5`. Each source photo is scored by the one model that did not train on it.
- Calibration and thresholds: Platt scaling fitted on the source detections. The aggregate report threshold targets a false-report rate of `ALPHA` = 0.10 with confidence parameter `DELTA` = 0.10 (the `cluster` method of `routing.core.select_t_high`); the photo-level rule targets a share of photos with a false report of `ALPHA_PHOTO` = 0.20; the hide threshold targets a miss rate of `BETA` = 0.10 and falls back to 0.50 when no hide threshold can be certified.
- Duplicate exclusion: a 64-bit difference hash (dhash) of each photo, with near-duplicates defined as at most 4 differing bits. Source photos with a near-duplicate in another fold are dropped (44 of 700, leaving 656). Each target is then compared with the source and with every previously kept target, in the order CHV, Hard Hat Workers, SHWD, SH17, and its near-duplicates are dropped. The lists in `data/duplicates/` reproduce the counts of the `cleaning` block in `results/experiments_cv.json` exactly.

## Reproducing one manuscript table

The released detections reproduce the Source held-out and Hard Hat Workers rows of Table 5 of the paper (source-half aggregate reporting rule at alpha = 0.10, applied unchanged, YOLOX Nano, S, M and L). The package requires Python 3.12 or later, because numpy 2.5.3 does; it was tested with Python 3.14.7 on Windows.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python tools\reproduce_table.py
```

On Linux or macOS, use `.venv/bin/python` in place of `.venv\Scripts\python`. `tools/test_fresh_env.ps1` runs the same steps in a fresh copy of the package. The reproduction applies the duplicate-exclusion lists to the released detections, fits the rule with the study's own functions (`routing.experiments_cv.fit_rule` and `e1_transfer`), and compares every cell with the saved results and with the published table. The expected output ends with these rows and the line `ALL CELLS MATCH`:

| Dataset | Size | False-report rate (%) | Runs within target (%) | Routed share (%) |
|---|---|---|---|---|
| Source held-out | Nano / S / M / L | 7.2 (4.5 to 10.0) / 7.4 (5.3 to 10.0) / 7.4 (4.4 to 11.1) / 7.8 (4.9 to 10.4) | 94 / 94 / 90 / 86 | 19 / 16 / 13 / 13 |
| Hard Hat Workers | Nano / S / M / L | 6.3 (3.5 to 8.5) / 5.3 (3.9 to 6.5) / 6.3 (4.0 to 8.8) / 6.0 (4.4 to 7.4) | 100 / 100 / 100 / 100 | 23 / 15 / 11 / 9 |

The CHV, SHWD and SH17 rows of the same table need detections that are not released in this version.

## Running the full analysis

`python -m routing.experiments_cv` reruns every cross-validated experiment, and `PPE_VARIANT=s`, `m` or `l` selects the YOLOX size. It reads the fold caches of every dataset (`outputs/cache/folds/`), which this package does not include: the Construction Site Safety and Hard Hat Workers detections are released here in npz form, while those for CHV, SHWD and SH17, the RT-DETRv2 detections and the model checkpoints will follow on Zenodo where the dataset licences permit. The scripts in `analysis/` read the same caches and therefore run only when those files are available. The saved outputs of every analysis are in `results/` and `analysis/`.

## Citation

Please cite the paper and this package; `CITATION.cff` gives both references. Version 1.0.0 of this package is archived on Zenodo at https://doi.org/10.5281/zenodo.24100886.

## Licences

The code is under the MIT licence (`LICENSE`). The lists, settings, manifests and saved results are under CC0 1.0, files derived from Construction Site Safety are under CC BY 4.0, and files derived from Hard Hat Workers are under CC0 1.0; `LICENSE-DATA.md` gives the details.
