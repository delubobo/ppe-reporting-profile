# Data licences

This note covers every file in the package that is not code. The code (`routing/`, `analysis/*.py`, `tests/`, `tools/`) is under the MIT licence in `LICENSE`.

## Original images

No image from any dataset is redistributed. The original images and annotations remain with the dataset providers and must be obtained from them under their own terms (see the table below).

## Files released under CC0 1.0

The following files are dedicated to the public domain under CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/):

- file names, fold assignments and image sizes: `data/folds.json`, `data/manifests/`;
- duplicate-exclusion lists and their summary: `data/duplicates/`;
- class map, matching settings and environment records: `data/class_map.json`, `data/matching_settings.json`, `data/environment.json`, `data/yolox_source.json`;
- saved summary results: `results/` and `analysis/*.json`.

## Files derived from Construction Site Safety: CC BY 4.0

The detections and labels in `detections/css_v30/` are derived from the Construction Site Safety dataset of Roboflow Universe Projects and are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Attribution: Roboflow Universe Projects, "Construction Site Safety", Roboflow Universe, https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety. The provenance of the images is as stated by the uploader on that page.

## Files derived from Hard Hat Workers: CC0 1.0

The detections and labels in `detections/hard_hat_workers/` are derived from the Hard Hat Workers dataset (Xie, Northeastern University, China; Harvard Dataverse doi:10.7910/DVN/7CBGOS), which is dedicated to the public domain under CC0 1.0. The derived files are released under CC0 1.0 as well.

## CHV, SHWD and SH17

Detections and labels derived from CHV, SHWD and SH17 are described in the paper and summarized in `results/`, but they are not released in this version. They will follow, dataset by dataset, where the dataset terms permit.

## Licence pages read on 9 November 2026

| Dataset | Page read | What the page stated |
|---|---|---|
| Construction Site Safety (Roboflow Universe Projects) | https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety | Licence field: CC BY 4.0, linked to the Creative Commons CC BY 4.0 deed |
| Hard Hat Workers | Harvard Dataverse record doi:10.7910/DVN/7CBGOS (read through the Dataverse metadata interface) | Licence: CC0 1.0 (SPDX identifier CC0-1.0) |
| Hard Hat Workers | https://public.roboflow.com/object-detection/hard-hat-workers (redirects to the Roboflow Universe page of the same dataset) | Licence field: Public Domain, linked to the CC0 1.0 deed |
| SHWD | https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset | Repository licence: MIT (LICENSE file, copyright 2019 njvisionpower). The README carries no separate licence statement for the images |
| CHV | https://github.com/ZijianWang-ZW/PPE_detection | No licence file. The README states that the dataset is open for free use and asks users to cite the repository |
| SH17 | https://github.com/ahmadmughees/SH17dataset | README licence section: CC BY-NC-SA 4.0. The images were collected from Pexels and the README asks users to respect the Pexels terms |
