# Chest X-ray pneumonia classifier: an MLOps life cycle project

Course project for JHU Engineering for Professionals, AI/ML Ops (635.603.81).

The goal is to practice the MLOps life cycle end to end: data versioning, training, evaluation, model registry, deployment to a REST endpoint, consuming the API, monitoring ideas and responsible AI. The point is managing the life cycle, not building a clever model, so the model is trained with Google Cloud's low-code AutoML (Vertex AI, now called Agent Platform in the console).

**This is a learning project and decision-support research only. It is not a diagnostic tool.**

## Dataset

Kaggle "Chest X-Ray Images (Pneumonia)" (`paultimothymooney/chest-xray-pneumonia`): pediatric chest X-rays from a single source, labeled normal or pneumonia. Pneumonia filenames also say `bacteria` or `virus`, which gives three classes.

The dataset is **not in this repo** (`data/` is gitignored). Download it from Kaggle and extract it so the images are under `data/chest_xray/` (folders `train/`, `val/`, `test/`, each with `NORMAL/` and `PNEUMONIA/`).

Kaggle's default split is unsuitable (the validation folder has only 16 images), so this project builds its own split.

| Split | Images |
|---|---|
| Train | 4,160 |
| Validation | 836 |
| Test | 828 |
| **Total** | **5,824** |

Test split: 226 normal, 393 bacteria, 209 virus.

## Setup

Requires Anaconda (or Miniconda).

```bash
conda env create -f environment.yml
conda activate mlops-xray
python src/prepare_data.py
```

The script finds the data relative to the project root (`data/chest_xray/`) and writes everything to `data/prepared/`. To import into Cloud Storage, set `gcs_prefix` in the script's `__main__` block to your own bucket path.

## Data preparation (`src/prepare_data.py`)

The script runs these steps in order:

1. **Find the real images.** Walk the dataset, keep `.jpeg`, `.jpg` and `.png`, and skip `._` files, `__MACOSX` junk and the nested duplicate `chest_xray/chest_xray` folder.
2. **Label each image** from its filename (`normal`, `pneumonia_bacteria`, `pneumonia_virus`). The folder name is only a sanity check, and the script stops if they disagree.
3. **Patient group key** from the filename (`person5_bacteria`, `IM-0115`). The subtype is part of the key because person numbering restarts per class. A file with no recognizable ID becomes its own group, with a warning.
4. **Split about 70/15/15**, stratified by class and grouped by patient (`StratifiedGroupKFold`, seed 42, sorted input order, so it is reproducible).
   - *Grouped by patient:* one patient's images look alike, so splitting by image would leak information into the test set and inflate scores.
   - *Stratified:* keeps class proportions similar in every split despite the class imbalance.
5. **Remove byte-identical duplicates.** The raw data contains 32 exact duplicate images (5,856 files became 5,824). Duplicates are found by MD5 hash, and the copy in the most protected split (test, then validation, then train) is kept so nothing leaks across splits.
6. **Verify.** Valid split names, unique images, no patient in more than one split, class counts and percentages per split.
7. **Export.** Copy the images to `data/prepared/images/<label>/`, write `manifest.csv` (relative path, label, binary label, patient group, split, source path), and write two Vertex AI import CSVs, one for three classes and one for the binary task (bacteria and virus merged into `pneumonia`, same images and same splits).

## Cloud workflow

- Images and CSVs are uploaded to a Cloud Storage bucket (us-central1) under a versioned prefix (`v1/`), so a future `v2/` can coexist.
- Import CSV format: no header, one line per image, `TRAINING|VALIDATION|TEST,gs://bucket/path/image.jpeg,label`. The split comes from the CSV, so the same split is used by every model.
- Two Vertex AI image datasets (single-label classification): `xray-3class-v1` and `xray-2class-v1`.
- Two AutoML models, each with the minimum 8 node-hour budget: `xray-3class-automl-v1` and `xray-2class-v1`.

## Results

Evaluated on the held-out test split (828 images, confidence threshold 0.5).

| | Binary (normal vs. pneumonia) | 3-class |
|---|---|---|
| PR AUC | 0.998 | 0.909 |
| Log loss | 0.081 | 0.271 |
| Precision / recall (Vertex headline) | 98.7% / 98.7% | 84.9% / 82.9% |

- **Binary:** 598 of 602 pneumonia cases found (4 missed), and 7 of 226 normal images flagged as pneumonia.
- **3-class** per-class recall: normal 96.9%, bacteria 90.1%, **virus 56.9%**. Virus is often confused with bacteria (79 of 209), and 11 virus cases are called normal.
- Merging the 3-class model's bacteria and virus predictions gives 13 missed pneumonia cases (vs. 4 for the dedicated binary model) and the same 7 false alarms.

Accuracy alone is misleading here because the classes are imbalanced (73% of the test images are pneumonia) and a missed pneumonia costs more than a false alarm, so per-class recall and the confusion matrix matter most.

## Limitations and responsible AI

- Single pediatric source, so results may not carry over to other populations, scanners or hospitals.
- Labels come from the dataset and are not clinically verified.
- Near-perfect binary scores on one source can reflect shortcuts in the images (framing, brightness), not only lung findings.
- AutoML is a black box: the architecture is chosen by the service, and no image explanations were available.
- Small test counts: differences of a few tenths of a percent are noise.
- Decision support only. Not for diagnosis.

## Project status

Done: data preparation, split, cloud upload, dataset import, AutoML training and evaluation of both models.
Next: deploy a model to an endpoint, call the REST API, record a demo, then undeploy to stop endpoint billing.

## Repository layout

```
README.md
environment.yml
src/prepare_data.py     data preparation, split and export
data/                   raw and prepared data (gitignored)
```
