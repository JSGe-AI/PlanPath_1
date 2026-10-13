# PlanPath

PlanPath is a patient-level pathology visual question answering agent
framework. It combines whole-slide retrieval, multiscale region selection,
vision-language observation, tissue-guided navigation, cellular analysis,
quantitative measurement, and structured answer generation in one inference
workflow.

The repository contains the complete PlanPath inference and evaluation code.
Model weights, whole-slide images, derived features, predictions, and runtime
caches are not included.

## Repository Structure

```text
PlanPath/
├── planpath/                       PlanPath Python package
│   ├── core/                       Retrieval and multiscale analysis
│   ├── depth_calibration/          Invasion-depth calibration
│   ├── efficiency/                 Efficient selection policies
│   ├── evidence_fusion/            Cross-scale evidence fusion
│   ├── measurement/                Measurement controllers
│   ├── model/                      Public PlanPath workflow and routing
│   ├── nodal_calibration/          Nodal-evidence calibration
│   ├── quantitative/               Quantitative inference implementation
│   └── evaluation.py               Dataset-level evaluation metrics
├── configs/
│   ├── renal.json                  Renal-cancer configuration
│   ├── gastric.json                Gastric-cancer configuration
│   └── generic_no_tissue.json      Configuration without tissue navigation
├── datasets/
│   ├── Patient_VQA_Gastric.json    Gastric patient-level VQA dataset
│   └── Patient_VQA_Renal.json      Renal patient-level VQA dataset
├── scripts/
│   ├── run_planpath.py             Unified inference entry point
│   ├── evaluate_predictions.py     Evaluation entry point
│   ├── launch_*_shards.sh          Patient-level shard launchers
│   └── merge_planpath_shards.*     Shard-merging utilities
├── vendor/histoplus/               Pinned HistoPLUS source and license
├── requirements.txt                Python dependencies
└── pyproject.toml                  Package metadata
```

## Environment Requirements

- Python 3.10-3.12;
- a CUDA-capable PyTorch environment;
- OpenSlide and its system library;
- local TITAN and PLIP model files;
- OpenAI-compatible Qwen and Patho-R1 inference endpoints;
- HistoPLUS checkpoints when cellular analysis is enabled;
- CONCH features, cancer-specific tissue prototypes, and tissue maps when
  tissue navigation is enabled.

Install PlanPath in an isolated environment:

```bash
cd PlanPath
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The HistoPLUS source is pinned under `vendor/histoplus`, with its upstream
license and attribution preserved. Authorized users can install its separately
distributed checkpoints with:

```bash
HF_TOKEN=YOUR_TOKEN bash scripts/install_histoplus.sh
```

PlanPath uses Qwen for structured planning and answer synthesis and Patho-R1
for morphology observations. Set the local model paths and the exact served
model IDs and endpoints in `configs/renal.json` or `configs/gastric.json`.

## Patient-Level VQA Datasets

The `datasets/` directory includes two patient-level VQA datasets constructed
for PlanPath:

| Dataset | Patients | Questions |
| --- | ---: | ---: |
| `Patient_VQA_Gastric.json` | 104 | 923 |
| `Patient_VQA_Renal.json` | 109 | 700 |

For access to the original whole-slide images for research use, please contact
the corresponding author.

Each file is a JSON array grouped by `patient_id`. Every question contains a
question ID, question type, question text, answer choices, and reference
answer:

```json
[
  {
    "patient_id": "gastric001",
    "qa_pairs": [
      {
        "qa_id": "q1",
        "question_type": "short_answer",
        "question": "What is the histological type of the primary tumor in this patient?",
        "options": [],
        "answer": "Adenocarcinoma"
      }
    ]
  }
]
```

The supported question types are `single_choice`, `yes_no`, and
`short_answer`. The supplied renal and gastric configurations already point to
their corresponding files under `datasets/`.

### Slide List Organization

The dataset-facing slide list is organized as JSON Lines with one slide per
line and only two identifiers:

```json
{"patient_id":"gastric001","slide_id":"slide0001"}
{"patient_id":"gastric001","slide_id":"slide0002"}
{"patient_id":"gastric002","slide_id":"slide0003"}
```

`patient_id` associates questions with a patient, and `slide_id` identifies a
whole-slide image. The local data layer resolves each `slide_id` to the WSI and
its required feature artifacts. Set `manifest` in the selected configuration
to the resolved local runtime manifest; storage paths remain outside the VQA
dataset and the dataset-facing slide list.

## Running PlanPath

Before inference, update the selected configuration with the local runtime
manifest, model locations, service endpoints, and output directories.
Repository-relative paths are resolved from the PlanPath root.

Run the renal-cancer dataset:

```bash
python scripts/run_planpath.py \
  --config configs/renal.json \
  --overwrite
```

Run the gastric-cancer dataset:

```bash
python scripts/run_planpath.py \
  --config configs/gastric.json \
  --overwrite
```

The unified runner analyzes every question before inference. Supported gastric
component-percentage questions and renal independent-primary-focus count
questions are routed automatically to PlanPath's quantitative workflow. All
other questions use the standard workflow, and no separate quantitative run or
prediction merge is required.

### Patient-Level Sharding

Run either dataset in patient-level shards:

```bash
PYTHON_BIN=python bash scripts/launch_renal_shards.sh 3
PYTHON_BIN=python bash scripts/launch_gastric_shards.sh 3
```

Merge the completed shard outputs:

```bash
PYTHON_BIN=python bash scripts/merge_planpath_shards.sh 3 both
```

### Evaluation

Evaluate renal predictions:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/renal/predictions.jsonl \
  --vqa datasets/Patient_VQA_Renal.json \
  --output-dir outputs/renal/metrics
```

Evaluate gastric predictions:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/gastric/predictions.jsonl \
  --vqa datasets/Patient_VQA_Gastric.json \
  --output-dir outputs/gastric/metrics
```

Use `--allow-partial` to evaluate a completed subset. The evaluator writes:

- `metrics.json`: overall, per-question-type, patient-macro, error, and
  quantitative metrics;
- `metrics.csv`: a flat table of the same aggregate metrics;
- `evaluation_cases.jsonl`: per-question scores and parsed quantitative values.

Accuracy, normalized exact match, short-answer containment accuracy, and
short-answer token F1 are reported. For automatically routed quantitative
questions, the report also includes percentage-point mean absolute error,
accuracy within 5 and 10 percentage points, and exact renal focus-count
accuracy.

The inference runner writes predictions to the configured `output`, task
traces under `trace_root`, and reusable runtime data under `cache_root`.

## Tissue-Analysis Scope

The included tissue-analysis tools were constructed for **renal cancer and
gastric cancer**. Their tissue labels, prototype-bank organization, routing
rules, and visualization palettes are specific to these cancer types.

Evaluation on another cancer type requires either:

1. a cancer-specific tissue-category prototype bank with matching tissue
   labels, routing rules, and palette; or
2. tissue analysis to be disabled.

Run PlanPath without the tissue-analysis tool:

```bash
python scripts/run_planpath.py \
  --config configs/generic_no_tissue.json \
  --no-tissue-navigation \
  --overwrite
```

This command disables tissue-map loading and tissue-guided ROI selection.
