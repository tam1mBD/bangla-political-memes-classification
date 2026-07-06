# Bangla Political Memes Classification

> **A production-grade, multi-modal stacking ensemble system for detecting and classifying political content in Bangla memes — combining OCR, NLP, and Vision (CLIP) models into a FastAPI-powered REST API, tracked with MLflow and DVC.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Model Performance](#model-performance)
- [System Architecture](#system-architecture)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [FastAPI Inference Server](#fastapi-inference-server)
- [Configuration](#configuration)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [DVC Pipeline Reproducibility](#dvc-pipeline-reproducibility)
- [Research Notebooks](#research-notebooks)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project implements an **end-to-end machine learning system** to classify Bangla meme images as **Political** or **Non-Political**. It addresses the challenge of Bangla meme content moderation by combining:

- **OCR-based text extraction** from meme images (EasyOCR)
- **Bangla text preprocessing** and political keyword mining
- **Logistic Regression** and **Deep Neural Network** text classifiers
- **Fine-tuned CLIP vision-language model** for image understanding
- **Stacking Ensemble** meta-classifier that fuses predictions from all three base models
- **FastAPI REST API** with a web UI for real-time meme classification
- **MLflow + DagHub** for experiment tracking and model registry
- **DVC** for pipeline reproducibility and data versioning

| Category | Details |
|---|---|
| **Project Type** | ML Classification (Binary: Political / Non-Political) |
| **Target Language** | Bangla (Bengali) |
| **Target Users** | Researchers, data scientists, content moderation teams |
| **Status** | ✅ Trained & Evaluated — Production-ready inference API |

---

## Model Performance

Evaluated on the held-out test set. Metrics tracked via MLflow at DagHub.

| Metric | Score |
|---|---|
| **Accuracy** | **87.26%** |
| **Precision** | **90.84%** |
| **Recall** | **83.50%** |
| **F1 Score** | **87.01%** |

---

## System Architecture

```
Meme Image
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION                           │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐ │
│  │  EasyOCR (Text)  │──▶│ Text Preprocessing│──▶│ Political Word  │ │
│  │  Text Extraction │   │ & Feature Mining  │   │ Keyword Detector│ │
│  └──────────────────┘   └──────────────────┘   └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
    │                            │                         │
    ▼                            ▼                         ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ Logistic         │   │ Deep Neural      │   │  Fine-tuned CLIP      │
│ Regression       │   │ Network (Text    │   │  Vision-Language      │
│ (Text Features)  │   │  Features)       │   │  Model (Image)        │
└────────┬─────────┘   └────────┬─────────┘   └──────────┬───────────┘
         │                      │                         │
         └──────────────────────┼─────────────────────────┘
                                ▼
                   ┌────────────────────────┐
                   │  Stacking Ensemble     │
                   │  (Logistic Regression  │
                   │   Meta-Classifier)     │
                   └────────────┬───────────┘
                                ▼
                   ┌────────────────────────┐
                   │  Political / Non-       │
                   │  Political Prediction  │
                   └────────────────────────┘
```

---

## Pipeline Stages

The full training pipeline consists of **10 sequential stages**, managed by DVC and executable via `main.py`:

| Stage | Name | Description |
|---|---|---|
| 01 | **Data Ingestion** | Downloads and extracts the meme classification dataset (Train/Test splits with image folders and CSV labels) |
| 02 | **Text Extraction** | Runs EasyOCR on all meme images to extract Bangla/English text; outputs CSV with raw extracted text |
| 03 | **Text Preprocessing** | Cleans and normalizes extracted Bangla text (removes noise, punctuation, normalizes Unicode) |
| 04 | **Political Word Detection** | Mines political-domain-specific keywords from training data to build a political vocabulary |
| 05 | **Text Analysis** | Engineers numerical features: keyword match count, keyword density ratio; produces feature CSVs |
| 06 | **Text Classification** | Trains a Logistic Regression classifier on text features; saves `model.joblib` |
| 07 | **Neural Network (Text)** | Trains a PyTorch deep neural network on text features; saves `model.pth` |
| 08 | **CLIP Model** | Fine-tunes a CLIP vision-language model on meme images; saves `clip_model.pth` |
| 09 | **Ensemble** | Creates stacking meta-features from all three base models; trains Logistic Regression meta-classifier; saves `meta_model.joblib` |
| 10 | **Model Evaluation** | Evaluates the full ensemble on the test set; logs accuracy, precision, recall, F1 to MLflow and `scores.json` |

---

## Project Structure

```
bangla-political-memes-classification/
│
├── src/memeClassifier/           # Main Python package
│   ├── __init__.py               # Logger setup
│   ├── components/               # Stage-level logic
│   │   ├── st_01_data_ingestion.py
│   │   ├── st_02_text_extraction.py
│   │   ├── st_03_text_preprocessing.py
│   │   ├── st_04_find_political_word.py
│   │   ├── st_05_text_analysis.py
│   │   ├── st_06_classification_model_using_text.py
│   │   ├── st_07_neural_network_model_using_text.py
│   │   ├── st_08_clip_model.py
│   │   ├── st_09_ensemble.py
│   │   └── st_10_model_evaluation_with_mlflow.py
│   ├── pipeline/                 # Pipeline runners
│   │   ├── stage_01_data_ingestion.py  ... stage_10_model_evaluation_with_mlflow.py
│   │   └── meme_prediction.py    # Inference pipeline for the API
│   ├── config/                   # Configuration manager
│   ├── constants/                # Project constants
│   ├── entity/                   # Data entity/dataclass definitions
│   └── utils/                    # Shared utility functions
│
├── config/
│   └── config.yaml               # All artifact paths and stage configs
│
├── research/                     # Exploratory Jupyter Notebooks (01–10)
│   ├── 01_data_ingestion.ipynb
│   ├── 02_text_extraction.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_find_political_word.ipynb
│   ├── 05_text_analysis.ipynb
│   ├── 06_classification_model_using_text.ipynb
│   ├── 07_neural_network_model_using_text.ipynb
│   ├── 08_clip_model.ipynb
│   ├── 09_ensemble.ipynb
│   └── 10_model_evaluation_with_mlflow.ipynb
│
├── templates/
│   └── index.html                # Web UI for the FastAPI server
│
├── artifacts/                    # Generated by the pipeline (gitignored)
│   ├── data_ingestion/
│   ├── text_extraction/
│   ├── text_preprocessing/
│   ├── find_political_word/
│   ├── text_analysis/
│   ├── classification_model_using_text/
│   ├── neural_network_model_using_text/
│   ├── clip_model/
│   └── ensemble/
│
├── app.py                        # FastAPI inference server
├── main.py                       # Full pipeline runner (all 10 stages)
├── params.yaml                   # Model hyperparameters
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC reproducibility lock
├── scores.json                   # Latest test evaluation metrics
├── setup.py                      # Package installation config
├── requirements.txt              # Python dependencies
└── .github/workflows/            # CI/CD workflows
```

---

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch, torchvision |
| **Vision-Language** | CLIP (`transformers`, Hugging Face) |
| **OCR** | EasyOCR |
| **ML / Classical** | scikit-learn, joblib |
| **Data Processing** | pandas, numpy |
| **Image Processing** | Pillow, OpenCV |
| **Visualization** | matplotlib, seaborn |
| **API Server** | FastAPI, Uvicorn |
| **Experiment Tracking** | MLflow, DagHub |
| **Pipeline / Data Versioning** | DVC |
| **Config Management** | PyYAML |
| **Logging** | Python `logging` |

---

## Installation

**Prerequisites:**
- Python 3.10+
- pip
- (Recommended) CUDA-enabled GPU for EasyOCR and CLIP training

### Step 1 — Clone the repository
```bash
git clone https://github.com/tam1mBD/bangla-political-memes-classification.git
cd bangla-political-memes-classification
```

### Step 2 — Create and activate a virtual environment
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the `memeClassifier` package in editable mode (`-e .`) automatically.

---

## Running the Pipeline

### Option A — Run all 10 stages at once via Python
```bash
PYTHONPATH=src python main.py
```

### Option B — Reproduce pipeline with DVC (recommended for reproducibility)
```bash
# Pull DVC-tracked data/artifacts from remote
dvc pull

# Reproduce only changed stages
dvc repro
```

### Option C — Run individual stages
```bash
PYTHONPATH=src python -m memeClassifier.pipeline.stage_01_data_ingestion
PYTHONPATH=src python -m memeClassifier.pipeline.stage_02_text_extraction
# ... and so on up to stage_10
```

---

## FastAPI Inference Server

The project ships with a **production-ready FastAPI server** (`app.py`) for real-time meme classification.

### Start the server
```bash
PYTHONPATH=src uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI (`templates/index.html`) or health check JSON |
| `POST` | `/predict` | Accepts a meme image upload and returns classification results |
| `GET` | `/docs` | Interactive Swagger UI for testing |

### Example `/predict` Response

```json
{
  "status": "success",
  "extracted_text": "বাংলাদেশের রাজনীতি...",
  "cleaned_text": "বাংলাদেশ রাজনীতি...",
  "metrics": {
    "keyword_matches": 5,
    "density_ratio": 0.3125
  },
  "prediction": {
    "label": "Political",
    "confidence_percentage": 94.72
  },
  "base_models_breakdown": {
    "logistic_regression": {"non_political": 0.12, "political": 0.88},
    "deep_neural_network": {"non_political": 0.08, "political": 0.92},
    "vision_clip_model": {"non_political": 0.15, "political": 0.85}
  }
}
```

### Supported Image Formats
`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

---

## Configuration

All paths and hyperparameters are centralized in two files:

### `config/config.yaml` — Artifact paths for each stage
```yaml
data_ingestion:
  source_URL: https://drive.google.com/file/d/...
  local_data_file: artifacts/data_ingestion/data.zip

clip_model:
  model_save_path: artifacts/clip_model/clip_model.pth
# ... and so on
```

### `params.yaml` — Model hyperparameters
```yaml
CLIPModel:
  BATCH_SIZE: 16
  EPOCHS: 10
  LEARNING_RATE: 1.0e-05
  WEIGHT_DECAY: 1.0e-04
  DROPOUT: 0.3

TextModel:
  BATCH_SIZE: 32
  EPOCHS: 100
  LEARNING_RATE: 0.001
  DROPOUT: 0.3

NeuralNetworkModelUsingText:
  EPOCHS: 50
  BATCH_SIZE: 16
  LEARNING_RATE: 0.001
  HIDDEN_DIM: 64

Ensemble:
  MAX_LENGTH: 128
  BATCH_SIZE: 16
  C: 1.0
  CLASS_WEIGHT: "balanced"
  SOLVER: "lbfgs"
  MAX_ITER: 1000
```

---



## DVC Pipeline Reproducibility

The `dvc.yaml` defines all 10 pipeline stages with their dependencies, outputs, and parameters. Any change to source code, config, or params is automatically detected.

```bash
# View pipeline DAG
dvc dag

# Check pipeline status
dvc status

# Reproduce changed stages
dvc repro
```

---

## Research Notebooks

Development notebooks are located in the [`research/`](research/) directory and mirror the 10 pipeline stages. They are useful for understanding the logic, exploring data, and prototyping new ideas before integrating into the production pipeline.

| Notebook | Purpose |
|---|---|
| `01_data_ingestion.ipynb` | Dataset download and exploration |
| `02_text_extraction.ipynb` | EasyOCR text extraction experiments |
| `03_text_preprocessing.ipynb` | Bangla text cleaning prototyping |
| `04_find_political_word.ipynb` | Political keyword mining |
| `05_text_analysis.ipynb` | Feature engineering exploration |
| `06_classification_model_using_text.ipynb` | Logistic Regression training |
| `07_neural_network_model_using_text.ipynb` | Neural network training |
| `08_clip_model.ipynb` | CLIP model fine-tuning |
| `09_ensemble.ipynb` | Stacking ensemble experiments |
| `10_model_evaluation_with_mlflow.ipynb` | Full evaluation and MLflow logging |

---

## Contributing

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`)
2. Commit your changes with clear messages (`git commit -m "feat: add X"`)
3. Ensure any new pipeline stage has a corresponding research notebook
4. Run `dvc repro` to verify end-to-end reproducibility before submitting
5. Open a pull request with a concise description of the change

---

## License

This project is licensed under the terms specified in [LICENSE](LICENSE).

---

<p align="center">
  Made with ❤️ for Bangla NLP and multimodal content moderation research
</p>