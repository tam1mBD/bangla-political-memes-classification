from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class TextExtractionConfig:
    root_dir: Path
    train_csv_path: Path
    train_image_folder: Path
    test_csv_path: Path
    test_image_folder: Path
    extracted_train_csv: Path
    extracted_test_csv: Path

@dataclass(frozen=True)
class TextPreprocessingConfig:
    root_dir: Path
    input_train_csv: Path
    input_test_csv: Path
    output_train_csv: Path
    output_test_csv: Path

@dataclass(frozen=True)
class FindPoliticalWordConfig:
    root_dir: Path
    input_train_csv: Path
    output_csv: Path

@dataclass(frozen=True)
class TextAnalysisConfig:
    root_dir: Path
    input_train_csv: Path
    input_test_csv: Path
    political_words_csv: Path
    output_train_features: Path
    output_test_features: Path

@dataclass(frozen=True)
class ClassificationModelUsingTextConfig:
    root_dir: Path
    train_features_csv: Path
    test_features_csv: Path
    model_path: Path

@dataclass(frozen=True)
class NeuralNetworkModelUsingTextConfig:
    root_dir: Path
    train_features_csv: Path
    test_features_csv: Path
    model_path: Path
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int

@dataclass(frozen=True)
class ClipModelConfig:
    root_dir: Path
    train_csv_path: Path
    train_img_dir: Path
    test_csv_path: Path
    test_img_dir: Path
    model_save_path: Path

@dataclass(frozen=True)
class EnsembleConfig:
    root_dir: Path
    train_csv_path: Path
    test_csv_path: Path
    train_features_csv: Path
    test_features_csv: Path
    train_img_dir: Path
    test_img_dir: Path
    logistic_regression_model_path: Path
    neural_network_model_path: Path
    clip_model_path: Path
    meta_model_path: Path
    c: float
    class_weight: str
    solver: str
    max_iter: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    test_csv_path: Path
    test_features_csv: Path
    test_img_dir: Path
    logistic_regression_model_path: Path
    neural_network_model_path: Path
    clip_model_path: Path
    meta_model_path: Path
    mlflow_uri: str
    all_params: dict