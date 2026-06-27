from memeClassifier import logger
from memeClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from memeClassifier.pipeline.stage_02_text_extraction import TextExtractionTrainingPipeline
from memeClassifier.pipeline.stage_03_text_preprocessing import TextPreprocessingTrainingPipeline
from memeClassifier.pipeline.stage_04_find_political_word import PoliticalWordPipeline
from memeClassifier.pipeline.stage_05_text_analysis import TextAnalysisPipeline
from memeClassifier.pipeline.stage_06_classification_model_using_text import TextClassificationPipeline
from memeClassifier.pipeline.stage_07_neural_network_model_using_text import NeuralNetworkTrainingPipeline
from memeClassifier.pipeline.stage_08_clip_model import ClipModelPipeline
from memeClassifier.pipeline.stage_09_ensemble import EnsembleTrainingPipeline
from memeClassifier.pipeline.stage_10_model_evaluation_with_mlflow import ModelEvaluationPipeline


STAGES = [
    ("Data Ingestion", DataIngestionTrainingPipeline),
    ("Text Extraction", TextExtractionTrainingPipeline),
    ("Text Preprocessing", TextPreprocessingTrainingPipeline),
    ("Political Word Detection", PoliticalWordPipeline),
    ("Text Analysis", TextAnalysisPipeline),
    ("Text Classification", TextClassificationPipeline),
    ("Neural Network Training", NeuralNetworkTrainingPipeline),
    ("CLIP Model Training", ClipModelPipeline),
    ("Ensemble Training", EnsembleTrainingPipeline),
    ("Model Evaluation", ModelEvaluationPipeline),
]


def run_pipeline():
    for stage_name, pipeline_cls in STAGES:
        logger.info(f">>>>> stage {stage_name} started <<<<<")
        pipeline = pipeline_cls()
        pipeline.main()
        logger.info(f">>>>> stage {stage_name} completed <<<<<\n\nx==========x")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.exception(e)
        raise e