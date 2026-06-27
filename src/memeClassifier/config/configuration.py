import os
from memeClassifier.constants import *
from memeClassifier.utils.common import read_yaml, create_directories
from memeClassifier.entity.config_entity import (DataIngestionConfig,
                                                TextExtractionConfig,
                                                TextPreprocessingConfig,
                                                FindPoliticalWordConfig,
                                                TextAnalysisConfig,
                                                ClassificationModelUsingTextConfig,
                                                NeuralNetworkModelUsingTextConfig,
                                                ClipModelConfig,
                                                EnsembleConfig,
                                                ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_text_extraction_config(self) -> TextExtractionConfig:
        config = self.config.text_extraction

        create_directories([config.root_dir])

        text_extraction_config = TextExtractionConfig(
            root_dir=Path(config.root_dir),
            train_csv_path=Path(config.train_csv_path),
            train_image_folder=Path(config.train_image_folder),
            test_csv_path=Path(config.test_csv_path),
            test_image_folder=Path(config.test_image_folder),
            extracted_train_csv=Path(config.extracted_train_csv),
            extracted_test_csv=Path(config.extracted_test_csv)
        )

        return text_extraction_config
    
    def get_text_preprocessing_config(self) -> TextPreprocessingConfig:
        config = self.config.text_preprocessing

        create_directories([config.root_dir])

        text_preprocessing_config = TextPreprocessingConfig(
            root_dir=Path(config.root_dir),
            input_train_csv=Path(config.input_train_csv),
            input_test_csv=Path(config.input_test_csv),
            output_train_csv=Path(config.output_train_csv),
            output_test_csv=Path(config.output_test_csv)
        )

        return text_preprocessing_config
    
    def get_find_political_word_config(self) -> FindPoliticalWordConfig:
        config = self.config.find_political_word

        create_directories([config.root_dir])

        find_political_word_config = FindPoliticalWordConfig(
            root_dir=Path(config.root_dir),
            input_train_csv=Path(config.input_train_csv),
            output_csv=Path(config.output_csv)
        )

        return find_political_word_config
    
    def get_text_analysis_config(self) -> TextAnalysisConfig:
        config = self.config.text_analysis

        create_directories([config.root_dir])

        text_analysis_config = TextAnalysisConfig(
            root_dir=Path(config.root_dir),
            input_train_csv=Path(config.input_train_csv),
            input_test_csv=Path(config.input_test_csv),
            political_words_csv=Path(config.political_words_csv),
            output_train_features=Path(config.output_train_features),
            output_test_features=Path(config.output_test_features)
        )

        return text_analysis_config
    
    def get_classification_model_using_text_config(self) -> ClassificationModelUsingTextConfig:
        config = self.config.classification_model_using_text

        create_directories([config.root_dir])

        classification_config = ClassificationModelUsingTextConfig(
            root_dir=Path(config.root_dir),
            train_features_csv=Path(config.train_features_csv),
            test_features_csv=Path(config.test_features_csv),
            model_path=Path(config.model_path)
        )

        return classification_config
    
    def get_neural_network_model_using_text_config(self) -> NeuralNetworkModelUsingTextConfig:
        config = self.config.neural_network_model_using_text
        params = self.params.NeuralNetworkModelUsingText

        create_directories([config.root_dir])

        neural_network_config = NeuralNetworkModelUsingTextConfig(
            root_dir=Path(config.root_dir),
            train_features_csv=Path(config.train_features_csv),
            test_features_csv=Path(config.test_features_csv),
            model_path=Path(config.model_path),
            epochs=params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            learning_rate=params.LEARNING_RATE,
            hidden_dim=params.HIDDEN_DIM
        )

        return neural_network_config
    
    def get_clip_model_config(self) -> ClipModelConfig:
        config = self.config.clip_model

        create_directories([config.root_dir])

        clip_model_config = ClipModelConfig(
            root_dir=Path(config.root_dir),
            train_csv_path=Path(config.train_csv_path),
            train_img_dir=Path(config.train_img_dir),
            test_csv_path=Path(config.test_csv_path),
            test_img_dir=Path(config.test_img_dir),
            model_save_path=Path(config.model_save_path)
        )

        return clip_model_config
    
    def get_ensemble_config(self) -> EnsembleConfig:
        config = self.config.ensemble

        create_directories([config.root_dir])

        ensemble_config = EnsembleConfig(
            root_dir=Path(config.root_dir),
            train_csv_path=Path(config.train_csv_path),
            test_csv_path=Path(config.test_csv_path),
            train_features_csv=Path(config.train_features_csv),
            test_features_csv=Path(config.test_features_csv),
            train_img_dir=Path(config.train_img_dir),
            test_img_dir=Path(config.test_img_dir),
            logistic_regression_model_path=Path(config.logistic_regression_model_path),
            neural_network_model_path=Path(config.neural_network_model_path),
            clip_model_path=Path(config.clip_model_path),
            meta_model_path=Path(config.meta_model_path),
            c=self.params.Ensemble.C,
            class_weight=self.params.Ensemble.CLASS_WEIGHT if self.params.Ensemble.CLASS_WEIGHT != 'None' else None,
            solver=self.params.Ensemble.SOLVER,
            max_iter=self.params.Ensemble.MAX_ITER
        )

        return ensemble_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        model_evaluation_config = ModelEvaluationConfig(
            test_csv_path=Path(config.test_csv_path),
            test_features_csv=Path(config.test_features_csv),
            test_img_dir=Path(config.test_img_dir),
            logistic_regression_model_path=Path(config.logistic_regression_model_path),
            neural_network_model_path=Path(config.neural_network_model_path),
            clip_model_path=Path(config.clip_model_path),
            meta_model_path=Path(config.meta_model_path),
            mlflow_uri=config.mlflow_uri,
            all_params=self.params,
        )
        return model_evaluation_config