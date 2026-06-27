from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_03_text_preprocessing import TextPreprocessing
from memeClassifier import logger

STAGE_NAME = "Text Preprocessing Stage"


class TextPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        text_preprocessing_config = config.get_text_preprocessing_config()
        text_preprocessing = TextPreprocessing(config=text_preprocessing_config)
        text_preprocessing.initiate_text_preprocessing()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = TextPreprocessingTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
