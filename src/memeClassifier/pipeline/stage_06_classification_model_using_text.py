from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_06_classification_model_using_text import ClassificationModelUsingText
from memeClassifier import logger

STAGE_NAME = "Text Classification Stage"


class TextClassificationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        classification_config = config.get_classification_model_using_text_config()
        classifier = ClassificationModelUsingText(config=classification_config)
        classifier.initiate_model_training()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = TextClassificationPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
