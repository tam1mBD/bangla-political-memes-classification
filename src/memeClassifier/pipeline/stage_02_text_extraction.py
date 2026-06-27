from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_02_text_extraction import TextExtraction
from memeClassifier import logger

STAGE_NAME = "Text Extraction Stage"

class TextExtractionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        text_extraction_config = config.get_text_extraction_config()
        text_extraction = TextExtraction(config=text_extraction_config)
        text_extraction.initiate_text_extraction()

if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = TextExtractionTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e