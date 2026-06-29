from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_05_text_analysis import TextAnalysis
from memeClassifier import logger

STAGE_NAME = "Text Analysis Stage"


class TextAnalysisPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        text_analysis_config = config.get_text_analysis_config()
        text_analysis = TextAnalysis(config=text_analysis_config)
        text_analysis.initiate_text_analysis()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = TextAnalysisPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
