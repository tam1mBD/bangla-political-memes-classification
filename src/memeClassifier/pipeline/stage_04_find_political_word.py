from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_04_find_political_word import FindPoliticalWord
from memeClassifier import logger

STAGE_NAME = "Political Word Detection Stage"


class PoliticalWordPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        find_political_word_config = config.get_find_political_word_config()
        find_political_word = FindPoliticalWord(config=find_political_word_config)
        find_political_word.process()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = PoliticalWordPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
