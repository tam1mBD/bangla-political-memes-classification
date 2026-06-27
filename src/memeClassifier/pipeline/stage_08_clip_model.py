from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_08_clip_model import ClipModelTraining
from memeClassifier import logger

STAGE_NAME = "CLIP Model Training Stage"


class ClipModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        clip_model_config = config.get_clip_model_config()
        clip_model_training = ClipModelTraining(config=clip_model_config)
        clip_model_training.initiate_clip_model_training()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = ClipModelPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
