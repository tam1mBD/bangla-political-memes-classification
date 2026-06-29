from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_07_neural_network_model_using_text import NeuralNetworkModelUsingText
from memeClassifier import logger

STAGE_NAME = "Neural Network Training Stage"


class NeuralNetworkTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        neural_network_config = config.get_neural_network_model_using_text_config()
        classifier = NeuralNetworkModelUsingText(config=neural_network_config)
        classifier.initiate_model_training()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = NeuralNetworkTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
