from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_09_ensemble import EnsembleModel
from memeClassifier import logger

STAGE_NAME = "Ensemble Training Stage"


class EnsembleTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        ensemble_config = config.get_ensemble_config()
        
        # Needs to match NeuralNetwork config HIDDEN_DIM
        params = config.params.NeuralNetworkModelUsingText
        nn_hidden_dim = params.HIDDEN_DIM
        
        ensemble = EnsembleModel(config=ensemble_config, nn_hidden_dim=nn_hidden_dim)
        ensemble.initiate_ensemble_training()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = EnsembleTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
