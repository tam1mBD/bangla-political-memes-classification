from memeClassifier.config.configuration import ConfigurationManager
from memeClassifier.components.st_10_model_evaluation_with_mlflow import ModelEvaluation
from memeClassifier import logger

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        
        # Get NN hidden dim from params
        nn_hidden_dim = config.params.NeuralNetworkModelUsingText.HIDDEN_DIM
        
        evaluation = ModelEvaluation(eval_config, nn_hidden_dim=nn_hidden_dim)
        evaluation.evaluate()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = ModelEvaluationPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
