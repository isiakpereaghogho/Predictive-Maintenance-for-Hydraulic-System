from src.models.modelling import ModellingPipeline
from src.logger import setup_logger

logging = setup_logger()


class TrainingPipeline:
    def __init__(self):
        self.modeling_pipeline = ModellingPipeline()

    def train(self):
        logging.info("Starting training pipeline...")

        self.modeling_pipeline.run()

        logging.info("Training pipeline completed successfully")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.train()

    