from prefect import flow, task

from models.optimized_semantic_train import train_optimized_semantic_model
from models.model_registry import register_model


@task
def train(data_path: str):
    result = train_optimized_semantic_model(data_path)
    best_model = result[0]  # Unpack best_model from returned tuple
    return best_model


@flow
def ml_pipeline(
    data_path: str = "data/raw/goodreads_books_children.json",
):
    model = train(data_path)
    register_model(model, "semantic-rating-predictor")


if __name__ == "__main__":
    ml_pipeline()
