import mlflow
import mlflow.pyfunc
import pandas as pd
import os


class SemanticRatingInference:
    """Easy-to-use interface for the semantic rating prediction model."""

    def __init__(self, model_uri=None, run_id=None):
        """
        Initialize the inference engine.

        Args:
            model_uri: Full MLflow model URI
            run_id: MLflow run ID (will construct URI automatically)
        """
        if model_uri:
            self.model_uri = model_uri
        elif run_id:
            self.model_uri = f"runs:/{run_id}/semantic_rating_predictor"
        else:
            # Use latest model from Model Registry
            self.model_uri = "models:/semantic-rating-predictor/latest"

        print(f"Loading model from: {self.model_uri}")
        self.model = mlflow.pyfunc.load_model(self.model_uri)
        print("✅ Model loaded successfully!")

    def predict_single_book(
        self,
        title,
        description,
        authors=None,
        publication_year=None,
        num_pages=None,
        ratings_count=10,
        text_reviews_count=5,
    ):
        """
        Predict rating for a single book.

        Args:
            title: Book title
            description: Book description
            authors: Author name(s)
            publication_year: Year of publication
            num_pages: Number of pages
            ratings_count: Existing ratings count
            text_reviews_count: Existing reviews count

        Returns:
            Predicted rating (float)
        """

        # Create a DataFrame with the book data
        book_data = pd.DataFrame(
            {
                "title": [title],
                "description": [description],
                "authors": [authors or "Unknown"],
                "publication_year": [publication_year or 2020],
                "num_pages": [num_pages or 100],
                "ratings_count": [ratings_count],
                "text_reviews_count": [text_reviews_count],
                "average_rating": [3.5],  # Placeholder, not used for prediction
            }
        )

        # Make prediction
        prediction = self.model.predict(book_data)
        return float(prediction[0])

    def predict_batch(self, books_df):
        """
        Predict ratings for multiple books.

        Args:
            books_df: DataFrame with book data

        Returns:
            Array of predicted ratings
        """
        predictions = self.model.predict(books_df)
        return predictions

    def predict_with_confidence(self, title, description, **kwargs):
        """
        Predict rating with confidence metrics.

        Returns:
            Dict with prediction, confidence level, and explanation
        """
        prediction = self.predict_single_book(title, description, **kwargs)

        # Simple confidence based on prediction range
        if prediction < 2.5:
            confidence = "Low-rated book predicted"
        elif prediction > 4.0:
            confidence = "High-rated book predicted"
        else:
            confidence = "Average rating predicted"

        return {
            "predicted_rating": round(prediction, 2),
            "confidence": confidence,
            "explanation": f"Based on semantic content analysis, this book is predicted to receive {prediction:.2f} stars",
        }


def load_latest_model():
    # Set MLflow tracking URI for container environment
    if os.path.exists("/mlruns"):
        mlflow.set_tracking_uri("file:///mlruns")
    else:
        # Fallback for local development
        mlflow.set_tracking_uri("file:./mlruns")

    try:
        model_uri = "models:/semantic-rating-predictor/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading the latest version directly
        try:
            model_uri = "models:/semantic-rating-predictor/1"
            model = mlflow.pyfunc.load_model(model_uri)
            return model
        except Exception as e2:
            print(f"Error loading model version 1: {e2}")
            raise e2


def demo_predictions():
    """Demo the model with example books."""

    print("🔮 SEMANTIC RATING PREDICTION DEMO")
    print("=" * 50)

    # Load the model
    predictor = load_latest_model()

    # Test books
    test_books = [
        {
            "title": "The Very Hungry Caterpillar",
            "description": "A beautiful picture book about a caterpillar who eats through various foods before transforming into a butterfly. Perfect for teaching counting and days of the week.",
            "authors": "Eric Carle",
            "publication_year": 1969,
            "num_pages": 32,
        },
        {
            "title": "Magic School Bus: Inside the Human Body",
            "description": "Ms. Frizzle takes her class on an educational adventure through the human body. Learn about organs, blood, and how our bodies work in this fun science book.",
            "authors": "Joanna Cole",
            "publication_year": 1989,
            "num_pages": 48,
        },
        {
            "title": "A Bad Day for Everyone",
            "description": "A poorly written story with confusing plot and boring characters. Nothing interesting happens and the ending makes no sense.",
            "authors": "Unknown Author",
            "publication_year": 2023,
            "num_pages": 20,
        },
    ]

    for i, book in enumerate(test_books, 1):
        print(f"\n📚 Book #{i}: {book['title']}")
        print(f"Author: {book['authors']}")
        print(f"Description: {book['description'][:100]}...")

        result = predictor.predict_with_confidence(**book)

        print(f"🎯 Predicted Rating: {result['predicted_rating']}/5.0")
        print(f"💡 {result['confidence']}")
        print(f"📝 {result['explanation']}")


if __name__ == "__main__":
    demo_predictions()
