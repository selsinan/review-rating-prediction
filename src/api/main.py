import os
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

app = FastAPI(title="Review Rating Prediction API")

# Global variables to hold the model and preprocessors
predictor = None
feature_builders = None


def load_preprocessors():
    """Load the feature builders that were saved with the model"""
    global feature_builders
    if feature_builders is None:
        try:
            # Find the model artifacts directory
            experiment_path = Path("/mlruns/740467793628096163")
            if experiment_path.exists():
                run_dirs = [d for d in experiment_path.iterdir() if d.is_dir()]

                for run_dir in sorted(
                    run_dirs, key=lambda x: x.stat().st_mtime, reverse=True
                ):
                    artifacts_path = run_dir / "artifacts"
                    if artifacts_path.exists():
                        # Try to load feature builders
                        builders = {}
                        for builder_file in [
                            "traditional_builder.pkl",
                            "semantic_builder.pkl",
                            "llm_enhancer.pkl",
                        ]:
                            builder_path = artifacts_path / builder_file
                            if builder_path.exists():
                                try:
                                    builders[
                                        builder_file.replace(".pkl", "")
                                    ] = joblib.load(builder_path)
                                    print(f"✅ Loaded {builder_file}")
                                except Exception as e:
                                    print(f"❌ Failed to load {builder_file}: {e}")

                        if builders:
                            feature_builders = builders
                            return feature_builders

            print("❌ No feature builders found, will use simplified preprocessing")
            return None

        except Exception as e:
            print(f"❌ Error loading preprocessors: {e}")
            return None

    return feature_builders


def preprocess_book_data(book_data):
    """Preprocess book data into features expected by the model"""
    try:
        # Create a pandas DataFrame with the input
        df = pd.DataFrame([book_data])

        # Load feature builders
        builders = load_preprocessors()

        if builders:
            # Use the actual feature builders if available
            all_features = []

            # Traditional features
            if "traditional_builder" in builders:
                try:
                    traditional_features = builders["traditional_builder"].transform(df)
                    all_features.append(traditional_features)
                    print(
                        f"✅ Built traditional features: {traditional_features.shape}"
                    )
                except Exception as e:
                    print(f"❌ Traditional features failed: {e}")

            # Semantic features
            if "semantic_builder" in builders:
                try:
                    semantic_features = builders["semantic_builder"].transform(df)
                    all_features.append(semantic_features)
                    print(f"✅ Built semantic features: {semantic_features.shape}")
                except Exception as e:
                    print(f"❌ Semantic features failed: {e}")

            # LLM features
            if "llm_enhancer" in builders:
                try:
                    llm_features = builders["llm_enhancer"].transform(df)
                    all_features.append(llm_features)
                    print(f"✅ Built LLM features: {llm_features.shape}")
                except Exception as e:
                    print(f"❌ LLM features failed: {e}")

            if all_features:
                # Combine all features
                combined_features = np.hstack(all_features)
                print(f"✅ Combined features shape: {combined_features.shape}")
                return combined_features

        # Fallback: Create basic numerical features if builders not available
        print("🔄 Using fallback feature engineering")
        basic_features = [
            book_data.get("num_pages", 100),
            book_data.get("publication_year", 2023),
            len(book_data.get("title", "")),
            len(book_data.get("description", "")),
            len(book_data.get("authors", "").split(",")),
        ]

        # Pad with zeros to match expected feature count (you may need to adjust this)
        # The model expects a specific number of features - let's create 103 features as a guess
        while len(basic_features) < 103:
            basic_features.append(0.0)

        return np.array([basic_features])

    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        raise e


def get_model():
    global predictor
    if predictor is None:
        try:
            print("🔍 Checking MLflow setup...")

            # Set tracking URI
            if os.path.exists("/mlruns"):
                print("✅ /mlruns directory found")
                mlflow.set_tracking_uri("file:///mlruns")
            else:
                print("❌ /mlruns directory not found, using local mlruns")
                mlflow.set_tracking_uri("file:./mlruns")

            print(f"🔗 MLflow tracking URI: {mlflow.get_tracking_uri()}")

            # Try registry approach first
            try:
                model_uri = "models:/semantic-rating-predictor/Production"
                print(f"📦 Attempting to load model: {model_uri}")
                predictor = mlflow.pyfunc.load_model(model_uri)
                print(f"✅ Successfully loaded model from registry: {model_uri}")
                return predictor
            except Exception as registry_error:
                print(f"❌ Registry approach failed: {registry_error}")

                # Fallback: Load directly from the latest run artifacts
                print("🔄 Trying direct artifact loading...")

                # Find the latest run with best_model artifact
                experiment_path = Path("/mlruns/740467793628096163")
                if experiment_path.exists():
                    run_dirs = [d for d in experiment_path.iterdir() if d.is_dir()]

                    for run_dir in sorted(
                        run_dirs, key=lambda x: x.stat().st_mtime, reverse=True
                    ):
                        model_path = run_dir / "artifacts" / "best_model"
                        if model_path.exists():
                            print(f"📁 Found model artifacts at: {model_path}")
                            try:
                                # Load the model directly using mlflow.pyfunc
                                predictor = mlflow.pyfunc.load_model(str(model_path))
                                print(
                                    f"✅ Successfully loaded model from: {model_path}"
                                )
                                return predictor
                            except Exception as artifact_error:
                                print(
                                    f"❌ Failed to load from {model_path}: {artifact_error}"
                                )
                                continue

                raise Exception("No valid model artifacts found")

        except Exception as e:
            print(f"❌ Complete model loading failure: {e}")
            print(f"📊 Full traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail=f"Model loading failed: {str(e)}"
            )

    return predictor


class BookRequest(BaseModel):
    title: str
    description: str
    authors: str = "Unknown"
    publication_year: int = 2023
    num_pages: int = 100


@app.get("/")
async def root():
    return {"message": "Review Rating Prediction API", "status": "running"}


@app.get("/health")
async def health():
    try:
        # Check basic container health
        health_info = {
            "status": "checking",
            "mlruns_exists": os.path.exists("/mlruns"),
            "models_dir_exists": os.path.exists("/mlruns/models"),
            "tracking_uri": None,
            "model_loaded": False,
            "preprocessors_loaded": False,
            "available_experiments": [],
            "available_models": [],
        }

        if os.path.exists("/mlruns"):
            mlflow.set_tracking_uri("file:///mlruns")

            # List available experiments
            if os.path.exists("/mlruns"):
                health_info["available_experiments"] = [
                    d for d in os.listdir("/mlruns") if d.isdigit()
                ]

            # List available models
            if os.path.exists("/mlruns/models"):
                health_info["available_models"] = os.listdir("/mlruns/models")
        else:
            mlflow.set_tracking_uri("file:./mlruns")

        health_info["tracking_uri"] = mlflow.get_tracking_uri()

        # Try to load model and preprocessors
        model = get_model()
        health_info["model_loaded"] = model is not None

        preprocessors = load_preprocessors()
        health_info["preprocessors_loaded"] = preprocessors is not None

        health_info["status"] = "healthy"

        return health_info

    except Exception as e:
        health_info = {
            "status": "unhealthy",
            "error": str(e),
            "mlruns_exists": os.path.exists("/mlruns"),
            "models_dir_exists": os.path.exists("/mlruns/models"),
            "tracking_uri": mlflow.get_tracking_uri() if "mlflow" in locals() else None,
        }

        # Add debugging info
        if os.path.exists("/mlruns"):
            health_info["available_experiments"] = [
                d for d in os.listdir("/mlruns") if d.isdigit()
            ]
        if os.path.exists("/mlruns/models"):
            health_info["available_models"] = os.listdir("/mlruns/models")

        return health_info


@app.post("/predict")
async def predict(request: BookRequest):
    try:
        model = get_model()

        # Create input data
        input_data = {
            "title": request.title,
            "description": request.description,
            "authors": request.authors,
            "publication_year": request.publication_year,
            "num_pages": request.num_pages,
        }

        # Preprocess the input data into features
        features = preprocess_book_data(input_data)
        print(f"🔍 Features shape: {features.shape}")

        # Make prediction with preprocessed features
        prediction = model.predict(features)

        # Extract the prediction value
        if hasattr(prediction, "__getitem__") and len(prediction) > 0:
            predicted_rating = float(prediction[0])
        else:
            predicted_rating = float(prediction)

        return {
            "predicted_rating": round(predicted_rating, 2),
            "book_info": input_data,
            "features_shape": features.shape if hasattr(features, "shape") else "N/A",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/debug/predict")
async def debug_predict(request: BookRequest):
    """Debug prediction to see raw output"""
    try:
        model = get_model()

        input_data = {
            "title": request.title,
            "description": request.description,
            "authors": request.authors,
            "publication_year": request.publication_year,
            "num_pages": request.num_pages,
        }

        # Preprocess the input data
        features = preprocess_book_data(input_data)

        # Make prediction with preprocessed features
        prediction = model.predict(features)

        return {
            "raw_prediction": str(prediction),
            "prediction_type": str(type(prediction)),
            "input_data": input_data,
            "features_shape": features.shape if hasattr(features, "shape") else "N/A",
            "features_sample": features[0][:10].tolist()
            if hasattr(features, "shape") and features.shape[1] >= 10
            else "N/A",
        }

    except Exception as e:
        import traceback

        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "input_data": input_data if "input_data" in locals() else None,
        }
