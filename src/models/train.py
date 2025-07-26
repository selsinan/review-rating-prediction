import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import yaml
import sys
import os
from data.etl import load_goodreads_data, filter_children_books, preprocess_features
from features.build_features import FeatureBuilder

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def train_model(
    data_path: str, config_path: str, experiment_name: str = "children-books-rating"
):
    """Train rating prediction model with MLflow tracking."""

    # Load configuration
    config = load_config(config_path)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config["model"])
        mlflow.log_param("data_path", data_path)

        print("Loading and preprocessing data...")

        # Load and preprocess data
        df = load_goodreads_data(data_path, sample_size=config.get("sample_size"))
        print(f"Loaded {len(df)} total records")

        df = filter_children_books(df)
        print(f"Filtered to {len(df)} children's books")

        df = preprocess_features(df)
        print("Preprocessed features")

        # Remove rows with missing target
        initial_len = len(df)
        df = df.dropna(subset=["average_rating"])
        print(f"Removed {initial_len - len(df)} rows with missing ratings")

        # Filter for reasonable ratings (1-5)
        df = df[(df["average_rating"] >= 1) & (df["average_rating"] <= 5)]
        print(f"Final dataset size: {len(df)} books")

        mlflow.log_metric("total_samples", len(df))

        if len(df) < 100:
            raise ValueError("Not enough data samples after filtering")

        # Build features
        print("Building enhanced features...")
        feature_builder = FeatureBuilder()
        X = feature_builder.build_features(df, is_training=True)
        y = df["average_rating"]

        print(f"Built feature matrix: {X.shape}")

        # Split data
        test_size = config.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("num_features", X.shape[1])

        # Initialize model
        model_type = config["model"]["type"]
        model_params = config["model"]["params"]

        print(f"Training {model_type} model...")

        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(**model_params)
        elif model_type == "random_forest":
            model = RandomForestRegressor(**model_params)
        elif model_type == "ridge":
            model = Ridge(**model_params)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed. Run: pip install xgboost")
            model = xgb.XGBRegressor(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        model.fit(X_train, y_train)

        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        mlflow.log_metric("cv_rmse_mean", np.sqrt(-cv_scores.mean()))
        mlflow.log_metric("cv_rmse_std", np.sqrt(cv_scores.std()))

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate
        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)

        # Log metrics
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)

        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            # Save feature importance as artifact
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

            print("\nTop 15 Most Important Features:")
            print(feature_importance.head(15))

        # Save model and preprocessors
        print("Saving model and preprocessors...")

        # Log model based on type
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Save feature builder
        feature_builder.save_preprocessors("preprocessors")
        mlflow.log_artifacts("preprocessors", "preprocessors")

        # Log model info
        print("\nModel Training Complete!")
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"Test MAE: {test_metrics['mae']:.4f}")
        print(f"Test R²: {test_metrics['r2']:.4f}")

        # Performance interpretation - UPDATED THRESHOLDS
        if test_metrics["r2"] > 0.25:
            print("🎉 OUTSTANDING! Exceptional model performance!")
        elif test_metrics["r2"] > 0.20:
            print("🌟 EXCELLENT! Very strong model performance!")
        elif test_metrics["r2"] > 0.15:
            print("✅ GREAT! Good model performance!")
        elif test_metrics["r2"] > 0.10:
            print("📈 GOOD! Solid performance - room for improvement")
        elif test_metrics["r2"] > 0.05:
            print("⚠️ MODERATE! Consider advanced feature engineering")
        else:
            print("❌ POOR! Major improvements needed")

        return model, feature_builder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/goodreads_books_children.json")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--experiment", default="children-books-rating")

    args = parser.parse_args()

    train_model(args.data, args.config, args.experiment)
