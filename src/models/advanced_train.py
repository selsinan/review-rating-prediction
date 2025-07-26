import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PowerTransformer
import sys
import os
from data.etl import load_goodreads_data, filter_children_books, preprocess_features
from features.advanced_features import AdvancedFeatureBuilder

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def train_advanced_model(
    data_path: str, experiment_name: str = "advanced-rating-prediction"
):
    """Train advanced model with sophisticated preprocessing."""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print("Loading and preprocessing data...")

        # Load more data for better performance
        df = load_goodreads_data(data_path, sample_size=25000)
        print(f"Loaded {len(df)} total records")

        df = filter_children_books(df)
        df = preprocess_features(df)

        # Remove outliers in target
        Q1 = df["average_rating"].quantile(0.25)
        Q3 = df["average_rating"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"Removing outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]")
        df = df[
            (df["average_rating"] >= lower_bound)
            & (df["average_rating"] <= upper_bound)
        ]

        # Remove rows with missing target
        df = df.dropna(subset=["average_rating"])
        df = df[(df["average_rating"] >= 1) & (df["average_rating"] <= 5)]

        print(f"Final dataset size: {len(df)} books")
        mlflow.log_metric("total_samples", len(df))

        # Advanced feature engineering
        feature_builder = AdvancedFeatureBuilder()
        X = feature_builder.build_features(df, is_training=True)
        y = df["average_rating"]

        # Target transformation
        target_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        y_transformed = target_transformer.fit_transform(
            y.values.reshape(-1, 1)
        ).ravel()

        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variance before transform: {y.var():.4f}")
        print(f"Target variance after transform: {y_transformed.var():.4f}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=0.2, random_state=42
        )

        # Try multiple models
        models = {
            "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                random_state=42,
            ),
        }

        best_score = -np.inf
        best_model = None
        best_model_name = None

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            model.fit(X_train, y_train)
            y_pred_transformed = model.predict(X_test)

            # Transform predictions back to original scale
            y_pred = target_transformer.inverse_transform(
                y_pred_transformed.reshape(-1, 1)
            ).ravel()
            y_test_original = target_transformer.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            # Evaluate
            r2 = r2_score(y_test_original, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
            mae = mean_absolute_error(y_test_original, y_pred)

            print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            # Log metrics
            mlflow.log_metric(f"{model_name}_r2", r2)
            mlflow.log_metric(f"{model_name}_rmse", rmse)
            mlflow.log_metric(f"{model_name}_mae", mae)

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = model_name

        print(f"\nBest model: {best_model_name} with R² = {best_score:.4f}")

        # Log best model
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.log_metric("best_r2", best_score)

        # Performance interpretation
        if best_score > 0.3:
            print("✅ Good model performance!")
        elif best_score > 0.15:
            print("⚠️ Moderate model performance")
        else:
            print("❌ Poor model performance")

        return best_model, feature_builder, target_transformer


if __name__ == "__main__":
    train_advanced_model("data/raw/goodreads_books_children.json")
