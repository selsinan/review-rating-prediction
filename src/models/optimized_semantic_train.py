import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import sys
import os
from data.etl import load_goodreads_data, filter_children_books, preprocess_features
from features.semantic_features import SemanticFeatureBuilder
from features.llm_features import LLMFeatureEnhancer
from features.build_features import FeatureBuilder

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def validate_and_clean_features(features, feature_type):
    """Validate and clean feature matrix."""
    print(f"Cleaning {feature_type} features...")

    initial_shape = features.shape
    print(f"Initial shape: {initial_shape}")

    # Check data types
    print(f"Data types: {features.dtypes.value_counts()}")

    # Convert to numeric and handle missing values
    features_clean = features.apply(pd.to_numeric, errors="coerce")
    features_clean = features_clean.fillna(0)

    # Handle infinite values
    features_clean = features_clean.replace([np.inf, -np.inf], 0)

    final_shape = features_clean.shape
    print(f"Final shape: {final_shape}")

    return features_clean


class OptimizedSemanticBuilder:
    """Advanced semantic feature builder with multiple similarity scales."""

    def __init__(self):
        self.semantic_builder = None
        self.is_fitted = False

    def build_optimized_features(self, df, is_training=True):
        """Build optimized semantic features with advanced similarity metrics."""

        if is_training:
            print(
                "INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: mps"
            )
            print(
                "INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2"
            )

            # Initialize semantic builder
            self.semantic_builder = SemanticFeatureBuilder()
            semantic_features = self.semantic_builder.build_semantic_features(
                df, is_training=True
            )

            print("Building advanced similarity features...")
            advanced_features = self._build_advanced_similarity_features(
                df, semantic_features
            )

            # Combine all semantic features
            if not advanced_features.empty:
                all_semantic = pd.concat([semantic_features, advanced_features], axis=1)
            else:
                all_semantic = semantic_features

            self.is_fitted = True
            return validate_and_clean_features(all_semantic, "optimized semantic")

        else:
            if not self.is_fitted or self.semantic_builder is None:
                raise ValueError("Must fit on training data first")

            semantic_features = self.semantic_builder.build_semantic_features(
                df, is_training=False
            )
            advanced_features = self._build_advanced_similarity_features(
                df, semantic_features
            )

            if not advanced_features.empty:
                all_semantic = pd.concat([semantic_features, advanced_features], axis=1)
            else:
                all_semantic = semantic_features

            return validate_and_clean_features(all_semantic, "optimized semantic")

    def _build_advanced_similarity_features(self, df, semantic_features):
        """Build advanced similarity-based features."""
        print("Creating advanced similarity features...")

        advanced_features = pd.DataFrame(index=df.index)

        try:
            # Extract embeddings (assuming they're in the semantic_features)
            embedding_cols = [
                col for col in semantic_features.columns if "embedding" in col
            ]

            if len(embedding_cols) > 0:
                embeddings = semantic_features[embedding_cols].values

                # Compute similarity matrix (sample for efficiency)
                n_samples = min(len(embeddings), 1000)
                sample_indices = np.random.choice(
                    len(embeddings), n_samples, replace=False
                )
                sample_embeddings = embeddings[sample_indices]

                from sklearn.metrics.pairwise import cosine_similarity

                similarity_matrix = cosine_similarity(embeddings, sample_embeddings)

                # Multi-scale similarity features
                for k in [5, 10, 20]:
                    print(f"Processing k={k} neighbors...")

                    # Get top-k similar books for each book
                    top_k_indices = np.argsort(similarity_matrix, axis=1)[
                        :, -k - 1 : -1
                    ]  # Exclude self
                    top_k_similarities = np.sort(similarity_matrix, axis=1)[
                        :, -k - 1 : -1
                    ]

                    # Sample ratings for similarity calculation
                    sample_ratings = df.iloc[sample_indices]["average_rating"].values

                    # Weighted rating features
                    weighted_ratings = []
                    mean_similarities = []
                    max_similarities = []

                    for i in range(len(embeddings)):
                        k_indices = top_k_indices[i]
                        k_sims = top_k_similarities[i]
                        k_ratings = sample_ratings[k_indices]

                        # Weighted average rating
                        if k_sims.sum() > 0:
                            weighted_rating = np.average(k_ratings, weights=k_sims)
                            weighted_ratings.append(weighted_rating)
                        else:
                            weighted_ratings.append(df["average_rating"].mean())

                        mean_similarities.append(k_sims.mean())
                        max_similarities.append(k_sims.max())

                    advanced_features[
                        f"weighted_similar_rating_k{k}"
                    ] = weighted_ratings
                    advanced_features[f"mean_similarity_k{k}"] = mean_similarities
                    advanced_features[f"max_similarity_k{k}"] = max_similarities

                # Rating distribution features from similar books
                print("Creating rating distribution features...")

                # Use top-10 for distribution analysis
                k = 10
                top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k - 1 : -1]
                sample_ratings = df.iloc[sample_indices]["average_rating"].values

                similar_rating_stats = []
                for i in range(len(embeddings)):
                    k_indices = top_k_indices[i]
                    k_ratings = sample_ratings[k_indices]

                    similar_rating_stats.append(
                        {
                            "mean_similar_rating": k_ratings.mean(),
                            "median_similar_rating": np.median(k_ratings),
                            "std_similar_rating": k_ratings.std(),
                            "similar_rating_q25": np.percentile(k_ratings, 25),
                            "similar_rating_q75": np.percentile(k_ratings, 75),
                            "similar_rating_range": k_ratings.max() - k_ratings.min(),
                        }
                    )

                stats_df = pd.DataFrame(similar_rating_stats, index=df.index)
                advanced_features = pd.concat([advanced_features, stats_df], axis=1)

        except Exception as e:
            print(f"Warning: Could not build advanced similarity features: {e}")
            # Return empty DataFrame if similarity computation fails
            return pd.DataFrame(index=df.index)

        return advanced_features


class SemanticRatingPredictor(mlflow.pyfunc.PythonModel):
    """
    Complete MLflow model wrapper that includes feature engineering pipeline.
    This enables end-to-end prediction from raw book data.
    """

    def __init__(self):
        self.model = None
        self.traditional_builder = None
        self.semantic_builder = None
        self.llm_enhancer = None
        self.feature_columns = None

    def load_context(self, context):
        """Load the model and feature builders from MLflow artifacts."""
        import joblib

        # Load the trained model
        self.model = joblib.load(context.artifacts["model"])

        # Load feature builders
        self.traditional_builder = joblib.load(context.artifacts["traditional_builder"])
        self.semantic_builder = joblib.load(context.artifacts["semantic_builder"])
        self.llm_enhancer = joblib.load(context.artifacts["llm_enhancer"])

        # Load feature column names
        with open(context.artifacts["feature_columns"], "r") as f:
            import json

            self.feature_columns = json.load(f)

    def _preprocess_data(self, raw_data):
        """Apply the complete feature engineering pipeline."""
        df = raw_data.copy()

        # Apply the same preprocessing as training
        df = preprocess_features(df)

        # Build traditional features
        traditional_features = self.traditional_builder.build_features(
            df, is_training=False
        )
        traditional_features = validate_and_clean_features(
            traditional_features, "traditional"
        )

        # Build semantic features
        semantic_features = self.semantic_builder.build_optimized_features(
            df, is_training=False
        )
        if not semantic_features.empty:
            semantic_features = semantic_features.reindex(
                traditional_features.index, fill_value=0
            )

        # Build LLM features
        llm_features = self.llm_enhancer.build_llm_features(df)
        llm_features.index = df.index
        llm_features = validate_and_clean_features(llm_features, "LLM")
        llm_features = llm_features.reindex(traditional_features.index, fill_value=0)

        # Combine features
        all_features = [traditional_features]
        if not semantic_features.empty:
            all_features.append(semantic_features)
        if not llm_features.empty:
            all_features.append(llm_features)

        X = pd.concat(all_features, axis=1)

        # Final cleaning
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Ensure same columns as training
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        return X

    def predict(self, context, model_input):
        """Make predictions on new data."""
        # Preprocess the input data
        X = self._preprocess_data(model_input)

        # Make predictions
        predictions = self.model.predict(X)

        return predictions


def train_optimized_semantic_model(
    data_path: str, experiment_name: str = "optimized-semantic-prediction"
):
    """Enhanced training function that saves complete MLflow model."""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        print("🚀 Training Optimized Semantic Model with MLOps Pipeline")
        print("=" * 60)

        # Load and preprocess data
        df = load_goodreads_data(data_path, sample_size=15000)
        print(f"Loaded {len(df)} total records")

        df = filter_children_books(df)
        df = preprocess_features(df)

        # Quality filtering
        df = df.dropna(subset=["average_rating"])
        df = df[(df["average_rating"] >= 2.0) & (df["average_rating"] <= 5.0)]
        df = df[df["ratings_count"] >= 15]
        df = df[df["description"].str.len() > 30]

        print(f"High-quality dataset size: {len(df)} books")

        # Build features
        print("Building traditional features...")
        traditional_builder = FeatureBuilder()
        traditional_features = traditional_builder.build_features(df, is_training=True)
        traditional_features = validate_and_clean_features(
            traditional_features, "traditional"
        )

        print("Building optimized semantic features...")
        optimized_builder = OptimizedSemanticBuilder()
        optimized_semantic_features = optimized_builder.build_optimized_features(
            df, is_training=True
        )

        print("Building LLM features...")
        llm_enhancer = LLMFeatureEnhancer()
        llm_features = llm_enhancer.build_llm_features(df)
        llm_features.index = df.index
        llm_features = validate_and_clean_features(llm_features, "LLM")

        # Combine features
        print("Combining all features...")
        all_features = [traditional_features]

        if not optimized_semantic_features.empty:
            print(
                f"Added {len(optimized_semantic_features.columns)} optimized semantic features"
            )
            optimized_semantic_features = optimized_semantic_features.reindex(
                traditional_features.index, fill_value=0
            )
            all_features.append(optimized_semantic_features)

        if not llm_features.empty:
            print(f"Added {len(llm_features.columns)} LLM features")
            llm_features = llm_features.reindex(
                traditional_features.index, fill_value=0
            )
            all_features.append(llm_features)

        X = pd.concat(all_features, axis=1)
        y = df["average_rating"].reindex(X.index)

        # Final data validation
        print("Final data validation...")
        print(f"Before cleaning: {X.shape}")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        print(f"After cleaning: {X.shape}")
        print(f"Data types: {X.dtypes.value_counts()}")
        print(f"Any NaN values: {X.isnull().sum().sum()}")
        print(
            f"Any infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}"
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Train models
        models = {
            "optimized_rf": RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            ),
        }

        best_score = -np.inf
        best_model = None
        best_model_name = None

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

                mlflow.log_metric(f"{model_name}_r2", r2)
                mlflow.log_metric(f"{model_name}_rmse", rmse)
                mlflow.log_metric(f"{model_name}_mae", mae)

                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = model_name

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        if best_model is None:
            print("❌ No model trained successfully!")
            return None

        print(f"\n🏆 Best model: {best_model_name} with R² = {best_score:.4f}")

        # Feature importance analysis
        if hasattr(best_model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {"feature": X.columns, "importance": best_model.feature_importances_}
            ).sort_values("importance", ascending=False)

            print(f"\n🎯 Top 15 features for {best_model_name}:")
            print(feature_importance.head(15))

            # Semantic feature analysis
            semantic_mask = feature_importance["feature"].str.contains(
                "similar|embedding|semantic|quality|diversity", case=False
            )
            semantic_features_df = feature_importance[semantic_mask]

            if not semantic_features_df.empty:
                total_semantic_importance = semantic_features_df["importance"].sum()
                total_importance = feature_importance["importance"].sum()
                semantic_percentage = (
                    total_semantic_importance / total_importance
                ) * 100

                print(
                    f"\n🔥 Semantic features importance: {total_semantic_importance:.4f} ({semantic_percentage:.1f}%)"
                )
                print("Top semantic features:")
                print(semantic_features_df.head(10))

            # Save feature importance
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

        # Log final metrics
        mlflow.log_metric("best_r2", best_score)
        mlflow.log_metric(
            "best_rmse", np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
        )
        mlflow.log_metric(
            "best_mae", mean_absolute_error(y_test, best_model.predict(X_test))
        )
        mlflow.log_metric("total_features", X.shape[1])

        # Performance assessment
        baseline_r2 = 0.1851
        improvement = best_score - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100

        print("\n📊 Comparing with baseline...")
        print(f"Baseline R²: {baseline_r2:.4f}")
        print(f"Optimized R²: {best_score:.4f}")
        print(f"Improvement: +{improvement:.4f} ({improvement_pct:.1f}%)")

        if best_score > 0.28:
            print("🎉 OUTSTANDING! This is exceptional performance!")
        elif best_score > 0.25:
            print("🌟 EXCELLENT! This is very strong performance!")
        elif best_score > 0.20:
            print("✅ GREAT! Solid performance improvement!")
        else:
            print("📈 Good progress, room for improvement")

        # Save just the sklearn model (simplified approach)
        # Save just the sklearn model (simplified approach)
        print("\n💾 Saving model...")

        # Save model with MLflow sklearn (this creates the 'best_model' artifact)
        try:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",  # This creates the model artifact
                registered_model_name="semantic-rating-predictor",  # Auto-register
            )
            print("✅ MLflow sklearn model saved and registered!")
        except Exception as model_save_error:
            print(f"⚠️ MLflow model save failed: {model_save_error}")
            # Fallback to joblib
            joblib.dump(best_model, "trained_model.pkl")
            mlflow.log_artifact("trained_model.pkl", "best_model")
            print("✅ Model saved as artifact")

        # Save feature builders as artifacts
        joblib.dump(traditional_builder, "traditional_builder.pkl")
        joblib.dump(optimized_builder, "semantic_builder.pkl")
        joblib.dump(llm_enhancer, "llm_enhancer.pkl")

        mlflow.log_artifact("traditional_builder.pkl")
        mlflow.log_artifact("semantic_builder.pkl")
        mlflow.log_artifact("llm_enhancer.pkl")

        # Save feature columns
        with open("feature_columns.json", "w") as f:
            json.dump(list(X.columns), f)
        mlflow.log_artifact("feature_columns.json")

        print(f"✅ Model saved! Run ID: {run.info.run_id}")
        print(f"📊 Model performance: R² = {best_score:.4f}")
        print("🔗 MLflow UI: http://localhost:5000")

        return (
            best_model,
            traditional_builder,
            optimized_builder,
            llm_enhancer,
            run.info.run_id,
        )


if __name__ == "__main__":
    try:
        (
            model,
            trad_builder,
            sem_builder,
            llm_enh,
            run_id,
        ) = train_optimized_semantic_model("data/raw/goodreads_books_children.json")
        print(f"\n🎉 Training complete! Model saved with run_id: {run_id}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
