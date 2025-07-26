import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
from data.etl import load_goodreads_data, filter_children_books, preprocess_features
from features.semantic_features import SemanticFeatureBuilder
from features.build_features import FeatureBuilder

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def validate_features(X, feature_name=""):
    """Validate that all features are numeric."""
    print(f"Validating {feature_name} features...")
    print(f"Shape: {X.shape}")
    print(f"Data types: {X.dtypes.value_counts()}")

    # Check for non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"Non-numeric columns found: {list(non_numeric)}")
        # Convert to numeric
        for col in non_numeric:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Check for NaN values
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"NaN values found: {nan_count}")
        X = X.fillna(0)

    # Check for infinite values
    inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"Infinite values found: {inf_count}")
        X = X.replace([np.inf, -np.inf], 0)

    print(f"Final shape: {X.shape}")
    return X


def train_semantic_model(
    data_path: str, experiment_name: str = "semantic-only-prediction"
):
    """Train model with semantic features only (no LLM features)."""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print("🚀 Training Pure Semantic Model (No LLM Features)")
        print("=" * 60)

        # Load data
        df = load_goodreads_data(
            data_path, sample_size=8000
        )  # Reasonable size for testing
        print(f"Loaded {len(df)} total records")

        df = filter_children_books(df)
        df = preprocess_features(df)

        # Quality filtering
        df = df.dropna(subset=["average_rating"])
        df = df[(df["average_rating"] >= 2.0) & (df["average_rating"] <= 5.0)]
        df = df[df["ratings_count"] >= 10]
        df = df[df["description"].str.len() > 20]

        print(f"Final dataset size: {len(df)} books")

        # Build traditional features
        print("\n📊 Building traditional features...")
        traditional_builder = FeatureBuilder()
        traditional_features = traditional_builder.build_features(df, is_training=True)
        traditional_features = validate_features(traditional_features, "traditional")

        # Build semantic features ONLY
        print("\n🧠 Building semantic features...")
        semantic_builder = SemanticFeatureBuilder("all-MiniLM-L6-v2")
        semantic_features = semantic_builder.build_semantic_features(
            df, is_training=True
        )

        if not semantic_features.empty:
            semantic_features = validate_features(semantic_features, "semantic")
        else:
            print("❌ No semantic features generated!")
            return None, None, None

        # Create two feature sets for comparison
        # 1. Traditional only
        X_traditional = traditional_features.copy()

        # 2. Traditional + Semantic
        semantic_features = semantic_features.reindex(
            traditional_features.index, fill_value=0
        )
        X_combined = pd.concat([traditional_features, semantic_features], axis=1)

        # Target variable
        y = df["average_rating"].reindex(traditional_features.index)

        # Final validation
        X_traditional = validate_features(X_traditional, "traditional final")
        X_combined = validate_features(X_combined, "combined final")

        print("\n📈 Feature Summary:")
        print(f"Traditional features: {X_traditional.shape[1]}")
        print(f"Semantic features: {semantic_features.shape[1]}")
        print(f"Combined features: {X_combined.shape[1]}")
        print(f"Target shape: {y.shape}")

        # Ensure aligned indices
        common_index = X_combined.index.intersection(y.index)
        X_traditional = X_traditional.loc[common_index]
        X_combined = X_combined.loc[common_index]
        y = y.loc[common_index]

        print(
            f"Final aligned shapes - Traditional: {X_traditional.shape}, Combined: {X_combined.shape}, y: {y.shape}"
        )

        # Log feature counts
        mlflow.log_metric("traditional_features", X_traditional.shape[1])
        mlflow.log_metric("semantic_features", semantic_features.shape[1])
        mlflow.log_metric("total_features", X_combined.shape[1])

        # Split data
        X_trad_train, X_trad_test, y_train, y_test = train_test_split(
            X_traditional, y, test_size=0.2, random_state=42
        )

        X_comb_train, X_comb_test, _, _ = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )

        print(f"\nTraining set: {X_trad_train.shape}, Test set: {X_trad_test.shape}")

        # Models to test
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42
            ),
        }

        results = {}

        print("\n" + "=" * 60)
        print("🔥 TRAINING AND COMPARING MODELS")
        print("=" * 60)

        for model_name, model in models.items():
            print(f"\n🎯 Training {model_name}...")

            try:
                # 1. Train with traditional features only
                print("  📊 Traditional features only...")
                model_trad = model.__class__(**model.get_params())
                model_trad.fit(X_trad_train, y_train)
                y_pred_trad = model_trad.predict(X_trad_test)

                r2_trad = r2_score(y_test, y_pred_trad)
                rmse_trad = np.sqrt(mean_squared_error(y_test, y_pred_trad))
                mae_trad = mean_absolute_error(y_test, y_pred_trad)

                # 2. Train with traditional + semantic features
                print("  🧠 Traditional + Semantic features...")
                model_comb = model.__class__(**model.get_params())
                model_comb.fit(X_comb_train, y_train)
                y_pred_comb = model_comb.predict(X_comb_test)

                r2_comb = r2_score(y_test, y_pred_comb)
                rmse_comb = np.sqrt(mean_squared_error(y_test, y_pred_comb))
                mae_comb = mean_absolute_error(y_test, y_pred_comb)

                # Calculate improvement
                improvement = r2_comb - r2_trad
                improvement_pct = (
                    (improvement / abs(r2_trad)) * 100 if r2_trad != 0 else 0
                )

                print(f"\n  📈 {model_name} Results:")
                print(
                    f"    Traditional only  - R²: {r2_trad:.4f}, RMSE: {rmse_trad:.4f}, MAE: {mae_trad:.4f}"
                )
                print(
                    f"    With Semantic     - R²: {r2_comb:.4f}, RMSE: {rmse_comb:.4f}, MAE: {mae_comb:.4f}"
                )
                print(
                    f"    🚀 Improvement    - R²: +{improvement:.4f} ({improvement_pct:+.1f}%)"
                )

                # Store results
                results[model_name] = {
                    "traditional_r2": r2_trad,
                    "combined_r2": r2_comb,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct,
                    "model": model_comb,
                }

                # Log metrics
                mlflow.log_metric(f"{model_name}_traditional_r2", r2_trad)
                mlflow.log_metric(f"{model_name}_semantic_r2", r2_comb)
                mlflow.log_metric(f"{model_name}_improvement", improvement)
                mlflow.log_metric(f"{model_name}_improvement_pct", improvement_pct)

                # Feature importance analysis for combined model
                if hasattr(model_comb, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "feature": X_combined.columns,
                            "importance": model_comb.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)

                    print(f"\n  🎯 Top 10 features for {model_name}:")
                    print(feature_importance.head(10).to_string(index=False))

                    # Analyze semantic feature importance
                    semantic_mask = feature_importance["feature"].str.contains(
                        "embedding|similar|semantic|quality|diversity", case=False
                    )
                    semantic_importance = feature_importance[semantic_mask]

                    if not semantic_importance.empty:
                        total_semantic_imp = semantic_importance["importance"].sum()
                        total_imp = feature_importance["importance"].sum()
                        semantic_pct = (total_semantic_imp / total_imp) * 100

                        print("\n  🧠 Semantic Features Analysis:")
                        print(
                            f"    Total semantic importance: {total_semantic_imp:.4f} ({semantic_pct:.1f}%)"
                        )
                        print("    Top semantic features:")
                        print(semantic_importance.head(5).to_string(index=False))

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

        # Find best performing model
        if results:
            best_model_name = max(
                results.keys(), key=lambda x: results[x]["combined_r2"]
            )
            best_result = results[best_model_name]

            print("\n" + "=" * 60)
            print("🏆 FINAL RESULTS")
            print("=" * 60)
            print(f"Best model: {best_model_name}")
            print(f"Traditional R²: {best_result['traditional_r2']:.4f}")
            print(f"With Semantic R²: {best_result['combined_r2']:.4f}")
            print(
                f"Improvement: +{best_result['improvement']:.4f} ({best_result['improvement_pct']:+.1f}%)"
            )

            # Performance assessment
            if best_result["improvement"] > 0.05:
                print(
                    "🎉 EXCELLENT! Semantic features provide significant improvement!"
                )
            elif best_result["improvement"] > 0.02:
                print("✅ GOOD! Semantic features provide meaningful improvement!")
            elif best_result["improvement"] > 0:
                print("📈 MODEST! Semantic features provide some improvement!")
            else:
                print("❌ Semantic features didn't improve performance")

            # Log best results
            mlflow.log_metric("best_traditional_r2", best_result["traditional_r2"])
            mlflow.log_metric("best_semantic_r2", best_result["combined_r2"])
            mlflow.log_metric("best_improvement", best_result["improvement"])

            # Save best model
            mlflow.sklearn.log_model(best_result["model"], "best_semantic_model")

            print("=" * 60)

            return best_result["model"], traditional_builder, semantic_builder

        else:
            print("❌ No models trained successfully")
            return None, None, None


if __name__ == "__main__":
    train_semantic_model("data/raw/goodreads_books_children.json")
