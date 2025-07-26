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
from features.llm_features import LLMFeatureEnhancer
from features.build_features import FeatureBuilder


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def validate_features(X, feature_name=""):
    """Validate that all features are numeric."""
    print(f"Validating {feature_name} features...")
    print(f"Shape: {X.shape}")

    # Convert each column to numeric individually
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)

    print(f"Final shape: {X.shape}")
    return X


def train_semantic_llm_model(
    data_path: str, experiment_name: str = "semantic-llm-prediction"
):
    """Train model with semantic + LLM features."""

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print("🚀 Training Semantic + LLM Model")
        print("=" * 60)

        # Load data
        df = load_goodreads_data(data_path, sample_size=8000)
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

        # Build semantic features
        print("\n🧠 Building semantic features...")
        semantic_builder = SemanticFeatureBuilder("all-MiniLM-L6-v2")
        semantic_features = semantic_builder.build_semantic_features(
            df, is_training=True
        )

        if not semantic_features.empty:
            semantic_features = validate_features(semantic_features, "semantic")
        else:
            print("❌ No semantic features generated!")
            return None, None, None, None

        # Build LLM features
        print("\n🤖 Building LLM features...")
        llm_enhancer = LLMFeatureEnhancer()
        llm_features = llm_enhancer.build_llm_features(df)
        llm_features.index = df.index
        llm_features = validate_features(llm_features, "LLM")

        # Create feature sets for comparison
        print("\n🔧 Combining features...")

        # 1. Traditional only
        X_traditional = traditional_features.copy()

        # 2. Traditional + Semantic
        semantic_features_aligned = semantic_features.reindex(
            traditional_features.index, fill_value=0
        )
        X_semantic = pd.concat(
            [traditional_features, semantic_features_aligned], axis=1
        )

        # 3. Traditional + Semantic + LLM
        llm_features_aligned = llm_features.reindex(
            traditional_features.index, fill_value=0
        )
        X_full = pd.concat(
            [traditional_features, semantic_features_aligned, llm_features_aligned],
            axis=1,
        )

        # Target variable
        y = df["average_rating"].reindex(traditional_features.index)

        # Final validation - simple approach
        print("\n🔍 Final feature validation...")

        # Convert all features to numeric using apply
        X_traditional = X_traditional.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_semantic = X_semantic.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_full = X_full.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Handle infinite values
        X_traditional = X_traditional.replace([np.inf, -np.inf], 0)
        X_semantic = X_semantic.replace([np.inf, -np.inf], 0)
        X_full = X_full.replace([np.inf, -np.inf], 0)

        print("\n📈 Feature Summary:")
        print(f"Traditional features: {X_traditional.shape[1]}")
        print(f"Semantic features: {semantic_features_aligned.shape[1]}")
        print(f"LLM features: {llm_features_aligned.shape[1]}")
        print(f"Total features: {X_full.shape[1]}")
        print(f"Target shape: {y.shape}")

        # Ensure aligned indices
        common_index = X_full.index.intersection(y.index)
        X_traditional = X_traditional.loc[common_index]
        X_semantic = X_semantic.loc[common_index]
        X_full = X_full.loc[common_index]
        y = y.loc[common_index]

        print(
            f"Final aligned shapes - Traditional: {X_traditional.shape}, Semantic: {X_semantic.shape}, Full: {X_full.shape}, y: {y.shape}"
        )

        # Log feature counts
        mlflow.log_metric("traditional_features", X_traditional.shape[1])
        mlflow.log_metric("semantic_features", semantic_features_aligned.shape[1])
        mlflow.log_metric("llm_features", llm_features_aligned.shape[1])
        mlflow.log_metric("total_features", X_full.shape[1])

        # Split data (same split for all)
        indices = X_full.index
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        X_trad_train, X_trad_test = (
            X_traditional.loc[train_idx],
            X_traditional.loc[test_idx],
        )
        X_sem_train, X_sem_test = X_semantic.loc[train_idx], X_semantic.loc[test_idx]
        X_full_train, X_full_test = X_full.loc[train_idx], X_full.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        print(f"\nTraining set: {X_full_train.shape}, Test set: {X_full_test.shape}")

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
        print("🔥 TRAINING AND COMPARING ALL FEATURE COMBINATIONS")
        print("=" * 60)

        for model_name, model in models.items():
            print(f"\n🎯 Training {model_name}...")

            try:
                # 1. Traditional only
                print("  📊 Training with traditional features...")
                model_trad = model.__class__(**model.get_params())
                model_trad.fit(X_trad_train, y_train)
                y_pred_trad = model_trad.predict(X_trad_test)
                r2_trad = r2_score(y_test, y_pred_trad)

                # 2. Traditional + Semantic
                print("  🧠 Training with semantic features...")
                model_sem = model.__class__(**model.get_params())
                model_sem.fit(X_sem_train, y_train)
                y_pred_sem = model_sem.predict(X_sem_test)
                r2_sem = r2_score(y_test, y_pred_sem)

                # 3. Traditional + Semantic + LLM
                print("  🤖 Training with all features...")
                model_full = model.__class__(**model.get_params())
                model_full.fit(X_full_train, y_train)
                y_pred_full = model_full.predict(X_full_test)
                r2_full = r2_score(y_test, y_pred_full)
                rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
                mae_full = mean_absolute_error(y_test, y_pred_full)

                # Calculate improvements
                semantic_improvement = r2_sem - r2_trad
                llm_improvement = r2_full - r2_sem
                total_improvement = r2_full - r2_trad

                print(f"\n  📈 {model_name} Results:")
                print(f"    Traditional only       - R²: {r2_trad:.4f}")
                print(
                    f"    + Semantic features    - R²: {r2_sem:.4f} (+{semantic_improvement:.4f})"
                )
                print(
                    f"    + LLM features         - R²: {r2_full:.4f} (+{llm_improvement:.4f})"
                )
                print(f"    📊 RMSE: {rmse_full:.4f}, MAE: {mae_full:.4f}")
                print(
                    f"    🚀 Total Improvement   - R²: +{total_improvement:.4f} ({total_improvement/abs(r2_trad)*100:+.1f}%)"
                )

                # Store results
                results[model_name] = {
                    "traditional_r2": r2_trad,
                    "semantic_r2": r2_sem,
                    "full_r2": r2_full,
                    "semantic_improvement": semantic_improvement,
                    "llm_improvement": llm_improvement,
                    "total_improvement": total_improvement,
                    "rmse": rmse_full,
                    "mae": mae_full,
                    "model": model_full,
                }

                # Log metrics
                mlflow.log_metric(f"{model_name}_traditional_r2", r2_trad)
                mlflow.log_metric(f"{model_name}_semantic_r2", r2_sem)
                mlflow.log_metric(f"{model_name}_full_r2", r2_full)
                mlflow.log_metric(
                    f"{model_name}_semantic_improvement", semantic_improvement
                )
                mlflow.log_metric(f"{model_name}_llm_improvement", llm_improvement)

                # Feature importance analysis
                if hasattr(model_full, "feature_importances_"):
                    feature_importance = pd.DataFrame(
                        {
                            "feature": X_full.columns,
                            "importance": model_full.feature_importances_,
                        }
                    ).sort_values("importance", ascending=False)

                    print(f"\n  🎯 Top 15 features for {model_name}:")
                    print(feature_importance.head(15).to_string(index=False))

                    # Analyze feature type importance
                    semantic_mask = feature_importance["feature"].str.contains(
                        "embedding|similar|semantic", case=False
                    )
                    llm_mask = feature_importance["feature"].str.contains(
                        "score|quality|complexity|is_", case=False
                    )

                    semantic_importance = feature_importance[semantic_mask][
                        "importance"
                    ].sum()
                    llm_importance = feature_importance[llm_mask]["importance"].sum()
                    traditional_importance = feature_importance[
                        ~(semantic_mask | llm_mask)
                    ]["importance"].sum()
                    total_importance = feature_importance["importance"].sum()

                    print("\n  📊 Feature Type Importance:")
                    print(
                        f"    Traditional: {traditional_importance:.4f} ({traditional_importance/total_importance*100:.1f}%)"
                    )
                    print(
                        f"    Semantic:    {semantic_importance:.4f} ({semantic_importance/total_importance*100:.1f}%)"
                    )
                    print(
                        f"    LLM:         {llm_importance:.4f} ({llm_importance/total_importance*100:.1f}%)"
                    )

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Find best performing model
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]["full_r2"])
            best_result = results[best_model_name]

            print("\n" + "=" * 60)
            print("🏆 FINAL RESULTS")
            print("=" * 60)
            print(f"Best model: {best_model_name}")
            print(f"Traditional R²: {best_result['traditional_r2']:.4f}")
            print(
                f"+ Semantic R²:  {best_result['semantic_r2']:.4f} (+{best_result['semantic_improvement']:.4f})"
            )
            print(
                f"+ LLM R²:       {best_result['full_r2']:.4f} (+{best_result['llm_improvement']:.4f})"
            )
            print(f"Total improvement: +{best_result['total_improvement']:.4f}")
            print(f"Final RMSE: {best_result['rmse']:.4f}")
            print(f"Final MAE: {best_result['mae']:.4f}")

            # Performance assessment
            if best_result["full_r2"] > 0.28:
                print("🎉 OUTSTANDING! This is exceptional performance!")
            elif best_result["full_r2"] > 0.25:
                print("🌟 EXCELLENT! Very strong performance!")
            elif best_result["full_r2"] > 0.20:
                print("✅ GREAT! Solid performance improvement!")
            else:
                print("📈 Good progress, room for improvement")

            # Log best results
            mlflow.log_metric("best_full_r2", best_result["full_r2"])
            mlflow.log_metric(
                "best_total_improvement", best_result["total_improvement"]
            )

            # Save best model
            mlflow.sklearn.log_model(best_result["model"], "best_full_model")

            print("=" * 60)

            return (
                best_result["model"],
                traditional_builder,
                semantic_builder,
                llm_enhancer,
            )

        else:
            print("❌ No models trained successfully")
            return None, None, None, None


if __name__ == "__main__":
    train_semantic_llm_model("data/raw/goodreads_books_children.json")
