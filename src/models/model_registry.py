import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


def register_model(model, model_name):
    """
    Register a model artifact to MLflow Model Registry.

    Args:
        model: Trained model object or model URI.
        model_name: Name to register the model under.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # If model is a string, treat as URI; otherwise, log and register
    if isinstance(model, str):
        model_uri = model
    else:
        # Log model to MLflow and get URI
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, "model")
            model_uri = f"runs:/{run.info.run_id}/model"

    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"✅ Model registered as '{model_name}' version {model_version.version}")

    # Optionally promote to Production
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"🚀 Model version {model_version.version} transitioned to Production")
    return model_version


def list_run_artifacts_safely(run_id):
    """Safely list artifacts for a run."""
    try:
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        return [artifact.path for artifact in artifacts]
    except Exception as e:
        print(f"⚠️ Error listing artifacts for {run_id}: {e}")
        return []


def register_best_model(
    experiment_name="optimized-semantic-prediction",
    model_name="semantic-rating-predictor",
):
    """Register the best model in MLflow Model Registry."""

    client = MlflowClient()

    # Get the best run from the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Experiment '{experiment_name}' not found!")
        return None

    # Find the run with highest R²
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.best_r2 DESC"],
        max_results=1,
    )

    if len(runs) == 0:
        print("❌ No runs found!")
        return None

    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    r2_score = best_run["metrics.best_r2"]

    print("🏆 Best model found:")
    print(f"   Run ID: {run_id}")
    print(f"   R² Score: {r2_score:.4f}")

    # List available artifacts for this run
    artifact_names = list_run_artifacts_safely(run_id)
    print(f"📁 Available artifacts: {artifact_names}")

    # Find model artifact
    model_artifact = None
    for artifact in artifact_names:
        if "model" in artifact.lower():
            model_artifact = artifact
            break

    if model_artifact is None:
        print("❌ No model artifact found!")
        return None

    model_uri = f"runs:/{run_id}/{model_artifact}"
    print(f"🎯 Using model URI: {model_uri}")

    try:
        # Register new model version
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"✅ Model registered as '{model_name}' version {model_version.version}")

        # Add description using update_model_version
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Semantic-based children's book rating predictor. R² = {r2_score:.4f}",
        )

        # Transition to Production
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print(f"🚀 Model version {model_version.version} transitioned to Production")
        return model_version

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return None


def register_latest_good_model(min_r2=0.25):
    """Register the latest model with good performance from any experiment."""

    client = MlflowClient()

    print(f"🔍 Searching for models with R² > {min_r2}...")

    # Get all experiments
    experiments = client.search_experiments()
    all_good_runs = []

    for experiment in experiments:
        try:
            # Search for good runs in each experiment
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=50,
            )

            for _, run in runs.iterrows():
                r2_best = run.get("metrics.best_r2", 0) or 0
                r2_test = run.get("metrics.test_r2", 0) or 0
                r2_score = max(r2_best, r2_test)

                if r2_score > min_r2:
                    all_good_runs.append(
                        {
                            "run_id": run["run_id"],
                            "r2_score": r2_score,
                            "start_time": run["start_time"],
                        }
                    )

        except Exception as e:
            print(f"⚠️ Error searching experiment {experiment.name}: {e}")
            continue

    if not all_good_runs:
        print(f"❌ No models found with R² > {min_r2}")
        return None

    # Sort by R² score (best first)
    all_good_runs.sort(key=lambda x: x["r2_score"], reverse=True)

    print(f"🔍 Found {len(all_good_runs)} good models:")
    for run_info in all_good_runs[:5]:  # Show top 5
        print(f"   Run {run_info['run_id']}: R² = {run_info['r2_score']:.4f}")

    # Try to register the best model
    best_run = all_good_runs[0]
    run_id = best_run["run_id"]
    r2_score = best_run["r2_score"]

    # Check what model artifacts are available
    model_artifacts = list_run_artifacts_safely(run_id)
    model_artifacts = [a for a in model_artifacts if "model" in a.lower()]

    if not model_artifacts:
        print(f"❌ No model artifacts found in run {run_id}")
        return None

    model_artifact = model_artifacts[0]
    model_uri = f"runs:/{run_id}/{model_artifact}"

    print(f"🎯 Registering best model from artifact: {model_artifact}")
    print(f"📍 Model URI: {model_uri}")

    try:
        model_version = mlflow.register_model(
            model_uri=model_uri, name="semantic-rating-predictor"
        )

        # Add description
        client.update_model_version(
            name="semantic-rating-predictor",
            version=model_version.version,
            description=f"Best performing model with R² = {r2_score:.4f}",
        )

        # Transition to Production
        client.transition_model_version_stage(
            name="semantic-rating-predictor",
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print("✅ Model registered and promoted to Production!")
        print(f"📊 Performance: R² = {r2_score:.4f}")
        return model_version

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return None


def quick_register_latest():
    """Quick registration of the most recent good model."""

    print("🚀 Quick Model Registration")
    print("=" * 40)

    try:
        # Get the most recent runs from all experiments
        runs = mlflow.search_runs(
            experiment_ids=None, order_by=["start_time DESC"], max_results=20
        )

        print(f"🔍 Checking {len(runs)} recent runs...")

        for _, run in runs.iterrows():
            run_id = run["run_id"]
            r2_best = run.get("metrics.best_r2", 0) or 0
            r2_test = run.get("metrics.test_r2", 0) or 0
            r2_score = max(r2_best, r2_test)

            print(f"   {run_id[:8]}: R² = {r2_score:.4f}")

            if r2_score > 0.25:  # Good performance
                # Find model artifacts
                model_artifacts = list_run_artifacts_safely(run_id)
                model_artifacts = [a for a in model_artifacts if "model" in a.lower()]

                if model_artifacts:
                    model_artifact = model_artifacts[0]
                    model_uri = f"runs:/{run_id}/{model_artifact}"

                    print(f"🎯 Registering: {model_uri}")

                    try:
                        # Register
                        client = MlflowClient()
                        mv = mlflow.register_model(
                            model_uri, "semantic-rating-predictor"
                        )

                        # Promote to production
                        client.transition_model_version_stage(
                            name="semantic-rating-predictor",
                            version=mv.version,
                            stage="Production",
                            archive_existing_versions=True,
                        )

                        print(
                            f"✅ Success! Version {mv.version} in Production (R² = {r2_score:.4f})"
                        )
                        return mv

                    except Exception as e:
                        print(f"❌ Failed to register {run_id}: {e}")
                        continue

    except Exception as e:
        print(f"❌ Error in quick registration: {e}")

    print("❌ No suitable models found")
    return None


def manual_register_specific_run(run_id, r2_score):
    """Manually register a specific run."""

    print(f"🎯 Manual registration for run: {run_id}")

    client = MlflowClient()

    # List artifacts
    artifacts = list_run_artifacts_safely(run_id)
    print(f"📁 Artifacts: {artifacts}")

    # Find model artifacts
    model_artifacts = [a for a in artifacts if "model" in a.lower()]

    if not model_artifacts:
        print("❌ No model artifacts found!")
        return None

    model_artifact = model_artifacts[0]
    model_uri = f"runs:/{run_id}/{model_artifact}"

    try:
        print(f"🔧 Registering: {model_uri}")

        mv = mlflow.register_model(model_uri, "semantic-rating-predictor")

        # Add description
        client.update_model_version(
            name="semantic-rating-predictor",
            version=mv.version,
            description=f"Manually registered model with R² = {r2_score:.4f}",
        )

        # Promote to production
        client.transition_model_version_stage(
            name="semantic-rating-predictor",
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print(f"✅ Success! Version {mv.version} registered and promoted!")
        return mv

    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return None


def load_production_model(model_name="semantic-rating-predictor"):
    """Load the production model from registry."""

    model_uri = f"models:/{model_name}/Production"
    print(f"Loading production model: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Production model loaded as sklearn model!")
        return model
    except Exception as e:
        print(f"❌ Error loading as sklearn model: {e}")

        # Try loading as pyfunc
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Production model loaded as pyfunc model!")
            return model
        except Exception as pyfunc_error:
            print(f"❌ Error loading as pyfunc model: {pyfunc_error}")
            return None


def compare_model_versions(model_name="semantic-rating-predictor"):
    """Compare different versions of the model."""

    client = MlflowClient()

    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            print(f"❌ No versions found for model '{model_name}'")
            return

        print(f"📊 Model Versions for '{model_name}':")
        print("-" * 50)

        for version in versions:
            try:
                run = client.get_run(version.run_id)
                r2_score = run.data.metrics.get(
                    "best_r2", run.data.metrics.get("test_r2", "N/A")
                )

                print(f"Version {version.version}:")
                print(f"  Stage: {version.current_stage}")
                print(f"  R² Score: {r2_score}")
                print(f"  Run ID: {version.run_id}")
                print()
            except Exception as e:
                print(f"  Error getting info for version {version.version}: {e}")

    except Exception as e:
        print(f"❌ Error comparing versions: {e}")


if __name__ == "__main__":
    print("🚀 MLflow Model Registry Operations")
    print("=" * 50)

    # Try to register the best model from the optimized experiment
    model_version = register_best_model()

    if model_version is None:
        print("\n🔄 Trying to register latest good model...")
        model_version = register_latest_good_model()

    if model_version is None:
        print("\n🔄 Trying quick registration...")
        model_version = quick_register_latest()

    # If still no success, try manual registration of the best run we know about
    if model_version is None:
        print("\n🔧 Trying manual registration of best known run...")
        # Based on your output, this run has R² = 0.2830
        model_version = manual_register_specific_run(
            "33791454f5b84fad9027a21c4b5bb607", 0.2701
        )

    if model_version:
        print("\n" + "=" * 50)
        print("✅ MODEL REGISTRY OPERATIONS COMPLETE")
        print("=" * 50)

        # Compare versions
        compare_model_versions()

        # Test loading production model
        prod_model = load_production_model()

        if prod_model:
            print("🎉 MLOps pipeline setup complete!")
            print("You can now use the production model for inference.")
        else:
            print("⚠️ Model registered but loading failed")
    else:
        print("❌ Could not register any model")
