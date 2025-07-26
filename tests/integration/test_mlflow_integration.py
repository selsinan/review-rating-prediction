import pytest
import mlflow
import pandas as pd
import numpy as np
import json
from mlflow.tracking import MlflowClient

@pytest.fixture(scope="session")
def mlflow_client():
    """Provide MLflow client for tests."""
    return MlflowClient()

@pytest.fixture(scope="session") 
def model_name():
    """Provide model name for tests."""
    return "semantic-rating-predictor"

@pytest.fixture(scope="session")
def ensure_mlflow_model_exists(mlflow_client, model_name):
    """Ensure the MLflow model exists before running tests."""
    models = mlflow_client.search_registered_models()
    model_names = [m.name for m in models]
    
    if model_name not in model_names:
        pytest.skip(f"Model {model_name} not found in MLflow registry")


class TestMLflowModelIntegration:
    """Integration tests for MLflow model registry and inference."""
    
    @classmethod
    def setup_class(cls):
        """Setup for the test class."""
        cls.model_name = "semantic-rating-predictor"
        cls.client = MlflowClient()
    
    def test_model_registry_exists(self):
        """Test that the model is registered in MLflow."""
        models = self.client.search_registered_models()
        model_names = [m.name for m in models]
        
        assert self.model_name in model_names, f"Model {self.model_name} not found in registry"
    
    def test_model_loads_successfully(self):
        """Test that the registered model can be loaded."""
        model = mlflow.sklearn.load_model(f"models:/{self.model_name}/1")
        
        assert model is not None, "Model should load successfully"
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'n_features_in_'), "Model should have feature count"
    
    def test_feature_names_artifact_exists(self):
        """Test that feature names artifact exists and is loadable."""
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        assert len(versions) > 0, "Should have at least one model version"
        
        run_id = versions[0].run_id
        
        # Check if feature_columns.json exists
        artifacts = self.client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        
        assert "feature_columns.json" in artifact_names, "feature_columns.json should exist"
        
        # Load and validate feature names
        artifact_path = self.client.download_artifacts(run_id, "feature_columns.json")
        with open(artifact_path, 'r') as f:
            feature_names = json.load(f)
        
        assert isinstance(feature_names, list), "Feature names should be a list"
        assert len(feature_names) > 0, "Should have feature names"
        assert len(feature_names) == 511, "Should have 511 features"
    
    def test_model_prediction_with_correct_features(self):
        """Test model prediction with correctly named features."""
        # Load model
        model = mlflow.sklearn.load_model(f"models:/{self.model_name}/1")
        
        # Load feature names
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        run_id = versions[0].run_id
        artifact_path = self.client.download_artifacts(run_id, "feature_columns.json")
        
        with open(artifact_path, 'r') as f:
            feature_names = json.load(f)
        
        # Create test data with correct feature names
        n_samples = 3
        test_data = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, len(feature_names))),
            columns=feature_names
        )
        
        # Make predictions
        predictions = model.predict(test_data)
        
        assert len(predictions) == n_samples, "Should get predictions for all samples"
        assert all(isinstance(p, (int, float)) for p in predictions), "Predictions should be numeric"
        
        # Check reasonable range for book ratings
        assert all(0.5 <= p <= 5.5 for p in predictions), "Predictions should be in reasonable range"
    
    def test_model_performance_metrics(self):
        """Test that we can retrieve model performance metrics."""
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        run_id = versions[0].run_id
        
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        
        # Check that we have performance metrics
        assert 'best_r2' in metrics, "Should have R² metric"
        
        r2_score = metrics['best_r2']
        assert r2_score > 0.25, f"R² should be > 0.25, got {r2_score}"
        assert r2_score <= 1.0, f"R² should be <= 1.0, got {r2_score}"

@pytest.fixture
def sample_model():
    """Fixture to provide the loaded model for tests."""
    return mlflow.sklearn.load_model("models:/semantic-rating-predictor/1")

def test_model_is_production_ready(sample_model):
    """Test that the model meets production readiness criteria."""
    assert sample_model is not None
    assert hasattr(sample_model, 'n_features_in_')
    assert sample_model.n_features_in_ == 511