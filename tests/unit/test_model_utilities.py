import pytest
import pandas as pd
import numpy as np

def create_realistic_book_features(feature_names, n_samples=1):
    """Utility function to create realistic book features."""
    test_data = pd.DataFrame(index=range(n_samples), columns=feature_names)
    test_data = test_data.fillna(0.0).infer_objects(copy=False)  # Fix pandas warning
    
    np.random.seed(42)
    
    for col in feature_names:
        if 'num_pages' in col:
            test_data[col] = np.random.normal(150, 50, n_samples)
        elif 'ratings_count' in col:
            test_data[col] = np.random.exponential(1000, n_samples)
        elif 'score' in col:
            test_data[col] = np.random.uniform(2.0, 5.0, n_samples)
        else:
            test_data[col] = np.random.normal(0, 0.5, n_samples)
    
    return test_data

class TestModelUtils:
    """Unit tests for model utility functions."""
    
    def test_create_realistic_book_features(self):
        """Test feature creation utility."""
        feature_names = ['num_pages', 'ratings_count', 'some_score', 'other_feature']
        
        result = create_realistic_book_features(feature_names, n_samples=2)
        
        assert result.shape == (2, 4)
        assert list(result.columns) == feature_names
        assert not result.isnull().any().any()
        
        # Check realistic ranges
        assert result['num_pages'].min() > 0
        assert result['ratings_count'].min() >= 0
        assert 2.0 <= result['some_score'].min() <= 5.0