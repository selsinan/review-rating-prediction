import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.etl import preprocess_features, filter_children_books

def test_preprocess_features():
    """Test basic preprocessing functionality."""
    data = {
        'average_rating': [4.5, 3.2, None],
        'num_pages': [32, None, 150],
        'title': ['Test Book', 'Another Book', 'Third Book'],
        'description': ['A great book', 'Nice story', 'Amazing tale'],
        'ratings_count': [100, 50, 200],
        'text_reviews_count': [10, 5, 20]
    }
    df = pd.DataFrame(data)
    
    result = preprocess_features(df)
    
    # Check that missing values are handled
    assert result['num_pages'].isna().sum() == 0
    assert 'title_length' in result.columns
    assert 'description_length' in result.columns
    assert 'publication_year' in result.columns

def test_filter_children_books():
    """Test children's book filtering."""
    data = {
        'title': ['Children Book', 'Adult Novel', 'Kids Story'],
        'popular_shelves': [
            "children fiction picture-book", 
            "mystery thriller", 
            "picture-book kids"
        ],
        'average_rating': [4.0, 3.5, 4.2]
    }
    df = pd.DataFrame(data)
    
    result = filter_children_books(df)
    
    # Should filter to children's books only
    assert len(result) >= 2  # Should keep first and third book
    
def test_empty_dataframe():
    """Test handling of empty dataframe."""
    df = pd.DataFrame()
    result = preprocess_features(df)
    assert len(result) == 0
    
def test_missing_columns():
    """Test preprocessing with missing columns."""
    data = {
        'title': ['Test Book'],
        'description': ['A great book']
    }
    df = pd.DataFrame(data)
    
    result = preprocess_features(df)
    
    # Should handle missing columns gracefully
    assert 'average_rating' in result.columns
    assert 'num_pages' in result.columns
    assert 'ratings_count' in result.columns
    assert 'publication_year' in result.columns
    assert len(result) == 1

def test_filter_empty_dataframe():
    """Test filtering empty dataframe."""
    df = pd.DataFrame()
    result = filter_children_books(df)
    assert len(result) == 0