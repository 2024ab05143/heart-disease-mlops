import pytest
import pandas as pd
import numpy as np
from src.train import load_data, prepare_data

# Create a fake small dataset for testing
@pytest.fixture
def sample_data():
    data = {
        'age': [63, 37],
        'sex': [1, 1],
        'cp': [3, 2],
        'trestbps': [145, 130],
        'chol': [233, 250],
        'fbs': [1, 0],
        'restecg': [0, 1],
        'thalach': [150, 187],
        'exang': [0, 0],
        'oldpeak': [2.3, 3.5],
        'slope': [0, 0],
        'ca': [0, 0],
        'thal': [1, 2],
        'target': [1, 0] # 1=Disease, 0=No Disease
    }
    return pd.DataFrame(data)

def test_load_data_columns(tmp_path, sample_data):
    # Save fake data to a temporary file
    d = tmp_path / "heart.csv"
    sample_data.to_csv(d, index=False, header=False) # Saving without header to mimic raw data
    
    # Test loading
    df = load_data(d)
    assert len(df.columns) == 14
    assert 'target' in df.columns

def test_prepare_data_shape(sample_data):
    # Test if data splitting works
    X_train, X_test, y_train, y_test = prepare_data(sample_data)
    
    # Check if we got 4 outputs
    assert X_train is not None
    assert y_train is not None
    
    # Check if scaling worked (mean should be close to 0)
    assert np.abs(np.mean(X_train)) < 1.0