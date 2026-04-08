import os
import pickle
import pandas as pd
import pytest

def test_model_exists():
    """Check if model.pkl is generated"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, 'models', 'model.pkl')
    assert os.path.exists(model_path), "Model file is missing!"

def test_config_exists():
    """Check if config file is present"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, 'config', 'config.yaml')
    assert os.path.exists(config_path), "Config file is missing!"