"""
Unit tests for machine learning models.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import VCTMatchPredictor, ModelPerformance


class TestModelPerformance:
    """Test cases for ModelPerformance dataclass."""
    
    def test_create_model_performance(self):
        """Test creating ModelPerformance object."""
        perf = ModelPerformance(
            model_name='test_model',
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1=0.85,
            cross_val_scores=[0.83, 0.87, 0.84, 0.86, 0.85],
            feature_importance={'feature1': 0.3, 'feature2': 0.7}
        )
        
        assert perf.model_name == 'test_model'
        assert perf.accuracy == 0.85
        assert len(perf.cross_val_scores) == 5
        assert perf.feature_importance is not None


class TestVCTMatchPredictor:
    """Test cases for VCTMatchPredictor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples), name='winner')
        
        return X, y
    
    def test_init(self):
        """Test VCTMatchPredictor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            
            assert predictor.models_dir == Path(temp_dir)
            assert predictor.models == {}
            assert predictor.model_performances == {}
            assert predictor.is_trained == False
    
    def test_create_random_forest(self):
        """Test Random Forest model creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            model = predictor.create_random_forest()
            
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_create_logistic_regression(self):
        """Test Logistic Regression model creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            model = predictor.create_logistic_regression()
            
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_create_svm(self):
        """Test SVM model creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            model = predictor.create_svm()
            
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_initialize_models(self, sample_data):
        """Test model initialization."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            predictor.initialize_models(input_dim=X.shape[1])
            
            assert len(predictor.models) >= 3  # At least RF, LR, SVM
            assert 'random_forest' in predictor.models
            assert 'logistic_regression' in predictor.models
            assert 'svm' in predictor.models
    
    def test_train_sklearn_model(self, sample_data):
        """Test training a sklearn model."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            predictor.feature_names = list(X.columns)
            
            model = predictor.create_random_forest()
            performance = predictor.train_sklearn_model(
                model, 'test_model', X_train, y_train, X_test, y_test
            )
            
            assert isinstance(performance, ModelPerformance)
            assert performance.model_name == 'test_model'
            assert 0 <= performance.accuracy <= 1
            assert 0 <= performance.f1 <= 1
            assert len(performance.cross_val_scores) == 5
    
    def test_train_all_models(self, sample_data):
        """Test training all models."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            
            performances = predictor.train_all_models(X_train, y_train, X_test, y_test)
            
            assert len(performances) >= 3
            assert predictor.is_trained == True
            
            # Check that all performances are valid
            for name, perf in performances.items():
                assert isinstance(perf, ModelPerformance)
                assert 0 <= perf.accuracy <= 1
                assert 0 <= perf.f1 <= 1
    
    def test_get_model_rankings(self, sample_data):
        """Test model ranking functionality."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            
            # Train models first
            predictor.train_all_models(X_train, y_train, X_test, y_test)
            
            rankings = predictor.get_model_rankings()
            
            assert len(rankings) >= 3
            assert all(isinstance(item, tuple) for item in rankings)
            assert all(len(item) == 2 for item in rankings)
            
            # Check that rankings are in descending order
            f1_scores = [item[1] for item in rankings]
            assert f1_scores == sorted(f1_scores, reverse=True)
    
    def test_predict_match(self, sample_data):
        """Test match prediction."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            
            # Train models first
            predictor.train_all_models(X_train, y_train, X_test, y_test)
            
            # Create sample team features
            team1_features = {
                'rating': 1200,
                'win_rate': 0.7,
                'round_win_rate': 0.65,
                'avg_combat_score': 250,
                'region': 'Americas'
            }
            
            team2_features = {
                'rating': 1100,
                'win_rate': 0.6,
                'round_win_rate': 0.58,
                'avg_combat_score': 230,
                'region': 'EMEA'
            }
            
            # Test prediction with random forest (should exist)
            prediction = predictor.predict_match(
                team1_features, team2_features, model_name='random_forest'
            )
            
            assert 'prediction' in prediction
            assert 'probability' in prediction
            assert 'confidence' in prediction
            assert 'model_used' in prediction
            
            assert prediction['prediction'] in [0, 1]
            assert 0 <= prediction['probability'] <= 1
            assert 0 <= prediction['confidence'] <= 1
            assert prediction['model_used'] == 'random_forest'
    
    def test_save_and_load_models(self, sample_data):
        """Test saving and loading models."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train and save models
            predictor1 = VCTMatchPredictor(models_dir=Path(temp_dir))
            predictor1.train_all_models(X_train, y_train, X_test, y_test)
            predictor1.save_models()
            
            # Load models in new predictor
            predictor2 = VCTMatchPredictor(models_dir=Path(temp_dir))
            predictor2.load_models()
            
            assert predictor2.is_trained == True
            assert len(predictor2.models) >= 3
            assert len(predictor2.model_performances) >= 3
            
            # Check that performance data was loaded correctly
            for name, perf in predictor2.model_performances.items():
                assert isinstance(perf, ModelPerformance)
                assert 0 <= perf.accuracy <= 1
    
    @patch('builtins.print')
    def test_print_performance_summary(self, mock_print, sample_data):
        """Test performance summary printing."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            predictor.train_all_models(X_train, y_train, X_test, y_test)
            
            predictor.print_performance_summary()
            
            # Verify that print was called (summary was displayed)
            assert mock_print.called
    
    def test_create_ensemble_model(self, sample_data):
        """Test ensemble model creation."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            
            # Train base models first
            predictor.train_all_models(X_train, y_train, X_test, y_test)
            
            # Create ensemble
            ensemble = predictor.create_ensemble_model(X_train, y_train)
            
            assert ensemble is not None
            assert 'ensemble' in predictor.models
            assert hasattr(ensemble, 'predict')
            assert hasattr(ensemble, 'predict_proba')
    
    def test_invalid_model_prediction(self, sample_data):
        """Test prediction with invalid model name."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            predictor.train_all_models(X_train, y_train, X_test, y_test)
            
            team1_features = {'rating': 1200, 'win_rate': 0.7, 'round_win_rate': 0.65, 'avg_combat_score': 250, 'region': 'Americas'}
            team2_features = {'rating': 1100, 'win_rate': 0.6, 'round_win_rate': 0.58, 'avg_combat_score': 230, 'region': 'EMEA'}
            
            with pytest.raises(ValueError, match="Model invalid_model not found"):
                predictor.predict_match(team1_features, team2_features, model_name='invalid_model')
    
    def test_prediction_without_training(self):
        """Test prediction without training models first."""
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = VCTMatchPredictor(models_dir=Path(temp_dir))
            
            team1_features = {'rating': 1200, 'win_rate': 0.7, 'round_win_rate': 0.65, 'avg_combat_score': 250, 'region': 'Americas'}
            team2_features = {'rating': 1100, 'win_rate': 0.6, 'round_win_rate': 0.58, 'avg_combat_score': 230, 'region': 'EMEA'}
            
            with pytest.raises(ValueError, match="Models must be trained before making predictions"):
                predictor.predict_match(team1_features, team2_features)


if __name__ == '__main__':\n    pytest.main([__file__])