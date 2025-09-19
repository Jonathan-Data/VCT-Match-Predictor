#!/usr/bin/env python3
"""
Simple Test Script for VCT ML Optimizations
Tests core functionality without complex training
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing Basic Imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        
        print("  ✅ All core dependencies imported successfully")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_optimized_predictor_creation():
    """Test creating optimized predictor"""
    print("\n🚀 Testing Optimized Predictor Creation...")
    
    try:
        from models.optimized_ml_predictor import OptimizedVCTPredictor
        
        predictor = OptimizedVCTPredictor()
        
        # Check key attributes
        assert hasattr(predictor, 'optimal_k_smoothing'), "Missing smoothing parameter"
        assert hasattr(predictor, 'calibrated_model'), "Missing calibrated model"
        assert hasattr(predictor, 'stacking_model'), "Missing stacking model"
        
        print(f"  ✅ OptimizedVCTPredictor created successfully")
        print(f"  ✅ Default k smoothing: {predictor.optimal_k_smoothing}")
        
        return predictor
        
    except Exception as e:
        print(f"  ❌ Creation failed: {e}")
        return None

def test_bayesian_smoothing():
    """Test Bayesian H2H smoothing"""
    print("\n🎯 Testing Bayesian H2H Smoothing...")
    
    try:
        from models.optimized_ml_predictor import OptimizedVCTPredictor
        
        predictor = OptimizedVCTPredictor()
        
        # Mock team stats
        predictor.team_stats = {
            'Sentinels': {'win_rate': 0.75},
            'Fnatic': {'win_rate': 0.65}
        }
        
        # Mock H2H records  
        predictor.h2h_records = {
            ('Fnatic', 'Sentinels'): {
                'total_matches': 5,
                'Sentinels': 3,
                'Fnatic': 2
            }
        }
        
        # Test smoothing
        raw_prob = 3/5  # 60%
        smoothed_prob = predictor._calculate_smoothed_h2h_feature('Sentinels', 'Fnatic', k=5)
        
        print(f"  📊 Raw H2H: {raw_prob:.3f}")
        print(f"  📊 Smoothed: {smoothed_prob:.3f}")
        print(f"  📊 Difference: {abs(smoothed_prob - raw_prob):.3f}")
        
        assert 0 <= smoothed_prob <= 1, "Invalid probability"
        
        print("  ✅ Bayesian smoothing working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Smoothing test failed: {e}")
        return False

def test_feature_creation():
    """Test optimized feature creation"""
    print("\n🔢 Testing Feature Creation...")
    
    try:
        from models.optimized_ml_predictor import OptimizedVCTPredictor
        import numpy as np
        
        predictor = OptimizedVCTPredictor()
        
        # Mock all required data
        predictor.team_stats = {
            'Sentinels': {
                'win_rate': 0.75, 'recent_form_rating': 0.80, 'consistency_rating': 0.65,
                'international_experience': 3, 'current_streak': 2, 'win_streak': 5
            },
            'Fnatic': {
                'win_rate': 0.70, 'recent_form_rating': 0.75, 'consistency_rating': 0.70,
                'international_experience': 4, 'current_streak': -1, 'win_streak': 3
            }
        }
        
        predictor.team_player_stats = {
            'Sentinels': {'avg_rating': 1.15, 'star_player_factor': 0.25, 'depth_factor': 0.80, 'team_synergy': 0.75},
            'Fnatic': {'avg_rating': 1.10, 'star_player_factor': 0.30, 'depth_factor': 0.75, 'team_synergy': 0.85}
        }
        
        predictor.regional_performance = {
            'Americas': {'strength_rating': 0.85},
            'EMEA': {'strength_rating': 0.80}
        }
        
        predictor.tournament_performance = {
            'Sentinels': {'masters_rating': 0.75, 'champions_rating': 0.70, 'big_match_experience': 3},
            'Fnatic': {'masters_rating': 0.80, 'champions_rating': 0.75, 'big_match_experience': 4}
        }
        
        predictor.h2h_records = {
            ('Fnatic', 'Sentinels'): {'total_matches': 6, 'Sentinels': 4, 'Fnatic': 2, 'momentum': 0.2}
        }
        
        # Test feature creation
        features, feature_names = predictor.create_optimized_features('Sentinels', 'Fnatic')
        
        print(f"  📊 Features created: {len(feature_names)}")
        print(f"  📊 Feature shape: {features.shape}")
        
        # Verify key features exist
        expected = ['t1_win_rate', 't2_win_rate', 'smoothed_h2h_win_prob']
        for feat in expected:
            assert feat in feature_names, f"Missing {feat}"
        
        # Check smoothed H2H feature
        smoothed_idx = feature_names.index('smoothed_h2h_win_prob')
        smoothed_val = features[0][smoothed_idx]
        
        print(f"  🎯 Smoothed H2H value: {smoothed_val:.3f}")
        print("  ✅ Feature creation working correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stacking_components():
    """Test stacking classifier components"""
    print("\n🏗️ Testing Stacking Components...")
    
    try:
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        import numpy as np
        
        # Create synthetic data for testing
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        
        # Create base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        
        # Create stacking classifier with compatible CV
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=3,  # Use integer instead of TimeSeriesSplit for compatibility
            stack_method='predict_proba'
        )
        
        print("  ✅ StackingClassifier created")
        
        # Test training
        stacking.fit(X, y)
        print("  ✅ Stacking model trained")
        
        # Test calibration
        calibrated = CalibratedClassifierCV(stacking, method='isotonic', cv=3)
        calibrated.fit(X, y)
        print("  ✅ Calibration applied")
        
        # Test prediction
        pred_proba = calibrated.predict_proba(X[:5])
        print(f"  📊 Sample predictions: {pred_proba[0]}")
        
        print("  ✅ Stacking components working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Stacking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bayesian_search():
    """Test Bayesian optimization components"""
    print("\n🔍 Testing Bayesian Optimization...")
    
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        import numpy as np
        
        # Create small synthetic dataset
        X = np.random.random((50, 5))
        y = np.random.randint(0, 2, 50)
        
        # Define search space
        search_space = {
            'n_estimators': Integer(10, 50),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 5)
        }
        
        # Create BayesSearchCV
        bayes_search = BayesSearchCV(
            RandomForestClassifier(random_state=42),
            search_space,
            n_iter=5,  # Small for testing
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            random_state=42
        )
        
        print("  ✅ BayesSearchCV created")
        
        # Test fitting (this may take a moment)
        bayes_search.fit(X, y)
        print("  ✅ Bayesian optimization completed")
        print(f"  📊 Best score: {bayes_search.best_score_:.3f}")
        print(f"  📊 Best params: {bayes_search.best_params_}")
        
        print("  ✅ Bayesian optimization working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Bayesian optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_tests():
    """Run simplified test suite"""
    print("🚀 VCT ML Optimization - Simple Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Optimized Predictor", lambda: test_optimized_predictor_creation() is not None),
        ("Bayesian Smoothing", test_bayesian_smoothing),
        ("Feature Creation", test_feature_creation),
        ("Stacking Components", test_stacking_components),
        ("Bayesian Optimization", test_bayesian_search)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print()  # Add spacing
            result = test_func()
            if result:
                passed += 1
                print(f"  ✅ {test_name} PASSED")
            else:
                print(f"  ❌ {test_name} FAILED")
        except Exception as e:
            print(f"  💥 {test_name} CRASHED: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SIMPLE TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Core optimizations working!")
    elif passed >= total * 0.7:
        print("⚠️ Most tests passed. Ready for advanced testing.")
    else:
        print("❌ Several failures. Need debugging.")
    
    return passed, total

if __name__ == "__main__":
    passed, total = run_simple_tests()
    print(f"\n🏁 Simple test completed: {passed}/{total}")