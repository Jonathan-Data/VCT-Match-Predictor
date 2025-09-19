#!/usr/bin/env python3
"""
Comprehensive Test for Optimized VCT Predictor
Tests with realistic match data and full optimization pipeline
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_realistic_match_data():
    """Create realistic VCT match data for testing"""
    
    # Realistic team data based on VCT 2024
    teams_data = {
        'Sentinels': {
            'region': 'Americas',
            'win_rate': 0.72, 'recent_form_rating': 0.75, 'consistency_rating': 0.68,
            'international_experience': 4, 'current_streak': 3, 'win_streak': 7,
            'avg_rating': 1.12, 'star_player_factor': 0.28, 'depth_factor': 0.82, 'team_synergy': 0.78,
            'masters_rating': 0.75, 'champions_rating': 0.70, 'big_match_experience': 5
        },
        'Fnatic': {
            'region': 'EMEA',
            'win_rate': 0.68, 'recent_form_rating': 0.72, 'consistency_rating': 0.75,
            'international_experience': 6, 'current_streak': 2, 'win_streak': 4,
            'avg_rating': 1.08, 'star_player_factor': 0.32, 'depth_factor': 0.78, 'team_synergy': 0.85,
            'masters_rating': 0.82, 'champions_rating': 0.78, 'big_match_experience': 7
        },
        'Paper Rex': {
            'region': 'Pacific',
            'win_rate': 0.70, 'recent_form_rating': 0.78, 'consistency_rating': 0.62,
            'international_experience': 3, 'current_streak': 5, 'win_streak': 8,
            'avg_rating': 1.15, 'star_player_factor': 0.35, 'depth_factor': 0.70, 'team_synergy': 0.72,
            'masters_rating': 0.70, 'champions_rating': 0.65, 'big_match_experience': 4
        },
        'LOUD': {
            'region': 'Americas',
            'win_rate': 0.74, 'recent_form_rating': 0.80, 'consistency_rating': 0.78,
            'international_experience': 5, 'current_streak': 4, 'win_streak': 6,
            'avg_rating': 1.18, 'star_player_factor': 0.30, 'depth_factor': 0.85, 'team_synergy': 0.88,
            'masters_rating': 0.85, 'champions_rating': 0.80, 'big_match_experience': 6
        },
        'Team Liquid': {
            'region': 'EMEA',
            'win_rate': 0.65, 'recent_form_rating': 0.68, 'consistency_rating': 0.72,
            'international_experience': 4, 'current_streak': 1, 'win_streak': 3,
            'avg_rating': 1.05, 'star_player_factor': 0.25, 'depth_factor': 0.80, 'team_synergy': 0.75,
            'masters_rating': 0.68, 'champions_rating': 0.72, 'big_match_experience': 5
        }
    }
    
    # Realistic match results for training
    matches = []
    base_date = datetime(2024, 1, 1)
    
    # Generate match data with some H2H history
    match_pairs = [
        ('Sentinels', 'Fnatic', [1, 0, 1, 1, 0]),  # Sentinels 3-2
        ('LOUD', 'Paper Rex', [1, 1, 0, 1]),       # LOUD 3-1
        ('Team Liquid', 'Sentinels', [0, 1, 0, 0, 1]), # Sentinels 3-2
        ('Fnatic', 'LOUD', [1, 0, 1, 0]),          # Split 2-2
        ('Paper Rex', 'Team Liquid', [1, 1, 1]),   # Paper Rex 3-0
        ('Sentinels', 'LOUD', [0, 1, 1, 0, 1]),    # LOUD 3-2
    ]
    
    for i, (team1, team2, results) in enumerate(match_pairs):
        for j, result in enumerate(results):
            match_date = base_date + timedelta(days=i*30 + j*7)
            matches.append({
                'date': match_date.strftime('%Y-%m-%d'),
                'team1': team1,
                'team2': team2,
                'winner': team1 if result == 1 else team2,
                'score': '2-1' if np.random.random() > 0.5 else '2-0'
            })
    
    return teams_data, matches

def test_optimized_training_pipeline():
    """Test the full optimized training pipeline"""
    print("\n🚀 Testing Optimized Training Pipeline...")
    
    try:
        from models.optimized_ml_predictor import OptimizedVCTPredictor
        
        predictor = OptimizedVCTPredictor()
        teams_data, matches = create_realistic_match_data()
        
        # Setup data
        predictor.team_stats = {team: data for team, data in teams_data.items()}
        predictor.team_player_stats = {
            team: {
                'avg_rating': data['avg_rating'],
                'star_player_factor': data['star_player_factor'],
                'depth_factor': data['depth_factor'],
                'team_synergy': data['team_synergy']
            }
            for team, data in teams_data.items()
        }
        
        predictor.regional_performance = {
            'Americas': {'strength_rating': 0.85},
            'EMEA': {'strength_rating': 0.80},
            'Pacific': {'strength_rating': 0.75}
        }
        
        predictor.tournament_performance = {
            team: {
                'masters_rating': data['masters_rating'],
                'champions_rating': data['champions_rating'],
                'big_match_experience': data['big_match_experience']
            }
            for team, data in teams_data.items()
        }
        
        # Build H2H records from matches
        predictor.h2h_records = {}
        for match in matches:
            team1, team2 = match['team1'], match['team2']
            key = tuple(sorted([team1, team2]))
            
            if key not in predictor.h2h_records:
                predictor.h2h_records[key] = {
                    'total_matches': 0,
                    team1: 0,
                    team2: 0,
                    'momentum': 0.0
                }
            
            predictor.h2h_records[key]['total_matches'] += 1
            predictor.h2h_records[key][match['winner']] += 1
        
        print(f"  📊 Teams: {len(teams_data)}")
        print(f"  📊 Matches: {len(matches)}")
        print(f"  📊 H2H pairs: {len(predictor.h2h_records)}")
        
        # Test feature creation for multiple matchups
        test_matchups = [
            ('Sentinels', 'Fnatic'),
            ('LOUD', 'Paper Rex'),
            ('Team Liquid', 'Sentinels')
        ]
        
        for team1, team2 in test_matchups:
            features, feature_names = predictor.create_optimized_features(team1, team2)
            print(f"  ✅ Features for {team1} vs {team2}: {features.shape}")
        
        print("  ✅ Optimized training pipeline working correctly")
        return predictor, teams_data, matches
        
    except Exception as e:
        print(f"  ❌ Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_bayesian_optimization():
    """Test Bayesian hyperparameter optimization"""
    print("\n🔍 Testing Bayesian Hyperparameter Optimization...")
    
    try:
        predictor, teams_data, matches = test_optimized_training_pipeline()
        if not predictor:
            return False
        
        # Create training data
        X, y = [], []
        feature_names = None
        
        for match in matches[:15]:  # Use subset for faster testing
            team1, team2 = match['team1'], match['team2']
            features, names = predictor.create_optimized_features(team1, team2)
            
            if feature_names is None:
                feature_names = names
            
            X.append(features[0])
            y.append(1 if match['winner'] == team1 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  📊 Training data: {X.shape}")
        print(f"  📊 Class distribution: {np.bincount(y)}")
        
        # Test individual model optimization (quick version)
        print("  🔧 Testing Random Forest optimization...")
        rf_optimized = predictor.optimize_random_forest(X, y, n_iter=3)  # Reduced for speed
        print(f"  ✅ RF best score: {rf_optimized.best_score_:.3f}")
        
        print("  🔧 Testing Gradient Boosting optimization...")
        gb_optimized = predictor.optimize_gradient_boosting(X, y, n_iter=3)
        print(f"  ✅ GB best score: {gb_optimized.best_score_:.3f}")
        
        print("  ✅ Bayesian optimization working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Bayesian optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smoothing_optimization():
    """Test Bayesian smoothing parameter optimization"""
    print("\n🎯 Testing Smoothing Parameter Optimization...")
    
    try:
        predictor, teams_data, matches = test_optimized_training_pipeline()
        if not predictor:
            return False
        
        # Test k optimization with cross-validation
        X, y = [], []
        
        for match in matches[:10]:  # Small subset for speed
            team1, team2 = match['team1'], match['team2']
            features, _ = predictor.create_optimized_features(team1, team2)
            X.append(features[0])
            y.append(1 if match['winner'] == team1 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Test different k values
        k_values = [1, 3, 5, 10, 20]
        scores = []
        
        for k in k_values:
            predictor.optimal_k_smoothing = k
            
            # Re-create features with new k
            X_k = []
            for match in matches[:10]:
                team1, team2 = match['team1'], match['team2']
                features, _ = predictor.create_optimized_features(team1, team2)
                X_k.append(features[0])
            
            X_k = np.array(X_k)
            
            # Quick score estimation (simplified)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            cv_scores = cross_val_score(rf, X_k, y, cv=3)
            avg_score = cv_scores.mean()
            scores.append(avg_score)
            
            print(f"  📊 k={k}: CV score = {avg_score:.3f}")
        
        best_k = k_values[np.argmax(scores)]
        predictor.optimal_k_smoothing = best_k
        
        print(f"  ✅ Optimal k selected: {best_k}")
        print(f"  ✅ Best CV score: {max(scores):.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Smoothing optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stacking_ensemble():
    """Test advanced stacking ensemble"""
    print("\n🏗️ Testing Stacking Ensemble...")
    
    try:
        predictor, teams_data, matches = test_optimized_training_pipeline()
        if not predictor:
            return False
        
        # Create training data
        X, y = [], []
        for match in matches:
            team1, team2 = match['team1'], match['team2']
            features, _ = predictor.create_optimized_features(team1, team2)
            X.append(features[0])
            y.append(1 if match['winner'] == team1 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  📊 Training stacking ensemble: {X.shape}")
        
        # Train stacking ensemble
        stacking_model = predictor.create_stacking_ensemble(X, y)
        
        print("  ✅ Stacking ensemble created")
        
        # Test predictions
        predictions = stacking_model.predict_proba(X[:5])
        print(f"  📊 Sample predictions: {predictions[0]}")
        
        # Test calibration
        calibrated_model = predictor.calibrate_model(stacking_model, X, y)
        calibrated_preds = calibrated_model.predict_proba(X[:5])
        
        print(f"  📊 Calibrated predictions: {calibrated_preds[0]}")
        print("  ✅ Stacking ensemble with calibration working correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stacking ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_accuracy():
    """Test prediction accuracy on realistic matchups"""
    print("\n🎯 Testing Prediction Accuracy...")
    
    try:
        predictor, teams_data, matches = test_optimized_training_pipeline()
        if not predictor:
            return False
        
        # Train on historical data
        X, y = [], []
        for match in matches:
            team1, team2 = match['team1'], match['team2']
            features, _ = predictor.create_optimized_features(team1, team2)
            X.append(features[0])
            y.append(1 if match['winner'] == team1 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Quick training (simplified for testing)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Test predictions on same data (overfitting expected, but tests functionality)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        
        print(f"  📊 Training accuracy: {accuracy:.3f}")
        print(f"  📊 Training AUC: {auc:.3f}")
        
        # Test realistic predictions
        test_matchups = [
            ('Sentinels', 'Fnatic', "Close match between Americas and EMEA champions"),
            ('LOUD', 'Paper Rex', "High firepower vs tactical discipline"),
            ('Team Liquid', 'Sentinels', "Experience vs current form")
        ]
        
        print("\n  🔮 Realistic Match Predictions:")
        for team1, team2, desc in test_matchups:
            features, _ = predictor.create_optimized_features(team1, team2)
            proba = model.predict_proba(features)[0]
            
            confidence = max(proba)
            predicted_winner = team1 if proba[1] > 0.5 else team2
            
            print(f"    {team1} vs {team2}")
            print(f"    {desc}")
            print(f"    Prediction: {predicted_winner} ({confidence:.1%} confidence)")
            print()
        
        print("  ✅ Prediction accuracy testing completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Prediction accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("🚀 VCT ML Optimization - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Training Pipeline", lambda: test_optimized_training_pipeline()[0] is not None),
        ("Bayesian Optimization", test_bayesian_optimization),
        ("Smoothing Optimization", test_smoothing_optimization),
        ("Stacking Ensemble", test_stacking_ensemble),
        ("Prediction Accuracy", test_prediction_accuracy)
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
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Optimized predictor ready for production!")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Minor issues to address.")
    else:
        print("❌ Several failures. Needs debugging.")
    
    return passed, total

if __name__ == "__main__":
    passed, total = run_comprehensive_tests()
    print(f"\n🏁 Comprehensive test completed: {passed}/{total}")
