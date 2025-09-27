# VCT Prediction Performance Monitoring System

## Overview

The Performance Monitor tracks real-world prediction accuracy and provides detailed analytics comparing actual performance against expected model performance (85.3% accuracy from testing).

## Key Features

### 📊 Core Monitoring
- **SQLite Database**: Stores predictions, match results, and evaluations
- **Brier Score & Log Loss**: Advanced prediction quality metrics
- **Confidence Calibration**: Tracks accuracy by confidence tier
- **Time-based Analysis**: Performance trends over prediction lead times
- **Tournament Context**: Accuracy breakdown by tournament importance

### 📈 Performance Metrics
- **Overall Accuracy**: Basic win/loss prediction accuracy  
- **Performance vs Expected**: Compares to 85.3% model test accuracy
- **Confidence Tiers**: High/Medium/Low confidence accuracy breakdown
- **Temporal Analysis**: 0-6h, 6-24h, 24-72h, 72h+ accuracy buckets
- **Tournament Types**: High/Medium/Low importance tournament accuracy

### 🔍 Real-time Evaluation
- **Automatic Evaluation**: Matches predictions with actual results
- **Trend Detection**: 7-day rolling performance analysis
- **Performance Alerts**: Identifies improving/declining/stable trends
- **Calibration Analysis**: Shows if confidence levels match actual accuracy

## Usage Integration

### 1. Adding Predictions (from main prediction system)
```python
from src.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# When making a prediction
prediction_data = {
    'match_id': match['match_id'],
    'predicted_winner': prediction['predicted_winner'], 
    'team1_probability': prediction['team1_probability'],
    'team2_probability': prediction['team2_probability'],
    'confidence': prediction['confidence'],
    'confidence_level': prediction['confidence_level'],
    'prediction_timestamp': prediction['prediction_timestamp'],
    'match_timestamp': match['scheduled_time'],
    'match_context': prediction['match_context']
}

monitor.add_prediction_record(prediction_data)
```

### 2. Adding Match Results
```python
from src.performance_monitor import MatchResult

# When match results are available
result = MatchResult(
    match_id="vct_2025_001",
    team1="Sentinels",
    team2="Fnatic", 
    actual_winner="Sentinels",
    actual_score="2-1",
    match_date="2025-01-15T20:00:00",
    tournament="VCT Champions 2025",
    stage="Group Stage",
    region="International",
    source="vlr.gg"  # or "rib.gg" 
)

monitor.add_match_result(result)
```

### 3. Running Evaluation
```python
# Evaluate all new predictions against results
evaluation_results = monitor.evaluate_predictions()

# Get comprehensive performance summary  
summary = monitor.get_performance_summary(days_back=30)
```

## Expected Performance Baseline

Based on model testing:
- **Target Accuracy**: 85.3%
- **Acceptable Range**: 60-85%
- **Poor Performance**: < 60%
- **Excellent Performance**: > 80%

## Test Results Summary

From the test run:
```
📊 Performance Results:
- Evaluated: 3 predictions
- Overall Accuracy: 66.7% (2/3 correct)
- Expected Accuracy: 85.3%
- Performance Gap: -18.6%
- Brier Score: 0.101 (lower is better)

📈 Confidence Breakdown:
- Very High: 100.0% accuracy (1 prediction)
- High: 50.0% accuracy (2 predictions)
```

## Database Schema

### Tables Created:
1. **match_results**: Actual match outcomes
2. **predictions**: Prediction records with probabilities  
3. **performance_evaluations**: Matched predictions vs results

### Key Metrics Tracked:
- Prediction correctness (binary)
- Brier score (probability accuracy)
- Log loss (prediction quality)
- Hours ahead (prediction timing)
- Tournament/stage importance
- Confidence calibration

## Production Workflow

1. **Daily**: Run live predictor, add prediction records
2. **Post-Match**: Add match results when available  
3. **Weekly**: Run evaluation and generate performance summary
4. **Monthly**: Deep dive analysis and model performance review

## Performance Thresholds

```python
accuracy_thresholds = {
    'excellent': 0.80,   # > 80%
    'good': 0.70,        # 70-80%  
    'acceptable': 0.60,  # 60-70%
    'poor': 0.50         # < 60%
}
```

## Files Overview

- `src/performance_monitor.py`: Main monitoring class
- `data/performance_monitor.db`: SQLite database
- `logs/performance_monitor_YYYYMMDD.log`: Daily logs

## Integration with Existing Systems

The monitor integrates seamlessly with:
- `main_gui.py`: Automatic prediction recording
- `src/data_collection/rib_scraper.py`: Match result collection 
- `src/data_collection/vlr_scraper.py`: Alternative result source
- Existing data pipeline and model evaluation

This system provides comprehensive real-world performance tracking to ensure your VCT prediction model maintains expected accuracy in production.