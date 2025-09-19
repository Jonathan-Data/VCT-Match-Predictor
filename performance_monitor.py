#!/usr/bin/env python3
"""
VCT Prediction Performance Monitoring System
Tracks real-world prediction accuracy and provides detailed analytics
"""

import sys
import os
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MatchResult:
    """Represents an actual match result for validation."""
    match_id: str
    team1: str
    team2: str
    actual_winner: str
    actual_score: str
    match_date: str
    tournament: str
    stage: str
    region: str
    source: str  # Where the result was obtained

@dataclass
class PredictionPerformance:
    """Performance metrics for a prediction."""
    match_id: str
    predicted_winner: str
    actual_winner: str
    predicted_team1_prob: float
    predicted_team2_prob: float
    confidence: float
    confidence_tier: str
    prediction_correct: bool
    prediction_timestamp: str
    match_timestamp: str
    hours_ahead: float
    tournament_importance: float
    stage_importance: float

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_predictions: int
    correct_predictions: int
    overall_accuracy: float
    confidence_calibration: Dict[str, float]
    accuracy_by_confidence: Dict[str, float]
    accuracy_by_time_ahead: Dict[str, float]
    accuracy_by_tournament_type: Dict[str, float]
    brier_score: float
    log_loss: float
    expected_accuracy: float
    performance_vs_expected: float

class PerformanceMonitor:
    """
    Monitors and analyzes prediction performance against actual results.
    """
    
    def __init__(self, data_dir: str = "data", logs_dir: str = "logs"):
        """Initialize the performance monitor."""
        self.data_dir = Path(data_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize database
        self.db_path = self.data_dir / "performance_monitor.db"
        self._setup_database()
        
        # Performance thresholds
        self.accuracy_thresholds = {
            'excellent': 0.80,
            'good': 0.70,
            'acceptable': 0.60,
            'poor': 0.50
        }
        
        # Expected model accuracy from testing
        self.expected_accuracy = 0.853
        
        self.logger.info("📊 VCT Prediction Performance Monitor initialized")
    
    def _setup_logging(self):
        """Set up logging for performance monitoring."""
        log_file = self.logs_dir / f"performance_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_database(self):
        """Set up SQLite database for storing performance data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create match results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_results (
                    match_id TEXT PRIMARY KEY,
                    team1 TEXT NOT NULL,
                    team2 TEXT NOT NULL,
                    actual_winner TEXT NOT NULL,
                    actual_score TEXT,
                    match_date TEXT NOT NULL,
                    tournament TEXT,
                    stage TEXT,
                    region TEXT,
                    source TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    predicted_winner TEXT NOT NULL,
                    team1_probability REAL NOT NULL,
                    team2_probability REAL NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_tier TEXT,
                    tournament_importance REAL,
                    stage_importance REAL,
                    prediction_timestamp TEXT NOT NULL,
                    match_timestamp TEXT,
                    hours_ahead REAL,
                    model_version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    predicted_winner TEXT NOT NULL,
                    actual_winner TEXT NOT NULL,
                    prediction_correct INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    confidence_tier TEXT,
                    brier_score REAL,
                    log_loss REAL,
                    hours_ahead REAL,
                    tournament_importance REAL,
                    evaluation_timestamp TEXT NOT NULL,
                    FOREIGN KEY (match_id) REFERENCES match_results (match_id),
                    FOREIGN KEY (match_id) REFERENCES predictions (match_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("🗄️ Performance monitoring database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up database: {e}")
    
    def add_match_result(self, result: MatchResult) -> bool:
        """
        Add an actual match result to the database.
        
        Args:
            result: MatchResult object with actual match outcome
            
        Returns:
            bool: True if added successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO match_results 
                (match_id, team1, team2, actual_winner, actual_score, match_date, tournament, stage, region, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.match_id, result.team1, result.team2, result.actual_winner,
                result.actual_score, result.match_date, result.tournament,
                result.stage, result.region, result.source
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Added match result: {result.team1} vs {result.team2} -> {result.actual_winner}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding match result: {e}")
            return False
    
    def add_prediction_record(self, prediction: Dict[str, Any], match_id: str = None) -> bool:
        """
        Add a prediction record to the database.
        
        Args:
            prediction: Prediction dictionary from live predictor
            match_id: Optional match ID override
            
        Returns:
            bool: True if added successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract prediction data
            pred_match_id = match_id or prediction.get('match_id', '')
            predicted_winner = prediction.get('predicted_winner', '')
            team1_prob = prediction.get('team1_probability', 0.0)
            team2_prob = prediction.get('team2_probability', 0.0)
            confidence = prediction.get('confidence', 0.0)
            confidence_tier = prediction.get('confidence_level', 'Unknown')
            
            # Extract context
            match_context = prediction.get('match_context', {})
            tournament_importance = match_context.get('tournament_importance', 0.8)
            stage_importance = match_context.get('stage_importance', 0.7)
            
            prediction_timestamp = prediction.get('prediction_timestamp', datetime.now().isoformat())
            match_timestamp = prediction.get('match_timestamp', '')
            
            # Calculate hours ahead if we have both timestamps
            hours_ahead = 0.0
            if match_timestamp and prediction_timestamp:
                try:
                    pred_time = datetime.fromisoformat(prediction_timestamp.replace('Z', '+00:00'))
                    match_time = datetime.fromisoformat(match_timestamp.replace('Z', '+00:00'))
                    hours_ahead = (match_time - pred_time).total_seconds() / 3600
                except:
                    pass
            
            cursor.execute('''
                INSERT INTO predictions 
                (match_id, predicted_winner, team1_probability, team2_probability, 
                 confidence, confidence_tier, tournament_importance, stage_importance,
                 prediction_timestamp, match_timestamp, hours_ahead, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pred_match_id, predicted_winner, team1_prob, team2_prob,
                confidence, confidence_tier, tournament_importance, stage_importance,
                prediction_timestamp, match_timestamp, hours_ahead, 'production_v1.0'
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Added prediction record for match {pred_match_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding prediction record: {e}")
            return False
    
    def evaluate_predictions(self) -> Dict[str, Any]:
        """
        Evaluate all predictions against actual results.
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            self.logger.info("📊 Evaluating predictions against actual results...")
            
            conn = sqlite3.connect(self.db_path)
            
            # Get predictions with matching results
            query = '''
                SELECT p.match_id, p.predicted_winner, r.actual_winner,
                       p.team1_probability, p.team2_probability, p.confidence,
                       p.confidence_tier, p.tournament_importance, p.stage_importance,
                       p.hours_ahead, p.prediction_timestamp, r.match_date
                FROM predictions p
                JOIN match_results r ON p.match_id = r.match_id
                WHERE p.match_id NOT IN (
                    SELECT match_id FROM performance_evaluations 
                    WHERE evaluation_timestamp > datetime('now', '-1 day')
                )
            '''
            
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                self.logger.info("No new predictions to evaluate")
                return {'message': 'No new predictions to evaluate'}
            
            self.logger.info(f"Evaluating {len(df)} predictions...")
            
            # Calculate performance metrics for each prediction
            evaluations = []
            
            for _, row in df.iterrows():
                prediction_correct = 1 if row['predicted_winner'] == row['actual_winner'] else 0
                
                # Calculate Brier score (for binary classification)
                if row['predicted_winner'] == row['actual_winner']:
                    # Predicted winner was correct
                    prob_actual = row['team1_probability'] if row['predicted_winner'] == 'team1' else row['team2_probability']
                    brier_score = (1 - prob_actual) ** 2
                else:
                    # Predicted winner was incorrect
                    prob_predicted = row['team1_probability'] if row['predicted_winner'] == 'team1' else row['team2_probability']
                    brier_score = prob_predicted ** 2
                
                # Calculate log loss
                prob_actual = row['team1_probability'] if row['actual_winner'] == 'team1' else row['team2_probability']
                prob_actual = max(min(prob_actual, 0.99), 0.01)  # Clip to avoid log(0)
                log_loss = -np.log(prob_actual) if prediction_correct else -np.log(1 - prob_actual)
                
                evaluation = PredictionPerformance(
                    match_id=row['match_id'],
                    predicted_winner=row['predicted_winner'],
                    actual_winner=row['actual_winner'],
                    predicted_team1_prob=row['team1_probability'],
                    predicted_team2_prob=row['team2_probability'],
                    confidence=row['confidence'],
                    confidence_tier=row['confidence_tier'],
                    prediction_correct=prediction_correct == 1,
                    prediction_timestamp=row['prediction_timestamp'],
                    match_timestamp=row['match_date'],
                    hours_ahead=row['hours_ahead'],
                    tournament_importance=row['tournament_importance'],
                    stage_importance=row['stage_importance']
                )
                
                evaluations.append(evaluation)
                
                # Store evaluation in database
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_evaluations 
                    (match_id, predicted_winner, actual_winner, prediction_correct,
                     confidence, confidence_tier, brier_score, log_loss, hours_ahead,
                     tournament_importance, evaluation_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['match_id'], row['predicted_winner'], row['actual_winner'],
                    prediction_correct, row['confidence'], row['confidence_tier'],
                    brier_score, log_loss, row['hours_ahead'],
                    row['tournament_importance'], datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            # Calculate aggregate metrics
            metrics = self._calculate_aggregate_metrics(evaluations)
            
            # Log results
            self.logger.info(f"✅ Evaluated {len(evaluations)} predictions")
            self.logger.info(f"📊 Overall accuracy: {metrics.overall_accuracy:.3f}")
            self.logger.info(f"📊 Brier score: {metrics.brier_score:.3f}")
            self.logger.info(f"📊 Performance vs expected: {metrics.performance_vs_expected:+.1%}")
            
            return {
                'evaluations_count': len(evaluations),
                'metrics': asdict(metrics),
                'individual_evaluations': [asdict(eval) for eval in evaluations]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating predictions: {e}")
            return {'error': str(e)}
    
    def _calculate_aggregate_metrics(self, evaluations: List[PredictionPerformance]) -> PerformanceMetrics:
        """Calculate aggregate performance metrics."""
        if not evaluations:
            return PerformanceMetrics(0, 0, 0.0, {}, {}, {}, {}, 0.0, 0.0, self.expected_accuracy, 0.0)
        
        total_predictions = len(evaluations)
        correct_predictions = sum(1 for eval in evaluations if eval.prediction_correct)
        overall_accuracy = correct_predictions / total_predictions
        
        # Confidence calibration
        confidence_tiers = {}
        accuracy_by_confidence = {}
        
        for eval in evaluations:
            tier = eval.confidence_tier
            if tier not in confidence_tiers:
                confidence_tiers[tier] = {'count': 0, 'correct': 0}
            confidence_tiers[tier]['count'] += 1
            if eval.prediction_correct:
                confidence_tiers[tier]['correct'] += 1
        
        for tier, data in confidence_tiers.items():
            accuracy_by_confidence[tier] = data['correct'] / data['count'] if data['count'] > 0 else 0.0
        
        # Time-based accuracy
        accuracy_by_time_ahead = {}
        time_buckets = {'0-6h': [], '6-24h': [], '24-72h': [], '72h+': []}
        
        for eval in evaluations:
            hours = eval.hours_ahead
            if hours <= 6:
                time_buckets['0-6h'].append(eval.prediction_correct)
            elif hours <= 24:
                time_buckets['6-24h'].append(eval.prediction_correct)
            elif hours <= 72:
                time_buckets['24-72h'].append(eval.prediction_correct)
            else:
                time_buckets['72h+'].append(eval.prediction_correct)
        
        for bucket, results in time_buckets.items():
            if results:
                accuracy_by_time_ahead[bucket] = sum(results) / len(results)
        
        # Tournament importance accuracy
        accuracy_by_tournament_type = {}
        tournament_buckets = {'high': [], 'medium': [], 'low': []}
        
        for eval in evaluations:
            importance = eval.tournament_importance
            if importance >= 0.9:
                tournament_buckets['high'].append(eval.prediction_correct)
            elif importance >= 0.75:
                tournament_buckets['medium'].append(eval.prediction_correct)
            else:
                tournament_buckets['low'].append(eval.prediction_correct)
        
        for bucket, results in tournament_buckets.items():
            if results:
                accuracy_by_tournament_type[bucket] = sum(results) / len(results)
        
        # Brier score and log loss
        brier_scores = []
        log_losses = []
        
        for eval in evaluations:
            # Brier score
            if eval.prediction_correct:
                prob_correct = eval.predicted_team1_prob if eval.predicted_winner == eval.actual_winner else eval.predicted_team2_prob
                brier = (1 - prob_correct) ** 2
            else:
                prob_incorrect = eval.predicted_team1_prob if eval.predicted_winner != eval.actual_winner else eval.predicted_team2_prob
                brier = prob_incorrect ** 2
            
            brier_scores.append(brier)
            
            # Log loss
            prob_actual = eval.predicted_team1_prob if eval.actual_winner == 'team1' else eval.predicted_team2_prob
            prob_actual = max(min(prob_actual, 0.99), 0.01)
            log_loss = -np.log(prob_actual)
            log_losses.append(log_loss)
        
        avg_brier_score = np.mean(brier_scores) if brier_scores else 0.0
        avg_log_loss = np.mean(log_losses) if log_losses else 0.0
        
        # Performance vs expected
        performance_vs_expected = (overall_accuracy - self.expected_accuracy) / self.expected_accuracy
        
        return PerformanceMetrics(
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            overall_accuracy=overall_accuracy,
            confidence_calibration={tier: data['correct'] / data['count'] for tier, data in confidence_tiers.items()},
            accuracy_by_confidence=accuracy_by_confidence,
            accuracy_by_time_ahead=accuracy_by_time_ahead,
            accuracy_by_tournament_type=accuracy_by_tournament_type,
            brier_score=avg_brier_score,
            log_loss=avg_log_loss,
            expected_accuracy=self.expected_accuracy,
            performance_vs_expected=performance_vs_expected
        )
    
    def get_performance_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary.
        
        Args:
            days_back: Number of days to look back for data
            
        Returns:
            Performance summary dictionary
        """
        try:
            self.logger.info(f"📊 Generating performance summary for last {days_back} days...")
            
            conn = sqlite3.connect(self.db_path)
            
            # Get recent evaluations
            query = '''
                SELECT *
                FROM performance_evaluations
                WHERE evaluation_timestamp > datetime('now', '-{} days')
                ORDER BY evaluation_timestamp DESC
            '''.format(days_back)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return {
                    'message': f'No performance data available for last {days_back} days',
                    'days_back': days_back,
                    'total_evaluations': 0
                }
            
            # Calculate summary statistics
            total_evaluations = len(df)
            correct_predictions = df['prediction_correct'].sum()
            overall_accuracy = correct_predictions / total_evaluations
            
            # Confidence tier breakdown
            confidence_breakdown = {}
            for tier in df['confidence_tier'].unique():
                tier_df = df[df['confidence_tier'] == tier]
                confidence_breakdown[tier] = {
                    'count': len(tier_df),
                    'accuracy': tier_df['prediction_correct'].mean(),
                    'avg_confidence': tier_df['confidence'].mean()
                }
            
            # Recent trend (last 7 days vs previous 7 days)
            recent_trend = {'status': 'stable', 'change': 0.0}
            if total_evaluations >= 10:  # Need enough data for trend analysis
                cutoff_date = datetime.now() - timedelta(days=7)
                recent_df = df[pd.to_datetime(df['evaluation_timestamp']) > cutoff_date]
                older_df = df[pd.to_datetime(df['evaluation_timestamp']) <= cutoff_date]
                
                if len(recent_df) > 0 and len(older_df) > 0:
                    recent_accuracy = recent_df['prediction_correct'].mean()
                    older_accuracy = older_df['prediction_correct'].mean()
                    change = recent_accuracy - older_accuracy
                    
                    recent_trend = {
                        'recent_accuracy': recent_accuracy,
                        'older_accuracy': older_accuracy,
                        'change': change,
                        'status': 'improving' if change > 0.05 else ('declining' if change < -0.05 else 'stable')
                    }
            
            # Performance vs expectations
            expected_accuracy = self.expected_accuracy
            performance_gap = overall_accuracy - expected_accuracy
            
            # Determine performance level
            performance_level = 'poor'
            for level, threshold in sorted(self.accuracy_thresholds.items(), key=lambda x: x[1], reverse=True):
                if overall_accuracy >= threshold:
                    performance_level = level
                    break
            
            summary = {
                'summary_period': f'Last {days_back} days',
                'generated_at': datetime.now().isoformat(),
                'total_evaluations': total_evaluations,
                'correct_predictions': int(correct_predictions),
                'overall_accuracy': overall_accuracy,
                'performance_level': performance_level,
                'expected_accuracy': expected_accuracy,
                'performance_gap': performance_gap,
                'performance_vs_expected': f"{performance_gap:+.1%}",
                'confidence_breakdown': confidence_breakdown,
                'recent_trend': recent_trend,
                'avg_brier_score': df['brier_score'].mean() if 'brier_score' in df.columns else None,
                'avg_log_loss': df['log_loss'].mean() if 'log_loss' in df.columns else None
            }
            
            self.logger.info(f"📊 Performance summary generated: {overall_accuracy:.3f} accuracy over {total_evaluations} predictions")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating performance summary: {e}")
            return {'error': str(e)}
    
    def add_mock_results_for_testing(self) -> int:
        """Add some mock match results for testing the monitoring system."""
        self.logger.info("🧪 Adding mock match results for testing...")
        
        mock_results = [
            MatchResult(
                match_id="vct_2025_001",
                team1="Sentinels",
                team2="Fnatic",
                actual_winner="Sentinels",
                actual_score="2-1",
                match_date=(datetime.now() - timedelta(hours=2)).isoformat(),
                tournament="VCT Champions 2025",
                stage="Group Stage",
                region="International",
                source="mock"
            ),
            MatchResult(
                match_id="vct_2025_002",
                team1="Paper Rex",
                team2="Team Liquid",
                actual_winner="Paper Rex",
                actual_score="2-0",
                match_date=(datetime.now() - timedelta(hours=6)).isoformat(),
                tournament="VCT Champions 2025",
                stage="Group Stage",
                region="International",
                source="mock"
            ),
            MatchResult(
                match_id="vct_2025_003",
                team1="G2 Esports",
                team2="NRG",
                actual_winner="G2 Esports",
                actual_score="3-1",
                match_date=(datetime.now() - timedelta(hours=12)).isoformat(),
                tournament="VCT Americas Kickoff",
                stage="Playoffs",
                region="Americas",
                source="mock"
            )
        ]
        
        added_count = 0
        for result in mock_results:
            if self.add_match_result(result):
                added_count += 1
        
        self.logger.info(f"✅ Added {added_count} mock match results")
        return added_count


def main():
    """Test the performance monitoring system."""
    print("📊 VCT Prediction Performance Monitor")
    print("=" * 50)
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Add some mock results for testing
    mock_results_added = monitor.add_mock_results_for_testing()
    print(f"\n🧪 Added {mock_results_added} mock match results for testing")
    
    # Load and add some mock predictions (simulating live predictor output)
    try:
        live_predictions_file = Path("data/live_predictions.json")
        if live_predictions_file.exists():
            with open(live_predictions_file, 'r') as f:
                data = json.load(f)
            
            predictions = data.get('predictions', [])
            added_predictions = 0
            
            for pred_data in predictions[:3]:  # Add first 3 predictions
                prediction = pred_data.get('prediction', {})
                match = pred_data.get('match', {})
                
                # Simulate prediction record
                prediction_record = {
                    'match_id': match.get('match_id', ''),
                    'predicted_winner': prediction.get('predicted_winner', ''),
                    'team1_probability': prediction.get('team1_probability', 0.5),
                    'team2_probability': prediction.get('team2_probability', 0.5),
                    'confidence': prediction.get('confidence', 0.5),
                    'confidence_level': prediction.get('confidence_level', 'Medium'),
                    'prediction_timestamp': prediction.get('prediction_timestamp', datetime.now().isoformat()),
                    'match_timestamp': match.get('scheduled_time', ''),
                    'match_context': {
                        'tournament_importance': 0.9,
                        'stage_importance': 0.8
                    }
                }
                
                if monitor.add_prediction_record(prediction_record):
                    added_predictions += 1
            
            print(f"📈 Added {added_predictions} prediction records from live predictions")
        
    except Exception as e:
        print(f"⚠️  Could not load live predictions: {e}")
    
    # Evaluate predictions
    print(f"\n📊 Evaluating predictions against results...")
    evaluation_results = monitor.evaluate_predictions()
    
    if 'error' in evaluation_results:
        print(f"❌ Evaluation failed: {evaluation_results['error']}")
    elif 'message' in evaluation_results:
        print(f"ℹ️  {evaluation_results['message']}")
    else:
        metrics = evaluation_results['metrics']
        print(f"✅ Evaluated {evaluation_results['evaluations_count']} predictions")
        print(f"📊 Overall accuracy: {metrics['overall_accuracy']:.1%}")
        print(f"📊 Expected accuracy: {metrics['expected_accuracy']:.1%}")
        print(f"📊 Performance vs expected: {metrics['performance_vs_expected']:+.1%}")
        print(f"📊 Brier score: {metrics['brier_score']:.3f}")
    
    # Get performance summary
    print(f"\n📈 Performance Summary (Last 30 days):")
    print("-" * 50)
    
    summary = monitor.get_performance_summary()
    
    if 'error' in summary:
        print(f"❌ Summary failed: {summary['error']}")
    elif 'message' in summary:
        print(f"ℹ️  {summary['message']}")
    else:
        print(f"Total Evaluations: {summary['total_evaluations']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"Performance Level: {summary['performance_level'].title()}")
        print(f"vs Expected: {summary['performance_vs_expected']}")
        print(f"Recent Trend: {summary['recent_trend']['status'].title()}")
        
        if summary['confidence_breakdown']:
            print(f"\nConfidence Tier Breakdown:")
            for tier, data in summary['confidence_breakdown'].items():
                print(f"  {tier}: {data['accuracy']:.1%} accuracy ({data['count']} predictions)")
    
    print(f"\n✅ Performance monitoring test completed")


if __name__ == "__main__":
    main()