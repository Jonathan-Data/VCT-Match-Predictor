"""
Data Preprocessing Pipeline for VCT 2025 Match Prediction

This module handles cleaning, transforming, and feature engineering of match and team data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import yaml
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VCTDataProcessor:
    """Data processor for VCT tournament and team data."""
    
    def __init__(self, config_path: str = None):
        """Initialize the data processor."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "teams.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.scalers = {}
        self.encoders = {}
        self.team_features = {}
    
    def load_kaggle_data(self) -> Dict[str, pd.DataFrame]:
        """Load all Kaggle datasets into DataFrames."""
        datasets = {}
        raw_dir = self.data_dir / "raw"
        
        for dataset_config in self.config['kaggle_datasets']:
            dataset_name = dataset_config['name']
            dataset_path = raw_dir / dataset_name
            
            if not dataset_path.exists():
                logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")
                continue
            
            # Try to find CSV files in the dataset directory
            csv_files = list(dataset_path.glob("*.csv"))
            if csv_files:
                # Load the main CSV file (assuming first one is primary)
                main_csv = csv_files[0]
                try:
                    df = pd.read_csv(main_csv)
                    datasets[dataset_name] = df
                    logger.info(f"Loaded {dataset_name}: {df.shape}")
                except Exception as e:
                    logger.error(f"Failed to load {main_csv}: {e}")
            else:
                logger.warning(f"No CSV files found in {dataset_path}")
        
        return datasets
    
    def load_team_stats(self) -> pd.DataFrame:
        """Load team statistics from VLR.gg scraper output."""
        team_stats_path = self.data_dir / "external" / "team_stats.csv"
        
        if team_stats_path.exists():
            df = pd.read_csv(team_stats_path)
            logger.info(f"Loaded team stats: {df.shape}")
            return df
        else:
            logger.warning(f"Team stats file not found at {team_stats_path}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'team_key', 'team_name', 'region', 'vlr_id', 'rating',
                'wins', 'losses', 'win_rate', 'rounds_won', 'rounds_lost',
                'round_win_rate', 'avg_combat_score'
            ])
    
    def clean_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize match data."""
        logger.info("Cleaning match data...")
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Standard cleaning operations
        # Remove rows with critical missing data
        if 'team1' in cleaned_df.columns and 'team2' in cleaned_df.columns:
            cleaned_df = cleaned_df.dropna(subset=['team1', 'team2'])
        
        # Convert date columns if present
        date_columns = [col for col in cleaned_df.columns if 'date' in col.lower()]
        for date_col in date_columns:
            try:
                cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col])
            except:
                logger.warning(f"Could not convert {date_col} to datetime")
        
        # Clean team names (standardize formatting)
        if 'team1' in cleaned_df.columns:
            cleaned_df['team1'] = cleaned_df['team1'].str.strip().str.lower()
        if 'team2' in cleaned_df.columns:
            cleaned_df['team2'] = cleaned_df['team2'].str.strip().str.lower()
        
        # Handle missing numerical values
        numerical_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        
        logger.info(f"Cleaned data shape: {cleaned_df.shape}")
        return cleaned_df
    
    def create_team_features(self, team_stats: pd.DataFrame) -> Dict[str, Dict]:
        """Create feature dictionary for each team."""
        features = {}
        
        for _, row in team_stats.iterrows():
            team_name = row['team_name'].lower() if pd.notna(row['team_name']) else 'unknown'
            features[team_name] = {
                'region': row.get('region', 'unknown'),
                'rating': row.get('rating', 0.0),
                'win_rate': row.get('win_rate', 0.5),
                'round_win_rate': row.get('round_win_rate', 0.5),
                'avg_combat_score': row.get('avg_combat_score', 200.0),
                'wins': row.get('wins', 0),
                'losses': row.get('losses', 0),
                'total_matches': row.get('wins', 0) + row.get('losses', 0)
            }
        
        self.team_features = features
        logger.info(f"Created features for {len(features)} teams")
        return features
    
    def engineer_match_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for match prediction."""
        logger.info("Engineering match features...")
        
        engineered_df = match_df.copy()
        
        # Add team-based features if we have team data
        if hasattr(self, 'team_features') and self.team_features:
            
            # Add team1 features
            for feature_name in ['rating', 'win_rate', 'round_win_rate', 'avg_combat_score']:
                engineered_df[f'team1_{feature_name}'] = engineered_df['team1'].map(
                    lambda x: self.team_features.get(str(x).lower(), {}).get(feature_name, 0.0)
                )
                engineered_df[f'team2_{feature_name}'] = engineered_df['team2'].map(
                    lambda x: self.team_features.get(str(x).lower(), {}).get(feature_name, 0.0)
                )
            
            # Calculate relative features (team1 vs team2)
            engineered_df['rating_diff'] = engineered_df['team1_rating'] - engineered_df['team2_rating']
            engineered_df['win_rate_diff'] = engineered_df['team1_win_rate'] - engineered_df['team2_win_rate']
            engineered_df['round_win_rate_diff'] = engineered_df['team1_round_win_rate'] - engineered_df['team2_round_win_rate']
            engineered_df['acs_diff'] = engineered_df['team1_avg_combat_score'] - engineered_df['team2_avg_combat_score']
        
        # Add regional matchup features
        if 'region' in engineered_df.columns or (hasattr(self, 'team_features') and self.team_features):
            try:
                engineered_df['team1_region'] = engineered_df['team1'].map(
                    lambda x: self.team_features.get(str(x).lower(), {}).get('region', 'unknown')
                )
                engineered_df['team2_region'] = engineered_df['team2'].map(
                    lambda x: self.team_features.get(str(x).lower(), {}).get('region', 'unknown')
                )
                
                # Same region matchup indicator
                engineered_df['same_region'] = (
                    engineered_df['team1_region'] == engineered_df['team2_region']
                ).astype(int)
                
            except:
                logger.warning("Could not create regional features")
        
        # Add time-based features if date column exists
        date_cols = [col for col in engineered_df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                engineered_df['month'] = pd.to_datetime(engineered_df[date_col]).dt.month
                engineered_df['day_of_week'] = pd.to_datetime(engineered_df[date_col]).dt.dayofweek
                engineered_df['is_weekend'] = (
                    pd.to_datetime(engineered_df[date_col]).dt.dayofweek >= 5
                ).astype(int)
            except:
                logger.warning("Could not create time-based features")
        
        logger.info(f"Engineered features shape: {engineered_df.shape}")
        return engineered_df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for ML models."""
        logger.info("Encoding categorical features...")
        
        encoded_df = df.copy()
        categorical_cols = encoded_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['team1', 'team2']:  # Keep team names for now
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    encoded_df[col] = self.encoders[col].fit_transform(encoded_df[col].fillna('unknown'))
                else:
                    # Handle new categories
                    unique_vals = set(encoded_df[col].fillna('unknown').unique())
                    known_vals = set(self.encoders[col].classes_)
                    new_vals = unique_vals - known_vals
                    
                    if new_vals:
                        # Add new categories to encoder
                        all_vals = list(known_vals) + list(new_vals)
                        self.encoders[col].classes_ = np.array(all_vals)
                    
                    encoded_df[col] = self.encoders[col].transform(encoded_df[col].fillna('unknown'))
        
        return encoded_df
    
    def prepare_training_data(
        self, 
        match_df: pd.DataFrame, 
        target_column: str = 'winner'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training ML models."""
        logger.info("Preparing training data...")
        
        # Features to exclude from training
        exclude_cols = ['team1', 'team2', target_column] + \
                      [col for col in match_df.columns if 'date' in col.lower()]
        
        feature_cols = [col for col in match_df.columns if col not in exclude_cols]
        
        X = match_df[feature_cols]
        y = match_df[target_column] if target_column in match_df.columns else None
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            if 'standard_scaler' not in self.scalers:
                self.scalers['standard_scaler'] = StandardScaler()
                X[numerical_cols] = self.scalers['standard_scaler'].fit_transform(X[numerical_cols])
            else:
                X[numerical_cols] = self.scalers['standard_scaler'].transform(X[numerical_cols])
        
        logger.info(f"Training data prepared: X shape {X.shape}, y shape {y.shape if y is not None else 'None'}")
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def save_preprocessors(self, output_dir: Path = None) -> None:
        """Save preprocessing objects for later use."""
        if output_dir is None:
            output_dir = self.processed_dir
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = output_dir / f"{name}.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {name} to {scaler_path}")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            encoder_path = output_dir / f"encoder_{name}.joblib"
            joblib.dump(encoder, encoder_path)
            logger.info(f"Saved encoder for {name} to {encoder_path}")
        
        # Save team features
        if self.team_features:
            import json
            features_path = output_dir / "team_features.json"
            with open(features_path, 'w') as f:
                json.dump(self.team_features, f, indent=2)
            logger.info(f"Saved team features to {features_path}")
    
    def process_full_pipeline(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Run the complete data processing pipeline."""
        logger.info("Starting full data processing pipeline...")
        
        # Load all data
        kaggle_datasets = self.load_kaggle_data()
        team_stats = self.load_team_stats()
        
        # Create team features
        self.create_team_features(team_stats)
        
        # Process each dataset and combine
        processed_matches = []
        
        for dataset_name, df in kaggle_datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Clean the data
            cleaned_df = self.clean_match_data(df)
            
            # Engineer features
            engineered_df = self.engineer_match_features(cleaned_df)
            
            # Add dataset source
            engineered_df['source_dataset'] = dataset_name
            
            processed_matches.append(engineered_df)
        
        if processed_matches:
            # Combine all datasets
            combined_df = pd.concat(processed_matches, ignore_index=True, sort=False)
            
            # Encode categorical features
            encoded_df = self.encode_categorical_features(combined_df)
            
            # Prepare training data
            target_col = 'winner' if 'winner' in encoded_df.columns else None
            if target_col:
                X, y = self.prepare_training_data(encoded_df, target_col)
            else:
                logger.warning("No target column found, returning features only")
                X, y = self.prepare_training_data(encoded_df, 'team1')  # Dummy target
                y = None
            
            # Save preprocessors
            self.save_preprocessors()
            
            # Save processed data
            processed_path = self.processed_dir / "processed_matches.csv"
            encoded_df.to_csv(processed_path, index=False)
            logger.info(f"Saved processed matches to {processed_path}")
            
            return X, y
        
        else:
            logger.error("No datasets were successfully processed")
            return pd.DataFrame(), pd.Series()

def main():
    """Main function to run the data processing pipeline."""
    processor = VCTDataProcessor()
    
    print("Starting VCT data processing pipeline...")
    X, y = processor.process_full_pipeline()
    
    if not X.empty:
        print(f"\nData processing completed successfully!")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape if y is not None else 'No target'}")
        print(f"Feature columns: {list(X.columns)[:10]}...")  # Show first 10
    else:
        print("Data processing failed - no data was processed")

if __name__ == "__main__":
    main()