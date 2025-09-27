# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a comprehensive machine learning project for predicting match outcomes in the VALORANT Champions Tour (VCT) 2025 tournament. The system combines data from Kaggle datasets with real-time team statistics scraped from VLR.gg to train multiple ML models for accurate match prediction.

## Essential Commands

### Initial Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Set up Kaggle API credentials (required for data download)
# Download kaggle.json from your Kaggle account settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Complete initial setup (downloads data, scrapes stats, trains models)
python main.py setup
```

### Core Development Commands
```bash
# Make match predictions
python main.py predict "Sentinels" "Fnatic"
python main.py predict "Paper Rex" "Team Liquid" --model random_forest

# List available teams
python main.py teams

# Check system status and data availability
python main.py status
```

### Testing and Quality Assurance
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=xml

# Run specific test categories
python -m pytest tests/ -m unit
python -m pytest tests/ -m integration
python -m pytest tests/ -m "not slow"

# Run linting and formatting
flake8 src/ tests/ main.py
black --check --diff src/ tests/ main.py
mypy src/ --ignore-missing-imports

# Format code
black src/ tests/ main.py
```

### Data Management
```bash
# Download only Kaggle datasets
python -c "from src.data_collection import KaggleDataDownloader; KaggleDataDownloader().download_all_datasets()"

# Scrape only VLR.gg team statistics
python -c "from src.data_collection import VLRScraper; s = VLRScraper(); stats = s.scrape_all_teams(); s.save_team_stats(stats)"

# Process data pipeline
python -c "from src.preprocessing import VCTDataProcessor; VCTDataProcessor().process_full_pipeline()"
```

## Architecture

### High-Level Structure
The project follows a modular architecture with clear separation of concerns:

- **Data Collection Layer** (`src/data_collection/`): Handles external data acquisition from Kaggle and VLR.gg
- **Preprocessing Layer** (`src/preprocessing/`): Transforms raw data into ML-ready features
- **Model Layer** (`src/models/`): Implements multiple ML algorithms and ensemble methods
- **Prediction Interface** (`src/prediction/`): Provides CLI and programmatic access to predictions
- **Configuration Management** (`config/`): YAML-based configuration for teams and datasets

### Key Components

#### Data Flow Pipeline
1. **KaggleDataDownloader**: Downloads tournament datasets from multiple Kaggle sources
2. **VLRScraper**: Scrapes real-time team statistics, ratings, and match history
3. **VCTDataProcessor**: Cleans, merges, and engineers features from multiple data sources
4. **VCTMatchPredictor**: Trains and manages multiple ML models (Random Forest, XGBoost, SVM, Neural Networks)
5. **Ensemble System**: Combines best-performing models using voting mechanisms

#### Model Architecture
The system implements a multi-model approach:
- Individual models: Random Forest, XGBoost, Logistic Regression, SVM, Neural Networks
- Ensemble model combining top performers
- Cross-validation and hyperparameter optimization
- Feature importance analysis for model interpretability

### Configuration System
All team data, Kaggle datasets, and regional information is managed through `config/teams.yaml`. This single source of truth contains:
- VCT 2025 team configurations with VLR.gg IDs and URLs
- Kaggle dataset references and IDs
- Regional groupings (Americas, EMEA, APAC, China)

### Testing Strategy
- **Unit tests**: Individual component testing with mocked dependencies
- **Integration tests**: End-to-end workflow testing
- **CLI tests**: Command-line interface validation
- **Performance tests**: Model accuracy and speed benchmarks

## Development Workflow

### Adding New Teams
Edit `config/teams.yaml`:
```yaml
teams:
  new_team_key:
    name: "New Team Name"
    region: "REGION"
    vlr_url: "https://www.vlr.gg/team/ID/team-name"
    vlr_id: 12345
```

### Model Development
New models should be added to `src/models/predictor_models.py` following the established pattern:
- Implement model creation method
- Add to `initialize_models()` method
- Include in training pipeline
- Add appropriate unit tests

### Data Source Integration
New data sources require:
- Downloader/scraper class in `src/data_collection/`
- Integration with `VCTDataProcessor`
- Configuration updates in `config/teams.yaml`
- Comprehensive unit tests

## CI/CD Pipeline

The GitHub Actions workflow includes:
- Multi-version Python testing (3.8-3.11)
- Code quality checks (flake8, black, mypy)
- Unit and integration test execution
- Security scanning with Bandit
- Dependency vulnerability checks with Safety
- Coverage reporting

## Data Dependencies

### External Requirements
- **Kaggle API**: Requires authenticated API credentials
- **VLR.gg**: Web scraping with rate limiting (2-second delays)
- **Internet connectivity**: Required for real-time data updates

### Data Storage Structure
```
data/
├── raw/                    # Kaggle datasets (tournament data)
├── processed/              # Cleaned and engineered features
└── external/               # VLR.gg scraped statistics
```

## Environment Considerations

### Required Environment Variables
- Kaggle credentials should be in `~/.kaggle/kaggle.json`
- No additional environment variables required for basic operation

### Performance Notes
- Initial setup can take 10-15 minutes depending on data download speeds
- Model training requires sufficient RAM (recommended 8GB+)
- Web scraping includes rate limiting to respect VLR.gg servers

### Error Handling
- Robust error handling for network failures during data collection
- Graceful degradation when data sources are unavailable
- Comprehensive logging throughout the pipeline

## Model Performance Tracking

The system maintains model performance metrics in `models/model_performances.json` and provides:
- Accuracy, precision, recall, and F1 scores
- Cross-validation results
- Feature importance rankings
- Model comparison utilities

Use `python main.py status` to view current model performance and system health.