# VCT 2025 Champions Match Predictor 🎮

A comprehensive machine learning project for predicting match outcomes in the VALORANT Champions Tour (VCT) 2025 tournament. This project combines data from Kaggle datasets with real-time team statistics scraped from VLR.gg to train multiple ML models for accurate match prediction.

## 🏆 Features

- **Multi-source Data Collection**: Automated downloading from Kaggle datasets and web scraping from VLR.gg
- **Advanced Preprocessing**: Comprehensive data cleaning, feature engineering, and transformation pipeline
- **Multiple ML Models**: Random Forest, XGBoost, Logistic Regression, SVM, and Neural Networks
- **Ensemble Learning**: Combines best-performing models for improved accuracy
- **Team Statistics**: Real-time stats for all 16 VCT Champions 2025 teams
- **Easy-to-use CLI**: Simple command-line interface for predictions and analysis
- **Regional Analysis**: Handles cross-regional matchups and regional advantages

## 🏅 Supported Teams

### Americas
- **G2 Esports**
- **NRG**
- **Sentinels**
- **MIBR**

### EMEA
- **Team Liquid**
- **GIANTX**
- **Fnatic**
- **Team Heretics**

### APAC
- **Paper Rex**
- **Rex Regum Qeon**
- **T1**
- **DRX**

### China
- **Bilibili Gaming**
- **Dragon Ranger Gaming**
- **Edward Gaming**
- **Xi Lai Gaming**

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Kaggle account and API credentials (for data download)
- Internet connection (for web scraping)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd valorant-vct-predictor-2025
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API credentials:**
   - Create a Kaggle account at [kaggle.com](https://kaggle.com)
   - Go to your account settings and create a new API token
   - Download the `kaggle.json` file
   - Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\\Users\\<username>\\.kaggle\\kaggle.json` (Windows)
   
   ```bash
   # Linux/Mac
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Run the initial setup:**
   ```bash
   python main.py setup
   ```

   This will:
   - Download all VCT 2025 datasets from Kaggle
   - Scrape team statistics from VLR.gg
   - Process and clean all data
   - Train multiple ML models
   - Create an ensemble model

## 📖 Usage

### Basic Commands

```bash
# Show help
python main.py --help

# List all available teams
python main.py teams

# Check system status
python main.py status

# Predict a match outcome
python main.py predict "Sentinels" "Fnatic"
python main.py predict "Paper Rex" "Team Liquid" --model random_forest
```

### Example Output

```
🥊 Match Prediction: Sentinels vs Fnatic
============================================================

🏆 Predicted Winner: Sentinels
📊 Win Probability: 67.3%
🎯 Confidence: 34.6%
🤖 Model Used: ensemble

📈 Team Comparison:
  Sentinels:
    Region: Americas
    Rating: 1247
    Win Rate: 0.73

  Fnatic:
    Region: EMEA
    Rating: 1189
    Win Rate: 0.68
```

## 🔧 Project Structure

```
valorant-vct-predictor-2025/
├── config/
│   └── teams.yaml              # Team configurations and Kaggle dataset info
├── data/
│   ├── raw/                    # Raw Kaggle datasets
│   ├── processed/              # Cleaned and engineered data
│   └── external/               # VLR.gg scraped data
├── models/                     # Trained ML models
├── notebooks/                  # Jupyter notebooks for analysis
├── src/
│   ├── data_collection/        # Kaggle downloader and VLR scraper
│   ├── preprocessing/          # Data cleaning and feature engineering
│   ├── models/                 # ML model implementations
│   └── prediction/             # Prediction interfaces
├── tests/                      # Unit and integration tests
├── main.py                     # Main CLI entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🤖 Machine Learning Models

### Available Models

1. **Random Forest** - Ensemble of decision trees, robust to overfitting
2. **XGBoost** - Gradient boosting framework, excellent for tabular data
3. **Logistic Regression** - Linear model for binary classification
4. **Support Vector Machine (SVM)** - Maximum margin classifier
5. **Neural Network** - Deep learning model with multiple layers
6. **Ensemble** - Combines best performing models using voting

### Features Used

- **Team Ratings**: VLR.gg team ratings
- **Win Rates**: Overall and round-specific win rates
- **Combat Scores**: Average combat scores per team
- **Regional Matchups**: Same/cross-region indicators
- **Recent Form**: Recent match performance trends
- **Head-to-head History**: Past matchup results

### Model Performance

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation scores
- Feature importance analysis

## 📊 Data Sources

### Kaggle Datasets

1. **Valorant Masters Bangkok 2025**
2. **Valorant Stage 2 2025 (All Regions)**
3. **Valorant Stage 1 2025 (All Regions)**
4. **Valorant Masters Toronto 2025**
5. **Valorant Kickoff 2025 (All Regions)**

### VLR.gg Web Scraping

- Real-time team statistics
- Player information
- Recent match results
- Team ratings and rankings

## 🛠️ Development

### Adding New Teams

Edit `config/teams.yaml` to add new teams:

```yaml
teams:
  new_team:
    name: "New Team Name"
    region: "REGION"
    vlr_url: "https://www.vlr.gg/team/ID/team-name"
    vlr_id: 12345
```

### Custom Models

Add new models in `src/models/predictor_models.py`:

```python
def create_custom_model(self, **kwargs):
    # Your custom model implementation
    return YourCustomModel(**kwargs)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Metrics

The system tracks multiple performance metrics:

- **Prediction Accuracy**: How often the model correctly predicts the winner
- **Confidence Calibration**: How well the predicted probabilities match actual outcomes
- **Feature Importance**: Which factors most influence predictions
- **Cross-regional Performance**: How well the model handles different regional matchups

## 🔍 Troubleshooting

### Common Issues

1. **Kaggle API Error**
   ```
   Solution: Ensure kaggle.json is properly configured and you have accepted dataset terms
   ```

2. **VLR.gg Scraping Timeout**
   ```
   Solution: The scraper includes rate limiting. If issues persist, increase delay in VLRScraper
   ```

3. **No Training Data**
   ```
   Solution: Run 'python main.py setup' to download and process data
   ```

4. **Model Not Found**
   ```
   Solution: Check 'python main.py status' and retrain models if needed
   ```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests: `python -m pytest`
6. Commit changes: `git commit -am 'Add feature'`
7. Push to branch: `git push origin feature-name`
8. Create a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Format code with `black`: `black src/`
- Lint with `flake8`: `flake8 src/`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Riot Games** for VALORANT and the VCT tournament
- **VLR.gg** for comprehensive VALORANT esports statistics
- **Kaggle** community for providing tournament datasets
- **Scikit-learn**, **XGBoost**, and **TensorFlow** teams for ML frameworks

## 📞 Support

- Create an issue for bug reports
- Join discussions for feature requests
- Check existing issues before creating new ones

## 🔮 Future Enhancements

- [ ] Real-time match prediction during live games
- [ ] Web dashboard with interactive visualizations
- [ ] Player-level statistics and analysis
- [ ] Integration with more data sources
- [ ] Advanced deep learning models
- [ ] Mobile app interface
- [ ] Tournament bracket predictions
- [ ] Live betting odds comparison

## 📊 Example Predictions

### Recent Predictions (Hypothetical)

| Match | Predicted Winner | Confidence | Actual Winner | Correct |
|-------|------------------|------------|---------------|---------|
| Sentinels vs Fnatic | Sentinels | 67% | Sentinels | ✅ |
| Paper Rex vs T1 | Paper Rex | 72% | T1 | ❌ |
| Team Liquid vs G2 | Team Liquid | 63% | Team Liquid | ✅ |

*Note: These are example predictions for demonstration purposes.*

---

**Made with ❤️ for the VALORANT esports community**