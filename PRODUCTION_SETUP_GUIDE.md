# VCT Prediction System - Production Setup Guide

## 🚀 Complete Production Deployment

This guide walks you through setting up the complete VCT prediction system for automated production use.

## 📋 System Overview

Your VCT prediction system consists of these components:

### ✅ Available Components
- **Enhanced Rib Scraper** (`enhanced_rib_scraper.py`) - Production-ready data collection
- **Performance Monitor** (`performance_monitor.py`) - Real-world accuracy tracking  
- **Automated Data Updater** (`automated_data_updater.py`) - Scheduled data updates
- **Production Deployment** (`production_deployment.py`) - System orchestration

### 🔧 Missing Components (Optional)
- **Live Predictor** - Can be recreated from existing model files
- **Model File** - Can be generated from your existing training data

## 🛠️ Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
cd /Users/jolauwer/Documents/valorant-vct-predictor-2025
pip3 install schedule pandas numpy scikit-learn requests beautifulsoup4 selenium joblib
```

### 2. Check System Status
```bash
python3 production_deployment.py --status
```

### 3. Test Data Collection
```bash
python3 production_deployment.py --data-only
```

### 4. Setup Automated Scheduling
```bash
python3 production_deployment.py --setup-cron
```

## 📊 Automated Data Updates

The `automated_data_updater.py` provides:

### Features
- **Multi-source Updates**: VLR.gg and rib.gg data collection
- **Configurable Intervals**: Default 6-8 hour update cycles
- **Automatic Retries**: Handles temporary failures gracefully
- **Rate Limiting**: Respects source website limits
- **Data Validation**: Ensures quality before storage
- **Performance Integration**: Links with monitoring system

### Default Schedule
- **VLR Updates**: Every 6 hours
- **Rib Updates**: Every 8 hours (offset)
- **Performance Monitoring**: Every 12 hours
- **Data Cleanup**: Daily at 2 AM

### Configuration
Edit `data/updater_config.json`:
```json
{
  "vlr_update_interval": 6,
  "rib_update_interval": 8,
  "enable_vlr": true,
  "enable_rib": true,
  "enable_performance_monitoring": true,
  "data_retention_days": 30
}
```

### Team List Management
Update monitored teams via `data/teams_to_update.json`:
```json
{
  "teams": [
    "Sentinels", "Fnatic", "Paper Rex", "Team Liquid",
    "G2 Esports", "NRG", "LOUD", "DRX", "NAVI"
  ],
  "updated_at": "2025-01-15T10:00:00"
}
```

## 📈 Performance Monitoring

The performance monitoring system provides:

### Real-time Tracking
- **Accuracy Monitoring**: Compare actual vs predicted results
- **Confidence Calibration**: Ensure confidence levels match reality
- **Time-based Analysis**: Track accuracy by prediction lead time
- **Tournament Context**: Performance breakdown by event importance

### Key Metrics
- **Overall Accuracy**: Basic win/loss prediction success rate
- **Brier Score**: Probability prediction quality (lower is better)
- **Log Loss**: Prediction confidence accuracy
- **Performance vs Expected**: Comparison to 85.3% test accuracy

### Database Schema
- **match_results**: Actual match outcomes
- **predictions**: Model predictions with probabilities
- **performance_evaluations**: Matched predictions vs results

### Usage
```bash
# Check current performance
python3 performance_monitor.py

# Add match results programmatically
python3 -c "
from performance_monitor import PerformanceMonitor, MatchResult
monitor = PerformanceMonitor()
result = MatchResult('match_001', 'Team1', 'Team2', 'Team1', '2-1', '2025-01-15', 'VCT', 'Groups', 'International', 'manual')
monitor.add_match_result(result)
"
```

## 🔄 Production Pipeline

The complete prediction pipeline runs in stages:

### Stage 1: Data Collection
- Updates team statistics from multiple sources
- Validates and stores new data
- Handles source failures gracefully

### Stage 2: Predictions 
- Generates predictions for upcoming matches
- Calculates confidence levels and betting recommendations
- Stores predictions for performance tracking

### Stage 3: Performance Monitoring
- Evaluates recent predictions against actual results
- Updates performance metrics
- Alerts on degraded performance

### Manual Pipeline Execution
```bash
# Run complete pipeline
python3 production_deployment.py --pipeline

# Run individual stages
python3 production_deployment.py --data-only
python3 production_deployment.py --predictions-only  
python3 production_deployment.py --monitor-only
```

## ⏰ Automated Scheduling

### Option 1: Cron Jobs (Recommended)
```bash
# Generate cron setup
python3 production_deployment.py --setup-cron

# Follow the instructions to add to crontab
crontab -e
```

### Option 2: Python Scheduler
```python
from automated_data_updater import AutomatedDataUpdater

updater = AutomatedDataUpdater()
updater.setup_schedule()
updater.run_scheduler()  # Runs continuously
```

### Option 3: System Service (Advanced)
Create `/etc/systemd/system/vct-predictor.service`:
```ini
[Unit]
Description=VCT Prediction System
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/Users/jolauwer/Documents/valorant-vct-predictor-2025
ExecStart=/usr/bin/python3 automated_data_updater.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 📁 Directory Structure

```
valorant-vct-predictor-2025/
├── data/                          # Data storage
│   ├── live_predictions.json      # Current predictions
│   ├── performance_monitor.db     # Performance tracking
│   ├── teams_to_update.json       # Team list config
│   └── updater_config.json        # Updater settings
├── logs/                          # System logs
│   ├── data_updater_YYYYMMDD.log  # Update logs
│   ├── performance_monitor_YYYYMMDD.log
│   └── production_YYYYMMDD.log    # Main system logs
├── models/                        # ML models
│   └── vct_model.joblib           # Trained prediction model
├── enhanced_rib_scraper.py        # Data collection
├── automated_data_updater.py      # Scheduled updates
├── performance_monitor.py         # Accuracy tracking
├── production_deployment.py       # System orchestration
└── PRODUCTION_SETUP_GUIDE.md      # This guide
```

## 🔧 Configuration Files

### `production_config.json`
```json
{
  "prediction_interval_hours": 6,
  "data_update_interval_hours": 8,
  "performance_check_interval_hours": 12,
  "enable_automated_updates": true,
  "enable_predictions": true,
  "enable_monitoring": true,
  "model_file": "vct_model.joblib",
  "log_level": "INFO"
}
```

### `data/updater_config.json`
```json
{
  "vlr_update_interval": 6,
  "rib_update_interval": 8,
  "max_retries": 3,
  "retry_delay": 300,
  "data_retention_days": 30,
  "enable_vlr": true,
  "enable_rib": true,
  "enable_performance_monitoring": true
}
```

## 📊 Monitoring and Maintenance

### Daily Checks
```bash
# System health
python3 production_deployment.py --status

# Recent performance 
python3 -c "
from performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
summary = monitor.get_performance_summary(days_back=7)
print(f'7-day accuracy: {summary.get(\"overall_accuracy\", 0):.1%}')
"
```

### Weekly Reviews
- Check prediction accuracy trends
- Review and update team monitoring list
- Verify data source availability
- Monitor disk usage and log retention

### Monthly Analysis
- Deep dive performance analysis
- Model retraining evaluation
- System optimization review
- Backup and disaster recovery validation

## 🚨 Troubleshooting

### Common Issues

1. **Scrapers Not Working**
   - Check internet connectivity
   - Verify target websites are accessible
   - Review rate limiting settings
   - Check for Cloudflare blocking

2. **Performance Degradation** 
   - Review recent match results accuracy
   - Check for data quality issues
   - Consider model retraining
   - Verify prediction input features

3. **Scheduling Issues**
   - Check cron job syntax
   - Verify file paths are absolute
   - Review log files for errors
   - Ensure Python environment is consistent

### Log Analysis
```bash
# Recent errors
tail -n 100 logs/production_$(date +%Y%m%d).log | grep ERROR

# Update success rate
grep "✅.*completed" logs/data_updater_$(date +%Y%m%d).log | wc -l

# Performance monitoring
tail -n 50 logs/performance_monitor_$(date +%Y%m%d).log
```

## 🎯 Expected Performance

Based on testing:
- **Target Accuracy**: 85.3%
- **Acceptable Range**: 60-85%
- **Performance Alerts**: < 60% or significant degradation
- **Confidence Calibration**: High confidence predictions should be >80% accurate

## 🔄 Next Steps

1. **Setup Automated Scheduling**: Use cron jobs for production
2. **Configure Team Lists**: Add/remove teams based on current VCT season
3. **Monitor Performance**: Track real-world accuracy vs expectations
4. **Data Source Expansion**: Add additional data sources as available
5. **Model Improvements**: Retrain models with new data periodically

## 📞 Support

For issues or improvements:
1. Check system logs in `logs/` directory
2. Verify component availability with `--status`
3. Review configuration files
4. Test individual components manually

The system is designed to be robust and self-recovering, with comprehensive logging for troubleshooting any issues that arise in production.