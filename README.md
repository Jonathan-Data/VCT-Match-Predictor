# VCT Prediction System

A comprehensive machine learning system for predicting Valorant Champions Tour (VCT) match outcomes with automated data collection, real-world performance monitoring, and an intuitive GUI interface.

## Features

- **Unified GUI Interface**: Single application with tabbed interface for all functionality
- **Automated Data Collection**: Scrapes team data from rib.gg and vlr.gg with fallback mechanisms
- **Live Match Predictions**: Generate predictions with confidence levels and betting recommendations
- **Performance Monitoring**: Track real-world accuracy vs model expectations (85.3% target)
- **Automated Scheduling**: Set up cron jobs for hands-off operation
- **Team Management**: Easily manage which teams to monitor and collect data for

## Quick Start

### 1. Install Dependencies
```bash
pip3 install tkinter pandas numpy scikit-learn requests beautifulsoup4 selenium joblib schedule
```

### 2. Run the GUI Application
```bash
python3 run_gui.py
```
or
```bash
python3 vct_gui.py
```

### 3. Basic Usage
1. **Dashboard Tab**: View system status and run quick actions
2. **Data Collection Tab**: Manage teams and collect data from sources
3. **Predictions Tab**: Generate match predictions or view upcoming matches
4. **Monitoring Tab**: Track prediction accuracy and add match results
5. **Settings Tab**: Configure automation and view system logs

## System Components

### Core Files
- `vct_gui.py` - Main GUI application (works reliably on macOS)
- `run_gui.py` - Simple launcher script
- `enhanced_rib_scraper.py` - Production-ready data scraper with fallback
- `performance_monitor.py` - Real-world accuracy tracking system
- `automated_data_updater.py` - Scheduled data updates
- `production_deployment.py` - System orchestration and cron job setup

### Data Structure
```
data/
├── teams_to_update.json       # List of teams to monitor
├── gui_settings.json          # GUI application settings
├── live_predictions.json      # Generated predictions
└── performance_monitor.db     # Performance tracking database

logs/
├── gui_YYYYMMDD.log          # GUI application logs
├── data_updater_YYYYMMDD.log # Data collection logs
└── production_YYYYMMDD.log   # System orchestration logs
```

## Usage Guide

### Data Collection
1. Go to the **Data Collection** tab
2. Manage your team list (default VCT teams included)
3. Select data sources (rib.gg recommended)
4. Click "Collect Selected Teams" or "Collect All Teams"
5. Monitor progress in the collection log

### Making Predictions
1. Go to the **Predictions** tab
2. Select Team 1 and Team 2 from dropdowns
3. Enter tournament name
4. Click "Predict Match" for single predictions
5. Use "Generate Live Predictions" for upcoming matches

### Performance Monitoring
1. Go to the **Monitoring** tab
2. Click "Refresh Performance" to see current stats
3. Add actual match results using "Add Match Result"
4. Click "Evaluate Predictions" to update accuracy metrics

### Automation Setup
1. Go to the **Settings** tab
2. Configure update intervals and automation settings
3. Click "Generate Cron Jobs" for automated scheduling
4. Follow the provided instructions to set up cron jobs

## Command Line Interface

For advanced users, individual components can be run from command line:

```bash
# Check system status
python3 production_deployment.py --status

# Run full pipeline
python3 production_deployment.py --pipeline

# Data collection only
python3 production_deployment.py --data-only

# Performance monitoring only
python3 production_deployment.py --monitor-only

# Generate cron job setup
python3 production_deployment.py --setup-cron
```

## Expected Performance

- **Target Accuracy**: 85.3% (based on model testing)
- **Acceptable Range**: 60-85%
- **Performance Monitoring**: Automatic alerts for degraded performance
- **Confidence Calibration**: High confidence predictions should be 80%+ accurate

## Production Deployment

### Automated Scheduling
The system can run automatically via cron jobs:

```bash
# Generate cron setup
python3 production_deployment.py --setup-cron

# Follow instructions to add to crontab
crontab -e
```

Default schedule:
- Full pipeline every 6 hours
- Data updates every 8 hours
- Performance checks every 12 hours

### Manual Operations
- **Data Updates**: Run when new tournaments start or team rosters change
- **Performance Reviews**: Weekly accuracy analysis recommended
- **System Maintenance**: Monthly cleanup of old logs and data

## Troubleshooting

### Common Issues

1. **GUI Won't Start**
   - Ensure tkinter is installed: `sudo apt-get install python3-tk` (Ubuntu/Debian)
   - Check all dependencies are installed

2. **Data Collection Fails**
   - Check internet connectivity
   - Verify rib.gg/vlr.gg are accessible
   - Review rate limiting settings in logs

3. **No Predictions Generated**
   - Ensure team data has been collected
   - Check that teams exist in the monitored list
   - Review error logs for model issues

4. **Performance Monitoring Empty**
   - Add match results manually or via automation
   - Ensure predictions have been generated
   - Check database connectivity

### Log Files
Check the `logs/` directory for detailed error information:
- `gui_YYYYMMDD.log` - GUI application issues
- `data_updater_YYYYMMDD.log` - Data collection problems
- `production_YYYYMMDD.log` - System-wide issues

## Development

### File Structure
- Single main GUI application with tabbed interface
- Modular backend components for data, predictions, and monitoring
- Clean separation between GUI and business logic
- Comprehensive logging and error handling

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request with clear description

## License

This project is for educational and research purposes. Please respect the terms of service of data sources (rib.gg, vlr.gg) when using the scraping functionality.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in the `logs/` directory
3. Test individual components via command line
4. Ensure all dependencies are properly installed

The system is designed to be robust and user-friendly, with comprehensive error handling and helpful status messages throughout the GUI interface.