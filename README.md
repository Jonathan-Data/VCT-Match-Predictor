# VCT Match Predictor

A full-stack machine learning application designed to forecast the outcomes of official VALORANT Champions Tour (VCT) matches. This project utilizes historical match data and team performance metrics to provide data-driven win probabilities.

---

## Project Overview

The VCT Match Predictor analyzes competitive data—including team win rates, recent form, and map-specific performance—to generate predictions. The application consists of a React-based frontend for user interaction and a Python Flask backend that handles data processing and model inference.

## Key Features

* Match Outcome Prediction: Calculates win probabilities for upcoming VCT matchups using a trained Random Forest classifier.
* Statistical Analysis: Displays side-by-side comparisons of team performance metrics.
* Data Integration: Utilizes historical match results and player statistics to inform the predictive model.
* Responsive Dashboard: A modern web interface for viewing predictions and historical trends.

---

## Tech Stack

### Frontend
* Framework: React (Vite)
* Styling: Tailwind CSS / Modular CSS
* State Management: React Hooks

### Backend and Machine Learning
* API Framework: Flask (Python)
* Machine Learning: Scikit-Learn
* Data Handling: Pandas, NumPy
* Model Type: Random Forest Classifier

---

## Installation and Setup

### Prerequisites
* Node.js (v16.0 or higher)
* Python (v3.10 or higher)
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/Jonathan-Data/VCT-Match-Predictor.git](https://github.com/Jonathan-Data/VCT-Match-Predictor.git)
cd VCT-Match-Predictor

cd server

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt

# Start the Flask server
python app.py
