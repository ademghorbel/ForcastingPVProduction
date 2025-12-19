# ⚡ Voltwise - Solar PV Production Dashboard

A real-time solar power production forecasting application with interactive weather analysis and machine learning predictions.

## 📋 Project Overview

This project provides:
- **Real-time weather data** integration from OpenWeatherMap API
- **ML-powered solar production forecasts** using fine-tuned XGBoost model with exogenous features
- **Interactive Streamlit dashboard** with live charts and analytics
- **24-hour production forecasts** with weather correlation analysis

**Status**: ✅ Fully operational with trained XGBoost model

---

## 📦 Project Structure

```
.
├── app.py                          # Main Streamlit dashboard application
├── config.py                       # Configuration and constants
├── weather_api.py                  # OpenWeatherMap API integration
├── model_utils.py                  # XGBoost model loading and prediction
├── best_model_exogenous.pkl        # Fine-tuned XGBoost model (trained)
├── project_scaler.pkl              # Feature scaler for preprocessing
├── BDDsfax.xlsx                    # Training dataset (Sfax, Tunisia)
├── projet_ML (1).ipynb             # ML model development notebook
├── .env                            # Environment variables (API keys)
├── requirements.txt                # Python dependencies
├── logo white.png                  # Dashboard logo
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8+
- Virtual environment: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac)
- OpenWeatherMap API key (free tier: https://openweathermap.org/api)

### 1. Get Your Weather API Key
1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for free account
3. Copy your API key

### 2. Configure Environment
Create `.env` file in project root:
```env
WEATHER_API_KEY=your_openweathermap_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Dashboard
```bash
streamlit run app.py
```

Dashboard opens at: `http://localhost:8501`

---

## 💡 How to Use

### Basic Workflow
1. **Enter Location**: City name + country code (e.g., "Sfax", "TN")
2. **Click "Fetch Weather & Predict"**
3. **View Results**:
   - Current conditions (temp, humidity, wind, clouds)
   - 24-hour production forecast chart
   - Detailed forecast table with predictions
   - Statistical analysis (avg, peak, total energy)

### Dashboard Sections

#### 🌤️ Current Conditions
- Temperature, humidity, wind speed, cloud coverage
- Current solar irradiance
- Current power production estimate

#### 📈 24-Hour Production Forecast
- Predicted solar power output (kW) with trend visualization
- High/medium/low production threshold indicators
- Hourly granularity for detailed planning

#### 🌡️ Weather Forecast
- Temperature trends (24 hours)
- Cloud coverage patterns
- Wind and humidity forecasts

#### 📊 Statistics & Analysis
- Average power (24h)
- Peak power prediction + time
- Total estimated energy (kWh)
- Average temperature with min/max

---

## 🤖 Machine Learning Model

### Model Details
- **Type**: XGBoost Regressor (fine-tuned)
- **Training Data**: Historical solar production data from Sfax, Tunisia
- **Dataset**: BDDsfax.xlsx (hourly measurements)
- **Model File**: `best_model_exogenous.pkl`

### Features Used (Exogenous)
- `hour` - Hour of day (0-23)
- `Température ambiante(℃)` - Ambient temperature
- `Humidité ambiante(%RH)` - Relative humidity
- `Vitesse vent(m/s)` - Wind speed
- `Irradiation transitoire pente(W/㎡)` - Solar irradiance
- `day_Monday` to `day_Sunday` - Day of week (one-hot encoded)

### Performance
- Uses real trained model from `best_model_exogenous.pkl`
- Exogenous features incorporate weather variables
- Non-negative predictions (physical constraint enforced)

---

## 📚 Development & Training

### Jupyter Notebook
The ML model development is documented in `projet_ML (1).ipynb`:
- Data exploration and preprocessing
- Feature engineering
- Model training and evaluation
- Hyperparameter tuning

### Training Data
- **Source**: `BDDsfax.xlsx`
- **Location**: Sfax, Tunisia
- **Variables**: Temperature, humidity, wind, irradiance, power output

To retrain the model:
1. Open `projet_ML (1).ipynb` in Jupyter
2. Update training parameters if needed
3. Run all cells to train XGBoost
4. Export model as `best_model_exogenous.pkl`

---

## 🔧 Configuration

### .env Variables
```env
WEATHER_API_KEY=your_api_key_here          # Required: OpenWeatherMap API key
```

### Model Configuration (config.py)
- `WEATHER_API_BASE_URL` - OpenWeatherMap current weather endpoint
- `WEATHER_FORECAST_URL` - OpenWeatherMap forecast endpoint
- `MODEL_PATH` - Path to trained model file
- `FEATURE_COLUMNS_EXOGENOUS` - Features expected by model

### Thresholds
- `IRRADIATION_THRESHOLD_HIGH` = 500 W/m²
- `IRRADIATION_THRESHOLD_MEDIUM` = 200 W/m²
- `TEMPERATURE_RANGE` = (5, 45)°C

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.28.1 |
| **ML Model** | XGBoost 2.0.3 |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Plotly 5.18.0 |
| **Weather API** | OpenWeatherMap REST API |
| **Python Version** | 3.8+ |

---

## 📋 Dependencies

See `requirements.txt` for full list:
- streamlit - Dashboard framework
- xgboost - ML model
- pandas - Data manipulation
- numpy - Numerical computing
- plotly - Interactive charts
- requests - HTTP requests
- python-dotenv - Environment management

Install all: `pip install -r requirements.txt`

---

## 🚨 Troubleshooting

### "Weather API Error: 404"
**Solution**: Check city spelling (e.g., use "Hammamet" not "hammemet")
- Try major cities first (Sfax, Tunis, Sousse)
- Use 2-letter country code (TN for Tunisia, FR for France)

### "Model not loaded"
**Solution**: Ensure `best_model_exogenous.pkl` exists in project root
- Check file exists: `ls *.pkl`
- Retrain if missing using `projet_ML (1).ipynb`

### "WEATHER_API_KEY not set"
**Solution**: 
1. Get free key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create `.env` file with: `WEATHER_API_KEY=your_key_here`
3. Restart app: `streamlit run app.py`

---

## 📝 License

Project for solar forecasting and energy management.

## 👥 Authors

Created for solar energy research and production forecasting.

---

**Last Updated**: December 2025  
**Model Status**: ✅ Fine-tuned XGBoost with exogenous features loaded  
**Dashboard Status**: ✅ Fully operational  
**AI Features**: Removed (focus on core forecasting)


---

## 🎯 Use Cases

### Solar Farm Operators
- Forecast daily production for planning
- Optimize battery charging/discharging
- Plan maintenance windows

### Energy Storage Managers
- Predict when to charge/discharge storage
- Maximize revenue from load shifting
- Meet grid demand forecasts

### Researchers
- Analyze weather impact on production
- Validate ML model predictions
- Develop improved forecasting models

---

## 📊 Model Performance & Example Predictions

### XGBoost Model Performance (Latest Results)

#### Training Phase Results (4622 samples)
| Metric | Baseline | Exogenous | Improvement |
|--------|----------|-----------|-------------|
| MAE (kW) | 0.2534 | 0.0205 | ↓ 91.9% |
| RMSE (kW) | 0.4997 | 0.0369 | ↓ 92.6% |
| R² Score | 0.8774 | 0.9993 | ↑ 12.2 pts |

#### Test Phase Results (Real-world generalization, 1150 samples: Dec 24 2024 - Jan 15 2025)
| Metric | Baseline | Exogenous | Fine-tuned |
|--------|----------|-----------|-----------|
| MAE (kW) | 0.3944 | 0.2734 | **0.2567** |
| RMSE (kW) | 0.6927 | 0.5581 | **0.5332** |
| R² Score | 0.6087 | 0.7460 | **0.7742** |

**Key Insight**: The exogenous model reduces prediction error by **30.7%** in production vs baseline. Fine-tuning adds an additional **6.1%** improvement.

#### Model Configuration
- **Framework**: XGBoost Regressor
- **Optimized Hyperparameters**:
  - `learning_rate=0.05`
  - `max_depth=5`
  - `n_estimators=100`
- **Cross-validation MAE**: 0.2262 kW
- **Input Features**: Hour, day of week, temperature, humidity, wind speed, solar irradiance

### Example Prediction

Given weather input:
```
Temperature: 28°C
Humidity: 45%
Wind Speed: 3.2 m/s
Cloud Coverage: 20% (Irradiance: ~800 W/m²)
Hour: 14:00
```

Expected output:
```
Predicted Power: 4.8 kW
Model Confidence: High (sunny conditions)
Expected Accuracy: ±0.26 kW (based on test MAE)
```

---

## 🔐 Security Notes

1. **Never commit `.env` file** to git (it's ignored)
2. **Keep API keys private** - regenerate if exposed
3. **Use environment variables** in production
4. **Monitor API usage** to avoid unexpected charges

---

## 📚 Documentation Files

- **README.md** (this file) - Complete usage guide
- **projet_ML (1).ipynb** - ML model training and validation
- **config.py** - All configuration constants
- **Code comments** - Inline documentation

---

## 🚦 Status Indicators

| Component | Status | Notes |
|-----------|--------|-------|
| Core Application | ✅ Working | Tested and validated |
| Weather API | ✅ Working | Requires valid API key |
| ML Model | ✅ Working | Pre-trained XGBoost |
| Gemini AI | ✅ Working | Requires API key |
| Claude AI | ✅ Working | Requires API key |
| GPT-4 AI | ✅ Working | Requires API key |

---

## 🔄 Next Steps

1. **First Run**: Test with demo location (Sfax, Tunisia)
2. **Add Your Location**: Update to your solar site
3. **Fine-tune Model**: Retrain with your local historical data
4. **Set Up Alerts**: Integrate with your monitoring system
5. **Deploy**: Host on cloud (Heroku, Azure, AWS)

---

## 💬 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `config.py` for configuration options
3. Check browser console (F12) for errors
4. Enable debug logging: `streamlit run app.py --logger.level=debug`

---

## 📝 License

This project is provided as-is for educational and commercial use.

---

**Last Updated**: December 18, 2025  
**Python Version**: 3.8+  
**Framework**: Streamlit 1.28.1
