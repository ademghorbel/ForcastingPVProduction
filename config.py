"""
Configuration for the PV Production Dashboard
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Weather API Settings
WEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Model Configuration
MODEL_PATH = "project_model.pkl"
SCALER_PATH = "project_scaler.pkl"

# Feature columns as they appear in training data
FEATURE_COLUMNS_TIMESTAMP_ONLY = [
    'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 
    'day_Thursday', 'day_Tuesday', 'day_Wednesday', 'hour'
]

FEATURE_COLUMNS_EXOGENOUS = [
    'hour', 'Vitesse vent(m/s)', 'Humidité ambiante(%RH)', 
    'Température ambiante(℃)', 'Irradiation transitoire pente(W/㎡)',
    'day_Friday', 'day_Monday', 'day_Saturday', 'day_Sunday', 
    'day_Thursday', 'day_Tuesday', 'day_Wednesday'
]

# Weather conditions mapping
WEATHER_CONDITIONS = {
    'sunny': ['Clear', 'Sunny'],
    'cloudy': ['Clouds', 'Overcast'],
    'rainy': ['Rain', 'Drizzle'],
    'stormy': ['Thunderstorm'],
    'snowy': ['Snow']
}

# Decision thresholds
IRRADIATION_THRESHOLD_HIGH = 500  # W/m²
IRRADIATION_THRESHOLD_MEDIUM = 200  # W/m²
TEMPERATURE_RANGE = (5, 45)  # Optimal temperature range in Celsius

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "openai", "anthropic", or "gemini"
LLM_MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
LLM_MODEL_OPENAI = "gpt-4"
LLM_MODEL_GEMINI = "gemini-1.5-flash"  # Updated: gemini-pro is deprecated
