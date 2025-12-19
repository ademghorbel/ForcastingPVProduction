"""
Gemini AI Agent for Battery Management Recommendations
Intelligent solar production analysis and battery management advice
"""
import os
from typing import Dict, List
import google.generativeai as genai
from config import GEMINI_API_KEY, LLM_MODEL_GEMINI


class AIDecisionAgent:
    """AI-powered decision support for battery management using Gemini"""
    
    def __init__(self):
        """Initialize the Gemini AI agent"""
        self.initialized = False
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini API"""
        try:
            if not GEMINI_API_KEY:
                print("⚠ GEMINI_API_KEY not set")
                return
            
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(LLM_MODEL_GEMINI)
            self.initialized = True
            print(f"✓ Gemini API initialized with model: {LLM_MODEL_GEMINI}")
        except Exception as e:
            print(f"✗ Error initializing Gemini: {e}")
    
    def get_battery_recommendations(
        self,
        current_prediction: float,
        forecast_predictions: List[float],
        current_weather: Dict,
        forecast_data: List[Dict],
        battery_capacity: float = 10.0,
        battery_soc: float = 50.0
    ) -> str:
        """
        Generate battery management recommendations using Gemini AI
        
        Args:
            current_prediction: Current power prediction in kW
            forecast_predictions: List of forecast predictions
            current_weather: Current weather conditions
            forecast_data: List of forecast data
            battery_capacity: Battery capacity in kWh
            battery_soc: Battery state of charge in %
        
        Returns:
            AI-generated recommendations as string
        """
        if not self.initialized:
            return "❌ Gemini Agent not initialized. Check GEMINI_API_KEY in .env file"
        
        # Prepare context for AI
        context = self._prepare_context(
            current_prediction,
            forecast_predictions,
            current_weather,
            forecast_data,
            battery_capacity,
            battery_soc
        )
        
        try:
            return self._get_gemini_recommendation(context)
        except Exception as e:
            return f"⚠️ Error generating recommendations: {str(e)}"
    
    def _prepare_context(
        self,
        current_prediction: float,
        forecast_predictions: List[float],
        current_weather: Dict,
        forecast_data: List[Dict],
        battery_capacity: float,
        battery_soc: float
    ) -> str:
        """Prepare detailed context for Gemini analysis"""
        
        avg_power = sum(forecast_predictions) / len(forecast_predictions) if forecast_predictions else 0
        max_power = max(forecast_predictions) if forecast_predictions else 0
        min_power = min(forecast_predictions) if forecast_predictions else 0
        
        context = f"""
You are a solar energy and battery management expert AI. Analyze this solar PV system data and provide specific, actionable battery management recommendations.

## CURRENT SYSTEM STATE
- Current Solar Production: {current_prediction:.2f} kW
- Battery Capacity: {battery_capacity:.1f} kWh
- Battery State of Charge: {battery_soc:.1f}%
- Current Temperature: {current_weather.get('temperature', 'N/A')}°C
- Current Humidity: {current_weather.get('humidity', 'N/A')}%
- Cloud Coverage: {current_weather.get('clouds', 'N/A')}%
- Condition: {current_weather.get('condition', 'N/A')} - {current_weather.get('description', 'N/A')}

## 24-HOUR FORECAST ANALYSIS
- Average Predicted Power: {avg_power:.2f} kW
- Peak Predicted Power: {max_power:.2f} kW
- Minimum Predicted Power: {min_power:.2f} kW
- Total Forecast Points: {len(forecast_predictions)}

## FORECAST TIMELINE
"""
        for i, (pred, fcast) in enumerate(zip(forecast_predictions[:8], forecast_data[:8])):  # First 8 hours
            context += f"\n- Hour {i}: {pred:.2f} kW (Temp: {fcast.get('temperature', 'N/A')}°C, Cloud: {fcast.get('clouds', 'N/A')}%)"
        
        context += f"""

## REQUESTED ANALYSIS & RECOMMENDATIONS

Based on the solar production forecast and current conditions, provide:

1. **CHARGING STRATEGY**: When should the battery be charged? (specific times or conditions)
   - Consider solar production peaks
   - Account for weather forecasts
   - Optimize for energy cost/availability

2. **DISCHARGING STRATEGY**: When should stored energy be used? (specific times or conditions)
   - Identify low production periods
   - Suggest energy usage patterns
   - Maximize self-consumption

3. **IMMEDIATE ACTIONS**: What should be done right now?
   - Current battery charge action
   - Priority adjustments needed

4. **RISK ALERTS**: Any concerns or warnings?
   - Potential grid issues
   - Battery degradation risks
   - Unusual patterns

5. **EFFICIENCY TIPS**: How to optimize this system?
   - Energy management improvements
   - Cost reduction opportunities

**Format your response as clear bullet points or numbered lists for easy reading in a dashboard.**
"""
        return context
    
    def _get_gemini_recommendation(self, context: str) -> str:
        """Get recommendation from Gemini"""
        try:
            response = self.model.generate_content(context)
            return response.text
        except Exception as e:
            return f"❌ Gemini API Error: {str(e)}"


def get_ai_recommendation(
    current_prediction: float,
    forecast_predictions: List[float],
    current_weather: Dict,
    forecast_data: List[Dict]
) -> str:
    """
    Convenience function to get AI recommendations from Gemini
    
    Args:
        current_prediction: Current power prediction
        forecast_predictions: List of forecast predictions
        current_weather: Current weather data
        forecast_data: List of forecast data
    
    Returns:
        AI-generated recommendation string
    """
    agent = AIDecisionAgent()
    return agent.get_battery_recommendations(
        current_prediction,
        forecast_predictions,
        current_weather,
        forecast_data
    )
