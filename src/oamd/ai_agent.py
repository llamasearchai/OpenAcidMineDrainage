"""AI Agent integration for OpenAcidMineDrainage using OpenAI API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .analyze import NUMERIC_PARAMETERS, anova_by_site, regression_by_parameter
from .data_models import Measurement
from .ml_forecast import forecast_for_site
from .reporting import generate_report


class AMDAnalysisAgent:
    """AI Agent for AMD data analysis and insights using OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        """Make a call to OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI API call failed: {str(e)}"

    def analyze_data_insights(self, data_summary: Dict[str, Any]) -> str:
        """Get AI-powered insights from water quality data."""
        system_prompt = """
        You are an expert environmental scientist specializing in acid mine drainage (AMD) analysis.
        Your role is to analyze water quality data, identify trends, and provide actionable insights
        for environmental protection and regulatory compliance.

        Key capabilities:
        - Statistical analysis of water quality parameters (pH, conductivity, metals)
        - Trend identification and anomaly detection
        - Impact assessment on natural assets
        - Predictive modeling and forecasting
        - Regulatory compliance evaluation
        """

        user_message = f"Analyze this AMD monitoring data and provide insights: {data_summary}"
        return self._call_openai(system_prompt, user_message)

    def check_regulatory_compliance(self, site_data: Dict[str, Any]) -> str:
        """AI-powered regulatory compliance assessment."""
        system_prompt = """
        You are a regulatory compliance specialist for environmental monitoring.
        Your role is to ensure AMD monitoring data meets all regulatory requirements
        and generate appropriate compliance documentation.

        Key responsibilities:
        - Verify data quality and completeness
        - Check against regulatory thresholds
        - Generate compliance reports
        - Identify potential violations
        - Recommend corrective actions
        """

        user_message = f"Check regulatory compliance for this site data: {site_data}"
        return self._call_openai(system_prompt, user_message)

    def monitor_and_alert(self, current_readings: Dict[str, Any]) -> str:
        """Intelligent monitoring and alerting system."""
        system_prompt = """
        You are an intelligent monitoring system for acid mine drainage sites.
        Your role is to continuously monitor water quality parameters and provide
        real-time insights and alerts.

        Key functions:
        - Real-time threshold monitoring
        - Anomaly detection and alerting
        - Risk assessment
        - Emergency response coordination
        - Trend monitoring and early warning
        """

        user_message = f"Monitor these current readings and provide alerts if needed: {current_readings}"
        return self._call_openai(system_prompt, user_message)

    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary with AI insights."""
        system_prompt = """
        You are an environmental science consultant preparing executive summaries
        for AMD monitoring reports. Create clear, actionable summaries that
        highlight key findings, risks, and recommendations.
        """

        user_message = f"""
        Generate an executive summary for these AMD monitoring results:
        {analysis_results}

        Include:
        - Key findings and trends
        - Environmental impact assessment
        - Compliance status
        - Recommendations for action
        - Risk levels and mitigation strategies
        """

        return self._call_openai(system_prompt, user_message)


def create_amd_agent() -> AMDAnalysisAgent:
    """Factory function to create AMD analysis agent."""
    return AMDAnalysisAgent()


# Integration functions for CLI usage
def ai_analyze_dataset(data_path: str, agent: AMDAnalysisAgent) -> str:
    """AI-powered dataset analysis."""
    try:
        # Load and summarize data for AI analysis
        import pandas as pd
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

        data_summary = {
            "total_records": len(df),
            "sites": df["site_id"].unique().tolist(),
            "parameters": NUMERIC_PARAMETERS,
            "date_range": {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            },
            "summary_stats": df[NUMERIC_PARAMETERS].describe().to_dict(),
        }

        return agent.analyze_data_insights(data_summary)
    except Exception as e:
        return f"Dataset analysis failed: {str(e)}"


def ai_forecast_with_insights(
    data_path: str,
    site_id: str,
    param: str,
    horizon: int,
    agent: AMDAnalysisAgent
) -> str:
    """AI-enhanced forecasting with insights."""
    try:
        # Get ML forecast first
        import pandas as pd
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        forecast_result = forecast_for_site(df, site_id, param, horizon)

        # Enhance with AI insights
        forecast_summary = {
            "site_id": site_id,
            "parameter": param,
            "horizon_days": horizon,
            "forecast_mae": forecast_result.mae,
            "predictions_count": len(forecast_result.predictions),
            "trend": "improving" if forecast_result.mae < 0.5 else "concerning",
        }

        insights = agent.analyze_data_insights(forecast_summary)
        return f"ML Forecast + AI Insights:\n{insights}"
    except Exception as e:
        return f"Forecast with insights failed: {str(e)}"
