
import os
import json
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.3-70b-versatile"          # fast + capable on Groq
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

AQI_CATEGORIES = {
    (0,   50):  ("Good",                           "green"),
    (51,  100): ("Moderate",                       "yellow"),
    (101, 150): ("Unhealthy for Sensitive Groups",  "orange"),
    (151, 200): ("Unhealthy",                      "red"),
    (201, 300): ("Very Unhealthy",                 "purple"),
    (301, 500): ("Hazardous",                      "maroon"),
}


def _aqi_category(aqi: float) -> str:
    for (lo, hi), (label, _) in AQI_CATEGORIES.items():
        if lo <= aqi <= hi:
            return label
    return "Hazardous"


def _groq_key() -> str:
    """Read key lazily so load_dotenv() has already run before first call."""
    key = os.getenv("GROQ_API_KEY", "")

 
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com/"
        )
    return key


# ── Core Groq caller ──────────────────────────────────────────────────────────

def _call_groq(
    prompt: str,
    system: str = "",
    max_tokens: int = 800,
    temperature: float = 0.4,
) -> str:
    """
    Single-turn call to Groq via its OpenAI-compatible endpoint.
    Raises RuntimeError on failure.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        GROQ_BASE_URL,
        headers={
            "Authorization": f"Bearer {_groq_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Groq response shape: {e}")


def _call_groq_chat(messages: list, max_tokens: int = 400) -> str:
    """
    Multi-turn call — messages already in OpenAI format
    (system + alternating user/assistant).
    """
    resp = requests.post(
        GROQ_BASE_URL,
        headers={
            "Authorization": f"Bearer {_groq_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
        },
        timeout=30,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Groq response shape: {e}")


# ── Feature 1: Health Risk Advisor ────────────────────────────────────────────

def get_health_advice(
    city: str,
    aqi: float,
    pm25: float | None = None,
    pm10: float | None = None,
    no2: float | None = None,
    user_profile: dict | None = None,
) -> dict:
    """
    Generate personalized health advice for a user based on current AQI.

    user_profile keys (all optional):
        age_group   : "child" | "adult" | "elderly"
        conditions  : list of strings e.g. ["asthma", "heart disease"]
        activity    : "outdoor_work" | "exercise" | "indoor" | "commuting"
    """
    category = _aqi_category(aqi)
    profile  = user_profile or {}

    age_group  = profile.get("age_group", "adult")
    conditions = ", ".join(profile.get("conditions", [])) or "none"
    activity   = profile.get("activity", "general outdoor")

    parts = []
    if pm25: parts.append(f"PM2.5={pm25:.1f} µg/m³")
    if pm10: parts.append(f"PM10={pm10:.1f} µg/m³")
    if no2:  parts.append(f"NO₂={no2:.1f} µg/m³")
    pollutants = ", ".join(parts) or "data unavailable"

    system = (
        "You are an expert public health advisor specialising in air quality and "
        "respiratory health in India. Give clear, actionable, empathetic advice. "
        "Respond ONLY with a valid JSON object — no markdown fences, no extra text."
    )
    prompt = f"""
Current air quality data for {city}:
- AQI: {aqi:.0f} ({category})
- Pollutants: {pollutants}
- Date/Time: {datetime.now().strftime('%A, %d %B %Y, %H:%M IST')}

User Profile:
- Age group: {age_group}
- Pre-existing conditions: {conditions}
- Planned activity: {activity}

Generate a JSON health advisory with exactly these keys:
{{
  "risk_level": "Low|Moderate|High|Very High|Hazardous",
  "summary": "2-sentence plain-language summary",
  "advice": ["up to 4 specific action items"],
  "precautions": ["up to 3 precautions tailored to the user profile"],
  "activities_to_avoid": ["up to 3 activities to avoid right now"],
  "mask_recommendation": "None|N95 recommended|N95 mandatory",
  "ventilation_advice": "one sentence on indoor air management",
  "when_to_seek_help": "one sentence on medical red flags"
}}
"""
    try:
        raw = _call_groq(prompt, system=system, max_tokens=600)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)
        result.update({
            "city": city,
            "aqi": aqi,
            "provider": "groq",
            "generated_at": datetime.utcnow().isoformat(),
        })
        return result
    except json.JSONDecodeError:
        logger.warning("Groq returned non-JSON health advice; wrapping as text.")
        return {
            "city": city, "aqi": aqi, "risk_level": category,
            "summary": raw[:500], "provider": "groq",
            "generated_at": datetime.utcnow().isoformat(),
            "advice": [], "precautions": [], "activities_to_avoid": [],
        }
    except RuntimeError as e:
        logger.error(f"Groq health advice failed: {e}")
        return {"error": str(e), "city": city, "aqi": aqi}


# ── Feature 2: AQI Chatbot ────────────────────────────────────────────────────

def chat_with_aqi_assistant(
    user_message: str,
    context: dict | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    Conversational AQI assistant powered by Groq.

    context dict (optional):
        city        : str
        current_aqi : float
        forecast    : list of {date, aqi} for next 7 days

    history: list of {role: "user"|"model", text: str}
    """
    ctx = context or {}
    city_info = ""
    if ctx.get("city") and ctx.get("current_aqi"):
        cat = _aqi_category(ctx["current_aqi"])
        city_info = (
            f"Current context: {ctx['city']} has AQI {ctx['current_aqi']:.0f} ({cat}). "
        )
        if ctx.get("forecast"):
            fc_str = "; ".join(
                f"{f['date']}: AQI {f['aqi']:.0f}" for f in ctx["forecast"][:3]
            )
            city_info += f"Forecast — {fc_str}."

    system = (
        "You are AQI Assistant, an expert on India's air quality, CPCB standards, "
        "ISRO satellite data (VEDAS), and health impacts of air pollution. "
        "You help users understand AQI readings, health risks, and what actions to take. "
        "Be concise (under 150 words), helpful, and use Indian context. "
        f"{city_info}"
        "At the end of your reply, suggest 2 short follow-up questions the user might ask "
        "as a JSON array in this EXACT format on the last line: "
        'SUGGESTIONS:["question 1","question 2"]'
    )

    # Build OpenAI-format message list
    messages = [{"role": "system", "content": system}]
    for h in (history or []):
        messages.append({
            "role": "assistant" if h["role"] == "model" else "user",
            "content": h["text"],
        })
    messages.append({"role": "user", "content": user_message})

    try:
        full_text = _call_groq_chat(messages, max_tokens=400)

        suggestions = []
        if "SUGGESTIONS:" in full_text:
            reply_part, sug_part = full_text.split("SUGGESTIONS:", 1)
            reply = reply_part.strip()
            try:
                suggestions = json.loads(sug_part.strip())
            except Exception:
                suggestions = []
        else:
            reply = full_text

        return {
            "reply": reply,
            "suggestions": suggestions,
            "provider": "groq",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Groq chatbot error: {e}")
        return {
            "reply": "I'm having trouble connecting right now. Please try again in a moment.",
            "suggestions": [],
            "error": str(e),
        }


# ── Feature 3: Health Report Generator ────────────────────────────────────────

def generate_health_report(
    city: str,
    aqi_stats: dict,
    pollutant_averages: dict,
    forecast_summary: dict,
    satellite_no2: float | None = None,
) -> dict:
    """
    Generate a structured health impact report for a city.

    aqi_stats          : {min, max, avg, trend}  (last 7 days)
    pollutant_averages : {pm25, pm10, no2, so2, co, o3}
    forecast_summary   : {avg_aqi, worst_day, best_day}
    satellite_no2      : float µmol/m² from ISRO VEDAS (optional)
    """
    no2_insight = (
        f"ISRO VEDAS satellite-derived NO₂ (downscaled via AI/ML): "
        f"{satellite_no2:.2f} µmol/m² column density."
        if satellite_no2 is not None else ""
    )

    system = (
        "You are an expert environmental health scientist writing a concise report "
        "for city health officials in India. Use CPCB standards as reference. "
        "Be factual, professional, and actionable. "
        "Respond ONLY with valid JSON — no markdown fences."
    )
    prompt = f"""
Generate a 7-day air quality health report for {city}.

Data Summary:
- AQI range: {aqi_stats.get('min','N/A')} to {aqi_stats.get('max','N/A')} (avg: {aqi_stats.get('avg','N/A')})
- AQI trend: {aqi_stats.get('trend','stable')}
- PM2.5 avg: {pollutant_averages.get('pm25','N/A')} µg/m³ (CPCB 24h standard: 60 µg/m³)
- PM10 avg:  {pollutant_averages.get('pm10','N/A')} µg/m³ (CPCB 24h standard: 100 µg/m³)
- NO₂ avg:   {pollutant_averages.get('no2','N/A')} µg/m³
- SO₂ avg:   {pollutant_averages.get('so2','N/A')} µg/m³
- Next 7 days forecast avg AQI: {forecast_summary.get('avg_aqi','N/A')}
- Worst forecast day: {forecast_summary.get('worst_day','N/A')}
{no2_insight}

Return JSON with exactly these keys:
{{
  "executive_summary": "3 sentences max",
  "compliance_status": {{
    "pm25": "compliant|non-compliant|borderline",
    "pm10": "compliant|non-compliant|borderline",
    "no2":  "compliant|non-compliant|borderline"
  }},
  "health_impacts": {{
    "general_population": "2 sentences",
    "vulnerable_groups": "2 sentences (children, elderly, respiratory patients)",
    "estimated_unhealthy_days": number
  }},
  "primary_pollutant": "pm25|pm10|no2|o3",
  "likely_sources": ["list","of","emission","sources"],
  "recommendations": {{
    "immediate":  ["up to 3 actions for next 24h"],
    "short_term": ["up to 3 actions for next week"],
    "policy":     ["up to 2 policy suggestions"]
  }},
  "satellite_insights": "1 sentence interpreting satellite NO2 data or null",
  "forecast_outlook": "1 sentence",
  "data_confidence": "High|Medium|Low"
}}
"""
    try:
        raw = _call_groq(prompt, system=system, max_tokens=900)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(raw)
        result.update({
            "city": city,
            "provider": "groq",
            "generated_at": datetime.utcnow().isoformat(),
            "report_period": "Last 7 days",
        })
        return result
    except json.JSONDecodeError:
        logger.warning("Groq health report returned non-JSON.")
        return {"error": "Report generation failed — invalid JSON", "raw": raw[:500]}
    except RuntimeError as e:
        logger.error(f"Groq report generation failed: {e}")
        return {"error": str(e)}


# ── Feature 4: Satellite NO₂ Interpretation ──────────────────────────────────

def interpret_satellite_no2(
    city: str,
    no2_column_density: float,
    ground_no2: float | None = None,
    downscale_method: str = "XGBoost",
) -> str:
    """
    SIH1734 core feature: explain downscaled satellite NO₂ data in plain language.

    no2_column_density : float — Sentinel-5P tropospheric NO₂ in µmol/m²
    ground_no2         : float — CPCB ground station NO₂ in µg/m³ (optional)
    downscale_method   : str   — ML model used (XGBoost / Random Forest / ANN)
    """
    ground_str = (
        f"CPCB ground station NO₂: {ground_no2:.1f} µg/m³"
        if ground_no2 else "No ground station data available."
    )
    system = (
        "You are an ISRO remote sensing scientist explaining satellite air quality data "
        "to a general audience in India. Be clear, avoid jargon, keep it under 120 words."
    )
    prompt = f"""
Satellite data for {city} (from ISRO VEDAS / Sentinel-5P):
- Tropospheric NO₂ column density: {no2_column_density:.4f} µmol/m²
- AI/ML downscaling method used: {downscale_method} (SIH1734 methodology)
- {ground_str}

In 3-4 sentences explain:
1. What this NO₂ level means for air quality
2. Why satellite data + AI downscaling is more useful than ground stations alone
3. Any health concern for residents of {city}
"""
    try:
        return _call_groq(prompt, system=system, max_tokens=200)
    except RuntimeError as e:
        logger.error(f"Groq NO2 interpretation failed: {e}")
        return f"Unable to generate satellite interpretation: {e}"