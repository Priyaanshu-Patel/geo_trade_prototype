"""
Claude-powered event analyzer.
Extracts structured event metadata from raw geopolitical text.
Uses prompt caching to minimize API costs across repeated calls.

Fallback: rule-based keyword extractor (no API key needed).
"""
import json
import re
import os
from typing import Optional

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quantitative geopolitical analyst specializing in how world events cascade through financial markets.

Extract structured information from raw news text. Be precise and quantitative.
Base intensity estimates on historical precedents:
- Intensity 0.9+: direct kinetic warfare, full trade embargo
- Intensity 0.7-0.9: major sanctions, border skirmishes, supply crises
- Intensity 0.5-0.7: diplomatic incidents, policy shifts, election outcomes
- Intensity 0.2-0.5: policy announcements, budget events, regulatory changes
- Intensity < 0.2: routine news

Always respond with valid JSON only. No markdown, no explanation."""

EXTRACTION_PROMPT = """Extract structured data from this geopolitical event text.

Event: {event_text}

Return JSON with exactly these fields:
{{
  "event_type": "<conflict|sanction|trade|policy|election|supply_disruption|regulatory|other>",
  "intensity": <float 0.0-1.0>,
  "affected_countries": [<ISO-2 codes, e.g. "RU","UA","IN">],
  "affected_sectors": [<from: energy|defense|tech|finance|commodities|currency|shipping|infrastructure>],
  "directly_affected_assets": [<specific tickers if mentioned, e.g. "XOM","HAL.NS">],
  "summary": "<one factual sentence, max 100 chars>",
  "market_sentiment": "<bearish|neutral|bullish>"
}}"""


# ── Rule-based fallback ───────────────────────────────────────────────────────

KEYWORD_RULES = {
    "conflict": ["war", "attack", "invasion", "military", "troops", "offensive", "bombing", "missile"],
    "sanction": ["sanction", "embargo", "ban", "restrict", "freeze", "penalty"],
    "supply_disruption": ["opec", "production cut", "supply cut", "shortage", "disruption", "outage"],
    "trade": ["trade deal", "fta", "tariff", "import", "export", "agreement"],
    "policy": ["budget", "interest rate", "federal reserve", "rate hike", "stimulus", "fiscal"],
    "election": ["election", "vote", "referendum", "president", "prime minister"],
    "regulatory": ["regulation", "compliance", "cop26", "net zero", "climate", "reform"],
}

SECTOR_KEYWORDS = {
    "energy":  ["oil", "gas", "energy", "opec", "petroleum", "coal", "renewables", "solar"],
    "defense": ["defense", "military", "arms", "weapon", "hal", "lockheed", "boeing", "army", "navy"],
    "tech":    ["tech", "semiconductor", "chip", "software", "it", "tcs", "infosys", "apple"],
    "finance": ["bank", "interest rate", "fed", "rbi", "finance", "credit", "currency"],
    "commodities": ["wheat", "corn", "food", "commodity", "metal", "steel", "copper"],
    "shipping": ["shipping", "port", "freight", "cargo", "vessel", "maritime"],
    "infrastructure": ["infrastructure", "rail", "road", "bridge", "construction"],
}

INTENSITY_MODIFIERS = {
    "major": 0.2, "significant": 0.15, "massive": 0.25, "minor": -0.15,
    "limited": -0.1, "full": 0.2, "partial": -0.1, "historic": 0.15,
}


def extract_event_rule_based(text: str) -> dict:
    """Keyword-based fallback extractor. No API required."""
    text_lower = text.lower()

    event_type = "other"
    for etype, keywords in KEYWORD_RULES.items():
        if any(k in text_lower for k in keywords):
            event_type = etype
            break

    affected_sectors = [
        sector for sector, keywords in SECTOR_KEYWORDS.items()
        if any(k in text_lower for k in keywords)
    ] or ["other"]

    base_intensity = {
        "conflict": 0.75, "sanction": 0.65, "supply_disruption": 0.60,
        "trade": 0.45, "policy": 0.40, "election": 0.50,
        "regulatory": 0.35, "other": 0.30,
    }.get(event_type, 0.30)

    for word, modifier in INTENSITY_MODIFIERS.items():
        if word in text_lower:
            base_intensity = min(1.0, max(0.0, base_intensity + modifier))

    sentiment = "bearish"
    if any(w in text_lower for w in ["deal", "agreement", "growth", "positive", "boost", "gain"]):
        sentiment = "bullish"
    elif any(w in text_lower for w in ["neutral", "stable", "unchanged"]):
        sentiment = "neutral"

    return {
        "event_type": event_type,
        "intensity": round(base_intensity, 2),
        "affected_countries": [],
        "affected_sectors": affected_sectors,
        "directly_affected_assets": [],
        "summary": text[:100],
        "market_sentiment": sentiment,
    }


# ── Claude-powered extractor ──────────────────────────────────────────────────

class EventAnalyzer:
    """
    Extracts structured event metadata using Claude API.
    Falls back to rule-based extraction if API unavailable.
    Uses prompt caching on the system prompt to reduce token costs.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            return self._client
        except ImportError:
            print("[EventAnalyzer] anthropic package not installed. Using rule-based fallback.")
            return None

    def _call(self, prompt: str) -> str:
        client = self._get_client()
        if client is None:
            raise RuntimeError("No Anthropic client")

        # Cache the system prompt to save tokens across multiple calls
        response = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse(self, text: str) -> dict:
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    def extract(self, event_text: str) -> dict:
        """
        Extract structured metadata from raw event text.
        Returns dict with event_type, intensity, affected_sectors, etc.
        Falls back to rule-based on API failure.
        """
        try:
            prompt = EXTRACTION_PROMPT.format(event_text=event_text)
            raw = self._call(prompt)
            result = self._parse(raw)
            result["intensity"] = max(0.0, min(1.0, float(result.get("intensity", 0.5))))
            return result
        except Exception as e:
            print(f"[EventAnalyzer] Claude failed ({e}), using rule-based fallback.")
            return extract_event_rule_based(event_text)
