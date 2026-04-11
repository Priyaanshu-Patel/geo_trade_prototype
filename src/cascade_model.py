"""
Empirical cascade model: maps (event_type, sector) → expected return.
Grounded in historical research documented in table.csv and financial literature.

This replaces asking the LLM to guess return numbers, which is unreliable.
Claude is used only for the high-level event understanding (event_analyzer.py).
The quantitative market impact comes from this empirical matrix.
"""
import numpy as np
from typing import Optional

# ── Empirical Impact Matrix ───────────────────────────────────────────────────
#
# Values = expected return (fraction) for an event of intensity=1.0
# Sign: positive = prices rise, negative = prices fall
# Sources: table.csv research + academic literature on geopolitical market impacts
#
# Key findings from table.csv:
#   - Defense stocks +81.4% abnormal returns post-conflict
#   - Oil/commodity prices +20-50% on supply disruption (Russia-Ukraine)
#   - Budget/energy sector: +3.16% post-India budget (renewables focus)
#   - Trade deals: IT +2-4%, exports +14-22%
#   - Infrastructure spending: shipping +3-9%

IMPACT_MATRIX = {
    # (event_type, sector) -> base expected return at intensity=1.0
    ("conflict",          "defense"):       +0.045,   # Wars drive defense budgets
    ("conflict",          "energy"):        +0.030,   # Supply disruption → oil up
    ("conflict",          "oil"):           +0.035,
    ("conflict",          "gold"):          +0.025,   # Safe-haven demand
    ("conflict",          "tech"):          -0.020,   # Supply chain risk
    ("conflict",          "finance"):       -0.018,   # Risk-off
    ("conflict",          "commodities"):   +0.020,   # Raw material scarcity
    ("conflict",          "currency"):      +0.012,   # USD strengthens (risk-off)
    ("conflict",          "shipping"):      -0.015,   # Route disruptions

    ("sanction",          "energy"):        +0.025,   # Target country supply cut
    ("sanction",          "finance"):       -0.025,   # Freeze assets, restrict flows
    ("sanction",          "currency"):      -0.040,   # Target currency collapse
    ("sanction",          "gold"):          +0.020,   # Alternative store of value
    ("sanction",          "commodities"):   +0.015,
    ("sanction",          "tech"):          -0.010,

    ("supply_disruption", "energy"):        +0.040,   # Direct supply shock
    ("supply_disruption", "oil"):           +0.045,
    ("supply_disruption", "commodities"):   +0.030,
    ("supply_disruption", "tech"):          -0.025,   # Semiconductor/input shortages
    ("supply_disruption", "shipping"):      +0.020,   # Freight rates spike
    ("supply_disruption", "defense"):       +0.010,
    ("supply_disruption", "gold"):          +0.015,
    ("supply_disruption", "currency"):      +0.008,   # USD safe-haven

    ("trade",             "tech"):          +0.025,   # FTA opens markets, removes double tax
    ("trade",             "commodities"):   +0.020,   # Export growth
    ("trade",             "finance"):       +0.015,
    ("trade",             "shipping"):      +0.018,   # More trade flow
    ("trade",             "energy"):        +0.010,
    ("trade",             "defense"):       -0.005,   # Peace signal

    ("policy",            "energy"):        +0.030,   # Budget allocations to renewables
    ("policy",            "infrastructure"):+0.025,   # Infra spending
    ("policy",            "shipping"):      +0.020,   # Port/waterway investments
    ("policy",            "finance"):       -0.020,   # Rate hikes hurt finance
    ("policy",            "tech"):          -0.015,   # Rate sensitivity
    ("policy",            "gold"):          +0.012,   # Uncertainty hedge
    ("policy",            "currency"):      +0.010,   # Rate hike strengthens USD

    ("election",          "finance"):       +0.015,   # Market-friendly outcome assumed
    ("election",          "energy"):        +0.008,
    ("election",          "tech"):          +0.010,
    ("election",          "defense"):       +0.012,
    ("election",          "currency"):      -0.010,

    ("regulatory",        "energy"):        +0.025,   # COP26 → renewables push
    ("regulatory",        "shipping"):      +0.020,   # Sagarmala-type programs
    ("regulatory",        "infrastructure"):+0.025,
    ("regulatory",        "finance"):       -0.010,
    ("regulatory",        "tech"):          +0.008,

    ("other",             "gold"):          +0.005,
    ("other",             "currency"):      +0.005,
}

# Ticker → sector mapping (must match ASSETS in data_pipeline.py)
TICKER_SECTOR = {
    "AAPL":    "tech",
    "XOM":     "energy",
    "LMT":     "defense",
    "TCS.NS":  "tech",
    "ONGC.NS": "energy",
    "HAL.NS":  "defense",
    "GLD":     "gold",
    "USO":     "oil",
    "UUP":     "currency",
}

# Sentiment multiplier: bearish events amplify negative impacts
SENTIMENT_MULTIPLIER = {
    "bearish": 1.25,
    "neutral": 1.0,
    "bullish": 0.75,
}


def predict_direct_impacts(event_metadata: dict, tickers: list) -> dict:
    """
    Compute direct (pre-cascade) impact for each ticker based on:
      base_impact = IMPACT_MATRIX[(event_type, sector)]
      scaled by intensity and sentiment modifier
    Returns dict: ticker -> expected return (fraction).
    """
    event_type = event_metadata.get("event_type", "other")
    intensity = float(event_metadata.get("intensity", 0.5))
    affected_sectors = event_metadata.get("affected_sectors", [])
    sentiment = event_metadata.get("market_sentiment", "bearish")
    directly_named = event_metadata.get("directly_affected_assets", [])

    sentiment_mult = SENTIMENT_MULTIPLIER.get(sentiment, 1.0)

    impacts = {}
    for ticker in tickers:
        sector = TICKER_SECTOR.get(ticker, "other")

        # Check if this sector is directly affected by the event
        sector_match = sector in affected_sectors or not affected_sectors

        base = IMPACT_MATRIX.get((event_type, sector), 0.0)

        if base == 0.0:
            # Try with generic event type
            base = IMPACT_MATRIX.get(("other", sector), 0.0)

        # Scale by intensity
        impact = base * intensity

        # Apply sentiment: bearish events amplify negative, dampen positive
        if sentiment == "bearish":
            impact = impact * 1.25 if impact < 0 else impact * 0.8
        elif sentiment == "bullish":
            impact = impact * 0.8 if impact < 0 else impact * 1.25

        # Only sectors explicitly named get full impact; others get half
        if affected_sectors and not sector_match:
            impact *= 0.3

        # Ticker directly named in event gets 1.5x boost
        if ticker in directly_named:
            impact *= 1.5

        impacts[ticker] = round(impact, 5)

    return impacts


def score_event(event_metadata: dict) -> str:
    """
    Return a human-readable summary of the event's expected market impact.
    Used for display in the dashboard.
    """
    event_type = event_metadata.get("event_type", "other")
    intensity = event_metadata.get("intensity", 0.5)
    sentiment = event_metadata.get("market_sentiment", "neutral")
    sectors = event_metadata.get("affected_sectors", [])

    severity = "HIGH" if intensity > 0.7 else "MEDIUM" if intensity > 0.4 else "LOW"
    return (
        f"[{severity}] {event_type.upper()} event (intensity={intensity:.2f}, {sentiment}) "
        f"affecting: {', '.join(sectors) or 'general market'}"
    )
