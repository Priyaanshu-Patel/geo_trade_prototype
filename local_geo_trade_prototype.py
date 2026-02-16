import json
import re
import subprocess
import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from tqdm import trange

# =====================================================
# CONFIG
# =====================================================

ASSETS = [
    "AAPL", "MSFT", "GOOG", "TSLA",
    "XOM", "BP",
    "EURUSD", "USDINR",
    "GOLD", "SILVER"
]

INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST = 0.0005
SLIPPAGE = 0.0005

OLLAMA_MODEL = "phi3:mini"

random.seed(42)
np.random.seed(42)

# =====================================================
# OLLAMA HELPERS
# =====================================================

def ollama_chat(prompt: str, model: str = OLLAMA_MODEL) -> str:
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(prompt)

    if stderr.strip():
        print("⚠️ Ollama stderr:", stderr)

    return stdout.strip()


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found:\n" + text)

    cleaned = match.group(0)
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)

# =====================================================
# SIMPLE EVENT PARSER
# =====================================================

def simple_event_parser(text: str) -> Dict[str, Any]:
    t = text.lower()

    if "sanction" in t:
        return {"entities": ["XOM", "BP"], "event_type": "sanction", "intensity": 0.7, "summary": text}
    if "conflict" in t or "military" in t:
        return {"entities": ["GOLD", "SILVER"], "event_type": "conflict", "intensity": 0.8, "summary": text}
    if "election" in t:
        return {"entities": ["USDINR", "EURUSD"], "event_type": "election", "intensity": 0.5, "summary": text}
    if "supply" in t:
        return {"entities": ["XOM"], "event_type": "supply", "intensity": 0.6, "summary": text}

    return {"entities": [], "event_type": "other", "intensity": 0.3, "summary": text}

# =====================================================
# MARKET SIMULATION
# =====================================================

def simulate_market(n_steps: int) -> pd.DataFrame:
    prices = pd.DataFrame(index=range(n_steps), columns=ASSETS, dtype=float)

    init = np.linspace(100, 200, len(ASSETS))
    vols = np.linspace(0.0005, 0.005, len(ASSETS))

    prices.iloc[0] = init

    for t in range(1, n_steps):
        prices.iloc[t] = prices.iloc[t - 1] * np.exp(
            np.random.normal(0, vols)
        )

    return prices

# =====================================================
# GRAPH
# =====================================================

class DynamicGraph:
    def __init__(self, assets: List[str]):
        self.G = nx.Graph()
        for a in assets:
            self.G.add_node(a)

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                self.G.add_edge(assets[i], assets[j], weight=0.05)

    def update(self, event: Dict[str, Any], correlations: Dict):
        ents = [e.upper() for e in event["entities"]]

        for u, v, d in self.G.edges(data=True):
            w = d["weight"]
            corr = correlations.get(tuple(sorted((u, v))), 0.0)

            boost = 0.3 * corr
            if u in ents or v in ents:
                boost += 0.5 * event["intensity"]

            d["weight"] = 0.9 * w + boost

    def summary(self, k: int = 6) -> str:
        edges = sorted(
            self.G.edges(data=True),
            key=lambda x: x[2]["weight"],
            reverse=True
        )[:k]
        return "; ".join(f"{u}-{v}:{d['weight']:.2f}" for u, v, d in edges)

# =====================================================
# GNN SURROGATE
# =====================================================

def gnn_surrogate_predict(graph, event, recent_prices):

    last_returns = (
        recent_prices.iloc[-1] / recent_prices.iloc[-2] - 1
        if len(recent_prices) > 1
        else pd.Series(0, index=ASSETS)
    )

    prompt = f"""
You MUST return valid JSON only.
No explanations.

Graph:
{graph.summary()}

Event:
{event["summary"]}

Recent returns:
{last_returns.to_dict()}

Return JSON:
{{ "ASSET": {{ "pred_return": float, "confidence": float }} }}
"""

    try:
        raw = ollama_chat(prompt)
        return extract_json(raw)
    except Exception:
        return {a: {"pred_return": 0.0, "confidence": 0.0} for a in ASSETS}

# =====================================================
# RL SURROGATE
# =====================================================

def rl_surrogate_action(predictions, portfolio, cash):

    prompt = f"""
You MUST return valid JSON only.
No explanations.

Predictions:
{predictions}

Portfolio:
{portfolio}
Cash: {cash}

Rules:
- Max 3 trades
- Max 20% per asset

Return JSON:
{{ "actions": {{ "ASSET": {{ "action": "buy|sell|hold", "size": float }} }} }}
"""

    try:
        raw = ollama_chat(prompt)
        return extract_json(raw)
    except Exception:
        return {"actions": {}}

# =====================================================
# ACTION SANITIZER (CRITICAL)
# =====================================================

def sanitize_actions(actions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    clean = {}

    for a, cmd in actions.items():
        if not isinstance(cmd, dict):
            continue

        action = cmd.get("action", "hold")
        size = cmd.get("size", 0.0)

        if action not in ("buy", "sell", "hold"):
            action = "hold"

        try:
            size = float(size)
        except Exception:
            size = 0.0

        clean[a] = {"action": action, "size": size}

    return clean

# =====================================================
# EXECUTION
# =====================================================

def apply_actions(actions, prices, portfolio, cash):

    total_value = cash + sum(portfolio[a] * prices[a] for a in ASSETS)

    for a, cmd in actions.items():
        action = cmd["action"]
        size = cmd["size"]

        if action == "hold" or size <= 0:
            continue

        target_val = size * total_value
        target_units = target_val / prices[a]
        current_units = portfolio[a]

        if action == "buy":
            delta = max(0, target_units - current_units)
            cost = delta * prices[a] * (1 + TRANSACTION_COST + SLIPPAGE)
            if cost <= cash:
                portfolio[a] += delta
                cash -= cost

        elif action == "sell":
            delta = max(0, current_units - target_units)
            proceeds = delta * prices[a] * (1 - TRANSACTION_COST - SLIPPAGE)
            portfolio[a] -= delta
            cash += proceeds

    return portfolio, cash

# =====================================================
# MAIN
# =====================================================

def run_prototype(steps: int = 60):

    prices = simulate_market(steps + 2)

    correlations = {}
    for i in range(len(ASSETS)):
        for j in range(i + 1, len(ASSETS)):
            s1 = prices[ASSETS[i]].pct_change().fillna(0)
            s2 = prices[ASSETS[j]].pct_change().fillna(0)
            correlations[(ASSETS[i], ASSETS[j])] = float(np.corrcoef(s1, s2)[0, 1])

    graph = DynamicGraph(ASSETS)
    portfolio = {a: 0.0 for a in ASSETS}
    cash = INITIAL_CAPITAL
    history = []

    last_preds = {a: {"pred_return": 0.0, "confidence": 0.0} for a in ASSETS}

    event_times, buy_times, sell_times = [], [], []
    event_values, buy_values, sell_values = [], [], []

    events = [
        "Sanctions imposed on oil exporter",
        "Military conflict escalates in region",
        "Election creates political uncertainty",
        "Supply disruption in energy markets",
        "Diplomatic tensions ease"
    ]

    for t in trange(30, steps):

        if random.random() < 0.25:
            event = simple_event_parser(random.choice(events))
            graph.update(event, correlations)
            preds = gnn_surrogate_predict(graph, event, prices.iloc[t-20:t])
            last_preds = preds
            event_times.append(len(history))

        else:
            preds = last_preds

        if t % 5 == 0:
            policy = rl_surrogate_action(preds, portfolio, cash)
            actions = sanitize_actions(policy.get("actions", {}))
        else:
            actions = {}

        portfolio, cash = apply_actions(
            actions,
            prices.iloc[t + 1],
            portfolio,
            cash
        )

        total_value = cash + sum(
            portfolio[a] * prices.iloc[t + 1][a] for a in ASSETS
        )

        for cmd in actions.values():
            if cmd["action"] == "buy":
                buy_times.append(len(history))
                buy_values.append(total_value)
            if cmd["action"] == "sell":
                sell_times.append(len(history))
                sell_values.append(total_value)

        history.append(total_value)

    # =====================================================
    # PLOT
    # =====================================================

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=history, mode="lines", name="Portfolio Value"))

    fig.add_trace(go.Scatter(
        x=event_times,
        y=[history[i] for i in event_times],
        mode="markers",
        marker=dict(symbol="x", size=12),
        name="Event"
    ))

    fig.add_trace(go.Scatter(
        x=buy_times,
        y=buy_values,
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="green"),
        name="BUY"
    ))

    fig.add_trace(go.Scatter(
        x=sell_times,
        y=sell_values,
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="red"),
        name="SELL"
    ))

    fig.update_layout(
        title="Geopolitical Event Cascade Trading (Local LLM)",
        xaxis_title="Time",
        yaxis_title="Portfolio Value"
    )

    fig.write_html("prototype_backtest.html")
    print("✅ Backtest complete → prototype_backtest.html")


if __name__ == "__main__":
    run_prototype()
