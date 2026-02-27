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
import yfinance as yf

# =====================================================
# CONFIG
# =====================================================

ASSETS = [
    "TCS.NS",
    "INFY.NS",
    "HAL.NS",
    "BEL.NS",
    "RELIANCE.NS",
    "ONGC.NS",
    "GOLDBEES.NS",
    "SILVERBEES.NS",
    "INR=X"
]

INITIAL_CAPITAL = 1_000_000  # ₹10 lakh
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

OLLAMA_MODEL = "phi3:mini"

random.seed(42)
np.random.seed(42)

# =====================================================
# OLLAMA
# =====================================================

def ollama_chat(prompt: str, model: str = OLLAMA_MODEL) -> str:
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, _ = process.communicate(prompt)
    return stdout.strip()


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")

    cleaned = match.group(0)
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)

# =====================================================
# LOAD REAL MARKET DATA (DAILY)
# =====================================================

def load_market_data(start="2022-01-01", end="2023-01-01"):
    data = yf.download(
        ASSETS,
        start=start,
        end=end,
        progress=False,
        group_by="ticker"
    )

    # If MultiIndex (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        closes = pd.DataFrame({
            ticker: data[ticker]["Close"]
            for ticker in ASSETS
            if ticker in data
        })
    else:
        # Single ticker fallback
        closes = data["Close"]

    closes = closes.dropna()
    return closes


# =====================================================
# SIMPLE EVENT PARSER
# =====================================================

def simple_event_parser(text: str) -> Dict[str, Any]:
    text_l = text.lower()

    if "war" in text_l or "conflict" in text_l:
        return {"entities": ["HAL.NS", "BEL.NS", "GOLDBEES.NS"], "intensity": 0.8, "summary": text}
    if "oil" in text_l or "sanction" in text_l:
        return {"entities": ["ONGC.NS", "RELIANCE.NS"], "intensity": 0.7, "summary": text}
    if "rate hike" in text_l:
        return {"entities": ["INR=X"], "intensity": 0.5, "summary": text}

    return {"entities": [], "intensity": 0.3, "summary": text}

# =====================================================
# GRAPH
# =====================================================

class DynamicGraph:
    def __init__(self, assets):
        self.G = nx.Graph()
        for a in assets:
            self.G.add_node(a)

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                self.G.add_edge(assets[i], assets[j], weight=0.05)

    def update(self, event, correlations):
        ents = event["entities"]

        for u, v, d in self.G.edges(data=True):
            w = d["weight"]
            corr = correlations.get(tuple(sorted((u, v))), 0)

            boost = 0.3 * corr
            if u in ents or v in ents:
                boost += 0.6 * event["intensity"]

            d["weight"] = 0.9 * w + boost

    def summary(self, k=6):
        edges = sorted(self.G.edges(data=True),
                       key=lambda x: x[2]["weight"],
                       reverse=True)[:k]
        return "; ".join(f"{u}-{v}:{d['weight']:.2f}" for u, v, d in edges)

# =====================================================
# GNN SURROGATE (LLM)
# =====================================================

def gnn_surrogate_predict(graph, event, recent_prices):
    last_returns = recent_prices.pct_change().iloc[-1].to_dict()

    prompt = f"""
Return JSON only.

Graph:
{graph.summary()}

Event:
{event["summary"]}

Recent returns:
{last_returns}

Return:
{{ "ASSET": {{ "pred_return": float, "confidence": float }} }}
"""

    try:
        raw = ollama_chat(prompt)
        return extract_json(raw)
    except Exception:
        return {a: {"pred_return": 0, "confidence": 0} for a in ASSETS}

# =====================================================
# RL SURROGATE
# =====================================================

def rl_surrogate_action(predictions, portfolio, cash):

    prompt = f"""
Return JSON only.

Predictions:
{predictions}

Rules:
Max 3 trades.
Max 25% per asset.

Return:
{{ "actions": {{ "ASSET": {{ "action": "buy|sell|hold", "size": float }} }} }} 
"""

    try:
        raw = ollama_chat(prompt)
        return extract_json(raw)
    except Exception:
        return {"actions": {}}

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
            cost = delta * prices[a] * (1 + TRANSACTION_COST)
            if cost <= cash:
                portfolio[a] += delta
                cash -= cost

        if action == "sell":
            delta = max(0, current_units - target_units)
            proceeds = delta * prices[a] * (1 - TRANSACTION_COST)
            portfolio[a] -= delta
            cash += proceeds

    return portfolio, cash

# =====================================================
# MAIN BACKTEST
# =====================================================

def run_prototype():

    prices = load_market_data()
    dates = prices.index
    steps = len(prices)

    correlations = {}
    for i in range(len(ASSETS)):
        for j in range(i + 1, len(ASSETS)):
            s1 = prices[ASSETS[i]].pct_change().fillna(0)
            s2 = prices[ASSETS[j]].pct_change().fillna(0)
            correlations[(ASSETS[i], ASSETS[j])] = float(np.corrcoef(s1, s2)[0, 1])

    graph = DynamicGraph(ASSETS)
    portfolio = {a: 0 for a in ASSETS}
    cash = INITIAL_CAPITAL
    history = []

    last_preds = {a: {"pred_return": 0, "confidence": 0} for a in ASSETS}

    event_date = dates[int(len(dates) * 0.5)]

    event_times = []
    buy_times = []
    buy_vals = []
    sell_times = []
    sell_vals = []

    for t in trange(30, steps - 1):

        if dates[t] == event_date:
            event = simple_event_parser("India faces geopolitical conflict impacting defence and energy")
            graph.update(event, correlations)
            preds = gnn_surrogate_predict(graph, event, prices.iloc[t-20:t])
            last_preds = preds
            event_times.append(t)
        else:
            preds = last_preds

        if t % 5 == 0:
            policy = rl_surrogate_action(preds, portfolio, cash)
        else:
            policy = {"actions": {}}

        portfolio, cash = apply_actions(policy.get("actions", {}),
                                        prices.iloc[t + 1],
                                        portfolio,
                                        cash)

        total_value = cash + sum(portfolio[a] * prices.iloc[t + 1][a]
                                 for a in ASSETS)

        for a, cmd in policy.get("actions", {}).items():
            if cmd["action"] == "buy":
                buy_times.append(t)
                buy_vals.append(total_value)
            if cmd["action"] == "sell":
                sell_times.append(t)
                sell_vals.append(total_value)

        history.append(total_value)

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=history, mode="lines", name="Portfolio"))

    fig.add_trace(go.Scatter(x=event_times,
                             y=[history[i] for i in event_times],
                             mode="markers",
                             marker=dict(size=12, symbol="x"),
                             name="Event"))

    fig.add_trace(go.Scatter(x=buy_times,
                             y=buy_vals,
                             mode="markers",
                             marker=dict(symbol="triangle-up", color="green"),
                             name="BUY"))

    fig.add_trace(go.Scatter(x=sell_times,
                             y=sell_vals,
                             mode="markers",
                             marker=dict(symbol="triangle-down", color="red"),
                             name="SELL"))

    fig.update_layout(title="Indian Geopolitical Event Cascade Backtest (Daily)",
                      xaxis_title="Time",
                      yaxis_title="Portfolio Value (₹)")

    fig.write_html("indian_backtest.html")
    print("Backtest complete → indian_backtest.html")


if __name__ == "__main__":
    run_prototype()
