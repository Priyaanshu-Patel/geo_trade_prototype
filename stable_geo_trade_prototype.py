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
from sentence_transformers import SentenceTransformer

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
    """
    Robust JSON extraction for local LLMs.
    """
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found:\n" + text)

    cleaned = match.group(0)
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)

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
# NLP EVENT EXTRACTION
# =====================================================

#embedder = SentenceTransformer("all-MiniLM-L6-v2")

# def nlp_extract_event(text: str) -> Dict[str, Any]:
#     prompt = f"""
# You MUST return valid JSON only.
# No explanations. No comments. No trailing commas.

# Extract geopolitical event information.

# Text:
# \"\"\"{text}\"\"\"

# Return JSON:
# {{
#   "entities": [string],
#   "event_type": "conflict|sanction|policy|election|supply|other",
#   "intensity": float,
#   "summary": string
# }}
# """
#     raw = ollama_chat(prompt)
#     data = extract_json(raw)
#     data["embedding"] = embedder.encode(data["summary"]).tolist()
#     return data

def simple_event_parser(text: str) -> Dict[str, Any]:
    """
    Deterministic event parser (fast & safe).
    """
    text_l = text.lower()

    if "sanction" in text_l:
        return {"entities": ["XOM", "BP"], "event_type": "sanction", "intensity": 0.7, "summary": text}
    if "conflict" in text_l or "military" in text_l:
        return {"entities": ["GOLD", "OIL"], "event_type": "conflict", "intensity": 0.8, "summary": text}
    if "election" in text_l:
        return {"entities": ["USDINR", "EURUSD"], "event_type": "election", "intensity": 0.5, "summary": text}
    if "supply" in text_l:
        return {"entities": ["XOM"], "event_type": "supply", "intensity": 0.6, "summary": text}

    return {"entities": [], "event_type": "other", "intensity": 0.3, "summary": text}


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
# GNN SURROGATE (EVENT-DRIVEN)
# =====================================================

def gnn_surrogate_predict(
    graph: DynamicGraph,
    event: Dict[str, Any],
    recent_prices: pd.DataFrame
) -> Dict[str, Dict]:

    last_returns = (
        recent_prices.iloc[-1] / recent_prices.iloc[-2] - 1
        if len(recent_prices) > 1
        else pd.Series(0, index=ASSETS)
    )

    prompt = f"""
You MUST return valid JSON only.
No explanations. No comments.

You emulate a graph neural network.

Graph:
{graph.summary()}

Event:
{event["summary"]} (type={event["event_type"]}, intensity={event["intensity"]})

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

def rl_surrogate_action(
    predictions: Dict[str, Dict],
    portfolio: Dict[str, float],
    cash: float
) -> Dict[str, Any]:

    prompt = f"""
You MUST return valid JSON only.
No explanations. No comments.

You emulate a trading policy.

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
# EXECUTION
# =====================================================

def apply_actions(actions, prices, portfolio, cash):
    value = cash + sum(portfolio[a] * prices[a] for a in ASSETS)

    for a, cmd in actions.items():
        action, size = cmd["action"], cmd["size"]

        if action == "hold" or size <= 0:
            continue

        target_val = size * value
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

    events = [
        "Sanctions imposed on oil exporter",
        "Military conflict escalates in region",
        "Election creates political uncertainty",
        "Supply disruption in energy markets",
        "Diplomatic tensions ease"
    ]

    for t in trange(30, steps):
        if random.random() < 0.25:
            #event = nlp_extract_event(random.choice(events))
            event = simple_event_parser(random.choice(events))
            graph.update(event, correlations)
            preds = gnn_surrogate_predict(graph, event, prices.iloc[t-20:t])
            last_preds = preds
        
        else:
            preds = last_preds

        if t % 5 == 0:
            policy = rl_surrogate_action(preds, portfolio, cash)
        else:
            policy = {"actions": {}}

        portfolio, cash = apply_actions(
            policy["actions"],
            prices.iloc[t + 1],
            portfolio,
            cash
        )

        total_value = cash + sum(
            portfolio[a] * prices.iloc[t + 1][a] for a in ASSETS
        )
        history.append(total_value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history, mode="lines", name="Portfolio Value"))
    fig.update_layout(
        title="Geopolitical Event Cascade Trading (Local LLM)",
        xaxis_title="Time",
        yaxis_title="Portfolio Value"
    )
    fig.write_html("prototype_backtest.html")
    print("✅ Backtest complete → prototype_backtest.html")

if __name__ == "__main__":
    run_prototype()
