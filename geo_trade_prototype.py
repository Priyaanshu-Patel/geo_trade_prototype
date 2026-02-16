"""
geo_trade_prototype.py
A minimal end-to-end prototype using an LLM as surrogate for GNN & RL.
Run: OPENAI_API_KEY=... python geo_trade_prototype.py
"""

import os, time, json, math, random
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
from openai import OpenAI
client = OpenAI()
import plotly.graph_objects as go
from tqdm import trange

# ------------- Config -------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


EMBEDDER_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers
ASSETS = ["AAPL", "MSFT", "GOOG", "TSLA", "BP", "XOM", "EURUSD", "USDINR", "GOLD", "SILVER"]
N_ASSETS = len(ASSETS)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Trading params
TRANSACTION_COST = 0.0005  # 5 bps
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 100000.0

# ------------- Helpers -------------
def simulate_market(n_steps=240):
    """Simulate minute-by-minute prices for each asset using geometric random walk."""
    prices = pd.DataFrame(index=pd.RangeIndex(n_steps), columns=ASSETS, dtype=float)
    init = np.linspace(100, 200, N_ASSETS)  # starting prices vary
    vols = np.linspace(0.0005, 0.005, N_ASSETS)  # per-minute vol
    for i, a in enumerate(ASSETS):
        p = init[i]
        prices.iloc[0, i] = p
    for t in range(1, n_steps):
        shocks = np.random.normal(loc=0.0, scale=vols)
        prices.iloc[t] = prices.iloc[t-1].values * np.exp(shocks)
    return prices

# ------------- NLP Module -------------
embedder = SentenceTransformer(EMBEDDER_MODEL)

def nlp_extract_event(text: str) -> Dict[str, Any]:
    """
    Use LLM (if available) to extract structured event: entities, type, intensity.
    If OPENAI_API_KEY not set, fallback to a simple heuristic extractor.
    """
    if OPENAI_API_KEY:
        prompt = f"""Extract event information in JSON. Input text delimited by <<< >>>.
Output JSON with keys: entities (list of strings), event_type (one of: conflict, sanction, policy, election, supply_disruption, other),
intensity (0.0-1.0 float), summary (short).
<<<{text}>>>"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # replace with suitable model if needed
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=200
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        try:
            data = json.loads(raw)
        except Exception:
            # try to extract JSON block
            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
            else:
                data = {"entities": [], "event_type": "other", "intensity": 0.5, "summary": text[:140]}
        # ensure fields
        data.setdefault("entities", [])
        data.setdefault("event_type", "other")
        data.setdefault("intensity", 0.5)
        data.setdefault("summary", text[:140])
        data["embedding"] = embedder.encode(data["summary"]).tolist()
        return data
    else:
        # fallback: simple heuristics
        ents = []
        lower = text.lower()
        for c in ["russia","ukraine","china","us","india","oil","iran","israel","palestine","nato","eu"]:
            if c in lower:
                ents.append(c.upper())
        etype = "other"
        if "election" in lower or "vote" in lower: etype="election"
        if "sanction" in lower: etype="sanction"
        if "attack" in lower or "conflict" in lower or "bomb" in lower: etype="conflict"
        intensity = min(1.0, 0.2 + 0.8 * (len(ents)/3))
        summary = text[:140]
        return {"entities": ents, "event_type": etype, "intensity": intensity, "summary": summary, "embedding": embedder.encode(summary).tolist()}

# ------------- Graph builder -------------
class DynamicGraph:
    def __init__(self, assets: List[str]):
        self.assets = assets
        self.G = nx.Graph()
        for a in assets:
            self.G.add_node(a)
        # initialize small random edges based on sector-like grouping
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                base = 0.05 * (1 - abs(i-j)/len(assets))
                self.G.add_edge(assets[i], assets[j], weight=base)
    def update_with_event(self, event: Dict[str,Any], correlations: Dict):
        # boost edges for assets whose names match event entities (toy logic)
        entities = [e.upper() for e in event.get("entities", [])]
        for u,v,data in self.G.edges(data=True):
            boost = 0.0
            # correlation-based
            pair = tuple(sorted([u,v]))
            corr = correlations.get(pair, 0.0)
            boost += 0.5 * corr
            # event cooccurrence heuristic: if asset symbol appears in entities
            if u.upper() in entities or v.upper() in entities:
                boost += event.get("intensity",0.5) * 0.5
            # decayed old weight
            old = data.get("weight", 0.01)
            neww = 0.9*old + boost
            data["weight"] = neww
    def summary_for_prompt(self, top_k=5):
        # return compact textual summary: top edges by weight
        edges = sorted(self.G.edges(data=True), key=lambda x: x[2].get("weight",0), reverse=True)[:top_k]
        s = []
        for u,v,d in edges:
            s.append(f"{u}-{v}:{d.get('weight',0):.3f}")
        return "; ".join(s)

# ------------- GNN surrogate (LLM) -------------
def gnn_surrogate_predict(graph: DynamicGraph, event: Dict[str,Any], recent_prices: pd.DataFrame) -> Dict[str, Dict]:
    """
    Simulate a GNN by prompting the LLM with:
      - graph summary
      - event summary + intensity
      - recent price returns (last minute returns per asset)
    Output: dict asset -> {pred_return, confidence}
    """
    # prepare concise market state: last return for each asset
    last_ret = (recent_prices.iloc[-1] / recent_prices.iloc[-2] - 1).to_dict() if len(recent_prices)>=2 else {a:0.0 for a in ASSETS}
    market_str = ", ".join([f"{a}:{last_ret.get(a,0):.6f}" for a in ASSETS])
    graph_summary = graph.summary_for_prompt(top_k=6)
    prompt = (
        "You are emulating a trained graph neural network that predicts short-term returns "
        "for a list of assets given a graph summary and a geopolitical event. "
        "Return JSON mapping each asset to predicted_return (float, -0.05..0.05) and confidence (0..1). "
        f"Graph summary: {graph_summary}\n"
        f"Event summary: {event.get('summary')} | type={event.get('event_type')} | intensity={event.get('intensity'):.2f}\n"
        f"Market last returns: {market_str}\n"
        "Be conservative with magnitudes. Output only JSON."
    )
    if OPENAI_API_KEY:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=400)
        raw = resp["choices"][0]["message"]["content"]
        try:
            out = json.loads(raw)
        except Exception:
            # fallback: construct neutral predictions
            out = {a: {"pred_return": 0.0, "confidence": 0.2} for a in ASSETS}
    else:
        # deterministic toy surrogate
        out = {}
        for a in ASSETS:
            # if asset name matches any entity, small positive/negative bias
            bias = 0.0
            for ent in event.get("entities",[]):
                if ent.lower() in a.lower():
                    bias += 0.005 * (1 if event.get("event_type")!="conflict" else -1)
            # use average neighbor weight from graph
            neighs = list(graph.G.neighbors(a))
            avgw = np.mean([graph.G[a][n]["weight"] for n in neighs]) if neighs else 0.01
            pred = bias + 0.01 * event.get("intensity",0.5) * avgw
            out[a] = {"pred_return": float(np.clip(pred, -0.05, 0.05)), "confidence": float(min(1.0, 0.2 + avgw))}
    return out

# ------------- RL surrogate (LLM) -------------
def rl_surrogate_action(predictions: Dict[str,Dict], portfolio: Dict[str,float], cash: float) -> Dict[str, Any]:
    """
    Prompt-based RL surrogate that outputs discrete actions: buy/sell/hold and size fractions.
    Return {"actions": {asset: {"action":"buy/sell/hold","size":0.1}}, "notes": "..."}
    """
    # create a text summary of top predictions
    sorted_preds = sorted(predictions.items(), key=lambda kv: kv[1]["pred_return"], reverse=True)
    pred_text = ", ".join([f"{k}:{v['pred_return']:.4f} (c={v['confidence']:.2f})" for k,v in sorted_preds[:6]])
    prompt = (
        "You are emulating a trained RL trading policy. Given predicted short-term returns and confidences for assets, "
        "decide discrete actions per asset: buy (increase position), sell (decrease/close), or hold. "
        "Constraints: max per-asset exposure 20% of portfolio, prefer high-confidence predictions, and avoid >3 simultaneous trades.\n"
        f"Predictions: {pred_text}\n"
        f"Current portfolio (positions in units): {portfolio}\nCash: {cash:.2f}\n"
        "Output JSON: {actions: {ASSET: {action: 'buy'/'sell'/'hold', size: fraction_of_portfolio_to_allocate}}, notes: '...'}."
    )
    if OPENAI_API_KEY:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.0, max_tokens=300)
        raw = resp["choices"][0]["message"]["content"]
        try:
            out = json.loads(raw)
        except Exception:
            out = {"actions": {a: {"action":"hold","size":0.0} for a in ASSETS}, "notes":"parse-fail fallback"}
    else:
        # simple rules
        out = {"actions": {}, "notes": "rule-based fallback"}
        buys = 0
        for a,v in sorted_preds:
            r = v["pred_return"]
            c = v["confidence"]
            if r > 0.002 and c>0.25 and buys<3:
                out["actions"][a] = {"action":"buy", "size":min(0.2, 0.05 + 0.5*c)}
                buys += 1
            elif r < -0.002 and c>0.25:
                out["actions"][a] = {"action":"sell", "size":min(0.2, 0.05 + 0.5*c)}
            else:
                out["actions"][a] = {"action":"hold","size":0.0}
    return out

# ------------- Backtest engine -------------
def apply_actions(actions: Dict[str,Any], prices_t: pd.Series, portfolio: Dict[str,float], cash: float):
    """
    Simple execution: compute unit change based on fraction size relative to portfolio value.
    Returns updated portfolio, cash, trade_log
    """
    portfolio_value = sum(portfolio[a]*prices_t[a] for a in ASSETS) + cash
    trades = []
    for a,cmd in actions.items():
        act = cmd["action"]
        frac = float(cmd.get("size",0.0))
        if act=="hold" or frac<=0:
            continue
        max_alloc = 0.2 * portfolio_value
        target_alloc = frac * portfolio_value
        # compute target units
        target_units = target_alloc / prices_t[a]
        current_units = portfolio.get(a,0.0)
        if act=="buy":
            units_to_buy = max(0.0, target_units - current_units)
            cost = units_to_buy * prices_t[a] * (1+TRANSACTION_COST+SLIPPAGE)
            if cost > cash:
                units_to_buy = cash / (prices_t[a] * (1+TRANSACTION_COST+SLIPPAGE))
                cost = units_to_buy * prices_t[a] * (1+TRANSACTION_COST+SLIPPAGE)
            portfolio[a] = current_units + units_to_buy
            cash -= cost
            trades.append((a,"buy",units_to_buy,prices_t[a]))
        elif act=="sell":
            units_to_sell = max(0.0, current_units - target_units)
            proceeds = units_to_sell * prices_t[a] * (1-TRANSACTION_COST-SLIPPAGE)
            portfolio[a] = current_units - units_to_sell
            cash += proceeds
            trades.append((a,"sell",units_to_sell,prices_t[a]))
    return portfolio, cash, trades

# ------------- Main orchestration / simulation -------------
def run_prototype(sim_steps=120):
    prices = simulate_market(n_steps=sim_steps+2)
    # precompute simple pairwise correlations on first window
    correlations = {}
    corr_window = 30
    for i in range(N_ASSETS):
        for j in range(i+1, N_ASSETS):
            a = ASSETS[i]; b = ASSETS[j]
            s1 = prices[a].pct_change().fillna(0).iloc[:corr_window]
            s2 = prices[b].pct_change().fillna(0).iloc[:corr_window]
            correlations[(a,b)] = float(np.corrcoef(s1,s2)[0,1])
    graph = DynamicGraph(ASSETS)
    # initial portfolio
    portfolio = {a:0.0 for a in ASSETS}
    cash = INITIAL_CAPITAL
    history = []
    event_pool = [
        "Major protest in capital city against oil imports.",
        "Country X announces sanctions on Country Y.",
        "Elections scheduled next month; incumbent leads in polls.",
        "Supply disruption at major refinery after fire.",
        "Diplomatic talks ease tensions between countries."
    ]
    for t in trange(30, sim_steps):  # skip warmup
        # 1) pick or get next event sometimes
        event_text = random.choice(event_pool) if random.random() < 0.2 else None
        event_obj = None
        if event_text:
            event_obj = nlp_extract_event(event_text)
            graph.update_with_event(event_obj, correlations)
        else:
            # no event: small intensity null
            event_obj = {"entities":[],"event_type":"none","intensity":0.0,"summary":"no_event","embedding":embedder.encode("no_event").tolist()}
        # 2) call GNN surrogate
        recent_window = prices.iloc[max(0,t-30):t+1]
        preds = gnn_surrogate_predict(graph, event_obj, recent_window)
        # 3) call RL surrogate
        rl = rl_surrogate_action(preds, portfolio, cash)
        actions = rl.get("actions", {})
        # 4) apply actions using price at t+1 (next minute)
        prices_t1 = prices.iloc[t+1]
        portfolio, cash, trades = apply_actions(actions, prices_t1, portfolio, cash)
        total_val = cash + sum(portfolio[a]*prices_t1[a] for a in ASSETS)
        history.append({"t":t, "event": event_text, "trades": trades, "total_value": total_val})
    # finalize: compute PnL timeseries
    df = pd.DataFrame(history)
    df["total_value"] = df["total_value"].astype(float)
    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["total_value"].values, mode="lines+markers", name="Portfolio Value"))
    fig.update_layout(title="Prototype Backtest Portfolio Value", xaxis_title="Step", yaxis_title="Value (USD)")
    out_html = "prototype_backtest.html"
    fig.write_html(out_html)
    print(f"Backtest complete. Final value: {df['total_value'].iloc[-1]:.2f}. Plot saved to {out_html}")
    return df, graph

if __name__ == "__main__":
    df, graph = run_prototype(sim_steps=250)
