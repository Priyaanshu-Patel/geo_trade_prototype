"""
Backtesting engine with full performance metrics.

Run flow:
  1. Fetch prices for date range
  2. Build correlation graph (updated periodically)
  3. At each event date: extract metadata → predict cascade → generate signals → execute
  4. Snapshot portfolio daily
  5. Compute metrics vs equal-weight benchmark
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional


# ── Performance Metrics ───────────────────────────────────────────────────────

def compute_metrics(history: list, risk_free_rate: float = 0.05) -> dict:
    """Compute Sharpe, Sortino, max drawdown, etc. from portfolio history list."""
    if len(history) < 2:
        return {}

    values = [h["value"] for h in history]
    returns = pd.Series(values).pct_change().dropna()

    total_return = (values[-1] - values[0]) / values[0]
    n = len(values)
    ann_factor = 252 / max(n - 1, 1)
    ann_return = (1 + total_return) ** ann_factor - 1

    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf

    vol = returns.std() * np.sqrt(252)
    sharpe = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0

    downside = excess[excess < 0]
    sortino = (excess.mean() / downside.std()) * np.sqrt(252) if len(downside) > 1 and downside.std() > 0 else 0.0

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    win_rate = float((returns > 0).mean())

    return {
        "total_return_pct":  round(total_return * 100, 2),
        "annualized_return": round(ann_return * 100, 2),
        "volatility":        round(vol * 100, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "sortino_ratio":     round(sortino, 3),
        "max_drawdown_pct":  round(max_drawdown * 100, 2),
        "win_rate_pct":      round(win_rate * 100, 2),
        "final_value":       round(values[-1], 2),
    }


def benchmark_metrics(prices: pd.DataFrame, initial_cash: float = 100_000.0) -> tuple:
    """Equal-weight buy-and-hold baseline. Returns (history list, metrics dict)."""
    tickers = prices.columns.tolist()
    shares = (initial_cash / len(tickers)) / prices.iloc[0]

    history = []
    for date, row in prices.iterrows():
        val = sum(shares.get(t, 0) * row.get(t, 0) for t in tickers if not pd.isna(row.get(t, 0)))
        history.append({"date": str(date.date()), "value": val})

    return history, compute_metrics(history)


# ── Trading Signal Generator ──────────────────────────────────────────────────

MIN_IMPACT_TO_TRADE = 0.008   # Only trade if |predicted impact| > 0.8%
MAX_POSITION_SIZE = 0.18      # Max size per trade (as fraction of portfolio)


def impacts_to_signals(
    cascade_impacts: dict,   # ticker -> expected return (fraction)
    portfolio_value: float,
    current_positions: dict,  # ticker -> {weight_pct, ...}
) -> list:
    """
    Convert cascade impact predictions to trade signals.
    Rules:
    - |impact| > MIN_IMPACT_TO_TRADE → active signal
    - Positive impact → buy (unless already heavily positioned)
    - Negative impact → sell (if holding)
    - Size proportional to |impact| strength, capped at MAX_POSITION_SIZE
    """
    signals = []
    for ticker, impact in sorted(cascade_impacts.items(), key=lambda x: -abs(x[1])):
        if abs(impact) < MIN_IMPACT_TO_TRADE:
            continue

        current_weight = current_positions.get(ticker, {}).get("weight_pct", 0) / 100.0

        if impact > 0:
            # Buy signal: skip if already near max
            if current_weight >= 0.17:
                continue
            size = min(MAX_POSITION_SIZE, abs(impact) * 4)
            signals.append({
                "ticker": ticker,
                "action": "buy",
                "size_pct": round(size, 3),
                "rationale": f"Positive cascade: +{impact*100:.1f}% predicted",
            })

        else:
            # Sell signal: only if holding
            if current_weight < 0.01:
                continue
            size = min(current_weight, abs(impact) * 4)
            signals.append({
                "ticker": ticker,
                "action": "sell",
                "size_pct": round(size, 3),
                "rationale": f"Negative cascade: {impact*100:.1f}% predicted",
            })

    return signals


# ── Visualization ─────────────────────────────────────────────────────────────

def build_chart(
    portfolio_history: list,
    benchmark_history: list,
    trades: list,
    events: list,
    metrics: dict,
    bm_metrics: dict,
) -> go.Figure:
    """Multi-panel backtest chart: portfolio value, daily PnL, drawdown."""

    dates = [h["date"] for h in portfolio_history]
    values = [h["value"] for h in portfolio_history]
    b_dates = [h["date"] for h in benchmark_history]
    b_values = [h["value"] for h in benchmark_history]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Portfolio Value (USD)", "Daily Return %", "Drawdown %"),
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.06,
    )

    # Portfolio vs benchmark
    fig.add_trace(go.Scatter(x=dates, y=values, name="GeoTrade AI", line=dict(color="#2196F3", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=b_dates, y=b_values, name="Equal-Weight Benchmark", line=dict(color="#9E9E9E", width=1.5, dash="dash")), row=1, col=1)

    # Trade markers
    buy_dates  = [t.date for t in trades if t.action == "buy"]
    sell_dates = [t.date for t in trades if t.action == "sell"]
    buy_vals   = [next((h["value"] for h in portfolio_history if h["date"] == d), None) for d in buy_dates]
    sell_vals  = [next((h["value"] for h in portfolio_history if h["date"] == d), None) for d in sell_dates]

    if buy_vals:
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_vals, mode="markers", name="Buy",
                                 marker=dict(symbol="triangle-up", color="#4CAF50", size=11, line=dict(width=1, color="#333"))), row=1, col=1)
    if sell_vals:
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_vals, mode="markers", name="Sell",
                                 marker=dict(symbol="triangle-down", color="#F44336", size=11, line=dict(width=1, color="#333"))), row=1, col=1)

    # Daily returns bar
    vals_s = pd.Series(values)
    daily_ret = vals_s.pct_change().fillna(0) * 100
    bar_colors = ["#4CAF50" if r >= 0 else "#F44336" for r in daily_ret]
    fig.add_trace(go.Bar(x=dates, y=daily_ret.tolist(), name="Daily Return", marker_color=bar_colors, showlegend=False), row=2, col=1)

    # Drawdown
    cum = (1 + vals_s.pct_change().fillna(0)).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()) * 100
    fig.add_trace(go.Scatter(x=dates, y=dd.tolist(), name="Drawdown", fill="tozeroy",
                             line=dict(color="#F44336", width=1), fillcolor="rgba(244,67,54,0.15)", showlegend=False), row=3, col=1)

    # Event annotations (top 5)
    for ev in events[:5]:
        ev_date = ev.get("date")
        if ev_date in dates:
            idx = dates.index(ev_date)
            label = ev.get("summary", "")[:35]
            fig.add_vline(x=ev_date, line_dash="dot", line_color="#FF9800", opacity=0.7)
            fig.add_annotation(x=ev_date, y=values[idx] * 1.005, text=label,
                                showarrow=False, font=dict(size=8, color="#FF6F00"),
                                bgcolor="rgba(255,255,255,0.7)")

    # Metrics table in title
    alpha = metrics.get("total_return_pct", 0) - bm_metrics.get("total_return_pct", 0)
    subtitle = (
        f"<b>Strategy</b>: Return={metrics.get('total_return_pct', 0):+.1f}%  "
        f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}  "
        f"MaxDD={metrics.get('max_drawdown_pct', 0):.1f}%  "
        f"Sortino={metrics.get('sortino_ratio', 0):.2f}   |   "
        f"<b>Benchmark</b>: Return={bm_metrics.get('total_return_pct', 0):+.1f}%  "
        f"Sharpe={bm_metrics.get('sharpe_ratio', 0):.2f}   |   "
        f"<b>Alpha: {alpha:+.1f}%</b>"
    )

    fig.update_layout(
        title=dict(text=f"GeoTrade AI — Backtest Results<br><sup>{subtitle}</sup>", x=0.02, font=dict(size=14)),
        height=750,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        template="plotly_white",
        margin=dict(l=50, r=30, t=100, b=30),
    )
    return fig
