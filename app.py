"""
GeoTrade AI — Streamlit Web Dashboard

Tabs:
  1. Event Analysis  — paste a headline, see cascade predictions and graph
  2. Backtest        — run historical backtest, view metrics + chart
  3. Asset Network   — explore the live correlation graph

Run:
    streamlit run app.py
"""
import os
import json
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from src.data_pipeline import fetch_market_data, fetch_gdelt_events, compute_returns, compute_correlation_matrix, ASSETS, SECTOR_COLORS
from src.graph_engine import build_graph, propagate_cascade, graph_to_plotly, graph_summary
from src.event_analyzer import EventAnalyzer, extract_event_rule_based
from src.cascade_model import predict_direct_impacts, score_event
from src.portfolio import Portfolio
from src.backtester import compute_metrics, benchmark_metrics, impacts_to_signals, build_chart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoTrade AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 GeoTrade AI")
    st.caption("Geopolitical event cascade trading system")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Optional. Without a key, rule-based event extraction is used.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    use_claude = bool(api_key)
    st.caption("✅ Claude AI active" if use_claude else "⚙️ Rule-based mode (no API key)")

    st.divider()
    st.subheader("Date Range")
    start_date = st.date_input("Start", value=date(2023, 1, 1))
    end_date   = st.date_input("End",   value=date(2023, 12, 31))

    st.divider()
    initial_cash = st.number_input("Portfolio Size ($)", value=100_000, step=10_000, min_value=10_000)

    st.divider()
    st.subheader("Assets")
    all_tickers = list(ASSETS.keys())
    selected = st.multiselect("Select tickers", all_tickers, default=all_tickers)
    if not selected:
        selected = all_tickers


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers, start, end):
    return fetch_market_data(tickers, str(start), str(end))


@st.cache_data(ttl=600, show_spinner=False)
def load_gdelt(query, days_back=30):
    return fetch_gdelt_events(query, max_records=10, days_back=days_back)


def get_graph(prices_df):
    returns = compute_returns(prices_df)
    corr = compute_correlation_matrix(returns, window=min(60, len(returns)))
    return build_graph(corr, ASSETS)


# ── Load market data ──────────────────────────────────────────────────────────

with st.spinner("Loading market data..."):
    try:
        prices = load_prices(tuple(selected), start_date, end_date)
        prices = prices[[c for c in selected if c in prices.columns]]
        data_ok = len(prices) > 5
    except Exception as e:
        st.error(f"Could not load market data: {e}")
        data_ok = False
        prices = pd.DataFrame()

if data_ok:
    G = get_graph(prices)
else:
    G = None


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔍 Event Analysis", "📈 Backtest", "🕸️ Asset Network"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Event Analysis
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Geopolitical Event Cascade Analyzer")
    st.markdown("Paste a news headline or event description to see how it cascades through the asset network.")

    col_input, col_examples = st.columns([2, 1])

    with col_examples:
        st.markdown("**Quick examples:**")
        EXAMPLES = {
            "Russia-Ukraine escalation": "Russia launches major new offensive in eastern Ukraine. NATO announces sweeping energy sanctions.",
            "OPEC production cut": "Saudi Arabia and OPEC+ announce surprise 1 million bpd production cut starting next month.",
            "Taiwan Strait crisis": "China conducts live-fire military exercises surrounding Taiwan. US carrier group deployed.",
            "India budget (defense)": "India's Union Budget allocates record spending to defense, infrastructure, and energy transition.",
            "US Fed rate hike": "US Federal Reserve raises interest rates by 75 bps, signalling more hikes ahead to fight inflation.",
        }
        chosen = st.radio("", list(EXAMPLES.keys()), label_visibility="collapsed")
        if st.button("Load example"):
            st.session_state["event_text"] = EXAMPLES[chosen]

    with col_input:
        event_text = st.text_area(
            "Event description",
            value=st.session_state.get("event_text", ""),
            height=120,
            placeholder="e.g., Russia launches major offensive in eastern Ukraine...",
        )

    analyze_btn = st.button("Analyze Cascade", type="primary", disabled=not event_text)

    if analyze_btn and event_text:
        with st.spinner("Extracting event metadata..."):
            if use_claude:
                analyzer = EventAnalyzer()
                metadata = analyzer.extract(event_text)
            else:
                metadata = extract_event_rule_based(event_text)

        st.divider()

        # Metadata cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Event Type",  metadata.get("event_type", "—").upper())
        c2.metric("Intensity",   f"{metadata.get('intensity', 0):.2f}")
        c3.metric("Sentiment",   metadata.get("market_sentiment", "—").upper())
        c4.metric("Sectors",     ", ".join(metadata.get("affected_sectors", ["—"])))

        st.caption(f"**Summary:** {metadata.get('summary', event_text[:100])}")

        if G and data_ok:
            st.divider()
            tickers = prices.columns.tolist()

            with st.spinner("Computing cascade impacts..."):
                direct = predict_direct_impacts(metadata, tickers)
                cascaded = propagate_cascade(G, direct)

            st.subheader("Cascade Impact Predictions")
            st.caption("Combines empirical impact matrix × event intensity, propagated through correlation graph.")

            # Impact table
            impact_df = pd.DataFrame([
                {
                    "Ticker": t,
                    "Sector": ASSETS.get(t, {}).get("sector", "—"),
                    "Direct Impact": f"{direct.get(t, 0)*100:+.2f}%",
                    "After Cascade": f"{cascaded.get(t, 0)*100:+.2f}%",
                    "Signal": "🟢 BUY" if cascaded.get(t, 0) > 0.008 else ("🔴 SELL" if cascaded.get(t, 0) < -0.008 else "⬜ HOLD"),
                }
                for t in sorted(tickers, key=lambda x: -abs(cascaded.get(x, 0)))
            ])
            st.dataframe(impact_df, use_container_width=True, hide_index=True)

            # Graph with impacts visualized
            st.subheader("Asset Network — Event Impact")
            fig_graph = graph_to_plotly(G, cascaded, title=f"Cascade: {metadata.get('event_type','').upper()} event")
            st.plotly_chart(fig_graph, use_container_width=True)

        else:
            st.warning("Market data not available — load prices in the sidebar date range to see cascade visualization.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Backtest
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Historical Backtest")
    st.markdown(
        "Runs the full pipeline over the selected date range: "
        "events fire at fixed intervals, cascade impacts are predicted, and trades are executed."
    )

    from main import DEMO_EVENTS

    col_a, col_b = st.columns([1, 1])
    with col_a:
        live_events = st.checkbox("Fetch live events from GDELT (requires internet)", value=False)
        gdelt_query = st.text_input("GDELT search query", value="geopolitical conflict sanctions oil", disabled=not live_events)
    with col_b:
        graph_refresh = st.slider("Refresh correlation graph every N days", 10, 60, 20)

    run_btn = st.button("Run Backtest", type="primary", disabled=not data_ok)

    if run_btn and data_ok:
        tickers = prices.columns.tolist()

        with st.spinner("Running backtest..."):
            portfolio = Portfolio(initial_cash=initial_cash)
            returns = compute_returns(prices)
            corr = compute_correlation_matrix(returns, window=min(60, len(returns)))
            G_bt = build_graph(corr, ASSETS)

            if live_events:
                articles = load_gdelt(gdelt_query, days_back=365)
                bt_events = [
                    {"day_offset": i * max(1, len(prices) // max(len(articles), 1)), "text": a["title"], "focus": tickers}
                    for i, a in enumerate(articles)
                ]
            else:
                bt_events = DEMO_EVENTS

            analyzer = EventAnalyzer() if use_claude else None
            processed_events = []
            dates = prices.index.tolist()

            progress = st.progress(0, text="Simulating...")
            for step, (date, row) in enumerate(prices.iterrows()):
                progress.progress(step / len(prices), text=f"Day {step+1}/{len(prices)}")
                current_prices = row.dropna().to_dict()
                if not current_prices:
                    continue
                portfolio.snapshot(str(date.date()), current_prices)

                if step > 0 and step % graph_refresh == 0:
                    wr = compute_returns(prices.iloc[max(0, step - 60): step])
                    if len(wr) > 5:
                        G_bt = build_graph(compute_correlation_matrix(wr), ASSETS)

                for ev in bt_events:
                    target_idx = min(ev["day_offset"], len(dates) - 1)
                    if dates[target_idx] != date:
                        continue
                    if use_claude and analyzer:
                        meta = analyzer.extract(ev["text"])
                    else:
                        meta = extract_event_rule_based(ev["text"])
                    meta["date"] = str(date.date())

                    direct = predict_direct_impacts(meta, tickers)
                    cascaded = propagate_cascade(G_bt, direct)
                    pos_summary = portfolio.position_summary(current_prices)
                    signals = impacts_to_signals(cascaded, portfolio.get_value(current_prices), pos_summary)
                    portfolio.execute_signals(signals, current_prices, str(date.date()))

                    meta["summary"] = meta.get("summary", ev["text"][:80])
                    meta["cascade"] = cascaded
                    processed_events.append(meta)

            progress.empty()

        metrics = compute_metrics(portfolio.history)
        bm_hist, bm_metrics = benchmark_metrics(prices, initial_cash)
        alpha = metrics.get("total_return_pct", 0) - bm_metrics.get("total_return_pct", 0)

        # Metrics row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Return",   f"{metrics.get('total_return_pct', 0):+.1f}%",  f"vs {bm_metrics.get('total_return_pct',0):+.1f}%")
        m2.metric("Sharpe Ratio",   f"{metrics.get('sharpe_ratio', 0):.2f}",         f"bm {bm_metrics.get('sharpe_ratio',0):.2f}")
        m3.metric("Max Drawdown",   f"{metrics.get('max_drawdown_pct', 0):.1f}%")
        m4.metric("Sortino Ratio",  f"{metrics.get('sortino_ratio', 0):.2f}")
        m5.metric("Alpha",          f"{alpha:+.1f}%")
        m6.metric("Trades",         len(portfolio.trades))

        fig = build_chart(portfolio.history, bm_hist, portfolio.trades, processed_events, metrics, bm_metrics)
        st.plotly_chart(fig, use_container_width=True)

        # Trade log
        if portfolio.trades:
            st.subheader("Trade Log")
            trade_df = pd.DataFrame([
                {
                    "Date": t.date, "Ticker": t.ticker, "Action": t.action.upper(),
                    "Shares": round(t.shares, 2), "Price": f"${t.price:.2f}",
                    "Value": f"${t.value:,.0f}", "Rationale": t.rationale,
                }
                for t in portfolio.trades
            ])
            st.dataframe(trade_df, use_container_width=True, hide_index=True)

        # Events log
        if processed_events:
            st.subheader("Events Processed")
            ev_df = pd.DataFrame([
                {
                    "Date": e.get("date"), "Type": e.get("event_type","").upper(),
                    "Intensity": e.get("intensity"), "Sentiment": e.get("market_sentiment"),
                    "Summary": e.get("summary","")[:80],
                }
                for e in processed_events
            ])
            st.dataframe(ev_df, use_container_width=True, hide_index=True)

    elif not data_ok:
        st.warning("No market data loaded. Check date range and internet connection.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Asset Network
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Asset Correlation Network")
    st.markdown(
        "Dynamic graph showing how assets are connected by rolling correlation. "
        "Edges: green = positive correlation, red = negative. "
        "Node size reflects the number of connections."
    )

    if data_ok and G:
        col_g1, col_g2 = st.columns([2, 1])

        with col_g1:
            fig_graph = graph_to_plotly(G, title="Asset Correlation Network")
            st.plotly_chart(fig_graph, use_container_width=True)

        with col_g2:
            st.subheader("Network Stats")
            st.metric("Nodes (assets)", G.number_of_nodes())
            st.metric("Edges (|corr| > 0.25)", G.number_of_edges())

            degrees = dict(G.degree())
            if degrees:
                most_connected = max(degrees, key=degrees.get)
                st.metric("Most Connected", f"{most_connected} ({degrees[most_connected]} edges)")

            st.divider()
            st.subheader("Correlation Matrix")
            returns = compute_returns(prices)
            corr = compute_correlation_matrix(returns)
            st.dataframe(corr.round(2).style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1), use_container_width=True)

            st.divider()
            st.subheader("Live GDELT Events")
            if st.button("Fetch latest geopolitical events"):
                with st.spinner("Fetching from GDELT..."):
                    articles = load_gdelt("geopolitical war sanctions conflict", days_back=7)
                if articles:
                    for a in articles[:5]:
                        st.markdown(f"**{a['source']}** — {a['title']}")
                else:
                    st.info("No events fetched. Check internet connection.")
    else:
        st.warning("Market data required to render network. Adjust the date range in the sidebar.")
