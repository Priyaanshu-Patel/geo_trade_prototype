"""
GeoTrade AI — CLI backtest runner.

Usage:
    python main.py --start 2023-01-01 --end 2023-12-31
    python main.py --start 2022-01-01 --end 2022-12-31 --live-events
    python main.py --start 2023-01-01 --end 2023-12-31 --no-claude

Requires:
    ANTHROPIC_API_KEY env var (or --no-claude for rule-based fallback)
"""
import argparse
import os
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

from src.data_pipeline import fetch_market_data, fetch_gdelt_events, compute_returns, compute_correlation_matrix, ASSETS
from src.graph_engine import build_graph, propagate_cascade, graph_summary
from src.event_analyzer import EventAnalyzer
from src.cascade_model import predict_direct_impacts, score_event
from src.portfolio import Portfolio
from src.backtester import (
    compute_metrics, benchmark_metrics, impacts_to_signals, build_chart
)


# ── Sample events for offline / demo mode ────────────────────────────────────

DEMO_EVENTS = [
    {
        "day_offset": 5,
        "text": "Russia launches major offensive in eastern Ukraine, targeting key industrial cities. "
                "NATO nations announce sweeping sanctions package including energy and financial restrictions.",
        "focus": ["XOM", "USO", "GLD", "LMT", "HAL.NS"],
    },
    {
        "day_offset": 22,
        "text": "India and Pakistan exchange fire across the Line of Control following a coordinated "
                "terrorist attack in Kashmir. India mobilises border troops and suspends trade.",
        "focus": ["HAL.NS", "ONGC.NS", "TCS.NS"],
    },
    {
        "day_offset": 45,
        "text": "US Federal Reserve signals aggressive rate hike cycle to combat 40-year high inflation. "
                "Dollar strengthens significantly against emerging market currencies.",
        "focus": ["UUP", "TCS.NS", "AAPL", "GLD"],
    },
    {
        "day_offset": 68,
        "text": "Saudi Arabia and OPEC+ announce surprise production cut of 1 million barrels per day, "
                "tightening global oil supply heading into winter.",
        "focus": ["USO", "XOM", "ONGC.NS"],
    },
    {
        "day_offset": 90,
        "text": "Taiwan Strait tensions escalate sharply as China conducts live-fire military exercises "
                "surrounding the island. Technology supply chain risks mount.",
        "focus": ["AAPL", "TCS.NS", "GLD", "UUP"],
    },
    {
        "day_offset": 120,
        "text": "India's Union Budget announces record ₹11.11 lakh crore infrastructure spending, "
                "with major allocations to defence, energy transition, and waterways.",
        "focus": ["HAL.NS", "ONGC.NS", "TCS.NS"],
    },
    {
        "day_offset": 150,
        "text": "India and Australia finalise ECTA free trade agreement, eliminating tariffs on "
                "85% of goods and removing double taxation for IT services.",
        "focus": ["TCS.NS", "AAPL"],
    },
]


def run_backtest(
    start_date: str,
    end_date: str,
    use_claude: bool = True,
    use_live_events: bool = False,
    initial_cash: float = 100_000.0,
    output_html: str = "backtest_result.html",
    graph_update_every: int = 20,
) -> dict:

    tickers = list(ASSETS.keys())
    print(f"\n[1/5] Fetching market data for {len(tickers)} assets ({start_date} → {end_date})")
    prices = fetch_market_data(tickers, start=start_date, end=end_date)
    tickers = prices.columns.tolist()
    print(f"      Available: {tickers} ({len(prices)} trading days)")

    if len(prices) < 10:
        raise ValueError("Insufficient market data. Check date range and internet connection.")

    print("\n[2/5] Building correlation graph")
    returns = compute_returns(prices)
    corr = compute_correlation_matrix(returns, window=min(60, len(returns)))
    G = build_graph(corr, ASSETS)
    print(f"      Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[3/5] Setting up event analyzer")
    analyzer = EventAnalyzer()
    if not use_claude:
        print("      Mode: rule-based (--no-claude)")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("      ANTHROPIC_API_KEY not set. Falling back to rule-based extraction.")

    dates = prices.index.tolist()

    if use_live_events:
        print("\n      Fetching live events from GDELT...")
        articles = fetch_gdelt_events("geopolitical conflict sanctions oil war", max_records=7, days_back=365)
        events = [
            {"day_offset": i * max(1, len(dates) // max(len(articles), 1)), "text": a["title"], "focus": tickers}
            for i, a in enumerate(articles)
        ]
        print(f"      Fetched {len(events)} GDELT events")
    else:
        events = DEMO_EVENTS

    portfolio = Portfolio(initial_cash=initial_cash)
    processed_events = []

    print(f"\n[4/5] Running backtest ({len(dates)} days, {len(events)} events)")
    for step, (date, row) in enumerate(prices.iterrows()):
        current_prices = row.dropna().to_dict()
        if not current_prices:
            continue

        portfolio.snapshot(str(date.date()), current_prices)

        # Periodically refresh the correlation graph
        if step > 0 and step % graph_update_every == 0:
            window_returns = compute_returns(prices.iloc[max(0, step - 60): step])
            if len(window_returns) > 5:
                corr = compute_correlation_matrix(window_returns)
                G = build_graph(corr, ASSETS)

        # Fire events scheduled for this step
        for ev in events:
            target_idx = min(ev["day_offset"], len(dates) - 1)
            if dates[target_idx] != date:
                continue

            print(f"\n  [Event] {date.date()}: {ev['text'][:90]}...")

            # Extract structured metadata
            if use_claude:
                metadata = analyzer.extract(ev["text"])
            else:
                from src.event_analyzer import extract_event_rule_based
                metadata = extract_event_rule_based(ev["text"])

            metadata["date"] = str(date.date())
            print(f"         {score_event(metadata)}")

            # Predict direct impacts via empirical matrix
            direct = predict_direct_impacts(metadata, tickers)

            # Propagate through correlation graph
            cascaded = propagate_cascade(G, direct)

            # Generate trading signals
            pos_summary = portfolio.position_summary(current_prices)
            signals = impacts_to_signals(cascaded, portfolio.get_value(current_prices), pos_summary)

            # Execute
            executed = portfolio.execute_signals(signals, current_prices, str(date.date()))
            for sig in executed:
                print(f"         Trade: {sig['action'].upper()} {sig['ticker']} ({sig['size_pct']*100:.1f}%)")

            metadata["summary"] = metadata.get("summary", ev["text"][:80])
            metadata["cascade"] = cascaded
            processed_events.append(metadata)

    print("\n[5/5] Computing performance metrics")
    metrics = compute_metrics(portfolio.history)
    bm_hist, bm_metrics = benchmark_metrics(prices, initial_cash)
    alpha = metrics.get("total_return_pct", 0) - bm_metrics.get("total_return_pct", 0)

    print("\n" + "=" * 55)
    print(f"  Strategy   | Return: {metrics.get('total_return_pct', 0):+6.1f}%  "
          f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  "
          f"MaxDD: {metrics.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Benchmark  | Return: {bm_metrics.get('total_return_pct', 0):+6.1f}%  "
          f"Sharpe: {bm_metrics.get('sharpe_ratio', 0):.2f}  "
          f"MaxDD: {bm_metrics.get('max_drawdown_pct', 0):.1f}%")
    print(f"  Alpha      | {alpha:+.1f}%")
    print(f"  Trades     | {len(portfolio.trades)}")
    print("=" * 55)

    fig = build_chart(portfolio.history, bm_hist, portfolio.trades, processed_events, metrics, bm_metrics)
    fig.write_html(output_html)
    print(f"\n  Chart saved → {output_html}")

    return {
        "metrics": metrics,
        "benchmark": bm_metrics,
        "alpha_pct": alpha,
        "trades": len(portfolio.trades),
        "events": len(processed_events),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoTrade AI — Backtest Runner")
    parser.add_argument("--start",       default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",         default="2023-12-31", help="End date   YYYY-MM-DD")
    parser.add_argument("--cash",        default=100_000.0, type=float, help="Initial cash (USD)")
    parser.add_argument("--output",      default="backtest_result.html", help="Output HTML path")
    parser.add_argument("--no-claude",   action="store_true", help="Skip Claude API, use rule-based fallback")
    parser.add_argument("--live-events", action="store_true", help="Fetch live events from GDELT")
    args = parser.parse_args()

    run_backtest(
        start_date=args.start,
        end_date=args.end,
        use_claude=not args.no_claude,
        use_live_events=args.live_events,
        initial_cash=args.cash,
        output_html=args.output,
    )
