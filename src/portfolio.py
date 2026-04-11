"""
Portfolio management: position tracking, trade execution, risk controls.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Transaction costs (one-way)
TRANSACTION_COST = 0.0008   # 8 bps (realistic for equities)
SLIPPAGE = 0.0005           # 5 bps

# Risk limits
MAX_POSITION_PCT = 0.20     # Max 20% of portfolio in any single asset
MAX_TRADES_PER_EVENT = 3
MIN_TRADE_SIZE = 500.0      # Minimum trade value in $


@dataclass
class Trade:
    date: str
    ticker: str
    action: str       # buy | sell
    shares: float
    price: float
    value: float      # notional
    cost: float       # total friction (slippage + commission)
    rationale: str = ""


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0):
        self.cash = float(initial_cash)
        self.initial_value = float(initial_cash)
        self.positions: dict = {}   # ticker -> shares
        self.trades: list = []
        self.history: list = []

    # ── Valuation ──────────────────────────────────────────────────────────

    def get_value(self, prices: dict) -> float:
        holdings = sum(
            shares * prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + holdings

    def get_weights(self, prices: dict) -> dict:
        total = self.get_value(prices)
        if total == 0:
            return {"cash": 1.0}
        weights = {
            ticker: (shares * prices.get(ticker, 0.0)) / total
            for ticker, shares in self.positions.items()
        }
        weights["cash"] = self.cash / total
        return weights

    def current_position_pct(self, ticker: str, prices: dict) -> float:
        total = self.get_value(prices)
        if total == 0:
            return 0.0
        return self.positions.get(ticker, 0.0) * prices.get(ticker, 0.0) / total

    # ── Execution ───────────────────────────────────────────────────────────

    def execute_trade(
        self,
        ticker: str,
        action: str,
        size_pct: float,
        price: float,
        date: str,
        rationale: str = "",
    ) -> bool:
        """
        Execute a trade. size_pct = fraction of total portfolio value to trade.
        Returns True if executed.
        """
        if price <= 0:
            return False

        total_value = self.get_value({ticker: price})
        target_notional = total_value * size_pct

        if target_notional < MIN_TRADE_SIZE:
            return False

        if action == "buy":
            # Enforce position limit
            current_pct = self.current_position_pct(ticker, {ticker: price})
            if current_pct >= MAX_POSITION_PCT:
                return False

            # Cap so we don't exceed max position
            max_additional = (MAX_POSITION_PCT - current_pct) * total_value
            target_notional = min(target_notional, max_additional, self.cash * 0.95)

            if target_notional < MIN_TRADE_SIZE:
                return False

            effective_price = price * (1 + SLIPPAGE + TRANSACTION_COST)
            shares = target_notional / effective_price
            cost = shares * (effective_price - price)

            self.cash -= shares * effective_price
            self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
            self.trades.append(Trade(date, ticker, "buy", shares, price, target_notional, cost, rationale))
            return True

        elif action == "sell":
            held = self.positions.get(ticker, 0.0)
            if held <= 0:
                return False

            effective_price = price * (1 - SLIPPAGE - TRANSACTION_COST)
            shares_to_sell = min(held, target_notional / effective_price)
            if shares_to_sell <= 0:
                return False

            proceeds = shares_to_sell * effective_price
            cost = shares_to_sell * (price - effective_price)
            self.cash += proceeds
            self.positions[ticker] = held - shares_to_sell
            if self.positions[ticker] < 1e-6:
                del self.positions[ticker]
            self.trades.append(Trade(date, ticker, "sell", shares_to_sell, price, shares_to_sell * price, cost, rationale))
            return True

        return False

    def execute_signals(self, signals: list, prices: dict, date: str) -> list:
        """
        Execute a list of signal dicts from the trading engine.
        Each signal: {ticker, action, size_pct, rationale}
        Respects MAX_TRADES_PER_EVENT.
        """
        executed = []
        count = 0
        for sig in signals:
            if count >= MAX_TRADES_PER_EVENT:
                break
            ticker = sig.get("ticker", "")
            action = sig.get("action", "hold")
            if action == "hold" or ticker not in prices:
                continue
            ok = self.execute_trade(
                ticker, action,
                size_pct=sig.get("size_pct", 0.05),
                price=prices[ticker],
                date=date,
                rationale=sig.get("rationale", ""),
            )
            if ok:
                executed.append(sig)
                count += 1
        return executed

    # ── Snapshot ────────────────────────────────────────────────────────────

    def snapshot(self, date: str, prices: dict) -> dict:
        value = self.get_value(prices)
        snap = {
            "date": date,
            "value": value,
            "cash": self.cash,
            "holdings": {t: s * prices.get(t, 0) for t, s in self.positions.items()},
            "pnl": value - self.initial_value,
            "pnl_pct": (value - self.initial_value) / self.initial_value,
        }
        self.history.append(snap)
        return snap

    def position_summary(self, prices: dict) -> dict:
        total = self.get_value(prices)
        return {
            ticker: {
                "shares": round(shares, 4),
                "value": round(shares * prices.get(ticker, 0), 2),
                "weight_pct": round(100 * shares * prices.get(ticker, 0) / total, 2) if total > 0 else 0,
            }
            for ticker, shares in self.positions.items()
        }
