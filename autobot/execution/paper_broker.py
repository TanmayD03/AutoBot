"""PaperBroker — default execution venue. Realistic fills with slippage and
Indian F&O cost model (brokerage, STT, exchange charges, GST, stamp duty approximation).
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Position:
    symbol: str
    qty: int
    entry: float
    stop: float
    target: float
    direction: int = 1  # long premium


@dataclass
class Fill:
    symbol: str
    qty: int
    price: float
    side: str
    charges: float


class PaperBroker:
    SLIPPAGE_PCT = 0.05 / 100

    def __init__(self, capital=100000.0):
        self.capital = capital
        self.positions: List[Position] = []
        self.fills: List[Fill] = []
        self.realized_pnl = 0.0

    @staticmethod
    def charges(turnover, sell_side=False):
        brokerage = min(20.0, turnover * 0.0003)
        stt = turnover * 0.000625 if sell_side else 0.0
        exch = turnover * 0.00035
        gst = (brokerage + exch) * 0.18
        stamp = turnover * 0.00003 if not sell_side else 0.0
        return brokerage + stt + exch + gst + stamp

    def buy(self, symbol, qty, price, stop, target):
        px = price * (1 + self.SLIPPAGE_PCT)
        cost = px * qty
        ch = self.charges(cost)

        # Circuit breaker: Reject if risk exceeds 60% of capital
        # (This catches math bugs in position sizing dynamically before they balloon)
        if cost + ch > self.capital * 0.60:
            return None

        if cost + ch > self.capital * 0.95:  # small buffer so it doesn't fail on margin edges
            return None

        self.capital -= cost + ch
        pos = Position(symbol, qty, px, stop, target)
        self.positions.append(pos)
        self.fills.append(Fill(symbol, qty, px, "BUY", ch))
        return pos

    def close(self, pos: Position, price):
        px = price * (1 - self.SLIPPAGE_PCT)
        proceeds = px * pos.qty
        ch = self.charges(proceeds, sell_side=True)
        self.capital += proceeds - ch
        pnl = (px - pos.entry) * pos.qty - ch
        self.realized_pnl += pnl
        self.fills.append(Fill(pos.symbol, pos.qty, px, "SELL", ch))
        self.positions.remove(pos)
        return pnl

    def close_partial(self, pos: Position, qty: int, price):
        """Close part of a position (scale-out). Reduces pos.qty in place and
        returns the realized PnL on just the closed portion. The position stays
        open (in self.positions) with the remaining qty."""
        qty = min(qty, pos.qty)
        if qty <= 0:
            return 0.0
        px = price * (1 - self.SLIPPAGE_PCT)
        proceeds = px * qty
        ch = self.charges(proceeds, sell_side=True)
        self.capital += proceeds - ch
        pnl = (px - pos.entry) * qty - ch
        self.realized_pnl += pnl
        self.fills.append(Fill(pos.symbol, qty, px, "SELL", ch))
        pos.qty -= qty
        if pos.qty <= 0:
            self.positions.remove(pos)
        return pnl
