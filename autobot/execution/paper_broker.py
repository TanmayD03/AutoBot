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
    entry_charges: float = 0.0


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
        if cost + ch > self.capital:
            return None
        self.capital -= cost + ch
        pos = Position(symbol, qty, px, stop, target, entry_charges=ch)
        self.positions.append(pos)
        self.fills.append(Fill(symbol, qty, px, "BUY", ch))
        return pos

    def close(self, pos: Position, price):
        px = price * (1 - self.SLIPPAGE_PCT)
        proceeds = px * pos.qty
        ch = self.charges(proceeds, sell_side=True)
        self.capital += proceeds - ch
        pnl = (px - pos.entry) * pos.qty - ch - pos.entry_charges
        self.realized_pnl += pnl
        self.fills.append(Fill(pos.symbol, pos.qty, px, "SELL", ch))
        self.positions.remove(pos)
        return pnl
