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
    px: float
    side: str
    charges: float

class PaperBroker:
    def __init__(self, capital: float):
        self.capital = capital
        self.positions: List[Position] = []
        self.fills: List[Fill] = []
        self.realized_pnl = 0.0

    def charges(self, proceeds: float, sell_side: bool = False) -> float:
        """Estimate NSE F&O transaction + regulatory + GST + STT costs."""
        c = 20.0  # Flat brokerage
        c += proceeds * 0.0005  # Exchange tx charge ~0.05%
        c += (c * 0.18)         # GST
        if sell_side:
            c += proceeds * 0.000625  # STT (sell only, 0.0625%)
        return c

    def buy(self, symbol: str, qty: int, px: float, stop: float, target: float):
        cost = px * qty
        ch = self.charges(cost, sell_side=False)
        if cost + ch > self.capital:
            return None
        self.capital -= cost + ch
        pos = Position(symbol, qty, px, stop, target, entry_charges=ch)
        self.positions.append(pos)
        self.fills.append(Fill(symbol, qty, px, "BUY", ch))
        return pos

    def close(self, pos: Position, px: float):
        proceeds = px * pos.qty
        ch = self.charges(proceeds, sell_side=True)
        self.capital += proceeds - ch
        pnl = (px - pos.entry) * pos.qty - ch - pos.entry_charges
        self.realized_pnl += pnl
        self.fills.append(Fill(pos.symbol, pos.qty, px, "SELL", ch))
        self.positions.remove(pos)
        return pnl
