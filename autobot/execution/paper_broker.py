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
class MultiLegPosition:
    """A spread or iron condor: multiple legs treated as one risk unit.
    legs: list of {"symbol", "side" ("BUY"/"SELL"), "entry", "qty"}."""
    legs: list
    max_loss_per_unit: float
    qty: int
    kind: str            # e.g. "DEBIT_SPREAD", "CREDIT_CONDOR"
    entry_net_cash: float  # cash flow at entry (positive=credit received, negative=debit paid)


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

    def buy_multi(self, legs, max_loss_per_unit, qty, kind="DEBIT_SPREAD"):
        """
        Open a multi-leg position (spread or iron condor).
        legs: list of {"symbol": str, "side": "BUY"/"SELL", "price": float}

        Risk-checked against max_loss_per_unit * qty — the TRUE worst case —
        not the net premium. A credit spread's cash inflow at entry is not
        its risk; a bug that inverted this would look "free" until assigned.
        """
        worst_case_risk = max_loss_per_unit * qty
        if worst_case_risk > self.capital * 0.60:
            return None  # same circuit breaker spirit as buy()

        total_flow = 0.0
        total_charges = 0.0
        built_legs = []
        for leg in legs:
            side = leg["side"]
            px = leg["price"] * (1 + self.SLIPPAGE_PCT) if side == "BUY" else leg["price"] * (1 - self.SLIPPAGE_PCT)
            turnover = px * qty
            ch = self.charges(turnover, sell_side=(side == "SELL"))
            total_charges += ch
            total_flow += (-turnover if side == "BUY" else turnover)
            built_legs.append({"symbol": leg["symbol"], "side": side, "entry": px, "qty": qty})
            self.fills.append(Fill(leg["symbol"], qty, px, side, ch))

        net_cash = total_flow - total_charges
        if -net_cash > self.capital * 0.95:  # a net-debit spread costing more than we have
            return None

        self.capital += net_cash  # credit adds to capital, debit subtracts
        pos = MultiLegPosition(legs=built_legs, max_loss_per_unit=max_loss_per_unit,
                                qty=qty, kind=kind, entry_net_cash=net_cash)
        self.positions.append(pos)
        return pos

    def close_multi(self, pos: MultiLegPosition, exit_prices: dict):
        """exit_prices: {symbol: current_price} for every leg. Unwinds each
        leg (a BUY leg is sold to close, a SELL leg is bought back)."""
        total_flow = 0.0
        for leg in pos.legs:
            price = exit_prices.get(leg["symbol"], leg["entry"])
            closing_side = "SELL" if leg["side"] == "BUY" else "BUY"
            px = price * (1 - self.SLIPPAGE_PCT) if closing_side == "SELL" else price * (1 + self.SLIPPAGE_PCT)
            turnover = px * pos.qty
            ch = self.charges(turnover, sell_side=(closing_side == "SELL"))
            total_flow += (turnover if closing_side == "SELL" else -turnover) - ch
            self.fills.append(Fill(leg["symbol"], pos.qty, px, closing_side, ch))

        self.capital += total_flow
        pnl = pos.entry_net_cash + total_flow
        self.realized_pnl += pnl
        self.positions.remove(pos)
        return pnl
