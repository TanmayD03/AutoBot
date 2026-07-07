"""Live monitoring terminal (Rich). Panels: macro, option-chain structure, signals,
sentiment feed, positions/P&L vs kill switch, trade log, system health.
Run: python -m autobot.terminal.dashboard
"""
import time
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Group


class Dashboard:
    def __init__(self):
        self.state = {"macro": {}, "chain": {}, "signals": [], "sentiment": [],
                      "positions": [], "trades": [], "day_pnl": 0.0,
                      "kill_limit": -20.0, "profit_lock": 50.0, "latency_ms": 0.0,
                      "status": "PAPER MODE"}

    def render(self):
        layout = Layout()
        layout.split_column(Layout(name="top", size=12), Layout(name="mid", size=12),
                            Layout(name="bottom"))
        macro = Table(title="Global Macro", expand=True)
        macro.add_column("Asset"); macro.add_column("Last"); macro.add_column("Chg %")
        for k, v in list(self.state["macro"].items())[:8]:
            chg = v.get("chg_pct", 0.0)
            macro.add_row(k, str(v.get("last", "-")), f"[{'green' if chg >= 0 else 'red'}]{chg:+.2f}%")
        chain = self.state["chain"]
        chain_txt = (f"Spot {chain.get('spot','-')} | PCR {chain.get('pcr','-')} | "
                     f"MaxPain {chain.get('max_pain','-')} | CallWall {chain.get('ceiling','-')} | "
                     f"PutWall {chain.get('floor','-')} | VIX {chain.get('vix','-')} | GEX {chain.get('gex','-')}")
        layout["top"].split_row(Layout(macro), Layout(Panel(chain_txt, title="Option Chain Structure")))
        sig = Table(title="Signals", expand=True)
        sig.add_column("Signal"); sig.add_column("Score"); sig.add_column("Conf")
        for s in self.state["signals"]:
            sig.add_row(s.name, f"{s.score:+.2f}", f"{s.confidence:.2f}")
        sent = Table(title="News Sentiment", expand=True)
        sent.add_column("Headline", overflow="fold"); sent.add_column("Impact")
        for h, imp in self.state["sentiment"][:5]:
            sent.add_row(h[:70], f"{imp:+.2f}")
        layout["mid"].split_row(Layout(sig), Layout(sent))
        pnl = self.state["day_pnl"]
        pos_lines = [f"{p.symbol} qty {p.qty} entry {p.entry:.2f} stop {p.stop:.2f} target {p.target:.2f}"
                     for p in self.state["positions"]] or ["(flat)"]
        trade_lines = [str(t) for t in self.state["trades"][-5:]] or ["(no trades yet)"]
        latency = self.state.get("latency_ms")
        latency_str = f"{latency:.0f} ms" if latency is not None else "N/A (flat, no active tick stream)"
        footer = Panel(Group(
            f"[bold]{self.state['status']}[/bold]  Day P&L: "
            f"[{'green' if pnl >= 0 else 'red'}]{pnl:+.2f}[/]  "
            f"(kill {self.state['kill_limit']:+.0f} / lock {self.state['profit_lock']:+.0f})  "
            f"latency {latency_str}",
            "Positions: " + " | ".join(pos_lines),
            "Recent trades: " + " | ".join(trade_lines)), title="Risk & Execution")
        layout["bottom"].update(footer)
        return layout

    def snapshot(self):
        """One-shot render for notebooks (Colab), where Live loops cannot display."""
        from rich.console import Console
        Console().print(self.render())

    def run(self, update_fn=None, refresh=2.0):
        with Live(self.render(), refresh_per_second=2) as live:
            while True:
                if update_fn:
                    update_fn(self.state)
                live.update(self.render())
                time.sleep(refresh)


if __name__ == "__main__":
    from ..data.market import MacroFeed
    dash = Dashboard()

    def update(state):
        try:
            state["macro"] = MacroFeed().snapshot()
            state["latency_ms"] = 200.0
        except Exception as e:
            state["status"] = f"feed error: {e}"
    dash.run(update, refresh=30.0)
