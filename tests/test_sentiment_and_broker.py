from autobot.sentiment.analyzer import SentimentEnsemble
from autobot.execution.paper_broker import PaperBroker


def test_war_headline_is_strongly_negative():
    s = SentimentEnsemble()
    r = s.score("US strikes Iran nuclear sites, war escalation feared")
    assert r["sentiment"] < -0.5 and r["category"] == "geopolitical"


def test_rally_headline_is_positive():
    s = SentimentEnsemble()
    assert s.score("Sensex hits record high as markets rally on stimulus")["sentiment"] > 0


def test_paper_broker_roundtrip_accounting():
    b = PaperBroker(capital=100000)
    pos = b.buy("NIFTY25000CE", 75, 100.0, stop=90.0, target=126.0)
    assert pos is not None and b.capital < 100000
    pnl = b.close(pos, 126.0)
    assert pnl > 0 and not b.positions


def test_paper_broker_rejects_oversized_order():
    b = PaperBroker(capital=1000)
    assert b.buy("NIFTY25000CE", 75, 100.0, 90.0, 126.0) is None
