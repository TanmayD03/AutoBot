import re
from dataclasses import dataclass
from typing import List

EVENT_LEXICON = {
    "geopolitical": (["war", "strike", "attack", "missile", "sanction", "iran", "conflict",
                      "invasion", "escalat", "tension", "troop"], -0.9),
    "rates": (["rate hike", "fed raises", "rbi hikes", "hawkish", "tightening", "inflation spike"], -0.5),
    "rates_dovish": (["rate cut", "dovish", "rbi cuts", "fed cuts", "easing", "pause"], 0.5),
    "oil_shock": (["crude surge", "oil spike", "brent above", "opec cut", "supply disruption"], -0.6),
    "macro_negative": (["recession", "default", "crash", "crisis", "plunge", "selloff", "bear market", "panic", "collapse"], -0.7),
    "macro_positive": (["record high", "rally", "upgrade", "beats estimates", "stimulus", "bull market", "surge", "breakout"], 0.6),
    "earnings_beat": (["blows past estimates", "strong earnings", "profit jump", "raises guidance", "smash estimates"], 0.5),
    "earnings_miss": (["misses estimates", "weak earnings", "profit drop", "cuts guidance", "disappoints"], -0.5),
}


@dataclass
class NewsImpact:
    headline: str
    impact: float  # [-1.0, 1.0]
    category: str


class SentimentEnsemble:
    def __init__(self):
        # In a real deployed environment, load HuggingFace models here
        # (e.g. ProsusAI/finbert, mrm8488/deberta-v3-small-finetuned-financial-news)
        pass

    def analyze_headline(self, headline: str) -> NewsImpact:
        """Fast lexicon fallback when transformers are too slow or offline."""
        headline_low = headline.lower()
        net_impact = 0.0
        matched_cat = "neutral"

        for cat, (keywords, weight) in EVENT_LEXICON.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', headline_low):
                    net_impact += weight
                    matched_cat = cat
                    break  # found one keyword in this category, move to next category

        # Cap between -1 and 1
        net_impact = max(-1.0, min(1.0, net_impact))
        return NewsImpact(headline, net_impact, matched_cat)

    def score(self, text: str) -> dict:
        """Compatibility wrapper for tests."""
        res = self.analyze_headline(text)
        return {"sentiment": res.impact, "event": res.category, "category": res.category}

    def aggregate_daily(self, headlines: List[str]) -> float:
        """Return the exponential moving average of the day's headline sentiment."""
        if not headlines:
            return 0.0
        impacts = [self.analyze_headline(h).impact for h in headlines]
        # Weight recent headlines more heavily if they are ordered chronologically
        total = sum(imp * (1.2 ** i) for i, imp in enumerate(impacts))
        div = sum(1.2 ** i for i in range(len(impacts)))
        return total / div
