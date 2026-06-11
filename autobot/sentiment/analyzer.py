"""Sentiment ensemble. Prefers transformer models (FinBERT-tone + DeBERTa-v3 financial)
when installed; falls back to a fast event-lexicon scorer so the demo runs anywhere.
Maps headlines -> (sentiment in [-1,1], event_category, market_impact).
Optional X/Twitter adapter (e.g. @deitaone) requires a paid API key — disabled by default.
"""

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


class SentimentEnsemble:
    def __init__(self):
        self._pipes = None
        try:
            from transformers import pipeline
            self._pipes = [
                pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone"),
                pipeline("sentiment-analysis",
                         model="mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"),
            ]
        except Exception:
            self._pipes = None  # lexicon fallback

    def _lexicon_score(self, text):
        t = text.lower()
        best_cat, best_score = "none", 0.0
        for cat, (kws, impact) in EVENT_LEXICON.items():
            if any(kw in t for kw in kws):
                if abs(impact) > abs(best_score):
                    best_cat, best_score = cat, impact
        return best_score, best_cat

    def score(self, headline: str):
        """Returns dict(sentiment, category, impact)."""
        lex_score, category = self._lexicon_score(headline)
        if self._pipes:
            vals = []
            for p in self._pipes:
                r = p(headline[:512])[0]
                sign = 1 if "pos" in r["label"].lower() else (-1 if "neg" in r["label"].lower() else 0)
                vals.append(sign * r["score"])
            model_score = sum(vals) / len(vals)
            sentiment = 0.6 * model_score + 0.4 * lex_score
        else:
            sentiment = lex_score
        return {"sentiment": sentiment, "category": category,
                "impact": sentiment * (1.5 if category == "geopolitical" else 1.0)}


def fetch_headlines(feeds, limit=20):
    import feedparser
    out = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            out += [e.title for e in d.entries[:limit]]
        except Exception:
            continue
    return out[:limit * len(feeds)]
