class SentimentEnsemble:
    def __init__(self):
        self.BULLISH_KEYWORDS = [
            ("ceasefire", 0.80), ("peace deal", 0.75), ("FII buying", 0.85),
            ("crude falls", 0.70), ("oil drops", 0.65), ("rate cut", 0.75),
            ("RBI holds", 0.40), ("GDP beats", 0.60), ("inflation falls", 0.55),
            ("trade deal", 0.65), ("tariff removed", 0.70), ("rupee strengthens", 0.50),
        ]
        self.BEARISH_KEYWORDS = [
            ("war escalation", -0.85), ("sanctions", -0.70), ("FII selling", -0.80),
            ("crude surges", -0.75), ("rate hike", -0.70), ("recession", -0.65),
            ("GDP miss", -0.60), ("inflation rises", -0.55), ("rupee weakens", -0.45),
            ("tariff", -0.50), ("border tension", -0.65), ("defaults", -0.60),
        ]

    def fetch_headlines(self):
        import os
        import requests
        from datetime import datetime, timedelta

        headlines = []
        finnhub_key = os.environ.get("FINNHUB_API_KEY")
        if finnhub_key:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                url = f"https://finnhub.io/api/v1/company-news?symbol=INDA&from={yesterday}&to={today}&token={finnhub_key}"
                res = requests.get(url, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    headlines.extend([item.get('headline', '') for item in data])
            except Exception:
                pass

        # Basic RSS fallback
        if not headlines:
            import xml.etree.ElementTree as ET
            feeds = [
                "https://feeds.reuters.com/reuters/businessNews",
                "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
            ]
            for feed in feeds:
                try:
                    res = requests.get(feed, timeout=5)
                    if res.status_code == 200:
                        root = ET.fromstring(res.content)
                        for item in root.findall('.//item/title'):
                            if item.text:
                                headlines.append(item.text)
                except Exception:
                    pass

        return headlines

    def score(self, headlines: list) -> tuple:
        total_score, total_weight = 0.0, 0.0
        matched = []
        for hl in headlines:
            hl_lower = hl.lower()
            for kw, score in self.BULLISH_KEYWORDS + self.BEARISH_KEYWORDS:
                if kw.lower() in hl_lower:
                    weight = 1.0
                    total_score  += score * weight
                    total_weight += weight
                    matched.append((kw, score))
        if total_weight == 0:
            return 0.0, 0.0
        final_score = max(-1.0, min(1.0, total_score / total_weight))
        conf = min(0.90, 0.40 + len(matched) * 0.08)
        return final_score, conf
