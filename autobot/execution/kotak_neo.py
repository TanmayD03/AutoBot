"""Kotak Neo live broker adapter (Tier-1 low-latency WebSocket feed + order routing).
Requires: pip install neo-api-client, plus env vars KOTAK_CONSUMER_KEY/SECRET/MOBILE/PASSWORD/MPIN.
LIVE MODE GUARD: refuses to start unless config mode=='live' AND
AUTOBOT_CONFIRM_LIVE=YES_I_ACCEPT_THE_RISK is exported.
"""
import os


class KotakNeoBroker:
    def __init__(self, mode="paper"):
        if mode != "live" or os.environ.get("AUTOBOT_CONFIRM_LIVE") != "YES_I_ACCEPT_THE_RISK":
            raise RuntimeError("Live mode blocked: set mode=live AND AUTOBOT_CONFIRM_LIVE env var.")
        from neo_api_client import NeoAPI
        self.client = NeoAPI(consumer_key=os.environ["KOTAK_CONSUMER_KEY"],
                             consumer_secret=os.environ["KOTAK_CONSUMER_SECRET"],
                             environment="prod")
        self.client.login(mobilenumber=os.environ["KOTAK_MOBILE"],
                          password=os.environ["KOTAK_PASSWORD"])
        self.client.session_2fa(OTP=os.environ["KOTAK_MPIN"])

    def subscribe_ticks(self, tokens, on_message):
        """Tier-1 low-latency tick + depth stream."""
        self.client.on_message = on_message
        self.client.subscribe(instrument_tokens=tokens, isIndex=False, isDepth=True)

    def buy_option(self, trading_symbol, qty):
        return self.client.place_order(
            exchange_segment="nse_fo", product="NRML", price="0", order_type="MKT",
            quantity=str(qty), validity="DAY", trading_symbol=trading_symbol,
            transaction_type="B", amo="NO", disclosed_quantity="0", market_protection="0",
            pf="N", trigger_price="0", tag="autobot")

    def sell_option(self, trading_symbol, qty):
        return self.client.place_order(
            exchange_segment="nse_fo", product="NRML", price="0", order_type="MKT",
            quantity=str(qty), validity="DAY", trading_symbol=trading_symbol,
            transaction_type="S", amo="NO", disclosed_quantity="0", market_protection="0",
            pf="N", trigger_price="0", tag="autobot")
