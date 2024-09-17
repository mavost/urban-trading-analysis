#!/usr/bin/env python
# coding: utf-8

# Build a trader bot which looks at sentiment of live news events and trades
# appropriately.
# [Github](https://github.com/nicknochnack/MLTradingBot)

# Author: Nick Renotte
# Version: 1.x
# License: This project is licensed under the MIT License

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import Asset

from datetime import datetime, timedelta
from alpaca_trade_api import REST
from dotenv import dotenv_values

from utils.finbert_utils import estimate_sentiment

env_dict = dotenv_values("../.env")

ALPACA_CREDS = {
    "BASE_URL": env_dict['ALP_API_URL'],
    "API_KEY": env_dict['ALP_API_KEY'],
    "API_SECRET": env_dict['ALP_API_SECRET'],
    "PAPER": True,
}


class MLTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
        print("INFO: initializing")
        self.sleeptime = "24H"
        self.last_trade = None

        self.api = REST(
            base_url=self.parameters["credentials"]["BASE_URL"],
            key_id=self.parameters["credentials"]["API_KEY"],
            secret_key=self.parameters["credentials"]["API_SECRET"],
        )

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.parameters["symbol"])
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        symbol = self.parameters["symbol"]
        print(f"INFO: getting news for {symbol}")

        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=symbol, start=three_days_prior, end=today)

        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)

        return probability, sentiment

    def on_trading_iteration(self):
        print("INFO: trading loop")
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > 0.999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.parameters["symbol"],
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=round(
                        last_price * self.parameters["trade_upside"], 2
                    ),
                    stop_loss_price=round(
                        last_price * self.parameters["trade_stop_loss"], 2
                    ),
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.parameters["symbol"],
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=round(
                        last_price / self.parameters["trade_upside"], 2
                    ),
                    stop_loss_price=round(
                        last_price / self.parameters["trade_stop_loss"], 2
                    ),
                )
                self.submit_order(order)
                self.last_trade = "sell"


strat_dict = {
    "broker": Alpaca(ALPACA_CREDS),
    "name": 'mlstrat',
    "parameters": {
        "cash_at_risk": 0.5,
        "credentials": ALPACA_CREDS,
        "symbol": "SPY",
        "trade_upside": 1.10,
        "trade_stop_loss": 0.95,
        "benchmark_asset": Asset(symbol="AAPL", asset_type="stock"),
    },
}

strategy = MLTrader(**strat_dict, force_start_immediately=True)

start_date = datetime(2022, 1, 1)
end_date = start_date + timedelta(days=365 * 2)

backtest_dict = {
    "datasource_class": YahooDataBacktesting,
    "backtesting_start": start_date,
    "backtesting_end": end_date,
    "benchmark_asset": Asset(symbol="AAPL", asset_type="stock"),
}

# strategy.backtest(
#     YahooDataBacktesting,
#     start_date,
#     end_date,
#     parameters={"symbol":"SPY", "cash_at_risk":.5}
# )
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
