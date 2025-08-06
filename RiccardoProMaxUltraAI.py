import logging
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import List
from pandas import DataFrame

logger = logging.getLogger(__name__)

class RiccardoProMaxUltraAI(IStrategy):
    timeframe = '1h'
    can_short = False
    startup_candle_count: int = 100

    minimal_roi = {
        "0": 0.08,
        "30": 0.04,
        "60": 0.02
    }
    stoploss = -0.015
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    _whitelist_cache = None
    _whitelist_last_updated = None

    @property
    def whitelist(self) -> List[str]:
        now = datetime.utcnow()
        if self._whitelist_cache and self._whitelist_last_updated and (now - self._whitelist_last_updated) < timedelta(hours=6):
            return self._whitelist_cache

        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            markets = response.json()
            filtered = [
                f"{d['symbol']}/USDT" for d in markets
                if d['symbol'].endswith('USDT')
                and float(d['quoteVolume']) > 50000000
                and not d['symbol'].endswith('UPUSDT')
                and not d['symbol'].endswith('DOWNUSDT')
            ]
            sorted_markets = sorted(filtered, key=lambda x: float(next(d['quoteVolume'] for d in markets if d['symbol'] == x.split('/')[0])), reverse=True)[:10]
            self._whitelist_cache = sorted_markets
            self._whitelist_last_updated = now
            return sorted_markets
        except Exception as e:
            logger.error(f"Error updating whitelist: {e}")
            return [
                "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
                "XRP/USDT", "ADA/USDT", "DOGE/USDT", "MATIC/USDT"
            ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def feature_engineering_expand_all(self, dataframe: DataFrame, period, **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-close_-1_pct"] = dataframe["close"].shift(1).pct_change()
        dataframe["%-close_-2_pct"] = dataframe["close"].shift(2).pct_change()
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-is_weekend"] = (dataframe["%-day_of_week"] >= 5).astype(int)
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["&-target"] = (
            dataframe["close"].shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            / dataframe["close"] - 1
        )
        return dataframe

    def bot_start(self, **kwargs) -> None:
        self._refresh_whitelist()
        self.notify_telegram("🤖 Bot avviato correttamente")

    def bot_loop_start(self, **kwargs) -> None:
        now = kwargs['current_time']
        if now.hour == 0 and now.minute < 5:
            self._refresh_whitelist()

    def _refresh_whitelist(self) -> None:
        new_list = self.whitelist
        with open(f"{self.config['user_data_dir']}/whitelist.json", 'w') as f:
            json.dump(new_list, f)
        self.notify_telegram(f"🔄 Whitelist aggiornata:\n{', '.join(new_list)}")

    def notify_telegram(self, message: str) -> None:
        if self.dp and hasattr(self.dp, 'send_telegram_message'):
            self.dp.send_telegram_message(message)
