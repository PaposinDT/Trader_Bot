import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy
from freqtrade.strategy import (CategoricalParameter, DecimalParameter, 
                                IntParameter, RealParameter)

__author__ = "Robert Roman"
__copyright__ = "Free For Use"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Robert Roman"
__email__ = "robertroman7@gmail.com"
__BTC_donation__ = "3FgFaG15yntZYSUzfEpxr5mDt1RArvcQrK"

class Bandtastic(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'

    # Optimal ROI from hyperopt
    minimal_roi = {
        "0": 0.348,
        "40": 0.129,
        "120": 0.062,
        "461": 0
    }

    # Optimal stoploss from hyperopt
    stoploss = -0.326

    # Optimal trailing stop from hyperopt
    trailing_stop = True
    trailing_stop_positive = 0.333
    trailing_stop_positive_offset = 0.402
    trailing_only_offset_is_reached = True

    # Optimal max_open_trades from hyperopt
    max_open_trades = 5

    startup_candle_count = 999

    # Hyperopt Buy Parameters - optimized values
    buy_fastema = IntParameter(low=1, high=236, default=203, space='buy', optimize=True, load=True)
    buy_slowema = IntParameter(low=1, high=250, default=163, space='buy', optimize=True, load=True)
    buy_rsi = IntParameter(low=15, high=70, default=17, space='buy', optimize=True, load=True)
    buy_mfi = IntParameter(low=15, high=70, default=16, space='buy', optimize=True, load=True)

    buy_rsi_enabled = CategoricalParameter([True, False], space='buy', default=True, optimize=True, load=True)
    buy_mfi_enabled = CategoricalParameter([True, False], space='buy', default=True, optimize=True, load=True)
    buy_ema_enabled = CategoricalParameter([True, False], space='buy', default=False, optimize=True, load=True)
    buy_trigger = CategoricalParameter(["bb_lower1", "bb_lower2", "bb_lower3", "bb_lower4"], 
                                     default="bb_lower2", space="buy", optimize=True, load=True)

    # Hyperopt Sell Parameters - optimized values
    sell_fastema = IntParameter(low=1, high=365, default=331, space='sell', optimize=True, load=True)
    sell_slowema = IntParameter(low=1, high=365, default=349, space='sell', optimize=True, load=True)
    sell_rsi = IntParameter(low=30, high=100, default=54, space='sell', optimize=True, load=True)
    sell_mfi = IntParameter(low=30, high=100, default=87, space='sell', optimize=True, load=True)

    sell_rsi_enabled = CategoricalParameter([True, False], space='sell', default=False, optimize=True, load=True)
    sell_mfi_enabled = CategoricalParameter([True, False], space='sell', default=False, optimize=True, load=True)
    sell_ema_enabled = CategoricalParameter([True, False], space='sell', default=True, optimize=True, load=True)
    sell_trigger = CategoricalParameter(["sell-bb_upper1", "sell-bb_upper2", "sell-bb_upper3", "sell-bb_upper4"], 
                                      default="sell-bb_upper4", space="sell", optimize=True, load=True)

    def rsi(self, dataframe, period=14):
        """Relative Strength Index (RSI) implementato con pandas"""
        delta = dataframe['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def mfi(self, dataframe, period=14):
        """Money Flow Index (MFI) implementato con pandas"""
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        money_flow = typical_price * dataframe['volume']
        
        positive_flow = money_flow.where(dataframe['close'] > dataframe['close'].shift(1), 0)
        negative_flow = money_flow.where(dataframe['close'] < dataframe['close'].shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period, min_periods=period).sum()
        
        money_ratio = positive_flow_sum / negative_flow_sum
        return 100 - (100 / (1 + money_ratio))

    def ema(self, series, period):
        """Exponential Moving Average (EMA) implementato con pandas"""
        return series.ewm(span=period, adjust=False).mean()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calcola tutti gli indicatori in un nuovo DataFrame temporaneo
        temp_df = dataframe.copy()
        
        # RSI e MFI
        temp_df['rsi'] = self.rsi(temp_df)
        temp_df['mfi'] = self.mfi(temp_df)

        # Bollinger Bands
        for stds in [1, 2, 3, 4]:
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(temp_df), window=20, stds=stds)
            temp_df[f'bb_lowerband{stds}'] = bollinger['lower']
            temp_df[f'bb_middleband{stds}'] = bollinger['mid']
            temp_df[f'bb_upperband{stds}'] = bollinger['upper']

        # Calcola tutte le EMA necessarie in un singolo passaggio
        ema_periods = set(
            list(self.buy_fastema.range) +
            list(self.buy_slowema.range) +
            list(self.sell_fastema.range) +
            list(self.sell_slowema.range)
        )
        
        # Crea un DataFrame temporaneo per le EMA
        ema_data = {}
        for period in ema_periods:
            ema_data[f'EMA_{period}'] = self.ema(temp_df['close'], period)
        
        # Unisci tutte le EMA al DataFrame principale in un'unica operazione
        ema_df = pd.DataFrame(ema_data, index=temp_df.index)
        temp_df = pd.concat([temp_df, ema_df], axis=1)

        return temp_df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # GUARDS
        if self.buy_rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi.value)
        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] < self.buy_mfi.value)
        if self.buy_ema_enabled.value:
            conditions.append(dataframe[f'EMA_{self.buy_fastema.value}'] > dataframe[f'EMA_{self.buy_slowema.value}'])

        # TRIGGERS
        if self.buy_trigger.value == 'bb_lower1':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband1'])
        if self.buy_trigger.value == 'bb_lower2':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband2'])
        if self.buy_trigger.value == 'bb_lower3':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband3'])
        if self.buy_trigger.value == 'bb_lower4':
            conditions.append(dataframe["close"] < dataframe['bb_lowerband4'])

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # GUARDS
        if self.sell_rsi_enabled.value:
            conditions.append(dataframe['rsi'] > self.sell_rsi.value)
        if self.sell_mfi_enabled.value:
            conditions.append(dataframe['mfi'] > self.sell_mfi.value)
        if self.sell_ema_enabled.value:
            conditions.append(dataframe[f'EMA_{self.sell_fastema.value}'] < dataframe[f'EMA_{self.sell_slowema.value}'])

        # TRIGGERS
        if self.sell_trigger.value == 'sell-bb_upper1':
            conditions.append(dataframe["close"] > dataframe['bb_upperband1'])
        if self.sell_trigger.value == 'sell-bb_upper2':
            conditions.append(dataframe["close"] > dataframe['bb_upperband2'])
        if self.sell_trigger.value == 'sell-bb_upper3':
            conditions.append(dataframe["close"] > dataframe['bb_upperband3'])
        if self.sell_trigger.value == 'sell-bb_upper4':
            conditions.append(dataframe["close"] > dataframe['bb_upperband4'])

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe
