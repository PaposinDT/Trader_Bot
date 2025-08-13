import logging
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
from functools import reduce

class Supertrend(IStrategy):
    """
    Supertrend strategy optimized to avoid DataFrame fragmentation warnings
    """
    
    INTERFACE_VERSION: int = 3
    
    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.087,
        "372": 0.058,
        "861": 0.029,
        "2221": 0
    }

    # Stoploss:
    stoploss = -0.265

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.144
    trailing_only_offset_is_reached = False

    timeframe = '1h'
    startup_candle_count = 199

    buy_m1 = IntParameter(1, 7, default=4)
    buy_m2 = IntParameter(1, 7, default=4)
    buy_m3 = IntParameter(1, 7, default=4)
    buy_p1 = IntParameter(7, 21, default=14)
    buy_p2 = IntParameter(7, 21, default=14)
    buy_p3 = IntParameter(7, 21, default=14)

    sell_m1 = IntParameter(1, 7, default=4)
    sell_m2 = IntParameter(1, 7, default=4)
    sell_m3 = IntParameter(1, 7, default=4)
    sell_p1 = IntParameter(7, 21, default=14)
    sell_p2 = IntParameter(7, 21, default=14)
    sell_p3 = IntParameter(7, 21, default=14)

    def calculate_atr(self, dataframe: DataFrame, period: int) -> DataFrame:
        """Calculate ATR without TA-Lib"""
        df = dataframe.copy()
        df['prev_close'] = df['close'].shift(1)
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['prev_close'])
        tr3 = abs(df['low'] - df['prev_close'])
        
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return df['TR'].rolling(window=period).mean()

    def supertrend(self, dataframe: DataFrame, multiplier: int, period: int) -> dict:
        """Calculate Supertrend indicator without TA-Lib"""
        df = dataframe.copy()
        df['ATR'] = self.calculate_atr(df, period)
        hl2 = (df['high'] + df['low']) / 2
        
        # Basic bands
        df['basic_ub'] = hl2 + multiplier * df['ATR']
        df['basic_lb'] = hl2 - multiplier * df['ATR']
        
        # Initialize columns
        df['final_ub'] = 0.0
        df['final_lb'] = 0.0
        st_col = np.zeros(len(df))
        stx_col = np.empty(len(df), dtype=object)
        
        # Calculate final bands and Supertrend
        for i in range(1, len(df)):
            # Final Upper Band
            if df['basic_ub'].iat[i] < df['final_ub'].iat[i-1] or df['close'].iat[i-1] > df['final_ub'].iat[i-1]:
                df.at[df.index[i], 'final_ub'] = df['basic_ub'].iat[i]
            else:
                df.at[df.index[i], 'final_ub'] = df['final_ub'].iat[i-1]
            
            # Final Lower Band
            if df['basic_lb'].iat[i] > df['final_lb'].iat[i-1] or df['close'].iat[i-1] < df['final_lb'].iat[i-1]:
                df.at[df.index[i], 'final_lb'] = df['basic_lb'].iat[i]
            else:
                df.at[df.index[i], 'final_lb'] = df['final_lb'].iat[i-1]
            
            # Supertrend
            if i >= period:
                if st_col[i-1] == df['final_ub'].iat[i-1]:
                    st_col[i] = df['final_ub'].iat[i] if df['close'].iat[i] <= df['final_ub'].iat[i] else df['final_lb'].iat[i]
                else:
                    st_col[i] = df['final_lb'].iat[i] if df['close'].iat[i] >= df['final_lb'].iat[i] else df['final_ub'].iat[i]
                
                # Supertrend direction
                if st_col[i] > 0:
                    stx_col[i] = 'up' if df['close'].iat[i] > st_col[i] else 'down'
        
        return {
            'ST': st_col,
            'STX': stx_col
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators efficiently without fragmenting the DataFrame"""
        temp_df = dataframe.copy()
        
        # Collect all unique parameter combinations
        param_combinations = set()
        for param_group in [(self.buy_m1, self.buy_p1), 
                          (self.buy_m2, self.buy_p2),
                          (self.buy_m3, self.buy_p3),
                          (self.sell_m1, self.sell_p1),
                          (self.sell_m2, self.sell_p2),
                          (self.sell_m3, self.sell_p3)]:
            for m in param_group[0].range:
                for p in param_group[1].range:
                    param_combinations.add((m, p))
        
        # Pre-allocate storage for all indicators
        st_data = {}
        stx_data = {}
        
        # Calculate all Supertrend indicators
        for m, p in param_combinations:
            result = self.supertrend(temp_df, m, p)
            st_data[f'supertrend_{m}_{p}_ST'] = result['ST']
            stx_data[f'supertrend_{m}_{p}_STX'] = result['STX']
        
        # Combine all indicators in one operation
        st_df = pd.DataFrame(st_data, index=temp_df.index)
        stx_df = pd.DataFrame(stx_data, index=temp_df.index)
        
        # Merge with original DataFrame
        temp_df = pd.concat([temp_df, st_df, stx_df], axis=1)
        
        return temp_df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe[f'supertrend_{self.buy_m1.value}_{self.buy_p1.value}_STX'] == 'up'),
            (dataframe[f'supertrend_{self.buy_m2.value}_{self.buy_p2.value}_STX'] == 'up'),
            (dataframe[f'supertrend_{self.buy_m3.value}_{self.buy_p3.value}_STX'] == 'up'),
            (dataframe['volume'] > 0)
        ]

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe[f'supertrend_{self.sell_m1.value}_{self.sell_p1.value}_STX'] == 'down'),
            (dataframe[f'supertrend_{self.sell_m2.value}_{self.sell_p2.value}_STX'] == 'down'),
            (dataframe[f'supertrend_{self.sell_m3.value}_{self.sell_p3.value}_STX'] == 'down'),
            (dataframe['volume'] > 0)
        ]

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1

        return dataframe
