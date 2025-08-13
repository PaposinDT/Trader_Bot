# UltimateHybridStrategyPro.py
import time
from functools import wraps
from collections import deque
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from freqtrade.persistence import Trade, LocalTrade
from freqtrade.enums import RunMode


class TradeStats:
    """Enhanced trade performance tracking with rolling metrics"""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.trade_results = deque(maxlen=window_size)
        self.consecutive_losses = 0
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'profit_factor': 1.0,
            'expectancy': 0.0,
            'win_rate': 0.0,
            'max_consecutive_losses': 0
        }

    def update(self, trade: Trade, exit_rate: Optional[float] = None) -> None:
        # Passa sempre un valore rate valido a calc_profit_ratio()
        if exit_rate is not None:
            profit = trade.calc_profit_ratio(exit_rate)
        else:
            profit = trade.calc_profit_ratio(trade.close_rate)

        is_profit = profit > 0

        self.stats['total_trades'] += 1
        self.trade_results.append(profit)

        if is_profit:
            self.stats['winning_trades'] += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.stats['max_consecutive_losses'] = max(
                self.stats['max_consecutive_losses'],
                self.consecutive_losses
            )

        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        if not self.trade_results:
            return

        wins = [x for x in self.trade_results if x > 0]
        losses = [x for x in self.trade_results if x < 0]

        self.stats['profit_factor'] = sum(wins) / abs(sum(losses)) if losses else float('inf')
        self.stats['win_rate'] = len(wins) / len(self.trade_results)
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        self.stats['expectancy'] = (self.stats['win_rate'] * avg_win) - ((1 - self.stats['win_rate']) * avg_loss)


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            dataframe = kwargs.get('dataframe', args[1] if len(args) > 1 else pd.DataFrame())
            try:
                df_shape = dataframe.shape
            except Exception:
                df_shape = None
            self.logger.debug(f"{func.__name__} executed in {duration:.2f}s | DF shape: {df_shape}")
            return result
        except Exception as e:
            try:
                self.logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            except Exception:
                print(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


def ema(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).clip(0, 100)
    return rsi


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr_series


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    if df.empty:
        return pd.Series([], dtype=float), pd.Series([], dtype=int)

    hl2 = (df['high'] + df['low']) / 2
    atr_series = atr(df, period)

    basic_upper = hl2 + multiplier * atr_series
    basic_lower = hl2 - multiplier * atr_series

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(df)):
        if (basic_upper.iat[i] < final_upper.iat[i - 1]) or (df['close'].iat[i - 1] > final_upper.iat[i - 1]):
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if (basic_lower.iat[i] > final_lower.iat[i - 1]) or (df['close'].iat[i - 1] < final_lower.iat[i - 1]):
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    direction.iat[0] = 1
    supertrend.iat[0] = final_lower.iat[0]

    for i in range(1, len(df)):
        close = df['close'].iat[i]
        prev_final_upper = final_upper.iat[i - 1]
        prev_final_lower = final_lower.iat[i - 1]

        if close > prev_final_upper:
            direction.iat[i] = 1
        elif close < prev_final_lower:
            direction.iat[i] = -1
        else:
            direction.iat[i] = direction.iat[i - 1]
            if direction.iat[i] == 1 and final_lower.iat[i] < final_lower.iat[i - 1]:
                final_lower.iat[i] = final_lower.iat[i - 1]
            if direction.iat[i] == -1 and final_upper.iat[i] > final_upper.iat[i - 1]:
                final_upper.iat[i] = final_upper.iat[i - 1]

        supertrend.iat[i] = final_lower.iat[i] if direction.iat[i] == 1 else final_upper.iat[i]

    direction = direction.fillna(1).astype(int)
    return supertrend, direction


class UltimateHybridStrategyPro5m(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False

    supertrend_period = IntParameter(7, 14, default=14, space='buy')
    supertrend_multiplier = DecimalParameter(2.0, 4.0, default=3.8, decimals=1, space='buy')
    entry_cooldown = IntParameter(1800, 7200, default=6422, space='buy')

    minimal_roi = {
        "0": 0.064,
        "21": 0.042,
        "72": 0.02,
        "160": 0
    }

    stoploss = -0.3
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False
    max_open_trades = 5

    exit_profit_pct = DecimalParameter(0.03, 0.1, default=0.09, decimals=2, space='sell')
    emergency_stop_pct = DecimalParameter(-0.1, -0.05, default=-0.09, decimals=2, space='sell')

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.trade_stats = TradeStats()
        self.logger = logging.getLogger(__name__)
        self.is_backtesting = config.get('runmode') == RunMode.BACKTEST
        self.last_entry = self._load_cooldown_state()
        self._init_strategy_params(config)

    def _init_strategy_params(self, config: dict) -> None:
        strategy_args = config.get('strategy_args', {})
        for param in [
            'supertrend_period', 'supertrend_multiplier',
            'entry_cooldown',
            'exit_profit_pct', 'emergency_stop_pct'
        ]:
            if param in strategy_args:
                param_obj = getattr(self, param, None)
                if param_obj and hasattr(param_obj, 'value'):
                    param_obj.value = strategy_args[param]
                else:
                    setattr(self, param, strategy_args[param])

    def _load_cooldown_state(self) -> Dict[str, float]:
        # Gestisce il cooldown in modo persistente (qui simulato con dict in memoria)
        # Puoi usare file o DB se vuoi persistente
        return {}

    def _save_cooldown_state(self, last_entry: Dict[str, float]) -> None:
        # Salva lo stato cooldown persistente (non implementato qui)
        pass

    def can_enter(self, pair: str, current_time: float) -> bool:
        last = self.last_entry.get(pair, 0)
        cooldown = self.entry_cooldown.value if hasattr(self.entry_cooldown, 'value') else self.entry_cooldown
        if current_time - last > cooldown:
            return True
        return False

    @log_execution_time
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        # Calcolo Supertrend e direzione
        st, direction = supertrend(dataframe, self.supertrend_period.value, self.supertrend_multiplier.value)
        dataframe['supertrend'] = st
        dataframe['supertrend_direction'] = direction

        # RSI
        dataframe['rsi'] = rsi(dataframe['close'], 14)

        # EMA 20
        dataframe['ema20'] = ema(dataframe['close'], 20)

        # ATR
        dataframe['atr'] = atr(dataframe, 14)

        return dataframe

    @log_execution_time
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        # Condizione: Supertrend bullish, RSI sotto 70, prezzo sopra EMA20, cooldown ok
        current_time = time.time()
        pair = metadata['pair']

        conditions = (
            (dataframe['supertrend_direction'] == 1) &
            (dataframe['rsi'] < 70) &
            (dataframe['close'] > dataframe['ema20'])
        )

        if self.can_enter(pair, current_time):
            dataframe.loc[conditions, 'enter_long'] = 1
        else:
            dataframe['enter_long'] = 0

        return dataframe

    @log_execution_time
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        # Condizione di uscita: prezzo supera target profit o scende sotto stop d'emergenza
        target_profit = 1 + (self.exit_profit_pct.value if hasattr(self.exit_profit_pct, 'value') else self.exit_profit_pct)
        emergency_stop = 1 + (self.emergency_stop_pct.value if hasattr(self.emergency_stop_pct, 'value') else self.emergency_stop_pct)

        entry_price = dataframe['close'].iloc[-1]  # ipotetico prezzo entrata

        dataframe['exit_long'] = 0
        # Potresti integrare qui logica di trailing stop e stoploss avanzato
        # Qui semplice: esci se close >= entry_price * target_profit o <= entry_price * emergency_stop
        dataframe.loc[dataframe['close'] >= entry_price * target_profit, 'exit_long'] = 1
        dataframe.loc[dataframe['close'] <= entry_price * emergency_stop, 'exit_long'] = 1

        return dataframe

    def custom_exit(self, trade: Trade, current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Personalizza l'uscita con trailing stop e gestione stoploss dinamica.
        Restituisce il motivo dell'uscita o None.
        """

        # Usa profit percentuale da trade
        profit_pct = current_profit

        # Se trailing stop abilitato e profitto oltre soglia, attiva trailing
        if self.trailing_stop:
            if profit_pct > self.trailing_stop_positive_offset:
                # Qui si potrebbe calcolare e aggiornare trailing stop (dipende da freqtrade)
                if profit_pct < self.trailing_stop_positive:
                    return "trailing_stop"

        # Uscita normale se profitto raggiunge target o stoploss
        if profit_pct >= (self.exit_profit_pct.value if hasattr(self.exit_profit_pct, 'value') else self.exit_profit_pct):
            return "take_profit"
        if profit_pct <= self.stoploss:
            return "stop_loss"

        # Nessuna uscita
        return None


    def informative_pairs(self):
        # Se vuoi aggiungere timeframe informativi o altri pair
        return []

