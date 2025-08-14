#!/usr/bin/env python3
import ccxt
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')


class EnhancedMomentumBot:
    """
    Enhanced Trading Bot with Support/Resistance Analysis

    Upgrades in this version:
    - Volatility-adaptive S/R clustering (ATR-based)
    - Confluence scoring across methods with multiplicative boosts
    - Time-decayed touch counting (recent touches weigh more)
    - Higher-timeframe trend filter (50/200 EMA on 4h by default)
    - Bullish RSI divergence detection near key support
    - Candlestick confirmation (Hammer, Doji, Bullish Engulfing)
    - ATR-based stop loss + R:R validation
    - Rule-based final signal gating (no point totals)
    """

    def __init__(self, trading_pair: str, exchange_name: str = 'binance'):
        self.trading_pair = trading_pair
        self.exchange_name = exchange_name

        # Configuration parameters
        self.config = {
            'trend_tolerance': 0.02,
            'strong_buy_threshold': 75,  # legacy (not used in new rule engine)
            'buy_threshold': 60,         # legacy (not used in new rule engine)
            'pullback_tolerance': 0.005,
            'rsi_threshold_high': 75,
            'rsi_threshold_low': 30,
            'crossover_lookback': 5,
            'volume_confirmation': True,

            # Support/Resistance parameters
            'sr_lookback_periods': 50,        # Periods to look back for S/R (base)
            'sr_min_touches': 2,              # Minimum touches to confirm S/R
            'sr_proximity_threshold': 0.01,   # Base % fallback if ATR not available
            'fibonacci_enabled': True,        # Enable Fibonacci retracements
            'pivot_order': 5,                 # Order for scipy peak detection
            'sr_strength_threshold': 3,       # Minimum strength for valid S/R
            'entry_buffer': 0.002,            # 0.2% buffer above support for entry
            'stop_loss_ratio': 0.015,         # legacy (replaced by ATR below)

            # === New config for enhancements ===
            'htf_timeframe': '4h',            # higher timeframe for trend filter
            'ema_fast': 50,
            'ema_slow': 200,

            'atr_period': 14,
            'atr_stop_multiplier': 2.0,       # Stop = Support - ATR * multiplier
            'min_reward_risk': 1.5,           # Minimum R:R to validate setup

            'time_decay_lambda': 0.15,        # Exponential decay for recent touches
            'vol_cluster_atr_mult': 1.0,      # How wide the clustering window is in ATRs
            'confluence_multipliers': {       # Multipliers by # of agreeing methods
                1: 1.0,
                2: 1.25,
                3: 1.55,
                4: 1.85,
                5: 2.2
            },
            'pattern_window': 5,              # check last N closed candles for patterns
            'divergence_lookback': 80,        # how far back to look for divergences
            'support_touch_atr_mult': 1.0,    # "near support" defined by ATR multiplier
        }

        self._cached_last_5m_atr = None  # used by entry calc when ATR from 5m is needed

        try:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                'apiKey': '',
                'secret': '',
                'timeout': 30000,
                'enableRateLimit': True,
            })
        except Exception as e:
            print(f"Error initializing exchange {exchange_name}: {e}")
            sys.exit(1)

        print(f"Initialized {self.__class__.__name__} for {trading_pair} on {exchange_name}")

    # ---------------------------
    # Data & indicators
    # ---------------------------
    def fetch_ohlcv_data(self, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from the exchange."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.trading_pair,
                timeframe=timeframe,
                limit=limit
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching {timeframe} data: {e}")
            return pd.DataFrame()

    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Average True Range (ATR)."""
        if df.empty:
            return pd.Series(dtype=float)
        if period is None:
            period = self.config['atr_period']
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr

    def calculate_ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    # ---------------------------
    # Swing points & S/R sources
    # ---------------------------
    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Identify swing highs and lows using scipy's peak detection.
        Returns tuples (index, price).
        """
        order = self.config['pivot_order']
        high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
        swing_highs = [(df.index[i], df['high'].iloc[i]) for i in high_indices]
        low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
        swing_lows = [(df.index[i], df['low'].iloc[i]) for i in low_indices]
        return swing_highs, swing_lows

    def calculate_support_resistance_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate S/R levels using multiple methods and score them with confluence & time-weighted touches.
        """
        if df.empty:
            return {'supports': [], 'resistances': [], 'current_price': np.nan}

        atr_series = self.calculate_atr(df)
        recent_atr = float(atr_series.iloc[-1]) if len(atr_series) else None

        swing_highs, swing_lows = self.identify_swing_points(df)

        # Method 1: Swing Point Clustering (ATR-adaptive)
        support_levels = self._cluster_price_levels([p for _, p in swing_lows], recent_atr, df['close'].iloc[-1])
        resistance_levels = self._cluster_price_levels([p for _, p in swing_highs], recent_atr, df['close'].iloc[-1])

        # Method 2: Psychological Levels (round numbers)
        psychological_levels = self._find_psychological_levels(df['close'].iloc[-1])

        # Method 3: Simplified Volume Profile
        volume_levels = self._calculate_volume_profile(df)

        # Method 4: Fibonacci Retracements
        fibonacci_levels = {}
        if self.config['fibonacci_enabled'] and len(swing_highs) > 0 and len(swing_lows) > 0:
            fibonacci_levels = self._calculate_fibonacci_levels(swing_highs, swing_lows)

        all_levels = {
            'support_swing': support_levels,
            'resistance_swing': resistance_levels,
            'psychological': psychological_levels,
            'volume_profile': volume_levels,
            'fibonacci': fibonacci_levels
        }

        scored_levels = self._score_sr_levels(df, all_levels, recent_atr)
        return scored_levels

    def _cluster_price_levels(self, prices: List[float], recent_atr: Optional[float], ref_price: float) -> List[Dict]:
        """
        Cluster similar price levels together using an ATR-adaptive proximity threshold.
        """
        if not prices:
            return []
        prices = sorted(prices)
        levels = []

        # ATR-adaptive threshold in price units
        if recent_atr and ref_price > 0:
            adaptive = (self.config['vol_cluster_atr_mult'] * recent_atr) / ref_price
        else:
            adaptive = self.config['sr_proximity_threshold']

        current_cluster = [prices[0]]
        for price in prices[1:]:
            cluster_avg = sum(current_cluster) / len(current_cluster)
            if abs(price - cluster_avg) / cluster_avg <= adaptive:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= self.config['sr_min_touches']:
                    levels.append({
                        'level': sum(current_cluster) / len(current_cluster),
                        'strength': len(current_cluster),
                        'touches': current_cluster.copy(),
                        'type': 'swing_cluster'
                    })
                current_cluster = [price]
        if len(current_cluster) >= self.config['sr_min_touches']:
            levels.append({
                'level': sum(current_cluster) / len(current_cluster),
                'strength': len(current_cluster),
                'touches': current_cluster.copy(),
                'type': 'swing_cluster'
            })
        return levels

    def _find_psychological_levels(self, current_price: float) -> List[Dict]:
        """Find psychological S/R levels (round numbers)."""
        levels = []
        if current_price >= 100:
            increments = [10, 50, 100]
        elif current_price >= 10:
            increments = [1, 5, 10]
        elif current_price >= 1:
            increments = [0.1, 0.5, 1]
        else:
            increments = [0.001, 0.01, 0.1]
        for increment in increments:
            lower = (int(current_price / increment)) * increment
            upper = lower + increment
            price_range = current_price * 0.2
            for lvl in [lower, upper]:
                if abs(lvl - current_price) <= price_range and lvl > 0:
                    levels.append({
                        'level': float(lvl),
                        'strength': 2,
                        'type': 'psychological',
                        'increment': increment
                    })
        return levels

    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> List[Dict]:
        """Simplified volume profile."""
        if df.empty:
            return []
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_profile = []
        for i in range(bins):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_mid = (bin_low + bin_high) / 2
            mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
            bin_volume = df.loc[mask, 'volume'].sum()
            if bin_volume > 0:
                volume_profile.append({
                    'level': float(bin_mid),
                    'strength': float(bin_volume),
                    'type': 'volume_profile',
                    'volume': float(bin_volume)
                })
        volume_profile.sort(key=lambda x: x['volume'], reverse=True)
        return volume_profile[:10]

    def _calculate_fibonacci_levels(self, swing_highs: List[Tuple], swing_lows: List[Tuple]) -> Dict:
        if not swing_highs or not swing_lows:
            return {}
        recent_high = max(swing_highs, key=lambda x: x[1])
        recent_low = min(swing_lows, key=lambda x: x[1])
        high_price = float(recent_high[1])
        low_price = float(recent_low[1])
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        levels = {}
        if high_price > low_price:
            for ratio in fib_ratios:
                fib_level = high_price - (high_price - low_price) * ratio
                levels[f'fib_{ratio}'] = {
                    'level': float(fib_level),
                    'strength': 3,
                    'type': 'fibonacci_retracement',
                    'ratio': ratio,
                    'from_high': high_price,
                    'from_low': low_price
                }
        return levels

    # ---------------------------
    # Scoring with Confluence + Time weighting
    # ---------------------------
    def _score_sr_levels(self, df: pd.DataFrame, all_levels: Dict, recent_atr: Optional[float]) -> Dict:
        """
        Score S/R by:
        - Clustering levels from all methods with ATR-adaptive proximity
        - Combining base strengths
        - Multiplying by confluence factor based on distinct methods agreeing
        - Adding time-decayed recent touches
        """
        current_price = float(df['close'].iloc[-1])
        if recent_atr and current_price > 0:
            prox_pct = (self.config['vol_cluster_atr_mult'] * recent_atr) / current_price
        else:
            prox_pct = self.config['sr_proximity_threshold']

        # Flatten all candidate levels
        candidates = []
        for category, levels in all_levels.items():
            if isinstance(levels, list):
                for lv in levels:
                    candidates.append({**lv, 'source': lv.get('type', category)})
            elif isinstance(levels, dict):
                for lv in levels.values():
                    candidates.append({**lv, 'source': lv.get('type', category)})

        # Cluster across sources (confluence clustering)
        candidates_sorted = sorted(candidates, key=lambda x: x['level'])
        clusters: List[List[Dict]] = []
        buffer: List[Dict] = []
        for lv in candidates_sorted:
            if not buffer:
                buffer.append(lv)
            else:
                cluster_avg = np.mean([b['level'] for b in buffer])
                if abs(lv['level'] - cluster_avg) / cluster_avg <= prox_pct:
                    buffer.append(lv)
                else:
                    clusters.append(buffer)
                    buffer = [lv]
        if buffer:
            clusters.append(buffer)

        # Score clusters
        scored_supports, scored_resistances = [], []
        for cluster in clusters:
            methods = set([c['source'] for c in cluster])
            base_strength = sum([float(c.get('strength', 1.0)) for c in cluster])

            # Time-decayed recent touches around the cluster mean
            level_mean = float(np.mean([c['level'] for c in cluster]))
            recent_touches = self._count_recent_touches(df, level_mean, prox_pct)
            # Confluence multiplier by number of distinct methods
            mcount = len(methods)
            mult = self.config['confluence_multipliers'].get(mcount, 1.0 + 0.3 * (mcount - 1))
            # Heavier boost if >2 methods
            confluence_boosted = base_strength * mult + recent_touches * 2.0

            # Save metadata
            level_info = {
                'level': level_mean,
                'methods': list(methods),
                'base_strength': base_strength,
                'recent_touches': recent_touches,
                'confluence_multiplier': mult,
                'total_strength': confluence_boosted,
                'distance_pct': abs(level_mean - current_price) / current_price,
                'type': 'confluence_zone',
            }

            if level_mean < current_price:
                scored_supports.append(level_info)
            else:
                scored_resistances.append(level_info)

        # Filter strong zones
        min_strength = self.config['sr_strength_threshold']
        valid_supports = [s for s in scored_supports if s['total_strength'] >= min_strength]
        valid_resistances = [r for r in scored_resistances if r['total_strength'] >= min_strength]

        # Sort by proximity first, then strength
        valid_supports.sort(key=lambda x: (x['distance_pct'], -x['total_strength']))
        valid_resistances.sort(key=lambda x: (x['distance_pct'], -x['total_strength']))

        return {
            'supports': valid_supports[:5],
            'resistances': valid_resistances[:5],
            'current_price': current_price
        }

    def _count_recent_touches(self, df: pd.DataFrame, level: float, prox_pct: Optional[float] = None, lookback: int = 120) -> float:
        """
        Count touches with exponential time decay.
        A touch occurs when candle range intersects [level*(1-prox), level*(1+prox)].
        """
        if df.empty:
            return 0.0
        if lookback > len(df):
            lookback = len(df)
        recent = df.tail(lookback)
        if prox_pct is None:
            prox_pct = self.config['sr_proximity_threshold']
        upper = level * (1 + prox_pct)
        lower = level * (1 - prox_pct)

        # Exponential decay by "age" (0 = most recent)
        lam = self.config['time_decay_lambda']
        weights_sum = 0.0
        for idx, (ts, candle) in enumerate(recent.iterrows()):
            age = lookback - 1 - idx  # older has larger age
            touched = (candle['low'] <= upper) and (candle['high'] >= lower)
            if touched:
                weight = np.exp(-lam * age)
                weights_sum += weight
        return float(weights_sum)

    # ---------------------------
    # Entry strategy & confirmations
    # ---------------------------
    def calculate_entry_strategy(self, sr_levels: Dict, atr_5m: Optional[float] = None) -> Dict:
        """
        Calculate entry/SL/targets using:
        - Nearest high-confluence support
        - ATR-based stop loss
        - Risk-defined targets and R:R validation
        """
        current_price = sr_levels['current_price']
        supports = sr_levels['supports']
        resistances = sr_levels['resistances']

        if not supports:
            return {
                'entry_valid': False,
                'reason': 'No valid support levels found'
            }

        # Choose the BEST nearby support by (strongest first within 5%)
        nearby_supports = [s for s in supports if s['distance_pct'] <= 0.05]
        if not nearby_supports:
            return {'entry_valid': False, 'reason': 'No nearby support levels found'}

        # Prefer high-confluence zones (more methods, higher strength, closest)
        nearby_supports.sort(key=lambda s: (-len(s.get('methods', [])), -s['total_strength'], s['distance_pct']))
        chosen = nearby_supports[0]
        support_level = chosen['level']

        entry_buffer = self.config['entry_buffer']
        entry_price = support_level * (1 + entry_buffer)

        # ATR-based stop loss from 5m data
        if atr_5m is None or atr_5m <= 0:
            # Fallback: percent-based if no ATR available
            stop_loss = support_level * (1 - self.config['stop_loss_ratio'])
            sl_method = 'percent_fallback'
        else:
            stop_loss = support_level - self.config['atr_stop_multiplier'] * atr_5m
            sl_method = 'atr_based'

        risk_amount = entry_price - stop_loss
        risk_pct = (risk_amount / entry_price) * 100 if entry_price > 0 else np.nan

        # Targets from resistances; compute R:R
        targets = []
        for i, r in enumerate(resistances[:3]):
            target_price = r['level']
            reward = target_price - entry_price
            if reward > 0 and risk_amount > 0:
                rr = reward / risk_amount
                targets.append({
                    'target': i + 1,
                    'price': target_price,
                    'reward_risk_ratio': rr,
                    'potential_profit_pct': ((target_price - entry_price) / entry_price) * 100
                })

        # Validate R:R
        rr_ok = False
        best_rr = 0.0
        if targets:
            best_rr = max(t['reward_risk_ratio'] for t in targets)
            rr_ok = best_rr >= self.config['min_reward_risk']

        return {
            'entry_valid': True,
            'support_level': support_level,
            'support_strength': chosen['total_strength'],
            'support_type': chosen['type'],
            'support_methods': chosen.get('methods', []),
            'high_confluence': len(chosen.get('methods', [])) >= 2 and chosen['total_strength'] >= self.config['sr_strength_threshold'] * 1.5,
            'confluence_score': chosen['total_strength'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'stop_loss_method': sl_method,
            'risk_pct': risk_pct,
            'targets': targets,
            'distance_to_support_pct': chosen['distance_pct'] * 100,
            'rr_ok': rr_ok,
            'best_rr': best_rr,
            'recommendation': self._generate_entry_recommendation(current_price, entry_price, targets)
        }

    def _generate_entry_recommendation(self, current_price: float, entry_price: float, targets: List[Dict]) -> str:
        price_diff_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0
        if not targets:
            return "HOLD - No clear targets identified"
        best_target = max(targets, key=lambda x: x['reward_risk_ratio'])
        if price_diff_pct > 2:
            return f"WAIT - Price {price_diff_pct:.1f}% above entry. Wait for pullback to support."
        elif price_diff_pct > -0.5:
            return f"BUY - Price near entry level. R:R {best_target['reward_risk_ratio']:.2f}"
        else:
            return f"STRONG_BUY - Price below entry. R:R {best_target['reward_risk_ratio']:.2f}"

    # ---------------------------
    # Momentum + confirmations
    # ---------------------------
    def analyze_momentum_signals(self, df: pd.DataFrame) -> Dict:
        """Simple EMA/RSI momentum snapshot (kept light)."""
        if len(df) < 21:
            return {"momentum_score": 0, "momentum_signal": "INSUFFICIENT_DATA"}
        df = df.copy()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        last = df.iloc[-1]
        score = 0
        if last['ema_9'] > last['ema_21']:
            score += 30
        if 40 <= last['rsi'] <= 70:
            score += 25
        elif 30 <= last['rsi'] <= 80:
            score += 15
        if last['close'] > last['ema_9']:
            score += 20
        if df['close'].pct_change(3).iloc[-1] > 0:
            score += 15
        momentum_signal = "BULLISH" if score >= 50 else "BEARISH" if score <= 30 else "NEUTRAL"
        return {"momentum_score": score, "momentum_signal": momentum_signal,
                "ema_9": last['ema_9'], "ema_21": last['ema_21'], "rsi": last['rsi']}

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ---------------------------
    # Higher timeframe trend filter
    # ---------------------------
    def _higher_timeframe_trend_ok(self, df_htf: pd.DataFrame) -> Tuple[bool, Dict]:
        """Uptrend if price > EMA50 and EMA50 > EMA200."""
        if df_htf.empty or len(df_htf) < max(self.config['ema_fast'], self.config['ema_slow']) + 5:
            return False, {"reason": "Insufficient HTF data"}
        close = df_htf['close']
        ema50 = self.calculate_ema(close, self.config['ema_fast'])
        ema200 = self.calculate_ema(close, self.config['ema_slow'])
        last_close = float(close.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])
        last_ema200 = float(ema200.iloc[-1])
        ok = (last_close > last_ema50) and (last_ema50 > last_ema200)
        return ok, {"last_close": last_close, "ema50": last_ema50, "ema200": last_ema200}

    # ---------------------------
    # Bullish divergence near support
    # ---------------------------
    def _detect_bullish_divergence(self, df: pd.DataFrame, support_level: float, recent_atr_5m: Optional[float]) -> bool:
        """
        Price makes lower low while RSI makes higher low, near support.
        Check last N bars for two swing lows in price and compare RSI lows.
        """
        if df.empty or len(df) < 30:
            return False
        lookback = min(self.config['divergence_lookback'], len(df))
        sub = df.tail(lookback).copy()
        sub['rsi'] = self.calculate_rsi(sub['close'])
        # find swing lows (use same pivot order but smaller for 5m)
        order = max(2, self.config['pivot_order'] // 2)
        low_idx = argrelextrema(sub['low'].values, np.less, order=order)[0]
        if len(low_idx) < 2:
            return False
        # Take last two swing lows
        i1, i2 = low_idx[-2], low_idx[-1]
        p1 = float(sub['low'].iloc[i1]); p2 = float(sub['low'].iloc[i2])
        r1 = float(sub['rsi'].iloc[i1]); r2 = float(sub['rsi'].iloc[i2])
        price_ll = p2 < p1
        rsi_hl = r2 > r1

        # "Near support" defined via ATR window around support
        if recent_atr_5m and support_level > 0:
            near = abs(p2 - support_level) <= self.config['support_touch_atr_mult'] * recent_atr_5m
        else:
            # fallback: 1% window
            near = abs(p2 - support_level) / support_level <= 0.01

        return bool(price_ll and rsi_hl and near)

    # ---------------------------
    # Candlestick confirmation
    # ---------------------------
    def _bullish_candle_confirmed(self, df: pd.DataFrame, support_level: float, recent_atr_5m: Optional[float]) -> bool:
        """
        Confirm if a bullish reversal pattern appears near support in the last closed candles.
        Patterns: Hammer, Doji, Bullish Engulfing.
        """
        if df.empty or len(df) < 5:
            return False
        window = min(self.config['pattern_window'], len(df)-1)
        slice_df = df.tail(window + 1).iloc[:-1]  # exclude the most recent forming candle
        for ts, row in slice_df.iterrows():
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            body = abs(c - o)
            range_ = h - l if h > l else 0.0
            if range_ == 0:
                continue

            # near support?
            if recent_atr_5m and support_level > 0:
                near = abs(c - support_level) <= self.config['support_touch_atr_mult'] * recent_atr_5m
            else:
                near = abs(c - support_level) / support_level <= 0.01

            if not near:
                continue

            upper_wick = h - max(c, o)
            lower_wick = min(c, o) - l

            # Hammer: small body near top, long lower shadow
            is_hammer = (lower_wick > 2 * body) and (upper_wick <= body)

            # Doji: tiny body relative to range
            is_doji = body <= 0.1 * range_

            # Bullish Engulfing requires previous candle
            # We need to access the next row (chronologically) → use rolling pairs
        # Re-check with rolling pairs for engulfing
        seq = slice_df.copy()
        prev = None
        engulfing_found = False
        for idx, row in seq.iterrows():
            if prev is not None:
                o1, c1 = float(prev['open']), float(prev['close'])
                o2, c2 = float(row['open']), float(row['close'])
                # bullish engulfing: prev red, current green, and current body engulfs prev body
                prev_red = c1 < o1
                curr_green = c2 > o2
                engulfs = (o2 < c1) and (c2 > o1)
                # near support?
                if recent_atr_5m and support_level > 0:
                    near = abs(c2 - support_level) <= self.config['support_touch_atr_mult'] * recent_atr_5m
                else:
                    near = abs(c2 - support_level) / support_level <= 0.01
                if prev_red and curr_green and engulfs and near:
                    engulfing_found = True
                    break
            prev = row

        # If not engulfing, evaluate individual candles for hammer/doji
        if engulfing_found:
            return True

        for ts, row in slice_df.iterrows():
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            body = abs(c - o)
            range_ = h - l if h > l else 0.0
            if range_ == 0:
                continue
            if recent_atr_5m and support_level > 0:
                near = abs(c - support_level) <= self.config['support_touch_atr_mult'] * recent_atr_5m
            else:
                near = abs(c - support_level) / support_level <= 0.01
            if not near:
                continue
            upper_wick = h - max(c, o)
            lower_wick = min(c, o) - l
            is_hammer = (lower_wick > 2 * body) and (upper_wick <= body)
            is_doji = body <= 0.1 * (h - l) if (h - l) > 0 else False
            if is_hammer or is_doji:
                return True

        return False

    # ---------------------------
    # Main analysis pipeline
    # ---------------------------
    def analyze_market_with_sr(self) -> Dict:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*70}")
        print(f"Enhanced Market Analysis with Support/Resistance - {timestamp}")
        print(f"Trading Pair: {self.trading_pair}")
        print(f"{'='*70}")

        # Fetch data
        df_1h = self.fetch_ohlcv_data('1h', limit=400)
        df_5m = self.fetch_ohlcv_data('5m', limit=500)
        df_htf = self.fetch_ohlcv_data(self.config['htf_timeframe'], limit=400)

        if df_1h.empty or df_5m.empty or df_htf.empty:
            return {"error": "Insufficient data for analysis"}

        # Cache 5m ATR for downstream use (stop, support proximity)
        atr_5m_series = self.calculate_atr(df_5m, self.config['atr_period'])
        self._cached_last_5m_atr = float(atr_5m_series.iloc[-1]) if len(atr_5m_series) else None

        print("Phase 1: Higher-Timeframe Trend Filter...")
        htf_ok, htf_details = self._higher_timeframe_trend_ok(df_htf)
        print(f"  HTF Uptrend: {htf_ok} | Close={htf_details.get('last_close', 'NA'):.6f} "
              f"| EMA50={htf_details.get('ema50', 'NA'):.6f} | EMA200={htf_details.get('ema200', 'NA'):.6f}")

        print("\nPhase 2: Calculating Support/Resistance Levels...")
        sr_levels = self.calculate_support_resistance_levels(df_1h)

        print(f"\n📊 Support Levels Found:")
        for i, support in enumerate(sr_levels['supports']):
            print(f"  S{i+1}: ${support['level']:.6f} (Strength: {support['total_strength']:.2f}, "
                  f"Methods: {len(support.get('methods', []))}, "
                  f"Distance: {support['distance_pct']*100:.2f}%)")

        print(f"\n📊 Resistance Levels Found:")
        for i, resistance in enumerate(sr_levels['resistances']):
            print(f"  R{i+1}: ${resistance['level']:.6f} (Strength: {resistance['total_strength']:.2f}, "
                  f"Methods: {len(resistance.get('methods', []))}, "
                  f"Distance: {resistance['distance_pct']*100:.2f}%)")

        print("\nPhase 3: Calculating Entry Strategy...")
        entry_strategy = self.calculate_entry_strategy(sr_levels, atr_5m=self._cached_last_5m_atr)
        if not entry_strategy['entry_valid']:
            return {
                "signal": "HOLD",
                "reason": entry_strategy['reason'],
                "sr_levels": sr_levels,
                "htf_uptrend": htf_ok,
                "timestamp": timestamp
            }

        # Display entry strategy
        print(f"\n🎯 Entry Strategy:")
        print(f"  Support Level: ${entry_strategy['support_level']:.6f} (Confluence: {entry_strategy['high_confluence']}, methods={len(entry_strategy.get('support_methods', []))})")
        print(f"  Entry Price:  ${entry_strategy['entry_price']:.6f}")
        print(f"  Stop Loss:    ${entry_strategy['stop_loss']:.6f} ({entry_strategy['stop_loss_method']})")
        print(f"  Risk:         {entry_strategy['risk_pct']:.2f}%")
        print(f"  Distance to Support: {entry_strategy['distance_to_support_pct']:.2f}%")

        if entry_strategy['targets']:
            print(f"\n🎯 Targets:")
            for target in entry_strategy['targets']:
                print(f"  Target {target['target']}: ${target['price']:.6f} "
                      f"(R:R {target['reward_risk_ratio']:.2f}, Profit: {target['potential_profit_pct']:.1f}%)")
        else:
            print("  No viable targets found.")

        # Confirmations
        print("\nPhase 4: Confirmation Signals (5m)...")
        divergence = self._detect_bullish_divergence(df_5m, entry_strategy['support_level'], self._cached_last_5m_atr)
        pattern_ok = self._bullish_candle_confirmed(df_5m, entry_strategy['support_level'], self._cached_last_5m_atr)
        print(f"  Bullish RSI Divergence near support: {divergence}")
        print(f"  Bullish Candlestick Pattern near support: {pattern_ok}")

        # Momentum snapshot (kept for additional color; not required by rules)
        print("\nPhase 5: Momentum Confirmation...")
        momentum_result = self.analyze_momentum_signals(df_5m)

        # Rule-based final decision
        combined_signal = self._combine_sr_momentum_signals(
            entry_strategy=entry_strategy,
            momentum_result=momentum_result,
            htf_uptrend=htf_ok,
            confirmations={'divergence': divergence, 'bullish_pattern': pattern_ok}
        )

        result = {
            "signal": combined_signal['signal'],
            "confidence": combined_signal['confidence'],
            "entry_strategy": entry_strategy,
            "momentum_analysis": momentum_result,
            "sr_levels": sr_levels,
            "htf_uptrend": htf_ok,
            "confirmations": {'divergence': divergence, 'bullish_pattern': pattern_ok},
            "recommendation": entry_strategy['recommendation'],
            "timestamp": timestamp
        }
        return result

    # ---------------------------
    # Final rule-based engine
    # ---------------------------
    def _combine_sr_momentum_signals(self, entry_strategy: Dict, momentum_result: Dict, htf_uptrend: bool, confirmations: Dict) -> Dict:
        """
        STRICT rule engine:
        STRONG_BUY requires:
          - Higher timeframe uptrend
          - High-confluence S/R zone
          - (Bullish divergence OR bullish candlestick pattern)
          - R:R >= min_reward_risk
        BUY requires:
          - Higher timeframe uptrend
          - Confluence (not necessarily "high")
          - One confirmation (divergence or pattern)
          - R:R >= min_reward_risk
        Else WAIT/HOLD
        """
        divergence = confirmations.get('divergence', False)
        pattern_ok = confirmations.get('bullish_pattern', False)
        rr_ok = entry_strategy.get('rr_ok', False)
        high_conf = entry_strategy.get('high_confluence', False)
        has_confluence = len(entry_strategy.get('support_methods', [])) >= 2

        # Confidence scaffold (0-100), still reported but not used for gating
        confidence = 20.0
        if htf_uptrend:
            confidence += 25.0
        if high_conf:
            confidence += 25.0
        elif has_confluence:
            confidence += 15.0
        if rr_ok:
            confidence += 15.0
        if momentum_result.get('momentum_signal') == 'BULLISH':
            confidence += 10.0
        if divergence or pattern_ok:
            confidence += 10.0
        confidence = min(100.0, max(0.0, confidence))

        # Gate logic
        if htf_uptrend and high_conf and rr_ok and (divergence or pattern_ok):
            signal = "STRONG_BUY"
        elif htf_uptrend and has_confluence and rr_ok and (divergence or pattern_ok):
            signal = "BUY"
        else:
            # differentiate WAIT vs HOLD:
            # WAIT if uptrend but missing one element; HOLD otherwise
            if htf_uptrend and (has_confluence or high_conf):
                signal = "WAIT"
            else:
                signal = "HOLD"

        return {
            "signal": signal,
            "confidence": confidence,
            "sr_contribution": None,
            "momentum_contribution": None
        }

    # ---------------------------
    # Runner
    # ---------------------------
    def run_enhanced_analysis(self):
        result = self.analyze_market_with_sr()
        if "error" in result:
            print(f"Error: {result['error']}")
            return result

        print(f"\n{'='*50}")
        print(f"🚨 FINAL ENHANCED RESULT 🚨")
        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Recommendation: {result['recommendation']}")
        if result['entry_strategy']['entry_valid']:
            print(f"Entry Price: ${result['entry_strategy']['entry_price']:.6f}")
            print(f"Stop Loss: ${result['entry_strategy']['stop_loss']:.6f}")
            if result['entry_strategy']['targets']:
                best_target = result['entry_strategy']['targets'][0]
                print(f"Primary Target: ${best_target['price']:.6f} (R:R {best_target['reward_risk_ratio']:.2f})")
        print(f"{'='*50}")
        return result


def main():
    """Main function to run the enhanced bot."""
    print("Enhanced Trading Bot with Support/Resistance Analysis")
    print("=" * 60)

    # Get trading pair from user
    if len(sys.argv) > 1:
        trading_pair = sys.argv[1]
    else:
        trading_pair = input("Enter trading pair (e.g., BTC/USDT): ").strip().upper()

    if not trading_pair or '/' not in trading_pair:
        print("Error: Valid trading pair required (e.g., BTC/USDT)")
        sys.exit(1)

    try:
        bot = EnhancedMomentumBot(trading_pair)

        print(f"\nEnhanced Configuration:")
        sr_config = {k: v for k, v in bot.config.items() if 'sr_' in k or 'fibonacci' in k or 'pivot' in k}
        # Also show key new risk/trend params
        extra = {k: v for k, v in bot.config.items() if k in ['atr_period', 'atr_stop_multiplier', 'min_reward_risk', 'htf_timeframe']}
        sr_config.update(extra)
        for key, value in sr_config.items():
            print(f"  {key}: {value}")

        bot.run_enhanced_analysis()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
