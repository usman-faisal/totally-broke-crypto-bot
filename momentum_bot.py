#!/usr/bin/env python3
# Requires: pip install mplfinance
import ccxt
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.signal import argrelextrema
import matplotlib
# Use Agg backend which is reliable for saving plots
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import mplfinance as mpf

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
            # --- Timeframes ---
            'htf_timeframe': '15m',           # Higher timeframe for trend filter

            # --- S/R Sensitivity & Precision ---
            'sr_lookback_periods': 150,       # Longer lookback on low TFs to get enough data
            'pivot_order': 3,                 # More sensitive to smaller price swings
            'vol_cluster_atr_mult': 0.4,      # Tighter clustering for more precise S/R zones
            'time_decay_lambda': 0.45,        # Heavily weigh recent touches over older ones
            'sr_strength_threshold': 3,       # Lower threshold to detect more levels
            'sr_proximity_threshold': 0.01,   # Fallback proximity when ATR not available (1%)
            'fibonacci_enabled': False,       # Fibonacci is less reliable on very low timeframes

            # --- Risk Management & Entry ---
            'atr_period': 14,
            'atr_stop_multiplier': 1.2,       # TIGHTER stop loss (critical for scalping)
            'min_reward_risk': 1.1,           # Lower R:R is acceptable for high-frequency scalps
            'entry_buffer': 0.0005,           # 0.05% buffer above support for entry (razor thin)
            'stop_loss_ratio': 0.01,          # Fallback percent stop when ATR unavailable (1%)
            
            # --- Confirmation Signal Timing ---
            'pattern_window': 3,              # Check last 3 closed candles for patterns
            'divergence_lookback': 40,        # Shorter lookback for recent divergences
            'support_touch_atr_mult': 0.8,    # "near support" defined by a tighter ATR multiplier

            # --- Legacy / Unused in new rules ---
            'ema_fast': 50,                   # These EMAs are now on the 15m chart
            'ema_slow': 200,
            'sr_min_touches': 2,
            'confluence_multipliers': {
                1: 1.0, 2: 1.25, 3: 1.55, 4: 1.85, 5: 2.2
            },
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
    def _detect_bullish_divergence(self, df: pd.DataFrame, support_level: float, recent_atr_5m: Optional[float]) -> Dict:
        """
        Price makes lower low while RSI makes higher low, near support.
        Check last N bars for two swing lows in price and compare RSI lows.
        """
        if df.empty or len(df) < 30:
            return {'found': False}
        lookback = min(self.config['divergence_lookback'], len(df))
        sub = df.tail(lookback).copy()
        sub['rsi'] = self.calculate_rsi(sub['close'])
        # find swing lows (use same pivot order but smaller for 5m)
        order = max(2, self.config['pivot_order'] // 2)
        low_idx = argrelextrema(sub['low'].values, np.less, order=order)[0]
        if len(low_idx) < 2:
            return {'found': False}
        # Take last two swing lows
        i1, i2 = low_idx[-2], low_idx[-1]
        p1 = float(sub['low'].iloc[i1]); p2 = float(sub['low'].iloc[i2])
        r1 = float(sub['rsi'].iloc[i1]); r2 = float(sub['rsi'].iloc[i2])
        ts1 = sub.index[i1]; ts2 = sub.index[i2]
        price_ll = p2 < p1
        rsi_hl = r2 > r1

        # "Near support" defined via ATR window around support
        if recent_atr_5m and support_level > 0:
            near = abs(p2 - support_level) <= self.config['support_touch_atr_mult'] * recent_atr_5m
        else:
            # fallback: 1% window
            near = abs(p2 - support_level) / support_level <= 0.01

        if price_ll and rsi_hl and near:
            return {
                'found': True,
                'price_points': ((ts1, p1), (ts2, p2)),
                'rsi_points': ((ts1, r1), (ts2, r2))
            }
        return {'found': False}

    # ---------------------------
    # Candlestick confirmation
    # ---------------------------
    def _bullish_candle_confirmed(self, df: pd.DataFrame, support_level: float, recent_atr_5m: Optional[float]) -> Dict:
        """
        Confirm if a bullish reversal pattern appears near support in the last closed candles.
        Patterns: Hammer, Doji, Bullish Engulfing.
        """
        if df.empty or len(df) < 5:
            return {'found': False}
        window = min(self.config['pattern_window'], len(df)-1)
        slice_df = df.tail(window + 1).iloc[:-1]  # exclude the most recent forming candle

        # Check Bullish Engulfing with rolling pairs
        prev = None
        for idx, row in slice_df.iterrows():
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
                    return {'found': True, 'timestamp': idx, 'pattern_type': 'Engulfing'}
            prev = row

        # If not engulfing, evaluate individual candles for hammer/doji
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
            # Hammer: small body near top, long lower shadow
            is_hammer = (lower_wick > 2 * body) and (upper_wick <= body)
            # Doji: tiny body relative to range
            is_doji = body <= 0.1 * range_
            if is_hammer:
                return {'found': True, 'timestamp': ts, 'pattern_type': 'Hammer'}
            if is_doji:
                return {'found': True, 'timestamp': ts, 'pattern_type': 'Doji'}

        return {'found': False}

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
        df_entry_exec = self.fetch_ohlcv_data('1m', limit=500)      # Execution chart
        df_sr_analysis = self.fetch_ohlcv_data('5m', limit=400)     # S/R chart
        df_htf = self.fetch_ohlcv_data(self.config['htf_timeframe'], limit=400) # HTF is now '15m'
        self._last_df_sr_analysis = df_sr_analysis

        if df_sr_analysis.empty or df_entry_exec.empty or df_htf.empty:
            return {"error": "Insufficient data for scalping analysis"}

        # Cache 5m ATR for downstream use (stop, support proximity)
        atr_5m_series = self.calculate_atr(df_sr_analysis, self.config['atr_period'])
        self._cached_last_5m_atr = float(atr_5m_series.iloc[-1]) if len(atr_5m_series) else None

        print("Phase 1: Higher-Timeframe Trend Filter...")
        htf_ok, htf_details = self._higher_timeframe_trend_ok(df_htf)
        print(f"  HTF Uptrend: {htf_ok} | Close={htf_details.get('last_close', 'NA'):.6f} "
              f"| EMA50={htf_details.get('ema50', 'NA'):.6f} | EMA200={htf_details.get('ema200', 'NA'):.6f}")

        print("\nPhase 2: Calculating Support/Resistance Levels...")
        sr_levels = self.calculate_support_resistance_levels(df_sr_analysis)

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
                "confidence": 0.0,
                "reason": entry_strategy['reason'],
                "sr_levels": sr_levels,
                "htf_uptrend": htf_ok,
                "entry_strategy": entry_strategy,
                "momentum_analysis": {},
                "confirmations": {'divergence': False, 'bullish_pattern': False, 'divergence_data': {}, 'pattern_data': {}},
                "recommendation": "No valid entry strategy",
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
        divergence_data = self._detect_bullish_divergence(df_sr_analysis, entry_strategy['support_level'], self._cached_last_5m_atr)
        pattern_data = self._bullish_candle_confirmed(df_sr_analysis, entry_strategy['support_level'], self._cached_last_5m_atr)
        divergence_found = bool(divergence_data.get('found', False))
        pattern_found = bool(pattern_data.get('found', False))
        print(f"  Bullish RSI Divergence near support: {divergence_found}")
        print(f"  Bullish Candlestick Pattern near support: {pattern_found}")

        # Momentum snapshot (kept for additional color; not required by rules)
        print("\nPhase 5: Momentum Confirmation...")
        momentum_result = self.analyze_momentum_signals(df_sr_analysis)

        # Rule-based final decision
        combined_signal = self._combine_sr_momentum_signals(
            entry_strategy=entry_strategy,
            momentum_result=momentum_result,
            htf_uptrend=htf_ok,
            confirmations={'divergence': divergence_found, 'bullish_pattern': pattern_found}
        )

        result = {
            "signal": combined_signal['signal'],
            "confidence": combined_signal['confidence'],
            "entry_strategy": entry_strategy,
            "momentum_analysis": momentum_result,
            "sr_levels": sr_levels,
            "htf_uptrend": htf_ok,
            "confirmations": {
                'divergence': divergence_found,
                'bullish_pattern': pattern_found,
                'divergence_data': divergence_data,
                'pattern_data': pattern_data
            },
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
    # Visualization
    # ---------------------------
    def plot_analysis(self, df: pd.DataFrame, analysis_result: Dict):
        if df is None or df.empty:
            print("WARNING: Dataframe for plotting is empty.")
            return None, None
        
        plot_df = df.tail(150).copy()
        
        # Ensure standard mplfinance column names
        if 'open' in plot_df.columns:
            plot_df = plot_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in plot_df.columns for col in required_cols):
            print(f"ERROR: Missing required columns for plotting. Found: {plot_df.columns.tolist()}")
            return None, None

        # --- Build Add-ons for the Plot ---

        # 1. Addplots (RSI, Pattern Markers)
        rsi_series = self.calculate_rsi(plot_df['Close'])
        apds = [mpf.make_addplot(rsi_series, panel=1, color='#8A2BE2', width=1.0, title='RSI')]

        pattern_data = analysis_result.get('confirmations', {}).get('pattern_data', {})
        if pattern_data.get('found'):
            ts = pattern_data.get('timestamp')
            if ts in plot_df.index:
                low_val = float(plot_df.loc[ts, 'Low'])
                marker_y = low_val * 0.995  # Place marker slightly below the low
                marker_series = pd.Series(np.nan, index=plot_df.index)
                marker_series.loc[ts] = marker_y
                apds.append(mpf.make_addplot(marker_series, type='scatter', marker='^', markersize=100, color='#2ca02c', panel=0))

        # 2. Horizontal Lines (S/R, Entry, Stop, Target)
        hlines_levels = []
        hlines_colors = []
        hlines_styles = []

        sr_levels = analysis_result.get('sr_levels', {})
        supports = sr_levels.get('supports', [])
        resistances = sr_levels.get('resistances', [])
        
        for s in supports:
            hlines_levels.append(float(s['level']))
            hlines_colors.append('#008000') # Green
            hlines_styles.append('dashed')

        for r in resistances:
            hlines_levels.append(float(r['level']))
            hlines_colors.append('#CC0000') # Red
            hlines_styles.append('dashed')

        entry_strategy = analysis_result.get('entry_strategy', {})
        if entry_strategy.get('entry_price'):
            hlines_levels.append(entry_strategy['entry_price'])
            hlines_colors.append('#1f77b4') # Blue
            hlines_styles.append('solid')
        if entry_strategy.get('stop_loss'):
            hlines_levels.append(entry_strategy['stop_loss'])
            hlines_colors.append('#d62728') # Darker Red
            hlines_styles.append('solid')
        if entry_strategy.get('targets') and entry_strategy['targets']:
            hlines_levels.append(entry_strategy['targets'][0]['price'])
            hlines_colors.append('#2ca02c') # Darker Green
            hlines_styles.append('solid')

        hlines_dict = dict(hlines=hlines_levels, colors=hlines_colors, linestyle=hlines_styles, linewidths=1.0) if hlines_levels else None

        # 3. Alines (Divergence)
        alines_list = []
        divergence_data = analysis_result.get('confirmations', {}).get('divergence_data', {})
        if divergence_data.get('found'):
            price_points = divergence_data.get('price_points')
            rsi_points = divergence_data.get('rsi_points')
            if price_points:
                alines_list.append(dict(alines=[price_points], colors=['#00ccff'], linewidths=1.5, panel=0))
            if rsi_points:
                alines_list.append(dict(alines=[rsi_points], colors=['#00cc66'], linewidths=1.5, panel=1))

        # --- Construct Final Plotting Keyword Arguments ---
        # Initialize kwargs WITHOUT 'alines'
        plot_kwargs = {
            'type': 'candle',
            'style': 'yahoo',
            'title': f"{self.trading_pair} - 5m Analysis\nSignal: {analysis_result.get('signal', 'N/A')}",
            'volume': True,
            'addplot': apds,
            'hlines': hlines_dict,
            'panel_ratios': (3, 1),
            'returnfig': True,
            'figratio': (16, 9),
            'figscale': 1.2,
            'tight_layout': True
        }

        # Conditionally ADD 'alines' to the dictionary if alines_list is not empty
        if alines_list:
            plot_kwargs['alines'] = alines_list

        try:
            fig, axes = mpf.plot(plot_df, **plot_kwargs)
            # Add text labels for entry/stop/target as they don't have built-in labels
            ax_main = axes[0]
            # Use iloc for integer-based positioning relative to the plotted data
            x_pos = len(plot_df) - 1 # Position text at the last candle
            if entry_strategy.get('entry_price'):
                ax_main.text(x_pos, entry_strategy['entry_price'], ' Entry', color='#1f77b4', va='center', fontsize=9)
            if entry_strategy.get('stop_loss'):
                ax_main.text(x_pos, entry_strategy['stop_loss'], ' Stop', color='#d62728', va='center', fontsize=9)
            if entry_strategy.get('targets') and entry_strategy['targets']:
                ax_main.text(x_pos, entry_strategy['targets'][0]['price'], ' Target', color='#2ca02c', va='center', fontsize=9)
            return fig, axes
        except Exception as e:
            print(f"FATAL PLOTTING ERROR: {e}")
            return None, None
        if df is None or df.empty:
            print("WARNING: Dataframe for plotting is empty.")
            return None, None
        
        plot_df = df.tail(150).copy()
        
        # Ensure standard mplfinance column names
        if 'open' in plot_df.columns:
            plot_df = plot_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in plot_df.columns for col in required_cols):
            print(f"ERROR: Missing required columns for plotting. Found: {plot_df.columns.tolist()}")
            return None, None

        # --- Build Add-ons for the Plot ---

        # 1. Addplots (RSI, Pattern Markers)
        rsi_series = self.calculate_rsi(plot_df['Close'])
        apds = [mpf.make_addplot(rsi_series, panel=1, color='#8A2BE2', width=1.0, title='RSI')]

        pattern_data = analysis_result.get('confirmations', {}).get('pattern_data', {})
        if pattern_data.get('found'):
            ts = pattern_data.get('timestamp')
            if ts in plot_df.index:
                low_val = float(plot_df.loc[ts, 'Low'])
                marker_y = low_val * 0.995  # Place marker slightly below the low
                marker_series = pd.Series(np.nan, index=plot_df.index)
                marker_series.loc[ts] = marker_y
                apds.append(mpf.make_addplot(marker_series, type='scatter', marker='^', markersize=100, color='#2ca02c', panel=0))

        # 2. Horizontal Lines (S/R, Entry, Stop, Target)
        hlines_levels = []
        hlines_colors = []
        hlines_styles = []

        sr_levels = analysis_result.get('sr_levels', {})
        supports = sr_levels.get('supports', [])
        resistances = sr_levels.get('resistances', [])
        
        for s in supports:
            hlines_levels.append(float(s['level']))
            hlines_colors.append('#008000') # Green
            hlines_styles.append('dashed')

        for r in resistances:
            hlines_levels.append(float(r['level']))
            hlines_colors.append('#CC0000') # Red
            hlines_styles.append('dashed')

        entry_strategy = analysis_result.get('entry_strategy', {})
        if entry_strategy.get('entry_price'):
            hlines_levels.append(entry_strategy['entry_price'])
            hlines_colors.append('#1f77b4') # Blue
            hlines_styles.append('solid')
        if entry_strategy.get('stop_loss'):
            hlines_levels.append(entry_strategy['stop_loss'])
            hlines_colors.append('#d62728') # Darker Red
            hlines_styles.append('solid')
        if entry_strategy.get('targets') and entry_strategy['targets'][0]:
            hlines_levels.append(entry_strategy['targets'][0]['price'])
            hlines_colors.append('#2ca02c') # Darker Green
            hlines_styles.append('solid')

        hlines_dict = dict(hlines=hlines_levels, colors=hlines_colors, linestyle=hlines_styles, linewidths=1.0) if hlines_levels else None

        # 3. Alines (Divergence)
        alines_list = []
        divergence_data = analysis_result.get('confirmations', {}).get('divergence_data', {})
        if divergence_data.get('found'):
            price_points = divergence_data.get('price_points')
            rsi_points = divergence_data.get('rsi_points')
            if price_points:
                alines_list.append(dict(alines=[price_points], colors=['#00ccff'], linewidths=1.5, panel=0))
            if rsi_points:
                alines_list.append(dict(alines=[rsi_points], colors=['#00cc66'], linewidths=1.5, panel=1))

        # --- Construct Final Plotting Keyword Arguments ---
        plot_kwargs = {
            'type': 'candle',
            'style': 'yahoo',
            'title': f"{self.trading_pair} - 5m Analysis\nSignal: {analysis_result.get('signal', 'N/A')}",
            'volume': True,
            'addplot': apds,
            'hlines': hlines_dict,
            'alines': alines_list if alines_list else None,
            'panel_ratios': (3, 1), # Main panel is 3x larger than RSI panel
            'returnfig': True,
            'figratio': (16, 9),
            'figscale': 1.2,
            'tight_layout': True
        }

        try:
            fig, axes = mpf.plot(plot_df, **plot_kwargs)
            # Add text labels for entry/stop/target as they don't have built-in labels
            ax_main = axes[0]
            x_pos = len(plot_df) # Position text on the right edge
            if entry_strategy.get('entry_price'):
                ax_main.text(x_pos, entry_strategy['entry_price'], ' Entry', color='#1f77b4', va='center', fontsize=9)
            if entry_strategy.get('stop_loss'):
                ax_main.text(x_pos, entry_strategy['stop_loss'], ' Stop', color='#d62728', va='center', fontsize=9)
            if entry_strategy.get('targets') and entry_strategy['targets'][0]:
                ax_main.text(x_pos, entry_strategy['targets'][0]['price'], ' Target', color='#2ca02c', va='center', fontsize=9)
            return fig, axes
        except Exception as e:
            print(f"FATAL PLOTTING ERROR: {e}")
            return None, None

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

        if hasattr(self, '_last_df_sr_analysis'):
            try:
                fig, axes = self.plot_analysis(self._last_df_sr_analysis, result)
                if fig is not None:
                    # Save the plot as an image file since we're using non-interactive backend
                    filename = f"{self.trading_pair.replace('/', '_')}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"📊 Analysis plot saved as: {filename}")
                    plt.close(fig)  # Close the figure to free memory
            except Exception as e:
                print(f"Plotting error: {e}")

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