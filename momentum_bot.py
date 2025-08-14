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
from matplotlib.patches import Rectangle

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')


class EnhancedMomentumBot:
    """
    Enhanced Trading Bot with Advanced Support/Resistance Analysis

    Key Improvements:
    - Multi-timeframe confluence (1m, 5m, 15m)
    - Dynamic ATR-based position sizing
    - Volume-weighted S/R levels
    - Order flow analysis
    - Momentum divergence detection
    - Enhanced risk management
    - Smart filtering for false signals
    - Advanced plotting with zones and confluence indicators
    """

    def __init__(self, trading_pair: str, exchange_name: str = 'binance'):
        self.trading_pair = trading_pair
        self.exchange_name = exchange_name

        # Enhanced configuration parameters
        self.config = {
            # --- Multi-Timeframe Analysis ---
            'execution_tf': '1m',             # Ultra-precise entry timing
            'analysis_tf': '5m',              # Main S/R analysis 
            'trend_tf': '15m',                # Trend confirmation
            'context_tf': '1h',               # Market context
            
            # --- Enhanced S/R Detection ---
            'sr_lookback_periods': 200,       # Extended lookback
            'pivot_order': 4,                 # More precise pivots
            'vol_cluster_atr_mult': 0.3,      # Tighter clustering
            'volume_weight_factor': 0.4,      # Volume importance
            'time_decay_lambda': 0.35,        # Recent bias
            'sr_strength_threshold': 4.0,     # Higher quality threshold
            'confluence_distance': 0.005,     # 0.5% for confluence
            
            # --- Advanced Risk Management ---
            'atr_period': 21,                 # More stable ATR
            'dynamic_stop_mult': 1.0,         # Base stop multiplier
            'volatility_adjustment': True,    # Adapt to market conditions
            'min_reward_risk': 1.8,           # Higher R:R requirement
            'max_risk_per_trade': 0.02,       # 2% max risk
            'position_size_method': 'atr',    # ATR-based sizing
            
            # --- Signal Quality Filters ---
            'min_volume_spike': 1.5,          # Volume confirmation
            'rsi_oversold': 35,               # RSI levels
            'rsi_overbought': 65,
            'macd_divergence_periods': 20,    # MACD analysis
            'momentum_lookback': 30,          # Momentum analysis
            
            # --- Order Flow Analysis ---
            'order_flow_periods': 10,         # Recent candles for flow
            'buyer_seller_ratio': 1.2,        # Bullish flow threshold
            
            # --- Pattern Recognition ---
            'pattern_confirmation_periods': 3,
            'pattern_strength_threshold': 0.7,
            
            # --- Legacy parameters (kept for compatibility) ---
            'entry_buffer': 0.0008,
            'stop_loss_ratio': 0.015,
            'ema_fast': 50,
            'ema_slow': 200,
            'fibonacci_enabled': True,
            'sr_proximity_threshold': 0.01,
            'sr_min_touches': 3,
            'confluence_multipliers': {
                1: 1.0, 2: 1.4, 3: 1.8, 4: 2.3, 5: 2.8, 6: 3.5
            },
        }

        self._cached_data = {}
        
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

        print(f"Initialized Enhanced {self.__class__.__name__} for {trading_pair} on {exchange_name}")

    # ---------------------------
    # Enhanced Data Management
    # ---------------------------
    def fetch_multi_timeframe_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data across multiple timeframes for comprehensive analysis."""
        timeframes = {
            '1m': 300,   # 5 hours of 1m data
            '5m': 400,   # ~33 hours of 5m data  
            '15m': 400,  # ~4 days of 15m data
            '1h': 200    # ~8 days of 1h data
        }
        
        data = {}
        for tf, limit in timeframes.items():
            try:
                df = self.fetch_ohlcv_data(tf, limit)
                if not df.empty:
                    # Add technical indicators immediately
                    df = self._add_technical_indicators(df)
                    data[tf] = df
                    print(f"✓ Fetched {len(df)} candles for {tf}")
                else:
                    print(f"⚠ No data for {tf}")
            except Exception as e:
                print(f"Error fetching {tf} data: {e}")
                
        self._cached_data = data
        return data

    def fetch_ohlcv_data(self, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Enhanced OHLCV data fetching with error handling."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.trading_pair,
                timeframe=timeframe,
                limit=limit
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Data quality check
            if len(df) < 50:
                print(f"Warning: Limited data for {timeframe} ({len(df)} candles)")
                
            return df
        except Exception as e:
            print(f"Error fetching {timeframe} data: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to dataframe."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Moving averages
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Volatility
        df['atr'] = self.calculate_atr(df)
        df['bb_upper'], df['bb_lower'] = self._bollinger_bands(df['close'])
        
        # Momentum
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._macd(df['close'])
        df['stoch_k'], df['stoch_d'] = self._stochastic(df)
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['money_flow'] = self._money_flow_index(df)
        
        # Order flow approximation
        df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'])).fillna(0.5)
        df['selling_pressure'] = 1 - df['buying_pressure']
        
        return df

    # ---------------------------
    # Enhanced Technical Indicators
    # ---------------------------
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Enhanced ATR calculation with dynamic period adjustment."""
        if df.empty:
            return pd.Series(dtype=float)
        if period is None:
            period = self.config['atr_period']
            
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()  # Use EMA for more responsive ATR
        
        return atr

    def _bollinger_bands(self, series: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = series.rolling(period).mean()
        rolling_std = series.rolling(period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, lower

    def _macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def _stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent

    def _money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mf_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Enhanced RSI calculation with smoothing."""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing method
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ---------------------------
    # Advanced S/R Analysis
    # ---------------------------
    def calculate_enhanced_sr_levels(self, df: pd.DataFrame) -> Dict:
        """Enhanced S/R calculation with multi-method confluence."""
        if df.empty:
            return {'supports': [], 'resistances': [], 'current_price': np.nan}

        current_price = float(df['close'].iloc[-1])
        atr_value = float(df['atr'].iloc[-1]) if 'atr' in df.columns else None

        # Method 1: Volume-weighted pivot points
        volume_pivots = self._calculate_volume_weighted_pivots(df)
        
        # Method 2: Order flow zones
        flow_zones = self._calculate_order_flow_zones(df)
        
        # Method 3: Institutional levels (psychological + Fibonacci)
        institutional_levels = self._calculate_institutional_levels(df)
        
        # Method 4: Dynamic support/resistance from trend changes
        dynamic_sr = self._calculate_dynamic_sr_levels(df)
        
        # Method 5: Volume profile POC and value areas
        volume_profile_levels = self._calculate_enhanced_volume_profile(df)
        
        all_methods = {
            'volume_pivots': volume_pivots,
            'flow_zones': flow_zones,
            'institutional': institutional_levels,
            'dynamic_sr': dynamic_sr,
            'volume_profile': volume_profile_levels
        }
        
        # Advanced confluence scoring
        scored_levels = self._advanced_confluence_scoring(df, all_methods, current_price, atr_value)
        
        return scored_levels

    def _calculate_volume_weighted_pivots(self, df: pd.DataFrame) -> List[Dict]:
        """Calculate pivot points weighted by volume."""
        swing_highs, swing_lows = self.identify_enhanced_swing_points(df)
        
        # Weight pivots by volume
        volume_weighted_pivots = []
        
        for timestamp, price in swing_highs + swing_lows:
            if timestamp in df.index:
                volume = float(df.loc[timestamp, 'volume'])
                volume_weight = volume / df['volume'].rolling(20).mean().loc[timestamp]
                
                pivot_type = 'resistance' if (timestamp, price) in swing_highs else 'support'
                
                volume_weighted_pivots.append({
                    'level': float(price),
                    'strength': float(volume_weight),
                    'type': f'volume_weighted_{pivot_type}',
                    'timestamp': timestamp,
                    'volume': volume
                })
        
        return volume_weighted_pivots

    def _calculate_order_flow_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Identify zones with significant buying/selling pressure."""
        zones = []
        window = self.config['order_flow_periods']
        
        for i in range(window, len(df)):
            slice_df = df.iloc[i-window:i]
            
            # Calculate net buying pressure
            avg_buying = slice_df['buying_pressure'].mean()
            avg_volume = slice_df['volume'].mean()
            
            if avg_buying > 0.65 and avg_volume > df['volume_sma'].iloc[i]:
                # Strong buying zone (support)
                zones.append({
                    'level': float(slice_df['low'].min()),
                    'strength': float(avg_buying * 3),
                    'type': 'order_flow_support',
                    'avg_volume': avg_volume
                })
            elif avg_buying < 0.35 and avg_volume > df['volume_sma'].iloc[i]:
                # Strong selling zone (resistance)
                zones.append({
                    'level': float(slice_df['high'].max()),
                    'strength': float((1 - avg_buying) * 3),
                    'type': 'order_flow_resistance',
                    'avg_volume': avg_volume
                })
        
        return zones

    def _calculate_institutional_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Calculate institutional-level S/R (psychological + Fibonacci)."""
        levels = []
        current_price = float(df['close'].iloc[-1])
        
        # Enhanced psychological levels
        psych_levels = self._find_enhanced_psychological_levels(current_price)
        levels.extend(psych_levels)
        
        # Multi-timeframe Fibonacci
        if self.config['fibonacci_enabled']:
            fib_levels = self._calculate_multi_tf_fibonacci(df)
            levels.extend(fib_levels)
            
        return levels

    def _find_enhanced_psychological_levels(self, current_price: float) -> List[Dict]:
        """Enhanced psychological level detection."""
        levels = []
        
        # Determine price range and increments
        if current_price >= 10000:
            increments = [100, 500, 1000, 5000]
        elif current_price >= 1000:
            increments = [10, 50, 100, 500]
        elif current_price >= 100:
            increments = [1, 5, 10, 50]
        elif current_price >= 10:
            increments = [0.1, 0.5, 1, 5]
        elif current_price >= 1:
            increments = [0.01, 0.05, 0.1, 0.5]
        else:
            increments = [0.001, 0.005, 0.01, 0.05]
        
        price_range = current_price * 0.15  # 15% range
        
        for increment in increments:
            # Find nearest round numbers
            lower_bound = int(current_price / increment) * increment
            upper_bound = lower_bound + increment
            
            for level in [lower_bound - increment, lower_bound, upper_bound, upper_bound + increment]:
                if abs(level - current_price) <= price_range and level > 0:
                    # Strength based on how "round" the number is
                    strength = 3.0 if increment in increments[-2:] else 2.0
                    
                    levels.append({
                        'level': float(level),
                        'strength': strength,
                        'type': 'psychological',
                        'increment': increment
                    })
        
        return levels

    def _calculate_multi_tf_fibonacci(self, df: pd.DataFrame) -> List[Dict]:
        """Multi-timeframe Fibonacci analysis."""
        levels = []
        
        # Recent swing high/low
        recent_periods = min(100, len(df))
        recent_df = df.tail(recent_periods)
        
        swing_high = float(recent_df['high'].max())
        swing_low = float(recent_df['low'].min())
        
        # Long-term swing high/low
        long_periods = min(200, len(df))
        long_df = df.tail(long_periods)
        
        long_high = float(long_df['high'].max())
        long_low = float(long_df['low'].min())
        
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
        
        # Recent Fibonacci retracements
        for ratio in fib_ratios:
            if swing_high > swing_low:
                fib_level = swing_high - (swing_high - swing_low) * ratio
                levels.append({
                    'level': float(fib_level),
                    'strength': 3.0 + ratio,  # Higher ratios get more weight
                    'type': 'fibonacci_recent',
                    'ratio': ratio
                })
        
        # Long-term Fibonacci retracements
        for ratio in fib_ratios:
            if long_high > long_low:
                fib_level = long_high - (long_high - long_low) * ratio
                levels.append({
                    'level': float(fib_level),
                    'strength': 2.5 + ratio,
                    'type': 'fibonacci_long',
                    'ratio': ratio
                })
        
        return levels

    def _calculate_dynamic_sr_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Calculate dynamic S/R from trend changes and consolidation zones."""
        levels = []
        
        # Find consolidation zones (low volatility periods)
        df['volatility'] = df['high'] - df['low']
        vol_threshold = df['volatility'].rolling(20).mean() * 0.7
        
        consolidation_zones = []
        current_zone = []
        
        for i, (ts, row) in enumerate(df.iterrows()):
            if row['volatility'] < vol_threshold.iloc[i]:
                current_zone.append((ts, row))
            else:
                if len(current_zone) >= 5:  # At least 5 periods of consolidation
                    zone_df = pd.DataFrame([r for _, r in current_zone])
                    zone_high = zone_df['high'].max()
                    zone_low = zone_df['low'].min()
                    zone_volume = zone_df['volume'].mean()
                    
                    # Both high and low of consolidation zone are potential S/R
                    levels.extend([
                        {
                            'level': float(zone_high),
                            'strength': float(len(current_zone) * 0.5),
                            'type': 'consolidation_resistance',
                            'volume': zone_volume
                        },
                        {
                            'level': float(zone_low),
                            'strength': float(len(current_zone) * 0.5),
                            'type': 'consolidation_support',
                            'volume': zone_volume
                        }
                    ])
                current_zone = []
        
        return levels

    def _calculate_enhanced_volume_profile(self, df: pd.DataFrame, bins: int = 50) -> List[Dict]:
        """Enhanced volume profile with POC and value area."""
        if df.empty:
            return []
            
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        volume_profile = []
        total_volume = 0
        
        # Calculate volume at each price level
        for i in range(bins):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_mid = (bin_low + bin_high) / 2
            
            # Volume calculation with time weighting (recent volume weighted more)
            bin_volume = 0
            for idx, row in df.iterrows():
                if row['low'] <= bin_high and row['high'] >= bin_low:
                    # Time decay weight
                    age = len(df) - df.index.get_loc(idx)
                    time_weight = np.exp(-0.1 * age / len(df))
                    bin_volume += row['volume'] * time_weight
            
            if bin_volume > 0:
                volume_profile.append({
                    'level': float(bin_mid),
                    'volume': float(bin_volume),
                    'strength': float(bin_volume),
                    'type': 'volume_profile'
                })
                total_volume += bin_volume
        
        # Sort by volume and identify key levels
        volume_profile.sort(key=lambda x: x['volume'], reverse=True)
        
        # POC (Point of Control) - highest volume
        if volume_profile:
            volume_profile[0]['type'] = 'volume_poc'
            volume_profile[0]['strength'] *= 2  # POC gets extra weight
        
        # Value Area (70% of volume)
        cumulative_volume = 0
        value_area_threshold = total_volume * 0.7
        
        for level in volume_profile:
            cumulative_volume += level['volume']
            if cumulative_volume <= value_area_threshold:
                level['type'] = 'volume_value_area'
                level['strength'] *= 1.5  # Value area gets extra weight
        
        return volume_profile[:15]  # Return top 15 levels

    def identify_enhanced_swing_points(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        """Enhanced swing point detection with volume confirmation."""
        order = self.config['pivot_order']
        
        # Basic swing point detection
        high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
        low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
        
        swing_highs = []
        swing_lows = []
        
        # Volume-confirmed swings (only keep swings with above-average volume)
        avg_volume = df['volume'].rolling(20).mean()
        
        for i in high_indices:
            if i < len(df) and df['volume'].iloc[i] > avg_volume.iloc[i] * 1.2:
                swing_highs.append((df.index[i], df['high'].iloc[i]))
        
        for i in low_indices:
            if i < len(df) and df['volume'].iloc[i] > avg_volume.iloc[i] * 1.2:
                swing_lows.append((df.index[i], df['low'].iloc[i]))
        
        return swing_highs, swing_lows

    def _advanced_confluence_scoring(self, df: pd.DataFrame, all_methods: Dict, current_price: float, atr_value: Optional[float]) -> Dict:
        """Advanced confluence scoring with multiple factors."""
        if atr_value and current_price > 0:
            proximity_threshold = (self.config['vol_cluster_atr_mult'] * atr_value) / current_price
        else:
            proximity_threshold = self.config['confluence_distance']
        
        # Flatten all levels
        all_levels = []
        for method_name, levels in all_methods.items():
            for level in levels:
                level['source_method'] = method_name
                all_levels.append(level)
        
        # Cluster levels by proximity
        clustered_levels = self._cluster_levels_by_proximity(all_levels, proximity_threshold, current_price)
        
        # Score each cluster
        scored_supports = []
        scored_resistances = []
        
        for cluster in clustered_levels:
            cluster_level = np.mean([l['level'] for l in cluster])
            
            # Base scoring factors
            method_count = len(set(l['source_method'] for l in cluster))
            base_strength = sum(l.get('strength', 1.0) for l in cluster)
            
            # Volume factor
            avg_volume = np.mean([l.get('volume', 0) for l in cluster if 'volume' in l])
            volume_factor = 1.0 + (avg_volume / df['volume'].mean() - 1) * 0.3 if avg_volume > 0 else 1.0
            
            # Distance factor (closer levels are stronger)
            distance_pct = abs(cluster_level - current_price) / current_price
            distance_factor = 1.0 / (1 + distance_pct * 10)  # Exponential decay with distance
            
            # Time decay for recent touches
            recent_touches = self._count_enhanced_recent_touches(df, cluster_level, proximity_threshold)
            
            # Confluence multiplier
            confluence_mult = self.config['confluence_multipliers'].get(method_count, 1.0 + 0.4 * (method_count - 1))
            
            # Final score calculation
            total_strength = (base_strength * confluence_mult * volume_factor * distance_factor) + recent_touches
            
            level_data = {
                'level': float(cluster_level),
                'methods': list(set(l['source_method'] for l in cluster)),
                'method_count': method_count,
                'base_strength': base_strength,
                'volume_factor': volume_factor,
                'distance_factor': distance_factor,
                'recent_touches': recent_touches,
                'confluence_multiplier': confluence_mult,
                'total_strength': total_strength,
                'distance_pct': distance_pct,
                'cluster_size': len(cluster),
                'type': 'enhanced_confluence_zone'
            }
            
            # Classify as support or resistance
            if cluster_level < current_price:
                scored_supports.append(level_data)
            else:
                scored_resistances.append(level_data)
        
        # Filter and sort
        min_strength = self.config['sr_strength_threshold']
        valid_supports = [s for s in scored_supports if s['total_strength'] >= min_strength]
        valid_resistances = [r for r in scored_resistances if r['total_strength'] >= min_strength]
        
        # Sort by total strength (best first)
        valid_supports.sort(key=lambda x: -x['total_strength'])
        valid_resistances.sort(key=lambda x: -x['total_strength'])
        
        return {
            'supports': valid_supports[:8],
            'resistances': valid_resistances[:8],
            'current_price': current_price,
            'analysis_quality': len(valid_supports) + len(valid_resistances)
        }

    def _cluster_levels_by_proximity(self, levels: List[Dict], threshold: float, ref_price: float) -> List[List[Dict]]:
        """Cluster levels by proximity using adaptive thresholding."""
        if not levels:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x['level'])
        
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Calculate dynamic threshold based on current cluster
            cluster_center = np.mean([l['level'] for l in current_cluster])
            adaptive_threshold = threshold * (1 + abs(cluster_center - ref_price) / ref_price)
            
            if abs(level['level'] - cluster_center) / cluster_center <= adaptive_threshold:
                current_cluster.append(level)
            else:
                if len(current_cluster) >= 1:  # Keep all clusters for now
                    clusters.append(current_cluster)
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters

    def _count_enhanced_recent_touches(self, df: pd.DataFrame, level: float, proximity_threshold: float) -> float:
        """Enhanced touch counting with volume weighting and time decay."""
        if df.empty:
            return 0.0
        
        lookback = min(150, len(df))
        recent_df = df.tail(lookback)
        
        upper_bound = level * (1 + proximity_threshold)
        lower_bound = level * (1 - proximity_threshold)
        
        weighted_touches = 0.0
        time_decay = self.config['time_decay_lambda']
        
        for idx, (timestamp, row) in enumerate(recent_df.iterrows()):
            # Check if candle touched the level
            if row['low'] <= upper_bound and row['high'] >= lower_bound:
                # Time decay weight (recent touches weighted more)
                age = lookback - 1 - idx
                time_weight = np.exp(-time_decay * age / lookback)
                
                # Volume weight
                volume_weight = 1.0
                if 'volume_ratio' in row:
                    volume_weight = min(2.0, max(0.5, row['volume_ratio']))
                
                # Wick vs body touch weight
                body_top = max(row['open'], row['close'])
                body_bottom = min(row['open'], row['close'])
                
                if body_bottom <= upper_bound and body_top >= lower_bound:
                    # Body touch - stronger
                    touch_weight = 1.5
                else:
                    # Wick only touch - weaker
                    touch_weight = 1.0
                
                total_weight = time_weight * volume_weight * touch_weight
                weighted_touches += total_weight
        
        return weighted_touches

    # ---------------------------
    # Enhanced Entry Strategy
    # ---------------------------
    def calculate_enhanced_entry_strategy(self, multi_tf_data: Dict[str, pd.DataFrame], sr_levels: Dict) -> Dict:
        """Enhanced entry strategy with multi-timeframe confirmation."""
        current_price = sr_levels['current_price']
        supports = sr_levels['supports']
        resistances = sr_levels['resistances']
        
        if not supports:
            return {'entry_valid': False, 'reason': 'No valid support levels found'}
        
        # Get data
        df_1m = multi_tf_data.get('1m', pd.DataFrame())
        df_5m = multi_tf_data.get('5m', pd.DataFrame())
        
        if df_1m.empty or df_5m.empty:
            return {'entry_valid': False, 'reason': 'Insufficient multi-timeframe data'}
        
        # Select best support with enhanced criteria
        best_support = self._select_optimal_support(supports, current_price, df_5m)
        if not best_support:
            return {'entry_valid': False, 'reason': 'No optimal support found'}
        
        support_level = best_support['level']
        
        # Enhanced position sizing
        position_data = self._calculate_enhanced_position_size(df_1m, df_5m, support_level, current_price)
        
        # Multi-timeframe entry timing
        entry_timing = self._calculate_entry_timing(df_1m, support_level, current_price)
        
        # Enhanced risk management
        risk_data = self._calculate_enhanced_risk_management(df_1m, df_5m, support_level, current_price)
        
        # Target calculation with multiple scenarios
        targets = self._calculate_enhanced_targets(resistances, current_price, risk_data['stop_loss'])
        
        # Validate overall setup
        setup_validation = self._validate_trading_setup(multi_tf_data, best_support, risk_data, targets)
        
        return {
            'entry_valid': setup_validation['valid'],
            'setup_quality': setup_validation['quality_score'],
            'support_data': best_support,
            'position_sizing': position_data,
            'entry_timing': entry_timing,
            'risk_management': risk_data,
            'targets': targets,
            'recommendation': self._generate_enhanced_recommendation(setup_validation, entry_timing, targets),
            'confidence_factors': setup_validation['confidence_factors']
        }

    def _select_optimal_support(self, supports: List[Dict], current_price: float, df: pd.DataFrame) -> Optional[Dict]:
        """Select the optimal support level using enhanced criteria."""
        if not supports:
            return None
        
        # Score each support
        scored_supports = []
        
        for support in supports:
            level = support['level']
            distance_pct = abs(level - current_price) / current_price
            
            # Skip supports too far away (>5%)
            if distance_pct > 0.05:
                continue
            
            # Enhanced scoring
            score = support['total_strength']
            
            # Proximity bonus (closer is better, but not too close)
            if 0.005 <= distance_pct <= 0.025:  # Sweet spot 0.5-2.5%
                score *= 1.3
            elif distance_pct <= 0.005:  # Too close
                score *= 0.8
            
            # Method diversity bonus
            method_bonus = 1 + (support['method_count'] - 1) * 0.15
            score *= method_bonus
            
            # Recent price action confirmation
            recent_confirmation = self._check_recent_price_action_at_level(df, level)
            score *= recent_confirmation
            
            scored_supports.append({
                **support,
                'selection_score': score,
                'distance_pct': distance_pct,
                'recent_confirmation': recent_confirmation
            })
        
        if not scored_supports:
            return None
        
        # Return the highest scored support
        return max(scored_supports, key=lambda x: x['selection_score'])

    def _check_recent_price_action_at_level(self, df: pd.DataFrame, level: float) -> float:
        """Check recent price action around a level for confirmation."""
        if df.empty:
            return 1.0
        
        recent_data = df.tail(20)
        confirmation_score = 1.0
        
        # Check for recent bounces
        touches = 0
        bounces = 0
        
        for i, row in recent_data.iterrows():
            level_tolerance = level * 0.008  # 0.8% tolerance
            
            if abs(row['low'] - level) <= level_tolerance:
                touches += 1
                # Check if it bounced (next few candles went higher)
                try:
                    idx_pos = recent_data.index.get_loc(i)
                    if idx_pos < len(recent_data) - 2:
                        next_candles = recent_data.iloc[idx_pos+1:idx_pos+3]
                        if next_candles['close'].min() > row['low']:
                            bounces += 1
                except:
                    pass
        
        if touches > 0:
            bounce_ratio = bounces / touches
            confirmation_score = 1.0 + bounce_ratio * 0.5  # Up to 50% bonus for good bounces
        
        return confirmation_score

    def _calculate_enhanced_position_size(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, support_level: float, current_price: float) -> Dict:
        """Calculate position size based on volatility and risk parameters."""
        
        # Get current ATR from both timeframes
        atr_1m = float(df_1m['atr'].iloc[-1]) if 'atr' in df_1m.columns else None
        atr_5m = float(df_5m['atr'].iloc[-1]) if 'atr' in df_5m.columns else None
        
        # Use 5m ATR as primary, 1m as backup
        primary_atr = atr_5m if atr_5m else atr_1m
        
        if not primary_atr:
            return {'method': 'fallback', 'risk_amount': current_price * 0.02}
        
        # Dynamic stop multiplier based on market volatility
        recent_volatility = df_5m['atr'].tail(10).std() / df_5m['atr'].tail(10).mean()
        volatility_multiplier = 1.0 + min(0.5, recent_volatility)  # Cap at 1.5x
        
        dynamic_stop_mult = self.config['dynamic_stop_mult'] * volatility_multiplier
        
        # Calculate stop distance
        stop_distance = primary_atr * dynamic_stop_mult
        
        # Position sizing
        max_risk = self.config['max_risk_per_trade']  # 2% max risk
        risk_per_unit = stop_distance
        
        return {
            'method': 'enhanced_atr',
            'atr_value': primary_atr,
            'volatility_adjustment': volatility_multiplier,
            'stop_distance': stop_distance,
            'risk_per_unit': risk_per_unit,
            'max_risk_pct': max_risk * 100,
            'recommended_risk_pct': min(max_risk * 100, 1.5),  # Conservative default
        }

    def _calculate_entry_timing(self, df_1m: pd.DataFrame, support_level: float, current_price: float) -> Dict:
        """Calculate optimal entry timing using 1m data."""
        
        if df_1m.empty:
            return {'timing': 'insufficient_data'}
        
        recent_1m = df_1m.tail(10)
        distance_to_support = (current_price - support_level) / current_price
        
        # Momentum analysis
        price_momentum = recent_1m['close'].pct_change(3).iloc[-1]  # 3-candle momentum
        volume_momentum = recent_1m['volume'].iloc[-1] / recent_1m['volume'].mean()
        
        # RSI condition
        current_rsi = recent_1m['rsi'].iloc[-1] if 'rsi' in recent_1m.columns else 50
        
        # Determine timing
        if distance_to_support <= 0.002:  # Very close to support (0.2%)
            if current_rsi < 40 and price_momentum < -0.005:  # Oversold and falling
                timing = 'WAIT_FOR_BOUNCE'
                confidence = 0.7
            else:
                timing = 'IMMEDIATE'
                confidence = 0.9
        elif distance_to_support <= 0.008:  # Close to support (0.8%)
            timing = 'PREPARE_ENTRY'
            confidence = 0.8
        elif distance_to_support <= 0.02:  # Moderate distance (2%)
            timing = 'MONITOR'
            confidence = 0.6
        else:  # Too far from support
            timing = 'WAIT_PULLBACK'
            confidence = 0.3
        
        return {
            'timing': timing,
            'confidence': confidence,
            'distance_to_support_pct': distance_to_support * 100,
            'price_momentum': price_momentum,
            'volume_momentum': volume_momentum,
            'rsi': current_rsi
        }

    def _calculate_enhanced_risk_management(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, support_level: float, current_price: float) -> Dict:
        """Enhanced risk management with dynamic stops and multiple scenarios."""
        
        # Get ATR values
        atr_1m = float(df_1m['atr'].iloc[-1]) if 'atr' in df_1m.columns else None
        atr_5m = float(df_5m['atr'].iloc[-1]) if 'atr' in df_5m.columns else None
        
        # Primary stop calculation (ATR-based)
        if atr_5m:
            # Market regime analysis
            volatility_regime = self._analyze_volatility_regime(df_5m)
            
            # Adjust stop multiplier based on regime
            if volatility_regime == 'low':
                stop_multiplier = 0.8
            elif volatility_regime == 'high':
                stop_multiplier = 1.4
            else:
                stop_multiplier = 1.0
            
            atr_stop = support_level - (atr_5m * stop_multiplier)
            stop_method = f'atr_5m_{volatility_regime}_regime'
        else:
            atr_stop = current_price * (1 - self.config['stop_loss_ratio'])
            stop_method = 'percentage_fallback'
        
        # Alternative stop levels
        technical_stop = self._calculate_technical_stop(df_5m, support_level)
        trailing_stop_trigger = current_price * 1.005  # Start trailing at 0.5% profit
        
        # Risk metrics
        entry_price = support_level * (1 + self.config['entry_buffer'])
        risk_amount = entry_price - atr_stop
        risk_pct = (risk_amount / entry_price) * 100 if entry_price > 0 else 0
        
        return {
            'stop_loss': atr_stop,  # Primary stop loss for target calculations
            'primary_stop': atr_stop,
            'stop_method': stop_method,
            'technical_stop': technical_stop,
            'trailing_stop_trigger': trailing_stop_trigger,
            'entry_price': entry_price,
            'risk_amount': risk_amount,
            'risk_pct': risk_pct,
            'max_acceptable_risk': 2.5,  # 2.5% max
            'risk_acceptable': risk_pct <= 2.5
        }

    def _analyze_volatility_regime(self, df: pd.DataFrame) -> str:
        """Analyze current volatility regime."""
        if 'atr' not in df.columns:
            return 'normal'
        
        current_atr = df['atr'].iloc[-1]
        atr_sma_20 = df['atr'].rolling(20).mean().iloc[-1]
        atr_sma_50 = df['atr'].rolling(50).mean().iloc[-1]
        
        if current_atr > atr_sma_20 * 1.3:
            return 'high'
        elif current_atr < atr_sma_50 * 0.7:
            return 'low'
        else:
            return 'normal'

    def _calculate_technical_stop(self, df: pd.DataFrame, support_level: float) -> float:
        """Calculate technical stop based on recent swing lows."""
        if df.empty:
            return support_level * 0.98
        
        recent_data = df.tail(20)
        recent_low = recent_data['low'].min()
        
        # Use the lower of: recent swing low or 2% below support
        technical_stop = min(recent_low, support_level * 0.98)
        
        return float(technical_stop)

    def _calculate_enhanced_targets(self, resistances: List[Dict], current_price: float, stop_loss: float) -> List[Dict]:
        """Calculate multiple target scenarios with R:R analysis."""
        targets = []
        risk_amount = current_price - stop_loss
        
        if risk_amount <= 0:
            return targets
        
        # Target from resistances
        for i, resistance in enumerate(resistances[:4]):
            target_price = resistance['level']
            reward = target_price - current_price
            
            if reward > 0:
                rr = reward / risk_amount
                targets.append({
                    'target_num': i + 1,
                    'price': target_price,
                    'reward_risk_ratio': rr,
                    'profit_pct': (reward / current_price) * 100,
                    'resistance_strength': resistance['total_strength'],
                    'type': 'resistance_based'
                })
        
        # Additional algorithmic targets
        algorithmic_targets = [1.5, 2.0, 2.5, 3.0]  # R:R ratios
        for i, rr in enumerate(algorithmic_targets):
            target_price = current_price + (risk_amount * rr)
            targets.append({
                'target_num': len(targets) + 1,
                'price': target_price,
                'reward_risk_ratio': rr,
                'profit_pct': ((target_price - current_price) / current_price) * 100,
                'resistance_strength': 0,
                'type': 'algorithmic'
            })
        
        # Sort by R:R ratio
        targets.sort(key=lambda x: x['reward_risk_ratio'])
        
        return targets

    def _validate_trading_setup(self, multi_tf_data: Dict, support_data: Dict, risk_data: Dict, targets: List[Dict]) -> Dict:
        """Comprehensive trading setup validation."""
        
        validation_score = 0
        max_score = 100
        confidence_factors = {}
        
        # Support quality (25 points)
        support_score = min(25, support_data['total_strength'] * 3)
        validation_score += support_score
        confidence_factors['support_quality'] = support_score / 25
        
        # Risk management (20 points)
        if risk_data['risk_acceptable']:
            risk_score = 20
            if risk_data['risk_pct'] <= 1.5:
                risk_score = 20  # Excellent risk
            elif risk_data['risk_pct'] <= 2.0:
                risk_score = 15  # Good risk
            else:
                risk_score = 10  # Acceptable risk
        else:
            risk_score = 0
        validation_score += risk_score
        confidence_factors['risk_management'] = risk_score / 20
        
        # Target quality (20 points)
        best_targets = [t for t in targets if t['reward_risk_ratio'] >= self.config['min_reward_risk']]
        if best_targets:
            best_rr = max(t['reward_risk_ratio'] for t in best_targets)
            target_score = min(20, best_rr * 7)  # Up to 20 points
        else:
            target_score = 0
        validation_score += target_score
        confidence_factors['target_quality'] = target_score / 20
        
        # Multi-timeframe alignment (15 points)
        mtf_score = self._check_multi_timeframe_alignment(multi_tf_data)
        validation_score += mtf_score
        confidence_factors['mtf_alignment'] = mtf_score / 15
        
        # Market structure (10 points)
        structure_score = self._analyze_market_structure(multi_tf_data.get('5m', pd.DataFrame()))
        validation_score += structure_score
        confidence_factors['market_structure'] = structure_score / 10
        
        # Momentum confirmation (10 points)
        momentum_score = self._check_momentum_confirmation(multi_tf_data.get('5m', pd.DataFrame()))
        validation_score += momentum_score
        confidence_factors['momentum'] = momentum_score / 10
        
        # Final validation
        is_valid = validation_score >= 60  # Require 60% minimum score
        quality_grade = 'A' if validation_score >= 85 else 'B' if validation_score >= 70 else 'C' if validation_score >= 60 else 'D'
        
        return {
            'valid': is_valid,
            'quality_score': validation_score,
            'max_score': max_score,
            'quality_grade': quality_grade,
            'confidence_factors': confidence_factors
        }

    def _check_multi_timeframe_alignment(self, multi_tf_data: Dict) -> float:
        """Check alignment across multiple timeframes."""
        alignment_score = 0.0
        
        # Check trend alignment (5m vs 15m vs 1h)
        for tf in ['5m', '15m', '1h']:
            df = multi_tf_data.get(tf)
            if df is not None and not df.empty and 'ema_21' in df.columns and 'ema_50' in df.columns:
                if df['ema_21'].iloc[-1] > df['ema_50'].iloc[-1]:
                    alignment_score += 5.0  # 5 points per aligned timeframe
        
        return min(15.0, alignment_score)

    def _analyze_market_structure(self, df: pd.DataFrame) -> float:
        """Analyze market structure for bullish bias."""
        if df.empty:
            return 0.0
        
        structure_score = 0.0
        
        # Higher lows pattern
        recent_lows = df['low'].tail(10)
        if len(recent_lows) >= 3:
            if recent_lows.iloc[-1] > recent_lows.iloc[-3]:
                structure_score += 5.0
        
        # Price above key EMAs
        if 'ema_21' in df.columns and df['close'].iloc[-1] > df['ema_21'].iloc[-1]:
            structure_score += 3.0
        
        # Volume trend
        if 'volume_ratio' in df.columns and df['volume_ratio'].tail(5).mean() > 1.1:
            structure_score += 2.0
        
        return min(10.0, structure_score)

    def _check_momentum_confirmation(self, df: pd.DataFrame) -> float:
        """Check momentum indicators for bullish confirmation."""
        if df.empty:
            return 0.0
        
        momentum_score = 0.0
        
        # RSI in bullish zone but not overbought
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if 40 <= rsi <= 65:
                momentum_score += 4.0
            elif 30 <= rsi <= 75:
                momentum_score += 2.0
        
        # MACD bullish
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                momentum_score += 3.0
        
        # Stochastic
        if 'stoch_k' in df.columns:
            stoch = df['stoch_k'].iloc[-1]
            if 20 <= stoch <= 70:
                momentum_score += 3.0
        
        return min(10.0, momentum_score)

    def _generate_enhanced_recommendation(self, validation: Dict, timing: Dict, targets: List[Dict]) -> str:
        """Generate comprehensive trading recommendation."""
        
        if not validation['valid']:
            return f"HOLD - Setup quality insufficient (Grade: {validation['quality_grade']})"
        
        timing_signal = timing['timing']
        confidence = timing['confidence']
        
        best_targets = [t for t in targets if t['reward_risk_ratio'] >= self.config['min_reward_risk']]
        
        if not best_targets:
            return "WAIT - No favorable risk/reward targets available"
        
        best_rr = max(t['reward_risk_ratio'] for t in best_targets)
        
        if timing_signal == 'IMMEDIATE':
            return f"STRONG BUY - Immediate entry (R:R {best_rr:.2f}, Grade: {validation['quality_grade']})"
        elif timing_signal == 'PREPARE_ENTRY':
            return f"BUY READY - Prepare entry orders (R:R {best_rr:.2f}, Grade: {validation['quality_grade']})"
        elif timing_signal == 'WAIT_FOR_BOUNCE':
            return f"BUY ON BOUNCE - Wait for support bounce (R:R {best_rr:.2f})"
        elif timing_signal == 'MONITOR':
            return f"WATCH - Monitor for better entry (R:R {best_rr:.2f})"
        else:
            return f"WAIT - Too far from entry zone (R:R {best_rr:.2f})"

    # ---------------------------
    # Enhanced Analysis Pipeline
    # ---------------------------
    def run_enhanced_market_analysis(self) -> Dict:
        """Main enhanced analysis pipeline."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*80}")
        print(f"🚀 ENHANCED MULTI-TIMEFRAME TRADING ANALYSIS 🚀")
        print(f"Trading Pair: {self.trading_pair}")
        print(f"Analysis Time: {timestamp}")
        print(f"{'='*80}")
        
        # Phase 1: Multi-timeframe data collection
        print("\nPhase 1: Multi-Timeframe Data Collection...")
        multi_tf_data = self.fetch_multi_timeframe_data()
        
        if not multi_tf_data or '5m' not in multi_tf_data:
            return {"error": "Insufficient multi-timeframe data"}
        
        # Phase 2: Enhanced S/R analysis
        print("\nPhase 2: Enhanced Support/Resistance Analysis...")
        sr_levels = self.calculate_enhanced_sr_levels(multi_tf_data['5m'])
        
        # Display S/R results
        print(f"\n📊 ENHANCED SUPPORT LEVELS (Top {len(sr_levels['supports'])}):")
        for i, support in enumerate(sr_levels['supports']):
            methods_str = ', '.join(support['methods'][:3]) + ('...' if len(support['methods']) > 3 else '')
            print(f"  S{i+1}: ${support['level']:.6f} | Strength: {support['total_strength']:.1f} | "
                  f"Methods: {support['method_count']} ({methods_str}) | "
                  f"Distance: {support['distance_pct']*100:.2f}%")
        
        print(f"\n📊 ENHANCED RESISTANCE LEVELS (Top {len(sr_levels['resistances'])}):")
        for i, resistance in enumerate(sr_levels['resistances']):
            methods_str = ', '.join(resistance['methods'][:3]) + ('...' if len(resistance['methods']) > 3 else '')
            print(f"  R{i+1}: ${resistance['level']:.6f} | Strength: {resistance['total_strength']:.1f} | "
                  f"Methods: {resistance['method_count']} ({methods_str}) | "
                  f"Distance: {resistance['distance_pct']*100:.2f}%")
        
        # Phase 3: Enhanced entry strategy
        print("\nPhase 3: Enhanced Entry Strategy Calculation...")
        entry_strategy = self.calculate_enhanced_entry_strategy(multi_tf_data, sr_levels)
        
        if not entry_strategy['entry_valid']:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": entry_strategy['reason'],
                "sr_levels": sr_levels,
                "timestamp": timestamp,
                "entry_strategy": entry_strategy
            }
        
        # Display entry strategy
        support_data = entry_strategy['support_data']
        timing_data = entry_strategy['entry_timing']
        risk_data = entry_strategy['risk_management']
        
        print(f"\n🎯 OPTIMAL ENTRY SETUP:")
        print(f"  Support Level: ${support_data['level']:.6f}")
        print(f"  Support Strength: {support_data['total_strength']:.1f} (Methods: {support_data['method_count']})")
        print(f"  Entry Price: ${risk_data['entry_price']:.6f}")
        print(f"  Stop Loss: ${risk_data['primary_stop']:.6f} ({risk_data['stop_method']})")
        print(f"  Risk: {risk_data['risk_pct']:.2f}% | Max Acceptable: {risk_data['max_acceptable_risk']}%")
        print(f"  Entry Timing: {timing_data['timing']} (Confidence: {timing_data['confidence']:.0%})")
        
        if entry_strategy['targets']:
            print(f"\n🎯 TARGET ANALYSIS:")
            for target in entry_strategy['targets'][:3]:  # Show top 3 targets
                print(f"  T{target['target_num']}: ${target['price']:.6f} | "
                      f"R:R {target['reward_risk_ratio']:.2f} | "
                      f"Profit: {target['profit_pct']:.1f}% | "
                      f"Type: {target['type']}")
        
        # Phase 4: Enhanced confirmations
        print("\nPhase 4: Enhanced Signal Confirmations...")
        confirmations = self._get_enhanced_confirmations(multi_tf_data, support_data, risk_data)
        
        for conf_type, conf_data in confirmations.items():
            print(f"  {conf_type.replace('_', ' ').title()}: {conf_data.get('status', 'Unknown')}")
        
        # Final signal generation
        final_signal = self._generate_final_enhanced_signal(entry_strategy, confirmations)
        
        print(f"\n{'='*60}")
        print(f"🚨 FINAL ENHANCED SIGNAL 🚨")
        print(f"Signal: {final_signal['signal']}")
        print(f"Confidence: {final_signal['confidence']:.1f}%")
        print(f"Setup Quality: {entry_strategy['setup_quality']:.1f}/100 ({entry_strategy.get('quality_grade', 'N/A')})")
        print(f"Recommendation: {entry_strategy['recommendation']}")
        print(f"{'='*60}")
        
        # Store for plotting
        self._last_analysis_data = {
            'multi_tf_data': multi_tf_data,
            'sr_levels': sr_levels,
            'entry_strategy': entry_strategy,
            'confirmations': confirmations,
            'final_signal': final_signal
        }
        
        return {
            "signal": final_signal['signal'],
            "confidence": final_signal['confidence'],
            "setup_quality": entry_strategy['setup_quality'],
            "quality_grade": entry_strategy.get('quality_grade', 'N/A'),
            "entry_strategy": entry_strategy,
            "sr_levels": sr_levels,
            "confirmations": confirmations,
            "recommendation": entry_strategy['recommendation'],
            "timestamp": timestamp,
            "multi_tf_analysis": True
        }

    def _get_enhanced_confirmations(self, multi_tf_data: Dict, support_data: Dict, risk_data: Dict) -> Dict:
        """Get comprehensive signal confirmations."""
        confirmations = {}
        
        df_5m = multi_tf_data.get('5m', pd.DataFrame())
        df_1m = multi_tf_data.get('1m', pd.DataFrame())
        
        # Volume confirmation
        confirmations['volume_confirmation'] = self._check_volume_confirmation(df_5m)
        
        # Momentum alignment
        confirmations['momentum_alignment'] = self._check_momentum_alignment(df_5m)
        
        # Candlestick patterns
        confirmations['candlestick_patterns'] = self._check_enhanced_patterns(df_5m, support_data['level'])
        
        # Divergence analysis
        confirmations['divergence_analysis'] = self._check_enhanced_divergences(df_5m, support_data['level'])
        
        # Market microstructure (1m)
        if not df_1m.empty:
            confirmations['microstructure'] = self._check_market_microstructure(df_1m, support_data['level'])
        
        return confirmations

    def _check_volume_confirmation(self, df: pd.DataFrame) -> Dict:
        """Enhanced volume analysis."""
        if df.empty or 'volume_ratio' not in df.columns:
            return {'status': 'INSUFFICIENT_DATA'}
        
        recent_volume = df['volume_ratio'].tail(5).mean()
        volume_trend = df['volume_ratio'].tail(10).rolling(3).mean().iloc[-1]
        
        if recent_volume > 1.5 and volume_trend > recent_volume * 0.8:
            status = 'STRONG_BULLISH'
        elif recent_volume > 1.2:
            status = 'BULLISH'
        elif recent_volume > 0.8:
            status = 'NEUTRAL'
        else:
            status = 'BEARISH'
        
        return {
            'status': status,
            'recent_volume_ratio': recent_volume,
            'volume_trend': volume_trend
        }

    def _check_momentum_alignment(self, df: pd.DataFrame) -> Dict:
        """Check momentum indicator alignment."""
        if df.empty:
            return {'status': 'INSUFFICIENT_DATA'}
        
        signals = []
        
        # RSI
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if 30 <= rsi <= 45:
                signals.append('RSI_OVERSOLD_BULLISH')
            elif 45 < rsi <= 65:
                signals.append('RSI_BULLISH')
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                signals.append('MACD_BULLISH')
        
        # Stochastic
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            if df['stoch_k'].iloc[-1] > df['stoch_d'].iloc[-1] and df['stoch_k'].iloc[-1] < 80:
                signals.append('STOCH_BULLISH')
        
        # Overall assessment
        bullish_count = len(signals)
        if bullish_count >= 3:
            status = 'STRONG_ALIGNMENT'
        elif bullish_count >= 2:
            status = 'GOOD_ALIGNMENT'
        elif bullish_count >= 1:
            status = 'PARTIAL_ALIGNMENT'
        else:
            status = 'NO_ALIGNMENT'
        
        return {
            'status': status,
            'bullish_signals': signals,
            'signal_count': bullish_count
        }

    def _check_enhanced_patterns(self, df: pd.DataFrame, support_level: float) -> Dict:
        """Enhanced candlestick pattern recognition."""
        if df.empty or len(df) < 5:
            return {'status': 'INSUFFICIENT_DATA'}
        
        patterns_found = []
        recent_candles = df.tail(5)
        
        for i, (ts, candle) in enumerate(recent_candles.iterrows()):
            # Check if near support
            near_support = abs(candle['low'] - support_level) / support_level <= 0.01
            
            if not near_support:
                continue
            
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            body = abs(c - o)
            total_range = h - l
            
            if total_range == 0:
                continue
            
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            
            # Hammer/Doji patterns
            if lower_shadow > 2 * body and upper_shadow <= body:
                patterns_found.append(('HAMMER', ts, 0.8))
            elif body <= 0.1 * total_range:
                patterns_found.append(('DOJI', ts, 0.6))
            
            # Bullish engulfing (need previous candle)
            if i > 0:
                prev_candle = recent_candles.iloc[i-1]
                prev_o, prev_c = prev_candle['open'], prev_candle['close']
                
                if (prev_c < prev_o and c > o and  # Previous red, current green
                    o < prev_c and c > prev_o):     # Current engulfs previous
                    patterns_found.append(('BULLISH_ENGULFING', ts, 0.9))
        
        if patterns_found:
            # Get strongest pattern
            strongest = max(patterns_found, key=lambda x: x[2])
            status = f'FOUND_{strongest[0]}'
            confidence = strongest[2]
        else:
            status = 'NO_PATTERNS'
            confidence = 0.0
        
        return {
            'status': status,
            'patterns': patterns_found,
            'confidence': confidence
        }

    def _check_enhanced_divergences(self, df: pd.DataFrame, support_level: float) -> Dict:
        """Enhanced divergence detection."""
        if df.empty or len(df) < 30:
            return {'status': 'INSUFFICIENT_DATA'}
        
        # Look for price vs RSI divergence
        lookback = min(40, len(df))
        recent_data = df.tail(lookback)
        
        if 'rsi' not in recent_data.columns:
            return {'status': 'NO_RSI_DATA'}
        
        # Find swing lows in both price and RSI
        price_lows = []
        rsi_lows = []
        
        for i in range(2, len(recent_data)-2):
            price = recent_data['low'].iloc[i]
            rsi = recent_data['rsi'].iloc[i]
            
            # Check if it's a local low
            if (price < recent_data['low'].iloc[i-1] and price < recent_data['low'].iloc[i+1] and
                price < recent_data['low'].iloc[i-2] and price < recent_data['low'].iloc[i+2]):
                price_lows.append((i, price))
                rsi_lows.append((i, rsi))
        
        # Check for divergence in last two lows
        divergences_found = []
        if len(price_lows) >= 2:
            last_two_price = price_lows[-2:]
            last_two_rsi = rsi_lows[-2:]
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (last_two_price[1][1] < last_two_price[0][1] and  # Price: lower low
                last_two_rsi[1][1] > last_two_rsi[0][1]):        # RSI: higher low
                
                # Check if recent low is near support
                recent_low_price = last_two_price[1][1]
                if abs(recent_low_price - support_level) / support_level <= 0.015:
                    divergences_found.append('BULLISH_RSI_DIVERGENCE')
        
        status = 'BULLISH_DIVERGENCE' if divergences_found else 'NO_DIVERGENCE'
        
        return {
            'status': status,
            'divergences': divergences_found,
            'price_lows_count': len(price_lows)
        }

    def _check_market_microstructure(self, df_1m: pd.DataFrame, support_level: float) -> Dict:
        """Analyze market microstructure on 1m timeframe."""
        if df_1m.empty:
            return {'status': 'INSUFFICIENT_DATA'}
        
        recent_1m = df_1m.tail(20)
        
        # Order flow analysis
        buying_pressure_avg = recent_1m['buying_pressure'].mean() if 'buying_pressure' in recent_1m.columns else 0.5
        
        # Price action near support
        touches_support = 0
        bounces_from_support = 0
        
        for i in range(len(recent_1m)):
            candle = recent_1m.iloc[i]
            if abs(candle['low'] - support_level) / support_level <= 0.005:  # Within 0.5%
                touches_support += 1
                # Check if next candle bounced
                if i < len(recent_1m) - 1:
                    next_candle = recent_1m.iloc[i + 1]
                    if next_candle['close'] > candle['low']:
                        bounces_from_support += 1
        
        # Volume spikes near support
        volume_spikes = 0
        if 'volume_ratio' in recent_1m.columns:
            for i in range(len(recent_1m)):
                candle = recent_1m.iloc[i]
                if (abs(candle['low'] - support_level) / support_level <= 0.005 and
                    candle.get('volume_ratio', 1) > 1.5):
                    volume_spikes += 1
        
        # Overall microstructure assessment
        if buying_pressure_avg > 0.6 and bounces_from_support >= touches_support * 0.7:
            status = 'BULLISH_STRUCTURE'
        elif buying_pressure_avg > 0.55:
            status = 'NEUTRAL_BULLISH'
        elif buying_pressure_avg < 0.4:
            status = 'BEARISH_STRUCTURE'
        else:
            status = 'NEUTRAL'
        
        return {
            'status': status,
            'buying_pressure': buying_pressure_avg,
            'support_touches': touches_support,
            'support_bounces': bounces_from_support,
            'volume_spikes': volume_spikes
        }

    def _generate_final_enhanced_signal(self, entry_strategy: Dict, confirmations: Dict) -> Dict:
        """Generate final signal based on all analysis."""
        
        # Base confidence from setup quality
        base_confidence = min(90, entry_strategy['setup_quality'])
        
        # Confirmation bonuses
        confirmation_bonus = 0
        
        # Volume confirmation
        vol_status = confirmations.get('volume_confirmation', {}).get('status', '')
        if vol_status == 'STRONG_BULLISH':
            confirmation_bonus += 8
        elif vol_status == 'BULLISH':
            confirmation_bonus += 5
        
        # Momentum alignment
        mom_status = confirmations.get('momentum_alignment', {}).get('status', '')
        if mom_status == 'STRONG_ALIGNMENT':
            confirmation_bonus += 7
        elif mom_status == 'GOOD_ALIGNMENT':
            confirmation_bonus += 4
        
        # Pattern confirmation
        pattern_status = confirmations.get('candlestick_patterns', {}).get('status', '')
        if 'FOUND_' in pattern_status:
            pattern_conf = confirmations['candlestick_patterns'].get('confidence', 0)
            confirmation_bonus += int(pattern_conf * 6)
        
        # Divergence bonus
        div_status = confirmations.get('divergence_analysis', {}).get('status', '')
        if div_status == 'BULLISH_DIVERGENCE':
            confirmation_bonus += 5
        
        # Microstructure bonus
        micro_status = confirmations.get('microstructure', {}).get('status', '')
        if micro_status == 'BULLISH_STRUCTURE':
            confirmation_bonus += 4
        
        # Final confidence calculation
        final_confidence = min(98, base_confidence + confirmation_bonus)
        
        # Signal determination
        timing = entry_strategy.get('entry_timing', {}).get('timing', '')
        
        if final_confidence >= 85 and timing in ['IMMEDIATE', 'PREPARE_ENTRY']:
            signal = 'STRONG_BUY'
        elif final_confidence >= 70 and timing in ['IMMEDIATE', 'PREPARE_ENTRY', 'WAIT_FOR_BOUNCE']:
            signal = 'BUY'
        elif final_confidence >= 60:
            signal = 'WATCH'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'confidence': final_confidence,
            'base_confidence': base_confidence,
            'confirmation_bonus': confirmation_bonus
        }

    # ---------------------------
    # Enhanced Plotting
    # ---------------------------
    def create_enhanced_plot(self, analysis_data: Dict) -> Tuple[Optional[plt.Figure], Optional[list]]:
        """Create enhanced multi-panel plot with comprehensive analysis visualization."""
        
        try:
            multi_tf_data = analysis_data['multi_tf_data']
            sr_levels = analysis_data['sr_levels']
            entry_strategy = analysis_data['entry_strategy']
            confirmations = analysis_data['confirmations']
            
            # Use 5m data for main plotting
            df = multi_tf_data.get('5m', pd.DataFrame())
            if df.empty:
                print("No 5m data available for plotting")
                return None, None
            
            # Prepare data (last 100 candles)
            plot_df = df.tail(100).copy()
            
            # Rename columns for mplfinance
            if 'open' in plot_df.columns:
                plot_df = plot_df.rename(columns={
                    'open': 'Open', 'high': 'High', 'low': 'Low', 
                    'close': 'Close', 'volume': 'Volume'
                })
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Define the plot layout
            gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], width_ratios=[3, 1],
                                hspace=0.3, wspace=0.3)
            
            # Main price chart
            ax_main = fig.add_subplot(gs[0, :])
            
            # Technical indicators
            ax_rsi = fig.add_subplot(gs[1, 0])
            ax_macd = fig.add_subplot(gs[2, 0])
            ax_volume = fig.add_subplot(gs[3, 0])
            
            # Analysis summary panel
            ax_summary = fig.add_subplot(gs[:, 1])
            
            # Plot candlesticks manually
            self._plot_candlesticks(ax_main, plot_df)
            
            # Add S/R levels
            self._add_sr_levels_to_plot(ax_main, sr_levels, plot_df)
            
            # Add entry/stop/target levels
            if entry_strategy['entry_valid']:
                self._add_entry_levels_to_plot(ax_main, entry_strategy, plot_df)
            
            # Add pattern markers
            self._add_pattern_markers(ax_main, plot_df, confirmations, sr_levels)
            
            # Plot technical indicators
            self._plot_technical_indicators(plot_df, ax_rsi, ax_macd, ax_volume)
            
            # Add analysis summary
            self._add_analysis_summary(ax_summary, analysis_data)
            
            # Formatting
            ax_main.set_title(f'{self.trading_pair} - Enhanced Multi-Timeframe Analysis (5m)', 
                            fontsize=16, fontweight='bold')
            ax_main.grid(True, alpha=0.3)
            ax_main.set_ylabel('Price', fontsize=12)
            
            # Remove x-axis labels from upper plots
            for ax in [ax_main, ax_rsi, ax_macd]:
                ax.set_xticklabels([])
            
            plt.tight_layout()
            return fig, [ax_main, ax_rsi, ax_macd, ax_volume, ax_summary]
            
        except Exception as e:
            print(f"Error creating enhanced plot: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _plot_candlesticks(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot candlesticks manually."""
        for i, (ts, row) in enumerate(df.iterrows()):
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Color determination
            color = 'green' if close_price > open_price else 'red'
            
            # Draw the wick
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.8)
            
            # Draw the body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Set x-axis limits
        ax.set_xlim(-1, len(df))

    def _add_sr_levels_to_plot(self, ax: plt.Axes, sr_levels: Dict, plot_df: pd.DataFrame):
        """Add S/R levels to the plot with enhanced visualization."""
        
        x_min, x_max = -1, len(plot_df)
        
        # Support levels
        for i, support in enumerate(sr_levels['supports'][:5]):
            level = support['level']
            strength = support['total_strength']
            methods = len(support['methods'])
            
            # Line thickness based on strength
            linewidth = 1 + min(3, strength / 5)
            alpha = 0.6 + min(0.4, methods / 5)
            
            ax.axhline(y=level, color='green', linestyle='--', 
                      linewidth=linewidth, alpha=alpha, label=f'S{i+1}')
            
            # Add label
            ax.text(x_max * 0.98, level, f'S{i+1}: {level:.6f}', 
                   verticalalignment='center', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   fontsize=9)
        
        # Resistance levels
        for i, resistance in enumerate(sr_levels['resistances'][:5]):
            level = resistance['level']
            strength = resistance['total_strength']
            methods = len(resistance['methods'])
            
            linewidth = 1 + min(3, strength / 5)
            alpha = 0.6 + min(0.4, methods / 5)
            
            ax.axhline(y=level, color='red', linestyle='--', 
                      linewidth=linewidth, alpha=alpha, label=f'R{i+1}')
            
            ax.text(x_max * 0.98, level, f'R{i+1}: {level:.6f}', 
                   verticalalignment='center', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                   fontsize=9)

    def _add_entry_levels_to_plot(self, ax: plt.Axes, entry_strategy: Dict, plot_df: pd.DataFrame):
        """Add entry, stop, and target levels to the plot."""
        
        x_max = len(plot_df)
        
        # Entry level
        if entry_strategy.get('risk_management', {}).get('entry_price'):
            entry_price = entry_strategy['risk_management']['entry_price']
            ax.axhline(y=entry_price, color='blue', linestyle='-', linewidth=2, alpha=0.8)
            ax.text(x_max * 0.02, entry_price, 'ENTRY', 
                   verticalalignment='center', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        # Stop loss
        if entry_strategy.get('risk_management', {}).get('primary_stop'):
            stop_price = entry_strategy['risk_management']['primary_stop']
            ax.axhline(y=stop_price, color='darkred', linestyle='-', linewidth=2, alpha=0.8)
            ax.text(x_max * 0.02, stop_price, 'STOP', 
                   verticalalignment='center', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8),
                   fontsize=10, fontweight='bold')
        
        # Targets
        targets = entry_strategy.get('targets', [])
        colors = ['darkgreen', 'green', 'lightgreen']
        for i, target in enumerate(targets[:3]):
            if target['reward_risk_ratio'] >= self.config['min_reward_risk']:
                target_price = target['price']
                color = colors[min(i, len(colors)-1)]
                ax.axhline(y=target_price, color=color, linestyle='-', linewidth=2, alpha=0.8)
                ax.text(x_max * 0.02, target_price, f'T{i+1}', 
                       verticalalignment='center', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                       fontsize=10, fontweight='bold')

    def _add_pattern_markers(self, ax: plt.Axes, plot_df: pd.DataFrame, confirmations: Dict, sr_levels: Dict):
        """Add pattern and confirmation markers."""
        
        # Divergence lines
        div_data = confirmations.get('divergence_analysis', {})
        if div_data.get('status') == 'BULLISH_DIVERGENCE':
            # Add a text marker for divergence
            ax.text(len(plot_df) * 0.5, sr_levels['current_price'] * 0.995, 
                   'BULLISH DIVERGENCE', 
                   horizontalalignment='center', verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                   fontsize=11, fontweight='bold')
        
        # Pattern markers
        pattern_data = confirmations.get('candlestick_patterns', {})
        if 'FOUND_' in pattern_data.get('status', ''):
            pattern_type = pattern_data['status'].replace('FOUND_', '')
            # Find recent pattern location (approximate)
            recent_low = plot_df['Low'].tail(10).min()
            ax.scatter(len(plot_df) - 5, recent_low, color='gold', s=200, marker='^', 
                      zorder=5, alpha=0.9, label=pattern_type)

    def _plot_technical_indicators(self, plot_df: pd.DataFrame, ax_rsi: plt.Axes, 
                                 ax_macd: plt.Axes, ax_volume: plt.Axes):
        """Plot technical indicators."""
        
        x_axis = range(len(plot_df))
        
        # RSI
        if 'rsi' in plot_df.columns:
            ax_rsi.plot(x_axis, plot_df['rsi'], color='purple', linewidth=1.5)
            ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3)
        
        # MACD
        if 'macd' in plot_df.columns and 'macd_signal' in plot_df.columns:
            ax_macd.plot(x_axis, plot_df['macd'], color='blue', linewidth=1.2, label='MACD')
            ax_macd.plot(x_axis, plot_df['macd_signal'], color='red', linewidth=1.2, label='Signal')
            histogram = plot_df['macd'] - plot_df['macd_signal']
            ax_macd.bar(x_axis, histogram, alpha=0.3, color='gray', width=0.8)
            ax_macd.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax_macd.set_ylabel('MACD')
            ax_macd.legend(loc='upper right', fontsize=8)
            ax_macd.grid(True, alpha=0.3)
        
        # Volume
        volume_colors = ['green' if plot_df['Close'].iloc[i] > plot_df['Open'].iloc[i] 
                        else 'red' for i in range(len(plot_df))]
        ax_volume.bar(x_axis, plot_df['Volume'], color=volume_colors, alpha=0.6, width=0.8)
        
        # Volume moving average
        if 'volume_sma' in plot_df.columns:
            ax_volume.plot(x_axis, plot_df['volume_sma'], color='orange', linewidth=1.5, 
                          label='Vol SMA')
            ax_volume.legend(loc='upper right', fontsize=8)
        
        ax_volume.set_ylabel('Volume')
        ax_volume.set_xlabel('Time')
        ax_volume.grid(True, alpha=0.3)

    def _add_analysis_summary(self, ax: plt.Axes, analysis_data: Dict):
        """Add comprehensive analysis summary panel."""
        
        ax.axis('off')
        
        # Get data
        final_signal = analysis_data['final_signal']
        entry_strategy = analysis_data['entry_strategy']
        confirmations = analysis_data['confirmations']
        sr_levels = analysis_data['sr_levels']
        
        # Title
        title = f"🚀 ENHANCED ANALYSIS SUMMARY 🚀"
        ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        
        # Signal box
        signal = final_signal['signal']
        confidence = final_signal['confidence']
        signal_color = {'STRONG_BUY': 'darkgreen', 'BUY': 'green', 'WATCH': 'orange', 'HOLD': 'red'}.get(signal, 'gray')
        
        signal_text = f"SIGNAL: {signal}\nCONFIDENCE: {confidence:.1f}%"
        ax.text(0.5, 0.85, signal_text, ha='center', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=signal_color, alpha=0.2),
               transform=ax.transAxes)
        
        # Setup quality
        quality = entry_strategy.get('setup_quality', 0)
        grade = entry_strategy.get('quality_grade', 'N/A')
        ax.text(0.5, 0.72, f"Setup Quality: {quality:.1f}/100 (Grade: {grade})",
               ha='center', va='top', fontsize=11, transform=ax.transAxes)
        
        # Key levels
        y_pos = 0.65
        ax.text(0.05, y_pos, "🎯 KEY LEVELS:", ha='left', va='top', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        
        if entry_strategy['entry_valid']:
            risk_data = entry_strategy['risk_management']
            y_pos -= 0.05
            ax.text(0.05, y_pos, f"Entry: ${risk_data['entry_price']:.6f}", ha='left', va='top',
                   fontsize=10, color='blue', transform=ax.transAxes)
            y_pos -= 0.04
            ax.text(0.05, y_pos, f"Stop: ${risk_data['primary_stop']:.6f}", ha='left', va='top',
                   fontsize=10, color='red', transform=ax.transAxes)
            y_pos -= 0.04
            ax.text(0.05, y_pos, f"Risk: {risk_data['risk_pct']:.2f}%", ha='left', va='top',
                   fontsize=10, transform=ax.transAxes)
        
        # Best targets
        targets = entry_strategy.get('targets', [])
        best_targets = [t for t in targets if t['reward_risk_ratio'] >= self.config['min_reward_risk']][:2]
        for i, target in enumerate(best_targets):
            y_pos -= 0.04
            ax.text(0.05, y_pos, f"T{i+1}: ${target['price']:.6f} (R:R {target['reward_risk_ratio']:.2f})", 
                   ha='left', va='top', fontsize=10, color='green', transform=ax.transAxes)
        
        # Confirmations
        y_pos -= 0.08
        ax.text(0.05, y_pos, "✅ CONFIRMATIONS:", ha='left', va='top', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        
        conf_items = [
            ('Volume', confirmations.get('volume_confirmation', {}).get('status', 'N/A')),
            ('Momentum', confirmations.get('momentum_alignment', {}).get('status', 'N/A')),
            ('Patterns', confirmations.get('candlestick_patterns', {}).get('status', 'N/A')),
            ('Divergence', confirmations.get('divergence_analysis', {}).get('status', 'N/A')),
        ]
        
        for name, status in conf_items:
            y_pos -= 0.04
            status_color = 'green' if any(x in status for x in ['BULLISH', 'STRONG', 'FOUND', 'GOOD']) else 'red' if 'NO' in status else 'orange'
            ax.text(0.05, y_pos, f"{name}: {status.replace('_', ' ')}", ha='left', va='top',
                   fontsize=9, color=status_color, transform=ax.transAxes)
        
        # S/R Summary
        y_pos -= 0.08
        ax.text(0.05, y_pos, "📊 S/R ANALYSIS:", ha='left', va='top', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        y_pos -= 0.04
        ax.text(0.05, y_pos, f"Supports: {len(sr_levels['supports'])} | Resistances: {len(sr_levels['resistances'])}", 
               ha='left', va='top', fontsize=9, transform=ax.transAxes)
        
        if sr_levels['supports']:
            best_support = sr_levels['supports'][0]
            y_pos -= 0.04
            ax.text(0.05, y_pos, f"Best Support: ${best_support['level']:.6f} (Str: {best_support['total_strength']:.1f})", 
                   ha='left', va='top', fontsize=9, color='green', transform=ax.transAxes)
        
        # Recommendation
        y_pos -= 0.08
        recommendation = entry_strategy.get('recommendation', 'N/A')
        ax.text(0.05, y_pos, "💡 RECOMMENDATION:", ha='left', va='top', fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        y_pos -= 0.04
        # Split long recommendations into multiple lines
        if len(recommendation) > 40:
            words = recommendation.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 40:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                ax.text(0.05, y_pos, line, ha='left', va='top', fontsize=9, 
                       color='darkblue', transform=ax.transAxes)
                y_pos -= 0.04
        else:
            ax.text(0.05, y_pos, recommendation, ha='left', va='top', fontsize=9, 
                   color='darkblue', transform=ax.transAxes)

    # ---------------------------
    # Main Runner
    # ---------------------------
    def run_enhanced_analysis(self):
        """Main execution function."""
        try:
            # Run the enhanced analysis
            result = self.run_enhanced_market_analysis()
            
            if "error" in result:
                print(f"Analysis Error: {result['error']}")
                return result
            
            # Create and save enhanced plot
            if hasattr(self, '_last_analysis_data'):
                try:
                    fig, axes = self.create_enhanced_plot(self._last_analysis_data)
                    if fig is not None:
                        # Save the enhanced plot
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"{self.trading_pair.replace('/', '_')}_enhanced_analysis_{timestamp}.png"
                        fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
                        print(f"\n📊 Enhanced analysis plot saved as: {filename}")
                        plt.close(fig)
                    else:
                        print("⚠ Could not generate plot")
                except Exception as e:
                    print(f"Plotting error: {e}")
            
            return result
            
        except Exception as e:
            print(f"Critical error in enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Critical analysis failure: {str(e)}"}


def main():
    """Enhanced main function."""
    print("🚀 ENHANCED TRADING BOT WITH ADVANCED ANALYSIS 🚀")
    print("=" * 70)

    # Get trading pair from user
    if len(sys.argv) > 1:
        trading_pair = sys.argv[1]
    else:
        trading_pair = input("Enter trading pair (e.g., BTC/USDT): ").strip().upper()

    if not trading_pair or '/' not in trading_pair:
        print("Error: Valid trading pair required (e.g., BTC/USDT)")
        sys.exit(1)

    try:
        # Initialize enhanced bot
        bot = EnhancedMomentumBot(trading_pair)

        print(f"\n🔧 Enhanced Configuration Summary:")
        key_configs = [
            'sr_lookback_periods', 'sr_strength_threshold', 'min_reward_risk',
            'max_risk_per_trade', 'atr_period', 'confluence_distance'
        ]
        for key in key_configs:
            print(f"  {key}: {bot.config.get(key, 'N/A')}")

        print(f"\n🚀 Starting Enhanced Multi-Timeframe Analysis...")
        print(f"📊 Analyzing: {trading_pair}")
        print(f"⏱ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run enhanced analysis
        result = bot.run_enhanced_analysis()
        
        # Final summary
        if result and "error" not in result:
            print(f"\n🎉 ENHANCED ANALYSIS COMPLETE! 🎉")
            print(f"📈 Final Signal: {result.get('signal', 'N/A')}")
            print(f"🎯 Confidence: {result.get('confidence', 0):.1f}%")
            print(f"⭐ Setup Quality: {result.get('setup_quality', 0):.1f}/100 ({result.get('quality_grade', 'N/A')})")
            print(f"💡 Recommendation: {result.get('recommendation', 'N/A')}")
        else:
            print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")

    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()