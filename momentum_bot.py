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
import talib
import matplotlib.pyplot as plt

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')


class EnhancedMomentumBot:
    """
    Enhanced Trading Bot with Advanced Support/Resistance Analysis
    
    Features:
    - Automatic support/resistance identification with confluence scoring
    - Multiple confirmation signals required for entry (HTF trend, divergence, candlesticks)
    - Dynamic ATR-based risk management
    - High-probability setup identification
    """
    
    def __init__(self, trading_pair: str, exchange_name: str = 'binance'):
        self.trading_pair = trading_pair
        self.exchange_name = exchange_name
        
        # Configuration parameters
        # In the __init__ method
        self.config = {
            # General Parameters
            'trend_tolerance': 0.02, # Kept the same
            'strong_buy_threshold': 80,
            'buy_threshold': 65,
            'volume_confirmation': True,

            # --- SCALPING S/R PARAMETERS ---
            'sr_lookback_periods': 75,          # Reduced lookback for faster analysis on 5m chart
            'sr_min_touches': 2,
            'sr_proximity_threshold': 0.005,    # Initial threshold, but will be made dynamic
            'fibonacci_enabled': False,         # DISABLED: Fibonacci is too noisy on low timeframes
            'pivot_order': 3,                   # Reduced order for more sensitive peak detection
            'sr_strength_threshold': 2,         # REDUCED: S/R levels on 5m are naturally weaker
            'entry_buffer': 0.0005,             # REDUCED: 0.05% buffer for a tight entry
            
            # --- SCALPING RISK & STRATEGY PARAMETERS ---
            'min_pullback_pct': 0.0015,         # REDUCED: Wait for a smaller 0.15% pullback
            'atr_period': 10,                   # Reduced ATR period for more reactivity
            'atr_stop_multiplier': 1.5,         # TIGHTER STOP: Use a smaller ATR multiplier for stops
            'min_reward_risk_ratio': 1.2,       # REDUCED: A 1.2 R:R is more realistic for scalping
            
            # Unchanged Parameters
            'crossover_lookback': 5,
            'rsi_threshold_high': 70,
            'rsi_threshold_low': 30,
            'confluence_boost_factor': 2.5,
            'time_decay_factor': 0.95,
            'candlestick_confirmation': True,
            'higher_tf_trend_filter': True,
            'divergence_confirmation': True,
            'min_confluence_distance': 0.001,   # REDUCED: 0.1% threshold for confluence
        }
        
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
    
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> float:
        """
        Calculate Average True Range for dynamic volatility measurement.
        """
        if period is None:
            period = self.config['atr_period']
        
        if len(df) < period + 1:
            return 0.0
        
        df = df.copy()
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        
        # Calculate ATR with Simple Moving Average
        atr = np.mean(tr[-period:])
        
        return atr
    
    def get_volatility_adjusted_threshold(self, df: pd.DataFrame) -> float:
        """
        Calculate a volatility-adjusted proximity threshold using ATR.
        """
        default_threshold = self.config['sr_proximity_threshold']
        atr = self.calculate_atr(df)
        
        if atr == 0:
            return default_threshold
        
        # Get current price level
        current_price = df['close'].iloc[-1]
        
        # Convert ATR to a percentage of current price
        atr_pct = atr / current_price
        
        # Scale the threshold: higher volatility = wider threshold
        # Base threshold + adjustment factor * ATR percentage
        adjusted_threshold = default_threshold + (0.5 * atr_pct)
        
        # Ensure threshold is reasonable (not too small or large)
        return max(0.003, min(adjusted_threshold, 0.03))
    
    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Identify swing highs and lows using scipy's peak detection.
        
        Returns:
            Tuple of (swing_highs, swing_lows) with (index, price) tuples
        """
        order = self.config['pivot_order']
        
        # Find swing highs
        high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
        swing_highs = [(df.index[i], df['high'].iloc[i]) for i in high_indices]
        
        # Find swing lows
        low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
        swing_lows = [(df.index[i], df['low'].iloc[i]) for i in low_indices]
        
        return swing_highs, swing_lows
    
    def calculate_support_resistance_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate support and resistance levels using multiple methods.
        
        Returns:
            Dictionary with support/resistance levels and their strengths
        """
        swing_highs, swing_lows = self.identify_swing_points(df)
        
        # Method 1: Swing Point Clustering
        support_levels = self._cluster_price_levels(df, [price for _, price in swing_lows])
        resistance_levels = self._cluster_price_levels(df, [price for _, price in swing_highs])
        
        # Method 2: Psychological Levels (round numbers)
        psychological_levels = self._find_psychological_levels(df['close'].iloc[-1])
        
        # Method 3: Volume Profile (enhanced)
        volume_levels = self._calculate_volume_profile(df)
        
        # Method 4: Fibonacci Retracements (enhanced with extensions)
        fibonacci_levels = {}
        if self.config['fibonacci_enabled'] and len(swing_highs) > 0 and len(swing_lows) > 0:
            fibonacci_levels = self._calculate_fibonacci_levels(swing_highs, swing_lows)
        
        # Combine and score all levels
        all_levels = {
            'support_swing': support_levels,
            'resistance_swing': resistance_levels,
            'psychological': psychological_levels,
            'volume_profile': volume_levels,
            'fibonacci': fibonacci_levels
        }
        
        # Score and filter levels with enhanced confluence detection
        scored_levels = self._score_sr_levels(df, all_levels)
        
        return scored_levels
    
    def _cluster_price_levels(self, df: pd.DataFrame, prices: List[float]) -> List[Dict]:
        """
        Cluster similar price levels together with adaptive proximity threshold.
        
        Returns:
            List of support/resistance levels with metadata
        """
        if not prices:
            return []
        
        prices = sorted(prices)
        levels = []
        
        # Use volatility-adjusted threshold
        proximity_threshold = self.get_volatility_adjusted_threshold(df)
        
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            # Check if price is within proximity of current cluster
            cluster_avg = sum(current_cluster) / len(current_cluster)
            
            if abs(price - cluster_avg) / cluster_avg <= proximity_threshold:
                current_cluster.append(price)
            else:
                # Finalize current cluster
                if len(current_cluster) >= self.config['sr_min_touches']:
                    levels.append({
                        'level': sum(current_cluster) / len(current_cluster),
                        'strength': len(current_cluster),
                        'touches': current_cluster.copy(),
                        'type': 'swing_cluster'
                    })
                current_cluster = [price]
        
        # Don't forget the last cluster
        if len(current_cluster) >= self.config['sr_min_touches']:
            levels.append({
                'level': sum(current_cluster) / len(current_cluster),
                'strength': len(current_cluster),
                'touches': current_cluster.copy(),
                'type': 'swing_cluster'
            })
        
        return levels
    
    def _find_psychological_levels(self, current_price: float) -> List[Dict]:
        """
        Find psychological support/resistance levels (round numbers).
        """
        levels = []
        
        # Determine the appropriate round number based on price magnitude
        if current_price >= 1000:
            increments = [50, 100, 500, 1000]
        elif current_price >= 100:
            increments = [5, 10, 25, 50, 100]
        elif current_price >= 10:
            increments = [0.5, 1, 2.5, 5, 10]
        elif current_price >= 1:
            increments = [0.05, 0.1, 0.25, 0.5, 1]
        else:
            increments = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
        
        for increment in increments:
            # Find nearest round levels above and below current price
            lower_level = (int(current_price / increment)) * increment
            upper_level = lower_level + increment
            
            # Add levels within reasonable range (±20% of current price)
            price_range = current_price * 0.2
            
            for level in [lower_level, upper_level]:
                if abs(level - current_price) <= price_range and level > 0:
                    # Assign higher strength to "rounder" numbers (larger increments)
                    strength_modifier = min(3, 1 + np.log10(increment) + 1) if increment >= 1 else 1
                    
                    levels.append({
                        'level': level,
                        'strength': 1 + strength_modifier,  # Base strength for psychological levels
                        'type': 'psychological',
                        'increment': increment
                    })
        
        return levels
    
    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> List[Dict]:
        """
        Calculate enhanced volume profile with weighted recency.
        """
        if df.empty or len(df) < 10:
            return []
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        volume_profile = []
        recency_weight = 1.0  # More recent periods have higher weight
        
        # Create recency weights (more recent bars have higher weight)
        num_periods = len(df)
        recency_weights = np.linspace(0.5, 1.0, num_periods)
        weighted_volume = df['volume'] * recency_weights
        
        for i in range(bins):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_mid = (bin_low + bin_high) / 2
            
            # Calculate weighted volume for this price range
            mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
            bin_volume = weighted_volume.loc[mask].sum()
            
            if bin_volume > 0:
                volume_profile.append({
                    'level': bin_mid,
                    'strength': bin_volume / weighted_volume.sum() * 10,  # Normalize and scale
                    'type': 'volume_profile',
                    'volume': bin_volume
                })
        
        # Sort by volume and return top levels
        volume_profile.sort(key=lambda x: x['volume'], reverse=True)
        return volume_profile[:8]  # Return top 8 volume levels
    
    def _calculate_fibonacci_levels(self, swing_highs: List[Tuple], swing_lows: List[Tuple]) -> Dict:
        """
        Calculate enhanced Fibonacci retracement and extension levels.
        """
        if not swing_highs or not swing_lows:
            return {}
        
        # Find the most recent significant swing points
        # Sort by recency, then by significance (price extremity)
        recent_highs = sorted(swing_highs, key=lambda x: x[0], reverse=True)[:5]
        recent_lows = sorted(swing_lows, key=lambda x: x[0], reverse=True)[:5]
        
        if not recent_highs or not recent_lows:
            return {}
            
        # Find the most recent swing high and low for trend direction
        most_recent_high = recent_highs[0]
        most_recent_low = recent_lows[0]
        
        # Determine the most significant swing points for Fibonacci calculation
        highest_high = max(recent_highs, key=lambda x: x[1])
        lowest_low = min(recent_lows, key=lambda x: x[1])
        
        # Fibonacci ratios (retracements and extensions)
        fib_ratios = {
            'r0': 0.0,
            'r236': 0.236,
            'r382': 0.382,
            'r5': 0.5,
            'r618': 0.618,
            'r786': 0.786,
            'r1': 1.0,
            'e1618': 1.618,  # Extension
            'e2618': 2.618   # Extension
        }
        
        levels = {}
        
        # If most recent swing is a high, calculate retracements downward
        if most_recent_high[0] > most_recent_low[0]:
            high_price = highest_high[1]
            low_price = lowest_low[1]
            
            for key, ratio in fib_ratios.items():
                if 'e' in key:  # Extension level
                    fib_level = low_price - (high_price - low_price) * (ratio - 1)
                else:  # Retracement level
                    fib_level = high_price - (high_price - low_price) * ratio
                
                # Assign strength based on common Fibonacci levels
                if ratio in [0.0, 0.5, 1.0]:
                    strength = 4.0  # Strongest levels
                elif ratio in [0.382, 0.618]:
                    strength = 3.5  # Very strong levels
                else:
                    strength = 2.5  # Standard levels
                
                levels[f'fib_{key}'] = {
                    'level': fib_level,
                    'strength': strength,
                    'type': 'fibonacci',
                    'ratio': ratio,
                    'direction': 'down'
                }
        # If most recent swing is a low, calculate retracements upward
        else:
            high_price = highest_high[1]
            low_price = lowest_low[1]
            
            for key, ratio in fib_ratios.items():
                if 'e' in key:  # Extension level
                    fib_level = high_price + (high_price - low_price) * (ratio - 1)
                else:  # Retracement level
                    fib_level = low_price + (high_price - low_price) * ratio
                
                # Assign strength based on common Fibonacci levels
                if ratio in [0.0, 0.5, 1.0]:
                    strength = 4.0  # Strongest levels
                elif ratio in [0.382, 0.618]:
                    strength = 3.5  # Very strong levels
                else:
                    strength = 2.5  # Standard levels
                
                levels[f'fib_{key}'] = {
                    'level': fib_level,
                    'strength': strength,
                    'type': 'fibonacci',
                    'ratio': ratio,
                    'direction': 'up'
                }
        
        return levels
    
    def _score_sr_levels(self, df: pd.DataFrame, all_levels: Dict) -> Dict:
        """
        Score all support/resistance levels with enhanced confluence detection.
        """
        current_price = df['close'].iloc[-1]
        
        # Flatten all levels into a single list
        all_flattened_levels = []
        for category, levels in all_levels.items():
            if isinstance(levels, list):
                for level in levels:
                    level_copy = level.copy()
                    level_copy['category'] = category
                    all_flattened_levels.append(level_copy)
            elif isinstance(levels, dict):
                for key, level in levels.items():
                    level_copy = level.copy()
                    level_copy['category'] = f"{category}_{key}"
                    all_flattened_levels.append(level_copy)
        
        # First pass: calculate the base strength for each level
        for level_info in all_flattened_levels:
            level_price = level_info['level']
            
            # Time-weighted strength for recent touches
            recent_touches_score = self._calculate_time_weighted_touches(df, level_price)
            level_info['recent_touches_score'] = recent_touches_score
            
            # Base strength + recent touches contribution
            level_info['base_strength'] = level_info['strength'] + recent_touches_score
            
            # Distance from current price
            level_info['distance_pct'] = abs(current_price - level_price) / current_price
            
            # Classification as support or resistance
            level_info['role'] = 'support' if level_price < current_price else 'resistance'
        
        # Second pass: detect confluence between different methods
        # Group nearby levels for confluence detection
        min_confluence_distance = self.config['min_confluence_distance']
        confluence_boost = self.config['confluence_boost_factor']
        
        # Sort by price for easier clustering
        all_flattened_levels.sort(key=lambda x: x['level'])
        
        # Find confluence clusters
        clusters = []
        current_cluster = []
        
        for i, level_info in enumerate(all_flattened_levels):
            if not current_cluster:
                current_cluster.append(level_info)
            else:
                prev_level = current_cluster[-1]['level']
                current_level = level_info['level']
                
                # If this level is close to the previous one, add to current cluster
                if abs(current_level - prev_level) / current_price <= min_confluence_distance:
                    current_cluster.append(level_info)
                else:
                    # Finalize current cluster and start a new one
                    if len(current_cluster) > 1:  # Only consider clusters with multiple levels
                        clusters.append(current_cluster)
                    current_cluster = [level_info]
        
        # Don't forget the last cluster
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        # Apply confluence boost to clusters
        for cluster in clusters:
            # Calculate average level price
            avg_price = sum(item['level'] for item in cluster) / len(cluster)
            
            # Count unique methods in this cluster
            unique_methods = set(item['category'].split('_')[0] for item in cluster)
            
            # Boost is stronger when more diverse methods confirm the level
            method_diversity_factor = min(len(unique_methods) * 0.5, 1.0)
            total_boost = confluence_boost * method_diversity_factor
            
            # Apply boost to each level in the cluster
            for level_info in cluster:
                level_info['confluence_boost'] = total_boost
                level_info['unique_methods_count'] = len(unique_methods)
                level_info['is_confluence_zone'] = True
        
        # Apply final scoring
        for level_info in all_flattened_levels:
            # Apply confluence boost if applicable
            if 'confluence_boost' in level_info:
                level_info['total_strength'] = level_info['base_strength'] * level_info['confluence_boost']
            else:
                level_info['total_strength'] = level_info['base_strength']
                level_info['is_confluence_zone'] = False
                level_info['unique_methods_count'] = 1
        
        # Separate into final supports and resistances
        supports = [l for l in all_flattened_levels if l['role'] == 'support']
        resistances = [l for l in all_flattened_levels if l['role'] == 'resistance']
        
        # Sort by strength and proximity
        supports.sort(key=lambda x: (x['distance_pct'], -x['total_strength']))
        resistances.sort(key=lambda x: (x['distance_pct'], -x['total_strength']))
        
        # Filter by minimum strength threshold
        min_strength = self.config['sr_strength_threshold']
        valid_supports = [s for s in supports if s['total_strength'] >= min_strength]
        valid_resistances = [r for r in resistances if r['total_strength'] >= min_strength]
        
        return {
            'supports': valid_supports[:8],  # Top 8 support levels
            'resistances': valid_resistances[:8],  # Top 8 resistance levels
            'current_price': current_price
        }
    
    def _calculate_time_weighted_touches(self, df: pd.DataFrame, level: float, lookback: int = 30) -> float:
        """
        Calculate time-weighted touches with exponential decay factor.
        More recent touches have higher weight.
        """
        if lookback > len(df):
            lookback = len(df)
        
        recent_data = df.tail(lookback)
        proximity = level * self.get_volatility_adjusted_threshold(df)
        
        total_weighted_touches = 0
        decay_factor = self.config['time_decay_factor']
        
        # More recent candles have higher weight (exponential decay)
        for i, (idx, candle) in enumerate(recent_data.iterrows()):
            weight = decay_factor ** (lookback - i - 1)  # Newer touches get higher weight
            
            if (candle['low'] <= level + proximity and 
                candle['high'] >= level - proximity):
                # Check how strong the touch is (closer = stronger)
                distance = min(
                    abs(candle['high'] - level),
                    abs(candle['low'] - level)
                )
                proximity_factor = 1.0 - (distance / (proximity + 1e-10))
                
                # Consider volume as an additional factor
                volume_factor = 1.0
                if 'volume' in candle and np.isfinite(candle['volume']) and candle['volume'] > 0:
                    avg_volume = df['volume'].mean()
                    volume_ratio = candle['volume'] / avg_volume
                    volume_factor = min(1.5, max(0.5, volume_ratio))
                
                # Combine all factors
                touch_score = weight * proximity_factor * volume_factor
                total_weighted_touches += touch_score
        
        return total_weighted_touches
    
    def detect_higher_timeframe_trend(self, timeframe: str = '4h') -> Dict:
        """
        Detect trend on higher timeframe for filtering trades.
        """
        try:
            df_htf = self.fetch_ohlcv_data(timeframe, limit=200)
            
            if len(df_htf) < 50:
                return {'trend': 'UNKNOWN', 'reason': 'Insufficient data'}
            
            # Calculate EMAs
            df_htf['ema50'] = df_htf['close'].ewm(span=50, adjust=False).mean()
            df_htf['ema200'] = df_htf['close'].ewm(span=200, adjust=False).mean()
            
            last_candle = df_htf.iloc[-1]
            price = last_candle['close']
            ema50 = last_candle['ema50']
            ema200 = last_candle['ema200']
            
            # Calculate price slopes for trending strength
            short_slope = (df_htf['ema50'].iloc[-1] - df_htf['ema50'].iloc[-10]) / df_htf['ema50'].iloc[-10]
            
            # Calculate trend strength
            trend_score = 0
            trend = 'NEUTRAL'
            reason = []
            
            # Determine trend based on price vs EMAs and EMA alignment
            if price > ema50 and ema50 > ema200:
                trend_score += 40
                reason.append("Price > EMA50 > EMA200")
                
                if short_slope > 0.005:
                    trend_score += 30
                    reason.append("Strong upward EMA slope")
                elif short_slope > 0:
                    trend_score += 15
                    reason.append("Positive EMA slope")
            
            elif price < ema50 and ema50 < ema200:
                trend_score -= 40
                reason.append("Price < EMA50 < EMA200")
                
                if short_slope < -0.005:
                    trend_score -= 30
                    reason.append("Strong downward EMA slope")
                elif short_slope < 0:
                    trend_score -= 15
                    reason.append("Negative EMA slope")
            
            # Final trend determination
            if trend_score >= 50:
                trend = 'STRONG_UPTREND'
            elif trend_score >= 20:
                trend = 'UPTREND'
            elif trend_score <= -50:
                trend = 'STRONG_DOWNTREND'
            elif trend_score <= -20:
                trend = 'DOWNTREND'
            
            return {
                'trend': trend,
                'score': trend_score,
                'reason': ', '.join(reason),
                'price': price,
                'ema50': ema50,
                'ema200': ema200
            }
            
        except Exception as e:
            print(f"Error detecting higher timeframe trend: {e}")
            return {'trend': 'ERROR', 'reason': str(e)}
    
    def detect_bullish_divergence(self, df: pd.DataFrame, support_level: float, lookback: int = 20) -> Dict:
        """
        Detect bullish divergence between price and RSI near support levels.
        """
        if len(df) < lookback + 14:  # Need enough data for RSI calculation
            return {'divergence': False, 'reason': 'Insufficient data'}
        
        # Calculate RSI
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Filter to lookback period
        recent_df = df.tail(lookback)
        
        # Find local price lows near the support level
        proximity_threshold = support_level * self.get_volatility_adjusted_threshold(df)
        
        price_lows = []
        for i in range(1, len(recent_df) - 1):
            if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and
                recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1] and
                abs(recent_df['low'].iloc[i] - support_level) < proximity_threshold * 3):
                price_lows.append((i, recent_df.index[i], recent_df['low'].iloc[i], recent_df['rsi'].iloc[i]))
        
        if len(price_lows) < 2:
            return {'divergence': False, 'reason': 'Insufficient price lows near support'}
        
        # Check for bullish divergence (lower price lows but higher RSI lows)
        divergences = []
        for i in range(len(price_lows) - 1):
            for j in range(i + 1, len(price_lows)):
                idx1, timestamp1, price1, rsi1 = price_lows[i]
                idx2, timestamp2, price2, rsi2 = price_lows[j]
                
                # Regular bullish divergence: price makes lower low but RSI makes higher low
                if price2 < price1 and rsi2 > rsi1:
                    strength = min(2.0, 1 + (rsi2 - rsi1) / 20)  # Bigger RSI difference = stronger signal
                    divergences.append({
                        'type': 'regular',
                        'strength': strength,
                        'price_diff_pct': (price1 - price2) / price1 * 100,
                        'rsi_diff': rsi2 - rsi1,
                        'recent_candle_idx': max(idx1, idx2)
                    })
                
                # Hidden bullish divergence: price makes higher low but RSI makes lower low
                elif price2 > price1 and rsi2 < rsi1:
                    strength = min(1.5, 0.7 + (price2 - price1) / price1 * 20)
                    divergences.append({
                        'type': 'hidden',
                        'strength': strength,
                        'price_diff_pct': (price2 - price1) / price1 * 100,
                        'rsi_diff': rsi1 - rsi2,
                        'recent_candle_idx': max(idx1, idx2)
                    })
        
        if divergences:
            # Sort by strength and recency
            divergences.sort(key=lambda x: (x['recent_candle_idx'], x['strength']), reverse=True)
            best_div = divergences[0]
            
            return {
                'divergence': True,
                'type': best_div['type'],
                'strength': best_div['strength'],
                'price_diff_pct': best_div['price_diff_pct'],
                'rsi_diff': best_div['rsi_diff']
            }
        
        return {'divergence': False, 'reason': 'No divergence detected'}
    
    def detect_candlestick_pattern(self, df: pd.DataFrame) -> Dict:
        """
        Detect bullish reversal candlestick patterns.
        """
        if len(df) < 3:
            return {'pattern': 'UNKNOWN', 'strength': 0}
                
        recent_candles = df.tail(3)
        
        # Extract open, high, low, close values for the most recent candles
        open_prices = recent_candles['open'].values
        high_prices = recent_candles['high'].values
        low_prices = recent_candles['low'].values
        close_prices = recent_candles['close'].values
        
        # Current candle (most recent)
        c_open = open_prices[-1]
        c_high = high_prices[-1]
        c_low = low_prices[-1]
        c_close = close_prices[-1]
        
        # Previous candle
        p_open = open_prices[-2]
        p_high = high_prices[-2]
        p_low = low_prices[-2]
        p_close = close_prices[-2]
        
        # Calculate candle body and shadow sizes
        c_body_size = abs(c_close - c_open)
        c_total_size = c_high - c_low
        c_upper_shadow = c_high - max(c_open, c_close)
        c_lower_shadow = min(c_open, c_close) - c_low
        
        p_body_size = abs(p_close - p_open)
        
        # Initialize pattern variables
        pattern = 'NONE'
        strength = 0
        
        # 1. Bullish Hammer
        if (c_close > c_open and  # Bullish candle
            c_lower_shadow > c_body_size * 2 and  # Long lower shadow
            c_upper_shadow < c_body_size * 0.5 and  # Short upper shadow
            c_low < p_low):  # New low
            
            pattern = 'HAMMER'
            strength = 8
        
        # 2. Bullish Engulfing
        elif (c_close > c_open and  # Current candle is bullish
            p_close < p_open and  # Previous candle is bearish
            c_open < p_close and  # Current open below previous close
            c_close > p_open and  # Current close above previous open
            c_body_size > p_body_size * 0.8):  # Current body engulfs previous
            
            pattern = 'BULLISH_ENGULFING'
            strength = 9
        
        # 3. Doji (near support)
        elif (c_body_size < c_total_size * 0.1 and  # Very small body
            c_total_size > 0):  # Avoid division by zero
            
            pattern = 'DOJI'
            strength = 5
        
        # 4. Piercing Line
        elif (c_close > c_open and  # Current candle is bullish
            p_close < p_open and  # Previous candle is bearish
            c_open < p_close and  # Current open below previous close
            c_close > (p_open + p_close) / 2):  # Closed above midpoint
            
            pattern = 'PIERCING_LINE'
            strength = 7
        
        # 5. Morning Star
        elif (len(recent_candles) >= 3 and
            close_prices[-3] < open_prices[-3] and  # First candle bearish
            abs(close_prices[-2] - open_prices[-2]) < p_body_size * 0.3 and  # Second candle small body
            c_close > c_open and  # Third candle bullish
            c_close > (open_prices[-3] + close_prices[-3]) / 2):  # Closed into first candle
            
            pattern = 'MORNING_STAR'
            strength = 10
        
        # 6. Tweezer Bottom
        elif (p_close < p_open and  # Previous candle bearish
            c_close > c_open and  # Current candle bullish
            abs(p_low - c_low) < c_total_size * 0.1):  # Similar lows
            
            pattern = 'TWEEZER_BOTTOM'
            strength = 6
        
        # 7. Three White Soldiers
        elif (len(recent_candles) >= 3 and
            all(close_prices[-i] > open_prices[-i] for i in range(1, min(4, len(close_prices) + 1))) and  # Bullish candles
            all(i < len(close_prices)-1 and close_prices[-i] > close_prices[-i-1] for i in range(1, min(3, len(close_prices)))) and  # Each closes higher
            all(i < len(open_prices)-1 and open_prices[-i] > open_prices[-i-1] for i in range(1, min(3, len(open_prices))))):  # Each opens higher
            
            pattern = 'THREE_WHITE_SOLDIERS'
            strength = 10
        
        # 8. Bullish Harami
        elif (p_close < p_open and  # Previous candle bearish
            c_close > c_open and  # Current candle bullish
            c_open > p_close and  # Current open above previous close
            c_close < p_open):  # Current close below previous open
            
            pattern = 'BULLISH_HARAMI'
            strength = 6
        
        # Adjust strength based on confirmation factors
        
        # Volume confirmation (if current volume > previous volume)
        if 'volume' in df.columns and df['volume'].iloc[-1] > df['volume'].iloc[-2] * 1.2:
            strength += 2
            
        # Trend confirmation (if this is a reversal of recent downtrend)
        # FIXED: Make sure we don't go out of bounds
        lookback = min(6, len(df))
        if lookback > 2:  # Ensure we have at least 3 candles (current + 2 previous) for trend check
            try:
                if all(df['close'].iloc[-i] < df['close'].iloc[-i-1] for i in range(2, lookback)):
                    strength += 2
            except IndexError:
                # Safely handle any unexpected index errors
                pass
        
        return {
            'pattern': pattern,
            'strength': strength
        }
    
    def calculate_entry_strategy(self, sr_levels: Dict, df_entry: pd.DataFrame, htf_trend: Dict) -> Dict:
        """
        Calculate enhanced entry strategy with dynamic ATR-based stops.
        
        Returns:
            Dictionary with entry point, stop loss, and targets
        """
        current_price = sr_levels['current_price']
        supports = sr_levels['supports']
        resistances = sr_levels['resistances']
        
        if not supports:
            return {
                'entry_valid': False,
                'reason': 'No valid support levels found'
            }
        
        # Find the strongest nearby support
        min_pullback = self.config['min_pullback_pct']
        valid_supports = []
        for support in supports:
            # Look for support within a reasonable range (e.g., 0.5% to 7% below current price)
            # This filters out supports the price is already sitting on.
            if min_pullback <= support['distance_pct'] <= 0.07:
                valid_supports.append(support)
        
        if not valid_supports:
            return {
                'entry_valid': False,
                'reason': 'No nearby support levels found'
            }
        
        # Sort by confluence and strength
        valid_supports.sort(key=lambda x: (not x.get('is_confluence_zone', False), 
                                          x['distance_pct'], 
                                          -x['total_strength']))
        
        nearest_support = valid_supports[0]
        support_level = nearest_support['level']
        entry_buffer = self.config['entry_buffer']
        
        # Calculate entry price (slightly above support)
        entry_price = support_level * (1 + entry_buffer)
        
        # Calculate ATR for dynamic stop loss
        atr = self.calculate_atr(df_entry)
        atr_multiplier = self.config['atr_stop_multiplier']
        
        # Calculate stop loss using ATR (more adaptive to market volatility)
        if atr > 0:
            stop_loss = support_level - (atr * atr_multiplier)
        else:
            # Fallback to fixed percentage if ATR calculation fails
            stop_loss = support_level * 0.985
        
        # Calculate risk (as percentage)
        risk_amount = entry_price - stop_loss
        risk_pct = (risk_amount / entry_price) * 100
        
        # Check if HTF trend aligns with the trade direction
        trend_aligned = htf_trend['trend'] in ['UPTREND', 'STRONG_UPTREND']
        
        # Calculate targets based on nearest resistances
        targets = []
        min_reward_risk = self.config['min_reward_risk_ratio']
        
        if resistances:
            for i, resistance in enumerate(resistances[:4]):  # Use first 4 resistances as targets
                target_price = resistance['level']
                risk = entry_price - stop_loss
                reward = target_price - entry_price
                
                if reward > 0:
                    reward_risk_ratio = reward / risk if risk > 0 else 0
                    targets.append({
                        'target': i + 1,
                        'price': target_price,
                        'reward_risk_ratio': reward_risk_ratio,
                        'potential_profit_pct': ((target_price - entry_price) / entry_price) * 100,
                        'resistance_strength': resistance.get('total_strength', 0),
                        'is_confluence': resistance.get('is_confluence_zone', False)
                    })
        
        # Check for bullish divergence at support
        divergence_result = self.detect_bullish_divergence(df_entry, support_level)
        
        # Check for bullish candlestick patterns
        candlestick_result = self.detect_candlestick_pattern(df_entry)
        
        # Determine if we have a high-probability setup
        has_min_reward_risk = any(t['reward_risk_ratio'] >= min_reward_risk for t in targets) if targets else False
        
        # Create setup quality assessment
        setup_quality = {
            'htf_trend_aligned': trend_aligned,
            'has_bullish_divergence': divergence_result.get('divergence', False),
            'has_candlestick_confirmation': candlestick_result['pattern'] != 'NONE' and candlestick_result['strength'] >= 6,
            'is_confluence_support': nearest_support.get('is_confluence_zone', False),
            'support_strength': nearest_support.get('total_strength', 0),
            'has_min_reward_risk': has_min_reward_risk
        }
        
        # Calculate overall setup score
        setup_score = 0
        if setup_quality['htf_trend_aligned']:
            setup_score += 25
        if setup_quality['has_bullish_divergence']:
            setup_score += 20
        if setup_quality['has_candlestick_confirmation']:
            setup_score += 15
        if setup_quality['is_confluence_support']:
            setup_score += 20
        if setup_quality['support_strength'] > 5:
            setup_score += 10
        if setup_quality['has_min_reward_risk']:
            setup_score += 10
        
        setup_quality['score'] = setup_score
        setup_quality['high_probability'] = setup_score >= 65
        
        return {
            'entry_valid': True,
            'support_level': support_level,
            'support_strength': nearest_support['total_strength'],
            'support_type': nearest_support.get('type', 'unknown'),
            'support_is_confluence': nearest_support.get('is_confluence_zone', False),
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_pct': risk_pct,
            'targets': targets,
            'distance_to_support_pct': nearest_support['distance_pct'] * 100,
            'atr': atr,
            'bullish_divergence': divergence_result,
            'candlestick_pattern': candlestick_result,
            'htf_trend': htf_trend['trend'],
            'setup_quality': setup_quality,
            'recommendation': self._generate_advanced_recommendation(
                current_price, entry_price, targets, setup_quality, 
                divergence_result, candlestick_result, htf_trend,
                {'support_level': support_level} # Pass the support level here
            )
        }
    
    def _generate_advanced_recommendation(self, 
                                       current_price: float, 
                                       entry_price: float, 
                                       targets: List[Dict], 
                                       setup_quality: Dict,
                                       divergence_result: Dict,
                                       candlestick_result: Dict,
                                       htf_trend: Dict,
                                       entry_strategy: Dict
                                       ) -> str:
        """Generate detailed entry recommendation based on advanced analysis."""
        price_diff_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Start with empty reasons list
        reasons = []
        
        # Check if we have any targets
        if not targets:
            return "HOLD - No clear resistance targets identified"
        
        # Get best target by reward:risk
        best_target = max(targets, key=lambda x: x['reward_risk_ratio']) if targets else None
        min_rr = self.config['min_reward_risk_ratio']
        support_level = entry_strategy['support_level']

        def get_confirmation_reasons():
            """Helper function to gather positive signals."""
            conf_reasons = []
            if setup_quality['htf_trend_aligned']:
                conf_reasons.append(f"HTF trend confirms ({htf_trend['trend']})")
            if setup_quality['is_confluence_support']:
                conf_reasons.append("Multi-method confluence support")
            if setup_quality['has_bullish_divergence']:
                div_type = divergence_result.get('type', 'regular')
                conf_reasons.append(f"{div_type.capitalize()} bullish divergence")
            if setup_quality['has_candlestick_confirmation']:
                conf_reasons.append(f"{candlestick_result['pattern']} pattern")
            if best_target and best_target['reward_risk_ratio'] >= min_rr:
                conf_reasons.append(f"Favorable R:R ({best_target['reward_risk_ratio']:.2f})")
            return conf_reasons

        if setup_quality['high_probability']:
            reasons = get_confirmation_reasons()
            if current_price > entry_price:
                signal = "WAIT"
                reasons.insert(0, f"Wait for price to pull back to the entry zone near ${entry_price:.6f}")
            elif current_price >= support_level:
                signal = "STRONG_BUY"
                reasons.insert(0, "Price has entered the ideal buy zone")
            else:
                signal = "HOLD"
                reasons.insert(0, f"Price is below support of ${support_level:.6f}. Awaiting reclaim.")

        elif setup_quality.get('score', 0) >= 50 and setup_quality['htf_trend_aligned']:
            reasons = get_confirmation_reasons()
            if current_price > entry_price:
                signal = "MONITOR"
                reasons.insert(0, f"Decent setup. Monitor for a pullback to entry near ${entry_price:.6f}")
            elif current_price >= support_level:
                signal = "BUY"
                reasons.insert(0, "Price has entered entry zone for a moderate-conviction setup")
            else:
                signal = "HOLD"
                reasons.insert(0, "Price has fallen below support.")
                
        else:
            signal = "HOLD"
            if not setup_quality['htf_trend_aligned']:
                reasons.append(f"Counter-trend setup ({htf_trend['trend']})")
            if best_target and best_target['reward_risk_ratio'] < min_rr:
                reasons.append(f"Poor R:R ({best_target['reward_risk_ratio']:.2f})")
            if not reasons:
                reasons.append("Insufficient confirmation signals for a trade.")

        return f"{signal} - {'; '.join(reasons)}"
    
    def analyze_market_with_sr(self) -> Dict:
        """
        Perform complete market analysis with advanced support/resistance integration.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*70}")
        print(f"Advanced Market Analysis with Enhanced Support/Resistance - {timestamp}")
        print(f"Trading Pair: {self.trading_pair}")
        print(f"{'='*70}")
        
        # Fetch data for various timeframes
        df_5m_structure = self.fetch_ohlcv_data('5m', limit=150)  # Use 5m for S/R structure
        df_1m_entry = self.fetch_ohlcv_data('1m', limit=200)      # Use 1m for entry signals
        
        if df_5m_structure.empty or df_1m_entry.empty:
            return {"error": "Insufficient data for analysis"}
        
        print("Phase 1: Analyzing Higher Timeframe Trend...")
        # Check higher timeframe trend (4h) for filtering
        htf_trend = self.detect_higher_timeframe_trend('15m')
        
        print(f"  Higher Timeframe Trend: {htf_trend['trend']}")
        print(f"  Reason: {htf_trend.get('reason', 'Not available')}")
        
        print("\nPhase 2: Calculating Enhanced Support/Resistance Levels...")
        
        # Calculate S/R levels on 1h timeframe for better reliability
        sr_levels = self.calculate_support_resistance_levels(df_5m_structure)
        
        # Display S/R levels
        print(f"\n📊 Support Levels Found:")
        for i, support in enumerate(sr_levels['supports']):
            confluence = "✓" if support.get('is_confluence_zone', False) else " "
            methods = support.get('unique_methods_count', 1)
            print(f"  S{i+1}: ${support['level']:.6f} (Strength: {support['total_strength']:.1f}, "
                  f"Distance: {support['distance_pct']*100:.2f}%, Type: {support['type']}, "
                  f"Confluence: {confluence} Methods: {methods})")
        
        print(f"\n📊 Resistance Levels Found:")
        for i, resistance in enumerate(sr_levels['resistances']):
            confluence = "✓" if resistance.get('is_confluence_zone', False) else " "
            methods = resistance.get('unique_methods_count', 1)
            print(f"  R{i+1}: ${resistance['level']:.6f} (Strength: {resistance['total_strength']:.1f}, "
                  f"Distance: {resistance['distance_pct']*100:.2f}%, Type: {resistance['type']}, "
                  f"Confluence: {confluence} Methods: {methods})")
        
        print("\nPhase 3: Calculating Advanced Entry Strategy...")
        entry_strategy = self.calculate_entry_strategy(sr_levels, df_1m_entry, htf_trend)
        
        if not entry_strategy['entry_valid']:
            print(f"⚠️ No valid entry: {entry_strategy.get('reason', 'Unknown reason')}")
            return {
                "signal": "HOLD",
                "reason": entry_strategy.get('reason', 'No valid entry setup'),
                "sr_levels": sr_levels,
                "htf_trend": htf_trend,
                "timestamp": timestamp
            }
        
        # Display entry strategy
        print(f"\n🎯 Entry Strategy Details:")
        print(f"  Support Level: ${entry_strategy['support_level']:.6f} ({entry_strategy['support_type']})")
        print(f"  Support Strength: {entry_strategy['support_strength']:.1f}")
        print(f"  Confluence Zone: {'Yes' if entry_strategy['support_is_confluence'] else 'No'}")
        print(f"  Entry Price: ${entry_strategy['entry_price']:.6f}")
        print(f"  ATR-based Stop Loss: ${entry_strategy['stop_loss']:.6f}")
        print(f"  Risk: {entry_strategy['risk_pct']:.2f}%")
        print(f"  Distance to Support: {entry_strategy['distance_to_support_pct']:.2f}%")
        
        # Display confirmations
        print(f"\n✅ Confirmations:")
        print(f"  Higher Timeframe Trend: {entry_strategy['htf_trend']}")
        
        if entry_strategy['bullish_divergence'].get('divergence', False):
            print(f"  Bullish Divergence: Yes ({entry_strategy['bullish_divergence'].get('type', 'regular')})")
        else:
            print(f"  Bullish Divergence: No")
        
        pattern = entry_strategy['candlestick_pattern']
        if pattern['pattern'] != 'NONE':
            print(f"  Candlestick Pattern: {pattern['pattern']} (Strength: {pattern['strength']}/10)")
        else:
            print(f"  Candlestick Pattern: None")
        
        # Display targets
        if entry_strategy['targets']:
            print(f"\n🎯 Targets:")
            for target in entry_strategy['targets']:
                confluence = "✓" if target.get('is_confluence', False) else " "
                print(f"  Target {target['target']}: ${target['price']:.6f} "
                      f"(R:R {target['reward_risk_ratio']:.2f}, "
                      f"Profit: {target['potential_profit_pct']:.1f}%, "
                      f"Strength: {target['resistance_strength']:.1f}, "
                      f"Confluence: {confluence})")
        
        # Display setup quality assessment
        setup = entry_strategy['setup_quality']
        print(f"\n📝 Setup Quality Assessment:")
        print(f"  Overall Score: {setup['score']}/100")
        print(f"  High-Probability Setup: {'Yes' if setup['high_probability'] else 'No'}")
        print(f"  HTF Trend Aligned: {'Yes' if setup['htf_trend_aligned'] else 'No'}")
        print(f"  Strong Confluence Support: {'Yes' if setup['is_confluence_support'] else 'No'}")
        print(f"  Bullish Divergence: {'Yes' if setup['has_bullish_divergence'] else 'No'}")
        print(f"  Candlestick Confirmation: {'Yes' if setup['has_candlestick_confirmation'] else 'No'}")
        print(f"  Favorable Reward-Risk: {'Yes' if setup['has_min_reward_risk'] else 'No'}")
        
        # Perform traditional momentum analysis on 5m for confirmation
        print("\nPhase 4: Momentum Confirmation Analysis...")
        momentum_result = self.analyze_momentum_signals(df_1m_entry)
        print(f"  Momentum Signal: {momentum_result['momentum_signal']}")
        print(f"  Momentum Score: {momentum_result['momentum_score']}/100")
        print(f"  RSI: {momentum_result['rsi']:.1f}")
        
        # Combine S/R analysis with momentum signals to generate final signal
        print("\nPhase 5: Final Signal Generation...")
        combined_signal = self._combine_sr_momentum_signals(entry_strategy, momentum_result)
        
        result = {
            "signal": combined_signal['signal'],
            "confidence": combined_signal['confidence'],
            "entry_strategy": entry_strategy,
            "momentum_analysis": momentum_result,
            "sr_levels": sr_levels,
            "htf_trend": htf_trend,
            "recommendation": entry_strategy['recommendation'],
            "timestamp": timestamp
        }
        
        return result
    
    def analyze_momentum_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum signals with additional indicators."""
        if len(df) < 21:
            return {"momentum_score": 0, "momentum_signal": "INSUFFICIENT_DATA"}
        
        # Calculate indicators
        df = df.copy()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        score = 0
        signals = []
        
        # EMA alignment
        if last_candle['ema_9'] > last_candle['ema_21']:
            score += 20
            signals.append("EMA9 > EMA21")
        else:
            signals.append("EMA9 < EMA21")
        
        # RSI check
        if 40 <= last_candle['rsi'] <= 60:
            score += 15  # Neutral territory
            signals.append(f"RSI neutral ({last_candle['rsi']:.1f})")
        elif 30 <= last_candle['rsi'] < 40:
            score += 10  # Oversold, potential bounce
            signals.append(f"RSI oversold ({last_candle['rsi']:.1f})")
        elif 60 < last_candle['rsi'] <= 70:
            score += 5   # Overbought but still bullish
            signals.append(f"RSI overbought ({last_candle['rsi']:.1f})")
        elif last_candle['rsi'] < 30:
            score += 5   # Heavily oversold
            signals.append(f"RSI extremely oversold ({last_candle['rsi']:.1f})")
        
        # MACD check
        if last_candle['macd_hist'] > 0:
            score += 15
            signals.append("MACD histogram positive")
            
            # MACD histogram increasing (additional bullishness)
            if last_candle['macd_hist'] > prev_candle['macd_hist']:
                score += 10
                signals.append("MACD histogram increasing")
        else:
            signals.append("MACD histogram negative")
        
        # Price vs EMA
        if last_candle['close'] > last_candle['ema_9']:
            score += 15
            signals.append("Price > EMA9")
        
        # Recent momentum (slope of price)
        recent_returns = df['close'].pct_change(3).iloc[-1] * 100
        if recent_returns > 1:
            score += 10
            signals.append(f"Strong recent uptrend ({recent_returns:.1f}%)")
        elif recent_returns > 0:
            score += 5
            signals.append(f"Mild recent uptrend ({recent_returns:.1f}%)")
        else:
            signals.append(f"Recent downtrend ({recent_returns:.1f}%)")
        
        # Volume trend
        if 'volume' in df.columns:
            recent_volume_ratio = df['volume'].iloc[-1] / df['volume'].iloc[-5:-1].mean()
            if recent_volume_ratio > 1.5 and last_candle['close'] > last_candle['open']:
                score += 10
                signals.append("Strong volume on bullish candle")
            elif recent_volume_ratio > 1.2 and last_candle['close'] > last_candle['open']:
                score += 5
                signals.append("Above average volume")
        
        # Determine final momentum signal
        if score >= 70:
            momentum_signal = "STRONGLY_BULLISH"
        elif score >= 50:
            momentum_signal = "BULLISH"
        elif score >= 30:
            momentum_signal = "NEUTRAL"
        else:
            momentum_signal = "BEARISH"
        
        return {
            "momentum_score": score,
            "momentum_signal": momentum_signal,
            "ema_9": last_candle['ema_9'],
            "ema_21": last_candle['ema_21'],
            "rsi": last_candle['rsi'],
            "macd": last_candle['macd'],
            "macd_signal": last_candle['macd_signal'],
            "macd_hist": last_candle['macd_hist'],
            "signals": signals
        }
    
    def _calculate_macd(self, price_series, fast=12, slow=26, signal=9):
        """Calculate MACD, Signal and Histogram."""
        ema_fast = price_series.ewm(span=fast, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, np.finfo(float).eps)
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _combine_sr_momentum_signals(self, entry_strategy: Dict, momentum_result: Dict) -> Dict:
        """
        Advanced signal generation based on stringent rule-based criteria.
        """
        # Start with a base confidence of 0
        confidence = 0
        setup_quality = entry_strategy.get('setup_quality', {})
        
        # Rule-based signal generation
        if not entry_strategy['entry_valid']:
            signal = "HOLD"
            confidence = 10
            reason = "No valid entry strategy"
        
        # STRONG_BUY requires all high-probability criteria to be met
        elif (setup_quality.get('high_probability', False) and
              setup_quality.get('htf_trend_aligned', False) and
              (setup_quality.get('has_bullish_divergence', False) or 
               setup_quality.get('has_candlestick_confirmation', False)) and
              setup_quality.get('has_min_reward_risk', False) and
              momentum_result['momentum_signal'] in ['BULLISH', 'STRONGLY_BULLISH']):
            
            signal = "STRONG_BUY"
            confidence = 90 + min(10, setup_quality.get('score', 0) - 65)  # Additional boost based on setup score
            reason = "High-probability setup with all confirmation criteria"
        
        # BUY requires higher timeframe trend and at least one confirmation
        elif (setup_quality.get('htf_trend_aligned', False) and 
              setup_quality.get('has_min_reward_risk', False) and
              (setup_quality.get('has_bullish_divergence', False) or 
               setup_quality.get('has_candlestick_confirmation', False) or
               setup_quality.get('is_confluence_support', False)) and
              momentum_result['momentum_signal'] in ['NEUTRAL', 'BULLISH', 'STRONGLY_BULLISH']):
            
            signal = "BUY"
            confidence = 65 + min(10, setup_quality.get('score', 0) - 40)
            reason = "Good setup with trend alignment and technical confirmation"
        
        # WAIT signal for setups that have potential but need more confirmation
        elif (setup_quality.get('htf_trend_aligned', False) or
              setup_quality.get('has_bullish_divergence', False) or
              (setup_quality.get('is_confluence_support', False) and 
               momentum_result['momentum_signal'] != 'BEARISH')):
            
            signal = "WAIT"
            confidence = 40 + min(15, setup_quality.get('score', 0) - 20)
            reason = "Partial confirmation - monitor for additional signals"
        
        # HOLD signal when conditions are not favorable
        else:
            signal = "HOLD"
            confidence = 20
            reason = "Insufficient confirmation signals"
            
            # Add specific reasons
            if not setup_quality.get('htf_trend_aligned', False):
                reason += "; Counter-trend setup"
                
            if momentum_result['momentum_signal'] == 'BEARISH':
                reason += "; Bearish momentum"
        
        return {
            "signal": signal,
            "confidence": min(100, max(0, confidence)),
            "reason": reason,
            "sr_contribution": setup_quality.get('score', 0) * 0.6,  # 60% weight to S/R analysis
            "momentum_contribution": momentum_result['momentum_score'] * 0.4  # 40% weight to momentum
        }
    
    def run_enhanced_analysis(self):
        """Run enhanced analysis with S/R levels."""
        result = self.analyze_market_with_sr()
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return result
        
        print(f"\n{'='*50}")
        print(f"🚨 FINAL ENHANCED RESULT 🚨")
        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Recommendation: {result['recommendation']}")
        
        if result.get('entry_strategy', {}).get('entry_valid', False):
            entry = result['entry_strategy']
            print(f"Entry Price: ${entry['entry_price']:.6f}")
            print(f"Stop Loss: ${entry['stop_loss']:.6f} (Dynamic ATR-based)")
            print(f"Risk: {entry['risk_pct']:.2f}%")
            
            # Show quality metrics for the setup
            setup = entry.get('setup_quality', {})
            if setup:
                print(f"\nSetup Quality Score: {setup.get('score', 0)}/100")
                print(f"High-Probability Setup: {'Yes' if setup.get('high_probability', False) else 'No'}")
            
            if entry.get('targets', []):
                best_target = max(entry['targets'], key=lambda x: x['reward_risk_ratio'])
                print(f"Best Target: ${best_target['price']:.6f} (R:R {best_target['reward_risk_ratio']:.2f})")
            
            # Show confirmation signals
            confirmations = []
            if setup.get('htf_trend_aligned', False):
                confirmations.append("Higher TF Trend")
            if setup.get('has_bullish_divergence', False):
                confirmations.append("Bullish Divergence")
            if setup.get('has_candlestick_confirmation', False):
                confirmations.append("Candlestick Pattern")
            if setup.get('is_confluence_support', False):
                confirmations.append("Confluence Support")
            
            if confirmations:
                print(f"Confirming Signals: {', '.join(confirmations)}")
        
        print(f"{'='*50}")
        
        return result


def main():
    """Main function to run the enhanced bot."""
    print("Enhanced Trading Bot with Advanced Support/Resistance Analysis")
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
        bot = EnhancedMomentumBot(trading_pair)
        
        print(f"\nEnhanced Configuration:")
        sr_config = {k: v for k, v in bot.config.items() if 'sr_' in k or 'fibonacci' in k or 'pivot' in k or 'atr' in k}
        for key, value in sr_config.items():
            print(f"  {key}: {value}")
        
        bot.run_enhanced_analysis()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()