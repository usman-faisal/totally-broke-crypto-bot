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
    Enhanced Trading Bot with institutional-grade quantitative analysis.
    
    This bot moves beyond simple indicators to a sophisticated, multi-layered
    approach for generating high-probability trading signals.

    Key Enhancements:
    1.  **Confluence-Based S/R Analysis**: Identifies and scores S/R zones where multiple
        analytical methods (swing points, volume profile, Fibonacci) converge.
    2.  **Adaptive S/R Detection**: Uses Average True Range (ATR) to dynamically adjust
        clustering sensitivity based on market volatility.
    3.  **Time-Weighted Strength**: Gives more importance to recent price interactions with S/R levels.
    4.  **Higher-Timeframe Trend Filter**: Ensures trades are only taken in alignment with the
        dominant market trend (e.g., daily or 4-hour).
    5.  **Advanced Confirmation Signals**:
        - **Bullish RSI Divergence**: Detects potential trend reversals near key support.
        - **Candlestick Pattern Recognition**: Waits for bullish confirmation patterns (e.g., Hammer, Engulfing)
          before signaling an entry.
    6.  **Dynamic Risk Management**:
        - **ATR-Based Stop-Loss**: Sets stop-losses outside of typical market noise.
        - **Reward/Risk Validation**: Ensures potential trades meet a minimum R:R ratio before they are considered valid.
    7.  **Rule-Based Signal Engine**: Replaces a simplistic scoring system with a strict,
        multi-conditional logic to generate STRONG_BUY signals only on A+ setups.
    """
    
    def __init__(self, trading_pair: str, exchange_name: str = 'binance'):
        self.trading_pair = trading_pair
        self.exchange_name = exchange_name
        
        # --- Enhanced Configuration Parameters ---
        self.config = {
            # Timeframes
            'higher_timeframe': '4h',             # Timeframe for primary trend analysis
            'entry_timeframe': '5m',              # Timeframe for entry signals
            'sr_timeframe': '1h',                 # Timeframe for robust S/R level calculation

            # Higher-Timeframe Trend Filter
            'htf_ema_short': 50,                  # Short EMA for HTF trend
            'htf_ema_long': 200,                  # Long EMA for HTF trend

            # S/R Analysis & Confluence
            'sr_lookback_periods': 250,           # Increased lookback for more S/R data
            'sr_min_touches': 2,                  # Minimum touches to initially form a cluster
            'pivot_order': 10,                    # Order for scipy peak detection (wider swings)
            'atr_period': 14,                     # Period for ATR calculation
            'atr_clustering_multiplier': 0.5,     # Proximity = 0.5 * ATR. Lower for tighter clusters.
            'confluence_multiplier': 2.5,         # Multiplier for strength of confluence zones
            'time_decay_factor': 0.97,            # Decay for time-weighting (e.g., 0.97^n candles ago)
            'volume_profile_bins': 30,            # Bins for volume profile analysis

            # Fibonacci Analysis
            'fibonacci_enabled': True,            # Enable Fibonacci retracements

            # Confirmation Signals
            'divergence_lookback': 30,            # Lookback period for RSI divergence detection
            'rsi_period': 14,

            # Risk Management
            'atr_stop_loss_multiplier': 2.0,      # Stop Loss = Support - (ATR * 2.0)
            'min_reward_to_risk_ratio': 1.5,      # Minimum R:R for a trade to be valid
            'entry_buffer_atr_multiplier': 0.2,   # Entry buffer = 0.2 * ATR above support
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
            
        print(f"Initialized {self.__class__.__name__} for {self.trading_pair} on {self.exchange_name}")

    def fetch_ohlcv_data(self, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """Fetch OHLCV data and calculate ATR and RSI."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.trading_pair,
                timeframe=timeframe,
                limit=limit
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            if df.empty: return df

            # Calculate essential indicators
            df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], period=self.config['atr_period'])
            df['rsi'] = self.calculate_rsi(df['close'], period=self.config['rsi_period'])
            return df
            
        except Exception as e:
            print(f"Error fetching {timeframe} data: {e}")
            return pd.DataFrame()

    # --- Core S/R Analysis Enhancements ---

    def calculate_support_resistance_levels(self, df: pd.DataFrame) -> Dict:
        """Calculates S/R levels using multiple methods and identifies confluence zones."""
        swing_highs, swing_lows = self.identify_swing_points(df)
        all_levels = []
        
        # Method 1: Swing Points
        all_levels.extend([{'level': p, 'type': 'swing_high'} for _, p in swing_highs])
        all_levels.extend([{'level': p, 'type': 'swing_low'} for _, p in swing_lows])

        # Method 2: Volume Profile
        volume_levels = self._calculate_volume_profile(df)
        all_levels.extend([{'level': lvl['level'], 'type': 'volume'} for lvl in volume_levels])
        
        # Method 3: Fibonacci Retracements
        if self.config['fibonacci_enabled'] and swing_highs and swing_lows:
            fib_levels = self._calculate_fibonacci_levels(df)
            all_levels.extend([{'level': lvl['level'], 'type': 'fibonacci'} for lvl in fib_levels.values()])

        # New Step: Cluster all levels to find confluence and score them
        scored_levels = self._cluster_and_score_levels(df, all_levels)
        
        current_price = df['close'].iloc[-1]
        supports = sorted([lvl for lvl in scored_levels if lvl['level'] < current_price], key=lambda x: x['level'], reverse=True)
        resistances = sorted([lvl for lvl in scored_levels if lvl['level'] >= current_price], key=lambda x: x['level'])
        
        return {
            'supports': supports[:7],
            'resistances': resistances[:7],
            'current_price': current_price
        }

    def _cluster_and_score_levels(self, df: pd.DataFrame, levels: List[Dict]) -> List[Dict]:
        """
        Volatility-adaptive clustering to find confluence zones and score them.
        This is a major enhancement combining Confluence, Time-Weighting, and Volatility-Adaptivity.
        """
        if not levels:
            return []

        atr = df['atr'].iloc[-1]
        # Volatility-Adaptive Proximity Threshold
        proximity_threshold = atr * self.config['atr_clustering_multiplier']
        
        levels.sort(key=lambda x: x['level'])
        
        clustered_zones = []
        current_cluster = [levels[0]]
        
        for level_info in levels[1:]:
            cluster_avg = np.mean([l['level'] for l in current_cluster])
            if abs(level_info['level'] - cluster_avg) <= proximity_threshold:
                current_cluster.append(level_info)
            else:
                clustered_zones.append(current_cluster)
                current_cluster = [level_info]
        clustered_zones.append(current_cluster)

        final_levels = []
        for zone in clustered_zones:
            zone_level = np.mean([l['level'] for l in zone])
            
            # Confluence Scoring
            methods_in_zone = set(l['type'] for l in zone)
            base_strength = len(zone)
            confluence_score = base_strength * self.config['confluence_multiplier'] if len(methods_in_zone) > 1 else base_strength

            # Time-Weighted Touch Scoring
            time_weighted_strength = self._count_recent_touches(df, zone_level, proximity_threshold)
            
            total_strength = confluence_score + time_weighted_strength
            
            final_levels.append({
                'level': zone_level,
                'strength': round(total_strength, 2),
                'methods': list(methods_in_zone),
                'is_confluence': len(methods_in_zone) > 1
            })

        return sorted(final_levels, key=lambda x: x['strength'], reverse=True)

    def _count_recent_touches(self, df: pd.DataFrame, level: float, proximity: float) -> float:
        """Counts recent touches with time-weighted decay."""
        touches_score = 0.0
        decay = self.config['time_decay_factor']
        recent_data = df.tail(self.config['sr_lookback_periods'])
        
        for i, candle in enumerate(recent_data.itertuples()):
            if candle.low <= level <= candle.high:
                candles_ago = len(recent_data) - 1 - i
                weight = decay ** candles_ago
                touches_score += weight

        return touches_score * 5 # Scale the score

    # --- Advanced Confirmation Signal Functions ---

    def _get_higher_timeframe_trend(self, df_htf: pd.DataFrame) -> str:
        """Determines the primary trend from a higher timeframe."""
        if df_htf.empty or len(df_htf) < self.config['htf_ema_long']:
            return 'INSUFFICIENT_DATA'

        short_ema = df_htf['close'].ewm(span=self.config['htf_ema_short'], adjust=False).mean().iloc[-1]
        long_ema = df_htf['close'].ewm(span=self.config['htf_ema_long'], adjust=False).mean().iloc[-1]
        current_price = df_htf['close'].iloc[-1]

        if current_price > short_ema and short_ema > long_ema:
            return 'UPTREND'
        elif current_price < short_ema and short_ema < long_ema:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'

    def _detect_bullish_divergence(self, df: pd.DataFrame) -> bool:
        """Detects bullish divergence between price lows and RSI lows."""
        lookback = self.config['divergence_lookback']
        if len(df) < lookback: return False

        data = df.tail(lookback)
        price_lows_indices = argrelextrema(data['low'].values, np.less, order=5)[0]
        rsi_lows_indices = argrelextrema(data['rsi'].values, np.less, order=5)[0]
        
        if len(price_lows_indices) < 2 or len(rsi_lows_indices) < 2:
            return False

        # Get the last two price and RSI lows
        last_price_low_idx = price_lows_indices[-1]
        prev_price_low_idx = price_lows_indices[-2]
        
        last_rsi_low_idx = -1
        # Find the RSI low that corresponds to the last price low
        for idx in reversed(rsi_lows_indices):
            if abs(idx - last_price_low_idx) < 3: # Allow a small window
                last_rsi_low_idx = idx
                break
        
        prev_rsi_low_idx = -1
        # Find the RSI low that corresponds to the previous price low
        for idx in reversed(rsi_lows_indices):
            if idx < last_rsi_low_idx and abs(idx - prev_price_low_idx) < 3:
                prev_rsi_low_idx = idx
                break

        if last_rsi_low_idx == -1 or prev_rsi_low_idx == -1:
            return False

        # Check for divergence condition
        price_makes_lower_low = data['low'].iloc[last_price_low_idx] < data['low'].iloc[prev_price_low_idx]
        rsi_makes_higher_low = data['rsi'].iloc[last_rsi_low_idx] > data['rsi'].iloc[prev_rsi_low_idx]

        return price_makes_lower_low and rsi_makes_higher_low

    def _identify_bullish_candlestick(self, df: pd.DataFrame) -> Optional[str]:
        """Identifies bullish reversal candlestick patterns on the last closed candle."""
        if len(df) < 2: return None
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body_size = abs(last['open'] - last['close'])
        candle_range = last['high'] - last['low']
        if candle_range == 0: return None
        
        # Hammer
        is_hammer = (
            (last['close'] > last['open']) and
            (body_size / candle_range < 0.3) and
            ((last['high'] - last['close']) / candle_range < 0.2) and
            ((last['open'] - last['low']) / candle_range > 0.6)
        )
        if is_hammer: return "Hammer"
        
        # Bullish Engulfing
        is_bullish_engulfing = (
            prev['close'] < prev['open'] and # Previous candle is red
            last['close'] > last['open'] and  # Current candle is green
            last['close'] > prev['open'] and
            last['open'] < prev['close']
        )
        if is_bullish_engulfing: return "Bullish Engulfing"
        
        # Doji (at support)
        is_doji = body_size / candle_range < 0.1
        if is_doji: return "Doji"

        return None
    
    # --- Improved Risk Management & Trade Setup ---

    def generate_trade_setup(self, sr_levels: Dict, df_entry: pd.DataFrame) -> Dict:
        """
        Calculates a full trade setup with dynamic SL/TP and validates R:R.
        """
        if not sr_levels['supports'] or df_entry.empty:
            return {'valid': False, 'reason': 'No valid support levels or entry data.'}

        current_price = sr_levels['current_price']
        primary_support = sr_levels['supports'][0]
        support_level = primary_support['level']
        
        # Only consider setups close to a strong support level
        if (current_price - support_level) / current_price > 0.03: # Price is >3% away
            return {'valid': False, 'reason': f"Price is too far from primary support ${support_level:.4f}."}

        atr = df_entry['atr'].iloc[-1]
        if atr is None or np.isnan(atr):
            return {'valid': False, 'reason': 'ATR is not available on entry timeframe.'}

        # Dynamic ATR-based Stop Loss and Entry
        entry_buffer = atr * self.config['entry_buffer_atr_multiplier']
        entry_price = support_level + entry_buffer
        stop_loss = support_level - (atr * self.config['atr_stop_loss_multiplier'])
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            return {'valid': False, 'reason': 'Invalid risk calculation (SL above entry).'}

        # Validate Take Profit against R:R ratio
        valid_targets = []
        if not sr_levels['resistances']:
            return {'valid': False, 'reason': 'No resistance levels found for targets.'}

        for i, resistance in enumerate(sr_levels['resistances']):
            target_price = resistance['level']
            reward_per_share = target_price - entry_price
            if reward_per_share > 0:
                rr_ratio = reward_per_share / risk_per_share
                valid_targets.append({
                    'target_num': i + 1,
                    'price': target_price,
                    'rr_ratio': round(rr_ratio, 2),
                })
        
        if not valid_targets:
            return {'valid': False, 'reason': 'No viable targets with positive reward.'}

        # Check if the *first* target meets the minimum R:R
        if valid_targets[0]['rr_ratio'] < self.config['min_reward_to_risk_ratio']:
            return {
                'valid': False,
                'reason': f"Primary target R:R is {valid_targets[0]['rr_ratio']:.2f}, "
                          f"below minimum of {self.config['min_reward_to_risk_ratio']}"
            }

        return {
            'valid': True,
            'primary_support': primary_support,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'targets': valid_targets,
            'risk_per_share': risk_per_share
        }

    # --- Main Analysis & Signal Generation Engine ---

    def analyze_market_with_sr(self) -> Dict:
        """
        The main orchestration function that performs the complete, multi-layered analysis.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*80}")
        print(f"QUANTITATIVE ANALYSIS REPORT - {timestamp}")
        print(f"Trading Pair: {self.trading_pair} | Primary Trend TF: {self.config['higher_timeframe']} | Entry TF: {self.config['entry_timeframe']}")
        print(f"{'='*80}")

        # 1. Fetch Data
        df_htf = self.fetch_ohlcv_data(self.config['higher_timeframe'], limit=self.config['htf_ema_long'] + 50)
        df_sr = self.fetch_ohlcv_data(self.config['sr_timeframe'], limit=self.config['sr_lookback_periods'])
        df_entry = self.fetch_ohlcv_data(self.config['entry_timeframe'], limit=200)

        if df_htf.empty or df_sr.empty or df_entry.empty:
            return {"signal": "HOLD", "reason": "Insufficient data for one or more timeframes."}

        # 2. Higher-Timeframe Trend Analysis
        print("\n[1] PRIMARY TREND ANALYSIS...")
        htf_trend = self._get_higher_timeframe_trend(df_htf)
        print(f"-> Higher-Timeframe ({self.config['higher_timeframe']}) Trend is: {htf_trend}")
        if htf_trend != 'UPTREND':
            print("-> Condition Failed: Primary trend is not bullish. No BUY signals will be generated.")
            return {"signal": "HOLD", "reason": f"Primary trend is {htf_trend}, not UPTREND."}

        # 3. S/R Confluence Zone Analysis
        print("\n[2] SUPPORT/RESISTANCE CONFLUENCE ANALYSIS...")
        sr_levels = self.calculate_support_resistance_levels(df_sr)
        self._log_sr_levels(sr_levels)
        
        # 4. Generate & Validate Trade Setup
        print("\n[3] TRADE SETUP VALIDATION (Entry, SL, TP, R:R)...")
        trade_setup = self.generate_trade_setup(sr_levels, df_entry)
        if not trade_setup['valid']:
            print(f"-> Invalid Trade Setup: {trade_setup['reason']}")
            return {"signal": "WAIT", "reason": trade_setup['reason'], "sr_levels": sr_levels}
        self._log_trade_setup(trade_setup)

        # 5. Confirmation Signal Analysis
        print("\n[4] ENTRY CONFIRMATION ANALYSIS...")
        is_divergence = self._detect_bullish_divergence(df_entry)
        bullish_pattern = self._identify_bullish_candlestick(df_entry)
        print(f"-> Bullish RSI Divergence Detected: {is_divergence}")
        print(f"-> Bullish Candlestick Pattern Found: {bullish_pattern if bullish_pattern else 'None'}")
        
        # 6. Final Signal Generation
        print("\n[5] FINAL SIGNAL GENERATION...")
        final_signal = self._generate_final_signal(htf_trend, trade_setup, is_divergence, bullish_pattern)
        
        print(f"\n{'='*30} FINAL RESULT {'='*30}")
        print(f"-> SIGNAL: {final_signal['signal']}")
        print(f"-> REASON: {final_signal['reason']}")
        print(f"{'='*80}")
        
        return {
            "signal": final_signal['signal'],
            "reason": final_signal['reason'],
            "trade_setup": trade_setup,
            "confirmations": {"divergence": is_divergence, "candlestick": bullish_pattern},
            "primary_trend": htf_trend,
            "sr_levels": sr_levels,
            "timestamp": timestamp
        }

    def _generate_final_signal(self, htf_trend: str, trade_setup: Dict, is_divergence: bool, bullish_pattern: Optional[str]) -> Dict:
        """
        A strict, rule-based engine to generate the final signal.
        """
        # A "High-Probability Setup" requires multiple conditions to be met.
        is_high_prob_setup = all([
            htf_trend == 'UPTREND',
            trade_setup['valid'],
            trade_setup['primary_support']['strength'] > 10, # Require a reasonably strong support
            trade_setup['primary_support']['is_confluence'], # Require confluence
            is_divergence or (bullish_pattern is not None)
        ])
        
        if is_high_prob_setup:
            reasons = []
            if trade_setup['primary_support']['is_confluence']: reasons.append("Confluence Support")
            if is_divergence: reasons.append("Bullish Divergence")
            if bullish_pattern: reasons.append(f"{bullish_pattern} Pattern")
            reason_str = " + ".join(reasons)
            return {
                "signal": "STRONG_BUY",
                "reason": f"High-probability setup confirmed: {reason_str}."
            }

        # A standard BUY signal might be a good setup but missing one confirmation element.
        is_standard_setup = all([
            htf_trend == 'UPTREND',
            trade_setup['valid'],
            trade_setup['primary_support']['strength'] > 5
        ])
        
        if is_standard_setup:
            return {
                "signal": "BUY",
                "reason": "Valid setup in an uptrend, awaiting strong confirmation (divergence/pattern)."
            }

        # Default to WAIT if a setup is forming but not yet valid.
        return {
            "signal": "WAIT",
            "reason": "Market conditions are favorable, but no valid trade setup meets all criteria yet."
        }

    # --- Utility and Helper Functions ---
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List, List]:
        order = self.config['pivot_order']
        high_indices = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]
        low_indices = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
        swing_highs = [(df.index[i], df['high'].iloc[i]) for i in high_indices]
        swing_lows = [(df.index[i], df['low'].iloc[i]) for i in low_indices]
        return swing_highs, swing_lows

    def _calculate_volume_profile(self, df: pd.DataFrame) -> List[Dict]:
        bins = self.config['volume_profile_bins']
        vp = df.groupby(pd.cut(df['close'], bins))['volume'].sum().sort_values(ascending=False)
        return [{'level': interval.mid, 'volume': vol} for interval, vol in vp.head(10).items()]

    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        lookback_df = df.tail(self.config['sr_lookback_periods'])
        high_price = lookback_df['high'].max()
        low_price = lookback_df['low'].min()
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        levels = {}
        if high_price > low_price:
            for ratio in fib_ratios:
                fib_level = high_price - (high_price - low_price) * ratio
                levels[f'fib_{ratio}'] = {'level': fib_level}
        return levels

    # --- Logging Helpers ---
    def _log_sr_levels(self, sr_levels):
        print(f"  Current Price: ${sr_levels['current_price']:.4f}")
        print("\n  Top Support Zones:")
        for s in sr_levels['supports']:
            conf_flag = ">> CONFLUENCE" if s['is_confluence'] else ""
            print(f"    - Level: ${s['level']:.4f} (Strength: {s['strength']:.2f}) Methods: {s['methods']} {conf_flag}")
        print("\n  Top Resistance Zones:")
        for r in sr_levels['resistances']:
            conf_flag = ">> CONFLUENCE" if r['is_confluence'] else ""
            print(f"    - Level: ${r['level']:.4f} (Strength: {r['strength']:.2f}) Methods: {r['methods']} {conf_flag}")
            
    def _log_trade_setup(self, setup):
        print(f"-> Valid Trade Setup Found:")
        print(f"  - Reacting to Support: ${setup['primary_support']['level']:.4f} (Strength: {setup['primary_support']['strength']:.2f})")
        print(f"  - Entry Price: ${setup['entry_price']:.4f}")
        print(f"  - Stop Loss: ${setup['stop_loss']:.4f} (Risk: ${setup['risk_per_share']:.4f})")
        for target in setup['targets'][:3]:
            print(f"  - Target {target['target_num']}: ${target['price']:.4f} (R:R Ratio: {target['rr_ratio']})")

def main():
    """Main function to run the enhanced bot."""
    print("Quantitative Trading Bot - Enhanced Signal Engine")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        trading_pair = sys.argv[1]
    else:
        trading_pair = input("Enter trading pair (e.g., BTC/USDT): ").strip().upper()
    
    if not trading_pair or '/' not in trading_pair:
        print("Error: Valid trading pair required (e.g., BTC/USDT)")
        sys.exit(1)
    
    try:
        bot = EnhancedMomentumBot(trading_pair)
        
        print("\nRunning with Enhanced Configuration:")
        for key, value in bot.config.items():
            print(f"  {key}: {value}")
            
        bot.analyze_market_with_sr()
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()