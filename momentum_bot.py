#!/usr/bin/env python3
import ccxt
import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.signal import argrelextrema

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')


class EnhancedMomentumBot:
    """
    Enhanced Trading Bot with Support/Resistance Analysis
    
    Features:
    - Automatic support/resistance identification
    - Entry calculations based on support levels
    - Multi-timeframe S/R analysis
    - Risk management based on S/R levels
    """
    
    def __init__(self, trading_pair: str, exchange_name: str = 'binance'):
        self.trading_pair = trading_pair
        self.exchange_name = exchange_name
        
        # Configuration parameters
        self.config = {
            'trend_tolerance': 0.02,
            'strong_buy_threshold': 75,
            'buy_threshold': 60,
            'pullback_tolerance': 0.005,
            'rsi_threshold_high': 75,
            'rsi_threshold_low': 30,
            'crossover_lookback': 5,
            'volume_confirmation': True,
            
            # Support/Resistance parameters
            'sr_lookback_periods': 50,      # Periods to look back for S/R
            'sr_min_touches': 2,            # Minimum touches to confirm S/R
            'sr_proximity_threshold': 0.01, # 1% proximity to consider a touch
            'fibonacci_enabled': True,      # Enable Fibonacci retracements
            'pivot_order': 5,               # Order for scipy peak detection
            'sr_strength_threshold': 3,     # Minimum strength for valid S/R
            'entry_buffer': 0.002,          # 0.2% buffer above support for entry
            'stop_loss_ratio': 0.015,       # 1.5% below support for stop loss
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
        support_levels = self._cluster_price_levels([price for _, price in swing_lows])
        resistance_levels = self._cluster_price_levels([price for _, price in swing_highs])
        
        # Method 2: Psychological Levels (round numbers)
        psychological_levels = self._find_psychological_levels(df['close'].iloc[-1])
        
        # Method 3: Volume Profile (simplified)
        volume_levels = self._calculate_volume_profile(df)
        
        # Method 4: Fibonacci Retracements
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
        
        # Score and filter levels
        scored_levels = self._score_sr_levels(df, all_levels)
        
        return scored_levels
    
    def _cluster_price_levels(self, prices: List[float]) -> List[Dict]:
        """
        Cluster similar price levels together.
        
        Returns:
            List of support/resistance levels with metadata
        """
        if not prices:
            return []
        
        prices = sorted(prices)
        levels = []
        proximity_threshold = self.config['sr_proximity_threshold']
        
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
        if current_price >= 100:
            increments = [10, 50, 100]
        elif current_price >= 10:
            increments = [1, 5, 10]
        elif current_price >= 1:
            increments = [0.1, 0.5, 1]
        else:
            increments = [0.001, 0.01, 0.1]
        
        for increment in increments:
            # Find nearest round levels above and below current price
            lower_level = (int(current_price / increment)) * increment
            upper_level = lower_level + increment
            
            # Add levels within reasonable range (±20% of current price)
            price_range = current_price * 0.2
            
            for level in [lower_level, upper_level]:
                if abs(level - current_price) <= price_range and level > 0:
                    levels.append({
                        'level': level,
                        'strength': 2,  # Base strength for psychological levels
                        'type': 'psychological',
                        'increment': increment
                    })
        
        return levels
    
    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> List[Dict]:
        """
        Calculate simplified volume profile levels.
        """
        if df.empty:
            return []
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        volume_profile = []
        
        for i in range(bins):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_mid = (bin_low + bin_high) / 2
            
            # Calculate volume for this price range
            mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
            bin_volume = df.loc[mask, 'volume'].sum()
            
            if bin_volume > 0:
                volume_profile.append({
                    'level': bin_mid,
                    'strength': bin_volume,
                    'type': 'volume_profile',
                    'volume': bin_volume
                })
        
        # Sort by volume and return top levels
        volume_profile.sort(key=lambda x: x['volume'], reverse=True)
        return volume_profile[:10]  # Return top 10 volume levels
    
    def _calculate_fibonacci_levels(self, swing_highs: List[Tuple], swing_lows: List[Tuple]) -> Dict:
        """
        Calculate Fibonacci retracement levels from recent significant moves.
        """
        if not swing_highs or not swing_lows:
            return {}
        
        # Find the most recent significant high and low
        recent_high = max(swing_highs, key=lambda x: x[1])
        recent_low = min(swing_lows, key=lambda x: x[1])
        
        high_price = recent_high[1]
        low_price = recent_low[1]
        
        # Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        levels = {}
        
        # Calculate retracement levels from high to low
        if high_price > low_price:
            for ratio in fib_ratios:
                fib_level = high_price - (high_price - low_price) * ratio
                levels[f'fib_{ratio}'] = {
                    'level': fib_level,
                    'strength': 3,  # Base strength for Fibonacci levels
                    'type': 'fibonacci_retracement',
                    'ratio': ratio,
                    'from_high': high_price,
                    'from_low': low_price
                }
        
        return levels
    
    def _score_sr_levels(self, df: pd.DataFrame, all_levels: Dict) -> Dict:
        """
        Score all support/resistance levels and filter by strength.
        """
        current_price = df['close'].iloc[-1]
        scored_supports = []
        scored_resistances = []
        
        # Combine all levels
        combined_levels = []
        for category, levels in all_levels.items():
            if isinstance(levels, list):
                combined_levels.extend(levels)
            elif isinstance(levels, dict):
                combined_levels.extend(levels.values())
        
        for level_info in combined_levels:
            level = level_info['level']
            base_strength = level_info['strength']
            
            # Additional scoring based on recent price action
            recent_touches = self._count_recent_touches(df, level)
            age_factor = 1.0  # Could be enhanced with time-based scoring
            
            total_strength = base_strength + recent_touches * 2
            
            # Classify as support or resistance based on current price
            if level < current_price:
                scored_supports.append({
                    **level_info,
                    'total_strength': total_strength,
                    'distance_pct': (current_price - level) / current_price,
                    'recent_touches': recent_touches
                })
            else:
                scored_resistances.append({
                    **level_info,
                    'total_strength': total_strength,
                    'distance_pct': (level - current_price) / current_price,
                    'recent_touches': recent_touches
                })
        
        # Filter by minimum strength and sort
        min_strength = self.config['sr_strength_threshold']
        
        valid_supports = [s for s in scored_supports if s['total_strength'] >= min_strength]
        valid_resistances = [r for r in scored_resistances if r['total_strength'] >= min_strength]
        
        # Sort by strength and proximity
        valid_supports.sort(key=lambda x: (x['distance_pct'], -x['total_strength']))
        valid_resistances.sort(key=lambda x: (x['distance_pct'], -x['total_strength']))
        
        return {
            'supports': valid_supports[:5],  # Top 5 support levels
            'resistances': valid_resistances[:5],  # Top 5 resistance levels
            'current_price': current_price
        }
    
    def _count_recent_touches(self, df: pd.DataFrame, level: float, lookback: int = 20) -> int:
        """
        Count how many times price has touched a level recently.
        """
        if lookback > len(df):
            lookback = len(df)
        
        recent_data = df.tail(lookback)
        proximity = level * self.config['sr_proximity_threshold']
        
        touches = 0
        for _, candle in recent_data.iterrows():
            if (candle['low'] <= level + proximity and 
                candle['high'] >= level - proximity):
                touches += 1
        
        return touches
    
    def calculate_entry_strategy(self, sr_levels: Dict) -> Dict:
        """
        Calculate entry strategy based on nearest support level.
        
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
        nearest_support = None
        for support in supports:
            # Look for support within reasonable distance (max 5% below current price)
            if support['distance_pct'] <= 0.05:
                nearest_support = support
                break
        
        if not nearest_support:
            return {
                'entry_valid': False,
                'reason': 'No nearby support levels found'
            }
        
        support_level = nearest_support['level']
        entry_buffer = self.config['entry_buffer']
        stop_loss_ratio = self.config['stop_loss_ratio']
        
        # Calculate entry point (slightly above support)
        entry_price = support_level * (1 + entry_buffer)
        
        # Calculate stop loss (below support)
        stop_loss = support_level * (1 - stop_loss_ratio)
        
        # Calculate targets based on nearest resistances
        targets = []
        if resistances:
            for i, resistance in enumerate(resistances[:3]):  # Use first 3 resistances as targets
                target_price = resistance['level']
                risk = entry_price - stop_loss
                reward = target_price - entry_price
                
                if reward > 0:
                    targets.append({
                        'target': i + 1,
                        'price': target_price,
                        'reward_risk_ratio': reward / risk if risk > 0 else 0,
                        'potential_profit_pct': ((target_price - entry_price) / entry_price) * 100
                    })
        
        # Calculate position sizing based on risk
        risk_per_trade = 0.02  # Risk 2% of account per trade
        risk_amount = entry_price - stop_loss
        risk_pct = (risk_amount / entry_price) * 100
        
        return {
            'entry_valid': True,
            'support_level': support_level,
            'support_strength': nearest_support['total_strength'],
            'support_type': nearest_support['type'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_pct': risk_pct,
            'targets': targets,
            'distance_to_support_pct': nearest_support['distance_pct'] * 100,
            'recommendation': self._generate_entry_recommendation(current_price, entry_price, targets)
        }
    
    def _generate_entry_recommendation(self, current_price: float, entry_price: float, targets: List[Dict]) -> str:
        """Generate entry recommendation based on analysis."""
        price_diff_pct = ((current_price - entry_price) / entry_price) * 100
        
        if not targets:
            return "HOLD - No clear targets identified"
        
        best_target = max(targets, key=lambda x: x['reward_risk_ratio'])
        
        if price_diff_pct > 2:
            return f"WAIT - Price {price_diff_pct:.1f}% above entry. Wait for pullback to support."
        elif price_diff_pct > -0.5:
            return f"BUY - Price near entry level. R:R ratio {best_target['reward_risk_ratio']:.2f}"
        else:
            return f"STRONG_BUY - Price below entry. Excellent R:R ratio {best_target['reward_risk_ratio']:.2f}"
    
    def analyze_market_with_sr(self) -> Dict:
        """
        Perform complete market analysis with support/resistance integration.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*70}")
        print(f"Enhanced Market Analysis with Support/Resistance - {timestamp}")
        print(f"Trading Pair: {self.trading_pair}")
        print(f"{'='*70}")
        
        # Fetch data for S/R analysis
        df_1h = self.fetch_ohlcv_data('1h', limit=100)
        df_5m = self.fetch_ohlcv_data('5m', limit=200)
        
        if df_1h.empty or df_5m.empty:
            return {"error": "Insufficient data for analysis"}
        
        print("Phase 1: Calculating Support/Resistance Levels...")
        
        # Calculate S/R levels on 1h timeframe for better reliability
        sr_levels = self.calculate_support_resistance_levels(df_1h)
        
        # Display S/R levels
        print(f"\n📊 Support Levels Found:")
        for i, support in enumerate(sr_levels['supports']):
            print(f"  S{i+1}: ${support['level']:.6f} (Strength: {support['total_strength']:.1f}, "
                  f"Distance: {support['distance_pct']*100:.2f}%, Type: {support['type']})")
        
        print(f"\n📊 Resistance Levels Found:")
        for i, resistance in enumerate(sr_levels['resistances']):
            print(f"  R{i+1}: ${resistance['level']:.6f} (Strength: {resistance['total_strength']:.1f}, "
                  f"Distance: {resistance['distance_pct']*100:.2f}%, Type: {resistance['type']})")
        
        print("\nPhase 2: Calculating Entry Strategy...")
        entry_strategy = self.calculate_entry_strategy(sr_levels)
        
        if not entry_strategy['entry_valid']:
            return {
                "signal": "HOLD",
                "reason": entry_strategy['reason'],
                "sr_levels": sr_levels,
                "timestamp": timestamp
            }
        
        # Display entry strategy
        print(f"\n🎯 Entry Strategy:")
        print(f"  Support Level: ${entry_strategy['support_level']:.6f} ({entry_strategy['support_type']})")
        print(f"  Entry Price: ${entry_strategy['entry_price']:.6f}")
        print(f"  Stop Loss: ${entry_strategy['stop_loss']:.6f}")
        print(f"  Risk: {entry_strategy['risk_pct']:.2f}%")
        print(f"  Distance to Support: {entry_strategy['distance_to_support_pct']:.2f}%")
        
        if entry_strategy['targets']:
            print(f"\n🎯 Targets:")
            for target in entry_strategy['targets']:
                print(f"  Target {target['target']}: ${target['price']:.6f} "
                      f"(R:R {target['reward_risk_ratio']:.2f}, "
                      f"Profit: {target['potential_profit_pct']:.1f}%)")
        
        # Perform traditional momentum analysis on 5m for confirmation
        print("\nPhase 3: Momentum Confirmation...")
        momentum_result = self.analyze_momentum_signals(df_5m)
        
        # Combine S/R analysis with momentum signals
        combined_signal = self._combine_sr_momentum_signals(entry_strategy, momentum_result)
        
        result = {
            "signal": combined_signal['signal'],
            "confidence": combined_signal['confidence'],
            "entry_strategy": entry_strategy,
            "momentum_analysis": momentum_result,
            "sr_levels": sr_levels,
            "recommendation": entry_strategy['recommendation'],
            "timestamp": timestamp
        }
        
        return result
    
    def analyze_momentum_signals(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum signals (simplified version of original logic)."""
        if len(df) < 21:
            return {"momentum_score": 0, "momentum_signal": "INSUFFICIENT_DATA"}
        
        # Calculate indicators
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        last_candle = df.iloc[-1]
        
        score = 0
        
        # EMA alignment
        if last_candle['ema_9'] > last_candle['ema_21']:
            score += 30
        
        # RSI check
        if 40 <= last_candle['rsi'] <= 70:
            score += 25
        elif 30 <= last_candle['rsi'] <= 80:
            score += 15
        
        # Price vs EMA
        if last_candle['close'] > last_candle['ema_9']:
            score += 20
        
        # Recent momentum
        if df['close'].pct_change(3).iloc[-1] > 0:
            score += 15
        
        momentum_signal = "BULLISH" if score >= 50 else "BEARISH" if score <= 30 else "NEUTRAL"
        
        return {
            "momentum_score": score,
            "momentum_signal": momentum_signal,
            "ema_9": last_candle['ema_9'],
            "ema_21": last_candle['ema_21'],
            "rsi": last_candle['rsi']
        }
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _combine_sr_momentum_signals(self, entry_strategy: Dict, momentum_result: Dict) -> Dict:
        """Combine S/R analysis with momentum signals."""
        base_confidence = 50  # Base confidence
        
        # S/R contribution (40% weight)
        sr_score = 0
        if entry_strategy['entry_valid']:
            # Strong support adds confidence
            sr_score += min(entry_strategy['support_strength'] * 5, 30)
            
            # Good risk/reward ratio adds confidence
            if entry_strategy['targets']:
                best_rr = max(t['reward_risk_ratio'] for t in entry_strategy['targets'])
                sr_score += min(best_rr * 10, 10)
        
        # Momentum contribution (30% weight)
        momentum_score = momentum_result['momentum_score'] * 0.3
        
        # Final confidence
        total_confidence = base_confidence + sr_score + momentum_score
        total_confidence = min(100, max(0, total_confidence))
        
        # Determine signal
        if total_confidence >= 75 and momentum_result['momentum_signal'] == 'BULLISH':
            signal = "STRONG_BUY"
        elif total_confidence >= 60 and momentum_result['momentum_signal'] in ['BULLISH', 'NEUTRAL']:
            signal = "BUY"
        elif total_confidence >= 40:
            signal = "WAIT"
        else:
            signal = "HOLD"
        
        return {
            "signal": signal,
            "confidence": total_confidence,
            "sr_contribution": sr_score,
            "momentum_contribution": momentum_score
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
        for key, value in sr_config.items():
            print(f"  {key}: {value}")
        
        bot.run_enhanced_analysis()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()