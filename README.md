# 5-Minute Momentum Signal Bot

A Python cryptocurrency trading signal bot that analyzes market momentum using technical indicators across multiple timeframes to generate BUY/HOLD signals.

## Overview

This bot continuously monitors a specified cryptocurrency pair and prints "BUY" signals when precise momentum conditions are met. The bot does NOT execute trades - it only provides signals for manual trading decisions.

## Features

- **Two-Phase Analysis**: 
  - Phase 1: 1h timeframe trend filter using 21-EMA
  - Phase 2: 5m timeframe entry signal using EMA crossover and RSI
- **Real-time Market Data**: Uses CCXT library to fetch live market data
- **Multiple Exchanges**: Supports Binance and other CCXT-compatible exchanges
- **Continuous Monitoring**: Runs analysis every 5 minutes
- **Clear Signal Output**: Prints prominent BUY/HOLD signals

## Technical Indicators

### Phase 1 (1h Chart)
- **21-period EMA**: Trend filter to ensure bullish market structure

### Phase 2 (5m Chart)
- **9-period EMA**: Fast moving average for crossover signals
- **21-period EMA**: Slow moving average for crossover signals  
- **14-period RSI**: Momentum oscillator to avoid overbought conditions

## Signal Logic

### Phase 1: High-Timeframe Trend Filter
- Current 1h close price must be >= 1h 21-EMA
- If this fails, signal is "HOLD" and Phase 2 is skipped

### Phase 2: Low-Timeframe Entry Signal (All must be true)
- **Condition A**: Bullish crossover - 9-EMA crosses above 21-EMA between last two 5m candles
- **Condition B**: Price pullback - Low of last 5m candle <= 9-EMA (price touched the moving average)
- **Condition C**: RSI confirmation - RSI < 70 (not overbought)

## Installation

1. Install Python 3.8+ and pip
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Arguments
```bash
# Run with trading pair as argument
python momentum_bot.py BTC/USDT

# Run interactively (will prompt for trading pair)
python momentum_bot.py
```

### Interactive Mode
The bot will ask you to choose between:
1. **Single Analysis**: Run analysis once and exit
2. **Continuous Monitoring**: Run analysis every 5 minutes indefinitely

### Example Output
```
5-Minute Momentum Signal Bot
========================================
Enter trading pair (e.g., BTC/USDT): BTC/USDT

Initialized MomentumBot for BTC/USDT on binance

Run mode:
1. Single analysis
2. Continuous monitoring
Choice (1/2): 2

Starting continuous analysis for BTC/USDT
Analysis interval: 5 minutes
Press Ctrl+C to stop

============================================================
Market Analysis - 2024-01-15 14:30:00
Trading Pair: BTC/USDT
============================================================
Phase 1: Checking 1h trend filter...
1h Close: 42150.50, 1h 21-EMA: 41980.25
Trend Filter: PASS

Phase 1 PASSED - Proceeding to Phase 2...
Phase 2: Checking 5m entry signal...
5m Analysis for last closed candle:
  Close: 42145.30
  Low: 42120.15
  9-EMA: 42125.40
  21-EMA: 42110.80
  RSI: 65.30

Condition A (Bullish Crossover): PASS
  Previous: 9-EMA(42118.20) < 21-EMA(42125.10)
  Current:  9-EMA(42125.40) > 21-EMA(42110.80)
Condition B (Price Pullback): PASS
  Low(42120.15) <= 9-EMA(42125.40)
Condition C (RSI < 70): PASS
  RSI: 65.30

Result: BUY (All conditions met!)

🚨 SIGNAL: BUY 🚨

Waiting 5 minutes for next analysis...
```

## Dependencies

- **ccxt**: Cryptocurrency exchange connectivity
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **ta-lib**: Technical analysis indicators (optional, fallback implementation included)
- **python-dotenv**: Environment variable management

## Supported Exchanges

The bot uses CCXT and supports all exchanges that provide OHLCV data, including:
- Binance (default)
- Coinbase Pro
- Kraken
- Bitfinex
- And many more

## Error Handling

The bot includes robust error handling for:
- Network connectivity issues
- Invalid trading pairs
- Insufficient data
- Exchange API errors
- Rate limiting

## Customization

You can modify the following parameters in the code:
- EMA periods (currently 9, 21 for 5m and 21 for 1h)
- RSI period (currently 14)
- RSI threshold (currently 70)
- Analysis interval (currently 5 minutes)
- Exchange (currently Binance)

## Disclaimer

This bot is for educational and analysis purposes only. It does not provide financial advice. Always do your own research and consider the risks before making any trading decisions.
