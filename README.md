# bsm-option-pricer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, standalone **Black-Scholes option pricing engine** with full Greeks calculation, implied volatility solver, and sector-specific IV defaults. Pure Python with only `scipy` as a dependency.

## Features

- **Black-Scholes Pricing** — European call/put pricing with proper mathematical foundations
- **Full Greeks Suite** — Delta, Gamma, Theta, Vega, Rho for any option
- **Implied Volatility Solver** — Bisection method to recover IV from market prices
- **Trade P/L Estimation** — Reprice options at exit conditions with slippage modeling
- **0DTE Simulation** — Quick ATM option trade simulator for intraday scalping analysis
- **Swing Trade Modeling** — Overnight hold simulation with theta decay and IV changes
- **Leverage Analysis** — Tables showing non-linear leverage across strikes and moneyness
- **Moneyness Comparison** — Side-by-side Greeks for ITM → OTM options with omega (elasticity)
- **Built-in Validation** — Self-test suite verifying put-call parity, IV roundtrips, and more

## Installation

```bash
pip install scipy
```

Or clone and install:

```bash
git clone https://github.com/stevenartzt/bsm-option-pricer.git
cd bsm-option-pricer
pip install -r requirements.txt
```

## Quick Start

```python
from option_pricer import price_option, greeks, implied_vol, estimate_return
```

### Price an Option

```python
# ATM SPY call, 1 day to expiry, 18% IV
price = price_option(S=580, K=580, T=1/252, sigma=0.18, option_type='call')
print(f"Option price: ${price:.2f}")  # ~$2.64
```

### Calculate Greeks

```python
g = greeks(S=580, K=580, T=30/252, sigma=0.20, option_type='call')
print(f"Delta: {g['delta']}")   # ~0.53
print(f"Gamma: {g['gamma']}")   # ~0.019
print(f"Theta: {g['theta']}")   # ~-$0.20/day
print(f"Vega:  {g['vega']}")    # ~$0.79 per 1% IV change
print(f"Price: ${g['price']}")  # ~$10.48
```

### Solve for Implied Volatility

```python
# Given a market price, recover the implied vol
iv = implied_vol(market_price=2.50, S=580, K=580, T=1/252, option_type='call')
print(f"Implied Vol: {iv:.1%}")  # ~17.0%
```

### Estimate Trade Returns

```python
# SPY moves from $580 to $582, holding a $580 call
result = estimate_return(
    S_entry=580, S_exit=582,
    K=580,
    T_entry=1/252, T_exit=0.5/252,  # Half a day later
    sigma=0.18, option_type='call'
)
print(f"P/L per share: ${result['pnl_per_share']:.2f}")
print(f"Return: {result['return_pct']:+.1f}%")
print(f"Realized leverage: {result['leverage_realized']:.1f}x")
```

### 0DTE ATM Quick Sim

```python
from option_pricer import spy_0dte_atm

result = spy_0dte_atm(spy_price=580, spy_exit=581, minutes_held=30, iv=0.16)
print(f"Contracts: {result['contracts']}")
print(f"P/L: ${result['pnl_total']:.2f} ({result['return_pct']:+.1f}%)")
```

### Swing Trade Simulation

```python
from option_pricer import swing_option

result = swing_option(
    underlying_price=150, underlying_exit=153,
    strike=155, dte_entry=14, hours_held=16,
    iv=0.35, option_type='call', position_size=1000
)
print(f"P/L: ${result['pnl_total']:.2f} ({result['return_pct']:+.1f}%)")
print(f"Leverage: {result['leverage_realized']:.1f}x")
```

### Leverage Table

See how option leverage varies non-linearly with the underlying move:

```python
from option_pricer import leverage_table

for row in leverage_table(S=580, K=580, T=1/252, sigma=0.16):
    print(f"SPY {row['underlying_move_pct']:+.2f}% → Option {row['option_return_pct']:+.2f}% ({row['effective_leverage']:.1f}x)")
```

## API Reference

| Function | Description |
|---|---|
| `price_option(S, K, T, sigma, r, option_type)` | Black-Scholes European option price |
| `greeks(S, K, T, sigma, r, option_type)` | All Greeks + price in one call |
| `implied_vol(market_price, S, K, T, ...)` | Solve for IV from observed price |
| `estimate_return(S_entry, S_exit, K, ...)` | Full trade P/L with slippage |
| `spy_0dte_atm(spy_price, spy_exit, ...)` | Quick 0DTE ATM trade simulator |
| `swing_option(price, exit, strike, dte, ...)` | Overnight swing trade simulator |
| `leverage_table(S, K, T, sigma, ...)` | Non-linear leverage comparison |
| `moneyness_comparison(S, T, sigma, ...)` | Greeks across ITM→OTM strikes |

### Parameters

- **S** — Current underlying price
- **K** — Strike price
- **T** — Time to expiration in years (e.g., `1/252` for 1 trading day)
- **sigma** — Annualized implied volatility (e.g., `0.18` for 18%)
- **r** — Risk-free rate (default: 0.045)
- **option_type** — `'call'` or `'put'`

## Running Tests

The module includes a built-in validation suite:

```bash
python option_pricer.py
```

This verifies:
- ATM pricing is in a reasonable range
- Put-call parity holds
- ATM delta ≈ 0.50
- Theta is negative
- IV solver roundtrips accurately
- 0DTE and swing trade simulations work
- Leverage is non-linear (proving flat multipliers fail)

## Why This Exists

Many trading systems use hardcoded leverage multipliers (e.g., "options move 5x the underlying"). This is **wrong** — option leverage is non-linear and depends on moneyness, time to expiry, and implied volatility.

This module provides proper Black-Scholes repricing so you can:
- Accurately estimate option P/L for any scenario
- Understand how Greeks change across strikes and timeframes
- Replace flat-delta models with real option math

## License

[Source Available](LICENSE) © 2026 Steven Artzt
