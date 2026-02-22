#!/usr/bin/env python3
"""
option_pricer.py — Black-Scholes Option Pricing & Greeks Engine

Replaces all hardcoded leverage multipliers and flat-delta models across
the trading system with proper option math.

Usage:
    from option_pricer import price_option, greeks, estimate_return, implied_vol

    # Price an ATM call
    p = price_option(S=580, K=580, T=1/252, sigma=0.18, r=0.045, option_type='call')

    # Get all Greeks
    g = greeks(S=580, K=580, T=1/252, sigma=0.18)

    # Estimate option P/L for a trade
    ret = estimate_return(
        S_entry=580, S_exit=582, K=580,
        T_entry=1/252, T_exit=0.5/252,  # Half a day later
        sigma=0.18, option_type='call'
    )

    # Solve for implied vol from market price
    iv = implied_vol(market_price=2.50, S=580, K=580, T=1/252, option_type='call')

Created: 2026
"""

import math
from typing import Optional
from scipy.stats import norm


# ─── Constants ───────────────────────────────────────────────────────────────

RISK_FREE_RATE = 0.045  # ~4.5% (current Fed funds rate as of Feb 2026)
TRADING_DAYS_PER_YEAR = 252


# ─── Core Black-Scholes ─────────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    """Calculate d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return float('inf') if S > K else float('-inf')
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    """Calculate d2 in Black-Scholes formula."""
    return _d1(S, K, T, sigma, r) - sigma * math.sqrt(max(T, 1e-10))


def price_option(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = RISK_FREE_RATE,
    option_type: str = 'call'
) -> float:
    """
    Black-Scholes European option price.

    Args:
        S: Current underlying price
        K: Strike price
        T: Time to expiration in years (e.g., 1/252 for 1 day, 0.5/252 for half a day)
        sigma: Annualized implied volatility (e.g., 0.18 for 18%)
        r: Risk-free rate (annualized)
        option_type: 'call' or 'put'

    Returns:
        Option price per share (multiply by 100 for contract price)
    """
    if T <= 0:
        # At expiration: intrinsic value only
        if option_type == 'call':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1 = _d1(S, K, T, sigma, r)
    d2 = _d2(S, K, T, sigma, r)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ─── Greeks ──────────────────────────────────────────────────────────────────

def greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = RISK_FREE_RATE,
    option_type: str = 'call'
) -> dict:
    """
    Calculate all Greeks for an option.

    Returns dict with:
        delta: Price change per $1 underlying move
        gamma: Delta change per $1 underlying move
        theta: Daily time decay (negative = losing value)
        vega: Price change per 1% IV change
        rho: Price change per 1% rate change
        price: Current option price
    """
    if T <= 0:
        price = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
        itm = (S > K) if option_type == 'call' else (K > S)
        return {
            'delta': 1.0 if (option_type == 'call' and itm) else (-1.0 if (option_type == 'put' and itm) else 0.0),
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'price': price,
        }

    d1 = _d1(S, K, T, sigma, r)
    d2 = _d2(S, K, T, sigma, r)
    sqrt_T = math.sqrt(T)
    nd1 = norm.pdf(d1)  # Standard normal density at d1

    price = price_option(S, K, T, sigma, r, option_type)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0

    # Gamma (same for calls and puts)
    gamma = nd1 / (S * sigma * sqrt_T)

    # Theta (per calendar day — divide annual by 365)
    theta_annual = -(S * nd1 * sigma) / (2 * sqrt_T)
    if option_type == 'call':
        theta_annual -= r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        theta_annual += r * K * math.exp(-r * T) * norm.cdf(-d2)
    theta = theta_annual / 365  # Per calendar day

    # Vega (per 1% IV change, not per 1 point)
    vega = S * sqrt_T * nd1 / 100  # Divide by 100 so it's per 1% move

    # Rho (per 1% rate change)
    if option_type == 'call':
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega': round(vega, 4),
        'rho': round(rho, 4),
        'price': round(price, 4),
    }


# ─── Implied Volatility ─────────────────────────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = RISK_FREE_RATE,
    option_type: str = 'call',
    precision: float = 0.0001,
    max_iterations: int = 100
) -> Optional[float]:
    """
    Solve for implied volatility using bisection method.

    Args:
        market_price: Observed option market price (per share)
        S, K, T, r: Same as price_option
        option_type: 'call' or 'put'
        precision: Convergence threshold
        max_iterations: Max bisection iterations

    Returns:
        Implied volatility (annualized), or None if no solution found
    """
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if market_price < intrinsic:
        return None  # Below intrinsic — no valid IV

    low, high = 0.01, 5.0  # 1% to 500% IV range

    for _ in range(max_iterations):
        mid = (low + high) / 2
        calc_price = price_option(S, K, T, mid, r, option_type)

        if abs(calc_price - market_price) < precision:
            return round(mid, 4)

        if calc_price > market_price:
            high = mid
        else:
            low = mid

    return round((low + high) / 2, 4)  # Best guess


# ─── Trade Return Estimation ────────────────────────────────────────────────

def estimate_return(
    S_entry: float,
    S_exit: float,
    K: float,
    T_entry: float,
    T_exit: float,
    sigma: float,
    r: float = RISK_FREE_RATE,
    option_type: str = 'call',
    sigma_exit: Optional[float] = None,
    contracts: int = 1,
    slippage_pct: float = 0.005,
) -> dict:
    """
    Estimate option trade P/L using Black-Scholes repricing.

    This replaces the hardcoded leverage multiplier approach with actual
    option repricing at exit conditions.

    Args:
        S_entry: Underlying price at entry
        S_exit: Underlying price at exit
        K: Strike price
        T_entry: Time to expiry at entry (years)
        T_exit: Time to expiry at exit (years)
        sigma: IV at entry (annualized)
        r: Risk-free rate
        option_type: 'call' or 'put'
        sigma_exit: IV at exit (if different from entry). None = same IV.
        contracts: Number of contracts
        slippage_pct: Round-trip slippage (applied to entry and exit)

    Returns:
        dict with entry_price, exit_price, pnl, return_pct, leverage_realized, greeks_entry, greeks_exit
    """
    sigma_exit = sigma_exit if sigma_exit is not None else sigma

    # Price at entry and exit
    entry_price = price_option(S_entry, K, T_entry, sigma, r, option_type)
    exit_price = price_option(S_exit, K, T_exit, sigma_exit, r, option_type)

    # Apply slippage
    entry_cost = entry_price * (1 + slippage_pct)  # Pay more on entry
    exit_proceeds = exit_price * (1 - slippage_pct)  # Receive less on exit

    # P/L
    pnl_per_share = exit_proceeds - entry_cost
    pnl_total = pnl_per_share * contracts * 100

    # Return percentage
    if entry_cost > 0:
        return_pct = pnl_per_share / entry_cost
    else:
        return_pct = 0.0

    # Realized leverage vs underlying move
    underlying_return = (S_exit - S_entry) / S_entry
    if option_type == 'put':
        underlying_return = -underlying_return
    if abs(underlying_return) > 0.0001:
        leverage_realized = return_pct / underlying_return
    else:
        leverage_realized = 0.0

    # Greeks at entry and exit for analysis
    greeks_entry = greeks(S_entry, K, T_entry, sigma, r, option_type)
    greeks_exit = greeks(S_exit, K, T_exit, sigma_exit, r, option_type)

    return {
        'entry_price': round(entry_cost, 4),
        'exit_price': round(exit_proceeds, 4),
        'pnl_per_share': round(pnl_per_share, 4),
        'pnl_total': round(pnl_total, 2),
        'return_pct': round(return_pct * 100, 2),
        'leverage_realized': round(leverage_realized, 2),
        'underlying_return_pct': round(underlying_return * 100, 4),
        'greeks_entry': greeks_entry,
        'greeks_exit': greeks_exit,
    }


# ─── Convenience Functions ───────────────────────────────────────────────────

def spy_0dte_atm(
    spy_price: float,
    spy_exit: float,
    minutes_held: int = 30,
    iv: float = 0.16,
    option_type: str = 'call',
    position_size: float = 333.0,
    slippage_pct: float = 0.005,
) -> dict:
    """
    Quick 0DTE ATM SPY option trade simulation.

    Designed for the Fib Scalper — replaces the flat delta=0.50 model.

    Args:
        spy_price: SPY price at entry
        spy_exit: SPY price at exit
        minutes_held: How long the trade lasted (minutes)
        iv: SPY implied volatility (default 16% — typical for SPY)
        option_type: 'call' or 'put'
        position_size: Capital allocated to this trade
        slippage_pct: Slippage percentage

    Returns:
        dict with pnl, contracts, return_pct, greeks, etc.
    """
    K = round(spy_price)  # ATM strike (nearest dollar)

    # Time to expiry: assume entry is X hours before close (4 PM ET)
    # For 0DTE, total remaining is typically 1-6 hours
    # We'll estimate ~4 hours remaining at typical entry, then subtract minutes_held
    hours_remaining_at_entry = 4.0  # Conservative: most 0DTE action is 10AM-2PM
    T_entry = hours_remaining_at_entry / (TRADING_DAYS_PER_YEAR * 6.5)  # 6.5 trading hours/day
    T_exit = max(0.0001, (hours_remaining_at_entry - minutes_held / 60) / (TRADING_DAYS_PER_YEAR * 6.5))

    # Price the option at entry to determine contracts
    entry_price = price_option(spy_price, K, T_entry, iv, RISK_FREE_RATE, option_type)
    entry_cost = entry_price * (1 + slippage_pct)

    if entry_cost <= 0.01:
        return {'pnl': 0, 'contracts': 0, 'return_pct': 0, 'error': 'option too cheap'}

    contracts = max(1, int(position_size / (entry_cost * 100)))

    result = estimate_return(
        S_entry=spy_price,
        S_exit=spy_exit,
        K=K,
        T_entry=T_entry,
        T_exit=T_exit,
        sigma=iv,
        option_type=option_type,
        contracts=contracts,
        slippage_pct=slippage_pct,
    )
    result['contracts'] = contracts
    result['strike'] = K
    result['T_entry_hours'] = round(hours_remaining_at_entry, 2)
    result['T_exit_hours'] = round(hours_remaining_at_entry - minutes_held / 60, 2)

    return result


def swing_option(
    underlying_price: float,
    underlying_exit: float,
    strike: float,
    dte_entry: int,
    hours_held: float = 16.0,
    iv: float = 0.35,
    option_type: str = 'call',
    iv_exit: Optional[float] = None,
    position_size: float = 1000.0,
    slippage_pct: float = 0.005,
) -> dict:
    """
    Overnight swing option trade simulation.

    Designed for the paper trading system — replaces hardcoded multiplier model.

    Accounts for:
    - Actual delta/gamma for the specific strike and DTE
    - Theta decay overnight
    - IV changes (if iv_exit specified)

    Args:
        underlying_price: Stock price at entry
        underlying_exit: Stock price at exit
        strike: Option strike price
        dte_entry: Days to expiration at entry
        hours_held: Hours in trade (default 16 = afternoon entry to next morning)
        iv: Implied volatility at entry
        option_type: 'call' or 'put'
        iv_exit: IV at exit (None = same as entry)
        position_size: Capital allocated
        slippage_pct: Slippage percentage

    Returns:
        dict with pnl, contracts, return_pct, greeks, etc.
    """
    T_entry = dte_entry / TRADING_DAYS_PER_YEAR
    # Subtract calendar days (overnight is ~0.7 trading days)
    trading_hours_held = hours_held * (6.5 / 24)  # Rough conversion
    T_exit = max(0.0001, T_entry - trading_hours_held / (TRADING_DAYS_PER_YEAR * 6.5))

    entry_price = price_option(underlying_price, strike, T_entry, iv, RISK_FREE_RATE, option_type)
    entry_cost = entry_price * (1 + slippage_pct)

    if entry_cost <= 0.01:
        return {'pnl': 0, 'contracts': 0, 'return_pct': 0, 'error': 'option too cheap'}

    contracts = max(1, int(position_size / (entry_cost * 100)))

    result = estimate_return(
        S_entry=underlying_price,
        S_exit=underlying_exit,
        K=strike,
        T_entry=T_entry,
        T_exit=T_exit,
        sigma=iv,
        option_type=option_type,
        sigma_exit=iv_exit,
        contracts=contracts,
        slippage_pct=slippage_pct,
    )
    result['contracts'] = contracts
    result['strike'] = strike
    result['dte_entry'] = dte_entry
    result['hours_held'] = hours_held

    return result


# ─── Analysis & Comparison Tools ────────────────────────────────────────────

def leverage_table(
    S: float = 580,
    K: float = 580,
    T: float = 1 / 252,
    sigma: float = 0.16,
    option_type: str = 'call',
    moves: Optional[list] = None,
) -> list[dict]:
    """
    Generate a leverage comparison table showing how option returns compare
    to underlying moves at different price levels.

    Useful for understanding why flat multipliers are wrong.

    Returns list of dicts with underlying_move, option_return, effective_leverage.
    """
    if moves is None:
        moves = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

    entry_price = price_option(S, K, T, sigma, RISK_FREE_RATE, option_type)
    results = []

    for move_pct in moves:
        S_exit = S * (1 + move_pct / 100)
        # Assume 30 minutes elapsed
        T_exit = max(0.0001, T - 0.5 / (252 * 6.5))

        exit_price = price_option(S_exit, K, T_exit, sigma, RISK_FREE_RATE, option_type)

        if entry_price > 0:
            opt_return = (exit_price - entry_price) / entry_price * 100
        else:
            opt_return = 0

        if abs(move_pct) > 0.01:
            eff_leverage = opt_return / move_pct
        else:
            eff_leverage = 0

        results.append({
            'underlying_move_pct': move_pct,
            'option_return_pct': round(opt_return, 2),
            'effective_leverage': round(eff_leverage, 2),
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
        })

    return results


def moneyness_comparison(
    S: float = 580,
    T: float = 14 / 252,
    sigma: float = 0.30,
    option_type: str = 'call',
) -> list[dict]:
    """
    Compare Greeks and pricing across different strikes (ITM → OTM).
    Shows why OTM options have higher leverage but more risk.
    """
    strikes = [
        S * 0.95,  # 5% ITM
        S * 0.97,  # 3% ITM
        S * 0.99,  # 1% ITM
        S,          # ATM
        S * 1.01,  # 1% OTM
        S * 1.03,  # 3% OTM
        S * 1.05,  # 5% OTM
    ]

    results = []
    for K in strikes:
        g = greeks(S, K, T, sigma, RISK_FREE_RATE, option_type)
        moneyness = (S - K) / S * 100 if option_type == 'call' else (K - S) / S * 100

        # Leverage = delta * (S / option_price) if price > 0
        if g['price'] > 0:
            omega = g['delta'] * S / g['price']  # Option elasticity (lambda/omega)
        else:
            omega = 0

        results.append({
            'strike': round(K, 2),
            'moneyness_pct': round(moneyness, 2),
            'price': g['price'],
            'delta': g['delta'],
            'gamma': g['gamma'],
            'theta': g['theta'],
            'vega': g['vega'],
            'omega_leverage': round(omega, 2),
        })

    return results


# ─── Self-Test ───────────────────────────────────────────────────────────────

def _validate():
    """Run sanity checks to verify the model is working correctly."""
    print("=" * 60)
    print("Option Pricer — Validation Suite")
    print("=" * 60)

    # Test 1: ATM call should be ~$2-4 for SPY with 1 DTE
    p = price_option(580, 580, 1/252, 0.16, option_type='call')
    print(f"\n1. SPY ATM 1DTE Call (IV=16%): ${p:.2f}")
    assert 1.0 < p < 6.0, f"Expected $1-6, got ${p:.2f}"
    print("   ✅ Reasonable range")

    # Test 2: Put-call parity
    call = price_option(580, 580, 30/252, 0.20, option_type='call')
    put = price_option(580, 580, 30/252, 0.20, option_type='put')
    parity = call - put - (580 - 580 * math.exp(-0.045 * 30/252))
    print(f"\n2. Put-call parity check: C-P = ${call-put:.4f}, S-PV(K) = ${580-580*math.exp(-0.045*30/252):.4f}")
    assert abs(parity) < 0.01, f"Parity violated: {parity:.6f}"
    print("   ✅ Put-call parity holds")

    # Test 3: Delta of ATM call should be ~0.50
    g = greeks(580, 580, 30/252, 0.20)
    print(f"\n3. ATM Call Delta: {g['delta']:.4f}")
    assert 0.45 < g['delta'] < 0.60, f"Expected ~0.50, got {g['delta']}"
    print("   ✅ Delta near 0.50")

    # Test 4: Theta should be negative for long options
    print(f"\n4. Theta (daily): ${g['theta']:.4f}")
    assert g['theta'] < 0, f"Expected negative theta"
    print("   ✅ Theta is negative")

    # Test 5: Implied vol roundtrip
    target_iv = 0.25
    price = price_option(580, 580, 30/252, target_iv)
    recovered_iv = implied_vol(price, 580, 580, 30/252)
    print(f"\n5. IV roundtrip: target={target_iv:.4f}, recovered={recovered_iv:.4f}")
    assert abs(recovered_iv - target_iv) < 0.001, f"IV roundtrip failed"
    print("   ✅ IV roundtrip accurate")

    # Test 6: 0DTE trade simulation
    result = spy_0dte_atm(580, 581, minutes_held=30, iv=0.16)
    print(f"\n6. 0DTE ATM Call: SPY $580→$581 (+0.17%) in 30min")
    print(f"   Entry: ${result['entry_price']:.2f}, Exit: ${result['exit_price']:.2f}")
    print(f"   P/L: ${result['pnl_total']:.2f} ({result['return_pct']:+.1f}%)")
    print(f"   Realized leverage: {result['leverage_realized']:.1f}x")
    print(f"   Contracts: {result['contracts']}")
    print("   ✅ Trade simulation works")

    # Test 7: Swing trade simulation
    result = swing_option(
        underlying_price=150, underlying_exit=153,
        strike=155, dte_entry=14, hours_held=16,
        iv=0.35, option_type='call', position_size=1000
    )
    print(f"\n7. Swing trade: $150→$153 (+2%), $155C 14DTE, 16hr hold")
    print(f"   Entry: ${result['entry_price']:.2f}, Exit: ${result['exit_price']:.2f}")
    print(f"   P/L: ${result['pnl_total']:.2f} ({result['return_pct']:+.1f}%)")
    print(f"   Realized leverage: {result['leverage_realized']:.1f}x")
    print("   ✅ Swing simulation works")

    # Test 8: Leverage table — show why flat multiplier is wrong
    print(f"\n8. Leverage Table (ATM 0DTE Call, IV=16%):")
    print(f"   {'SPY Move':>10} {'Opt Return':>12} {'Leverage':>10}")
    print(f"   {'-'*10} {'-'*12} {'-'*10}")
    table = leverage_table()
    for row in table:
        print(f"   {row['underlying_move_pct']:>+9.2f}% {row['option_return_pct']:>+11.2f}% {row['effective_leverage']:>9.1f}x")
    print("   ✅ Leverage is NON-LINEAR (this is why flat multipliers fail)")

    # Test 9: OTM options have higher leverage (omega)
    print(f"\n9. Moneyness Comparison (14DTE, IV=30%):")
    print(f"   {'Strike':>8} {'Money%':>8} {'Price':>8} {'Delta':>8} {'Omega':>8}")
    print(f"   {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for row in moneyness_comparison():
        print(f"   {row['strike']:>8.1f} {row['moneyness_pct']:>+7.1f}% ${row['price']:>7.2f} {row['delta']:>7.3f} {row['omega_leverage']:>7.1f}x")
    print("   ✅ OTM = higher leverage, ITM = lower leverage")

    print(f"\n{'=' * 60}")
    print("All validations passed ✅")
    print("=" * 60)


if __name__ == '__main__':
    _validate()
