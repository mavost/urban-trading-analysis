import numpy as np


def american_option_tree(S, K, T, r, sigma, option_type='call', steps=500):
    """Price an American option with a CRR binomial tree.

    Returns a float price.
    """
    if T <= 0 or steps <= 0:
        return float(max(S - K, 0) if option_type == 'call' else max(K - S, 0))
    if sigma is None or sigma <= 0:
        if option_type == 'call':
            return float(max(S - K * np.exp(-r * T), 0))
        else:
            return float(max(K * np.exp(-r * T) - S, 0))

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    denom = u - d
    if denom == 0:
        p = 0.5
    else:
        p = (np.exp(r * dt) - d) / denom
    p = min(max(p, 0.0), 1.0)
    df = np.exp(-r * dt)

    fs = np.zeros(steps + 1)
    S_t = S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
    if option_type == 'call':
        fs = np.maximum(S_t - K, 0.0)
    else:
        fs = np.maximum(K - S_t, 0.0)

    for i in range(steps - 1, -1, -1):
        S_t = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
        hold_value = df * (p * fs[:-1] + (1.0 - p) * fs[1:])
        if option_type == 'call':
            exercise_value = np.maximum(S_t - K, 0.0)
        else:
            exercise_value = np.maximum(K - S_t, 0.0)
        fs = np.maximum(hold_value, exercise_value)
    return float(fs[0])


def implied_vol_bisect_market(
    S,
    K,
    T,
    r,
    market_price,
    option_type='call',
    steps=300,
    tol=1e-4,
    maxiter=60,
    low=1e-6,
    high=5.0,
):
    """Find implied volatility by bisection using the American tree price function.

    Returns implied vol (float) or None if not found.
    """

    # price is increasing in sigma; bracket by expanding high if needed
    def price_at(sigma):
        return american_option_tree(
            S, K, T, r, sigma, option_type=option_type, steps=steps
        )

    p_low = price_at(low)
    p_high = price_at(high)

    # try expanding high until market_price <= p_high or reach cap
    if market_price > p_high:
        h = high
        for _ in range(10):
            h *= 2
            p_high = price_at(h)
            if market_price <= p_high:
                high = h
                break
        else:
            return None

    if market_price < p_low:
        return None

    a, b = low, high
    fa, fb = p_low - market_price, p_high - market_price

    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b

    for i in range(maxiter):
        m = 0.5 * (a + b)
        pm = price_at(m)
        fm = pm - market_price
        if abs(fm) < tol:
            return float(m)
        # choose side
        if fm > 0:
            b = m
        else:
            a = m
    return float(0.5 * (a + b))


def finite_diff_greeks(S, K, T, r, sigma, option_type='call', steps=300, eps=None):
    """Compute Greeks by central finite differences using the tree.

    Returns dict: delta, vega, theta
    """
    if eps is None:
        eps = max(1e-4, S * 1e-4)

    def price(S_, sigma_, T_):
        return american_option_tree(
            S_, K, T_, r, sigma_, option_type=option_type, steps=steps
        )

    price_c = price(S + eps, sigma, T)
    price_b = price(S - eps, sigma, T)
    delta = (price_c - price_b) / (2 * eps)

    sv = max(1e-4, sigma * 1e-4)
    price_sv = price(S, sigma + sv, T)
    price_sv_b = price(S, sigma - sv, T)
    vega = (price_sv - price_sv_b) / (2 * sv)

    dt = max(1.0 / 365.0, T * 1e-4)
    if T - dt <= 0:
        theta = (price(S, sigma, max(T - dt, 0.0)) - price(S, sigma, T)) / dt
    else:
        theta = (price(S, sigma, T - dt) - price(S, sigma, T)) / dt

    return {"delta": float(delta), "vega": float(vega), "theta": float(theta)}
