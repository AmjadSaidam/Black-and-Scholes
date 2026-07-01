# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An equity option pricing engine built around the Black-Scholes model, with a Streamlit dashboard for interactive exploration. The project models European call/put option pricing, Greeks (delta, gamma, theta), long-gamma delta hedging P&L, and GBM-based price simulation.

## Running the App

```bash
# Run the Streamlit dashboard from the project root
streamlit run app/dashboard.py
```

The dashboard requires a [Databento](https://databento.com) API key, entered via the sidebar at runtime. The `databento` package must be installed via conda (`conda install databento`) rather than pip.

## Architecture

The project is split into three layers:

**`code/` — Core quantitative models**
- `gbm.py`: `StochasticPriceForecast` — GBM simulation (`gbm_sim`), deterministic confidence bands (`gbm_confidence_interval`), log-normal price density, and analytical VaR. This is the base class for `Hedging`.
- `black_scholes_model.py`: All Black-Scholes logic.
  - `Time` dataclass: converts calendar dates + current day index into time-to-maturity (η = T − t) in years, using `pd.bdate_range` for business-day counting. `time_frame_in_years` defaults to 252.
  - `BlackScholesOptionPricing`: call/put pricing, delta/gamma/theta Greeks. Volatility is computed from historical returns (annualised). The `iv` attribute is set externally when implied volatility pricing is needed; the `implied` flag in `_d1d2` selects between `self.std` and `self.iv`.
  - `Hedging(StochasticPriceForecast)`: `delta_gamma_hedge` simulates the continuous mark-to-market P&L of a long-gamma delta hedge over GBM paths — the discrete approximation of `½(σ² − σ̃²) ∫ S²Γ e^{rτ} dt`.
  - Surface functions (`OptionPremiumSurface`, `GreekSurface`, `DeltaHedgeProfitSurface`): vectorised 2D sweeps over stock price × time / strike × drift, returning dict arrays for 3D plotting.

**`data/` — Market data**
- `databento_api_request.py`: `DatabentoAsset` wraps the Databento Historical client. `equity_schema` pulls OHLCV from `XNAS.ITCH`; `option_equity_schema` pulls the full option chain definition from `OPRA.PILLAR` using `stype_in='parent'`.

**`app/` — Streamlit dashboard**
- `dashboard.py`: Single-file app. Runs top-to-bottom on every interaction. Session state (`st.session_state`) is used to persist pulled data and simulation results across widget interactions. `multi_session_state()` initialises keys only if absent.
- `load_mathjax.js`: Injected via `st.components.v1.html` to enable LaTeX rendering in Streamlit using MathJax 2.7.5 from CDN.

## Key Conventions

- **Time representation**: day indices (integers) are passed as `current_time`; `Time.time_until_maturity()` converts them to fractional years using business-day counts. Surfaces sweep `np.linspace(0, contract_length, n)` as the time axis.
- **Volatility**: always annualised. Historical vol = `np.sqrt(np.var(returns) * 252)`. Implied vol is set on `bs.iv` before calling methods with `implied_volatility=True`.
- **Surface functions return dicts** with keys like `'price_axis'`, `'times_axis'`, `'value_axis'` (or `'greek_surface'`, `'pnl_surface'`) for direct unpacking into `plot3d()`.
- The `Hedging` class sets `hedge.loc = mu * hedge.days_in_year` externally in `DeltaHedgeProfitSurface` to override the drift for each surface point.