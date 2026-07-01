# app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
# general
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
# eda
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
# custom
from data.databento_api_request import DatabentoAsset
from code.gbm import StochasticPriceForecast
import code.black_scholes_model as bs

# =============================================
# LaTeX rendering
# =============================================
with open("app/load_mathjax.js", "r") as f:
    js = f.read()
    st.components.v1.html(f"<script>{js}</script>", height=0)

# =============================================
# Helper Functions
# =============================================
def multi_session_state(states_types: dict[str]):
    for state, type in states_types.items():
        if state not in st.session_state:
            setattr(st.session_state, state, type) 
        else:
            pass

def plot3d(X, Y, Z, Xt, Yt, Zt, column = None, key = None, title = ''):
    fig = go.Figure(data = [go.Surface(x = X, y = Y, z = Z)])
    fig.update_layout(scene = dict(xaxis_title = Xt, yaxis_title = Yt, zaxis_title = Zt), title = title)
    if column is not None:
        return column.plotly_chart(fig, key = key)
    st.plotly_chart(fig, key = key)

# =============================================
# Title
# =============================================
st.set_page_config(layout = 'wide')
st.title('Equity Option Pricing Engine')

# =============================================
# API 
# =============================================
api_key = st.sidebar.text_input('Databento API-key', 'YOUR-API-KEY')

# =============================================
# Datbento Data 
# =============================================
st.header('Data')

# define portfolio 
underlying = st.sidebar.text_input('Underlying Equity', 'NVDA') 
start_date_equity = st.sidebar.date_input('Starte Date Underlying', min_value = '2020-01-01')
start_date_option = st.sidebar.date_input('Start Date Option', min_value = '2026-01-01')
end_date = st.sidebar.date_input('End Date Equity/Option', min_value = '2026-01-02')
db_data = DatabentoAsset(api_key, 
                         underlying, 
                         end_date)

# define defualt states
data_states_types = {
    'pull_data': False, 
    'underlying_prices': None, 
    'underlying_option_chain': None
}
multi_session_state(data_states_types)

if st.button('Pull Feature/Label Data'):
    st.session_state.pull_data = True 

    # get portfolio returns and prices
    s_time_equity = pd.to_datetime(start_date_equity)
    s_time_option = pd.to_datetime(start_date_option)
    e_time = pd.to_datetime(end_date)
    if api_key is not None: 
        try:
            prices = db_data.equity_schema(s_time_equity, schema = 'OHLCV-1D')
            o_chain = db_data.option_equity_schema(start_date_option)
        except:
            # reset to default times
            def_e_time = pd.Timestamp.today() - pd.Timedelta(days = 1)
            def_s_time_equity = pd.Timestamp.today() - pd.Timedelta(days = 365)
            def_s_time_option = def_e_time - pd.Timedelta(days = 1)

            db_data.end_date = def_e_time
            prices = db_data.equity_schema(def_s_time_equity, schema = 'OHLCV-1D')
            o_chain = db_data.option_equity_schema(def_s_time_option)

    # print status 
    st.success(f'successfully retrieved {underlying} equity data and option chain data')

    # persist values 
    st.session_state.underlying_prices = prices
    st.session_state.underlying_option_chain = o_chain

prices = st.session_state.underlying_prices
underlying_returns = st.session_state.underlying_prices['close'].pct_change().fillna(0)
o_chain = st.session_state.underlying_option_chain

# print values 
with st.expander(f'{underlying} Prices'):
    st.dataframe(st.session_state.underlying_prices)

with st.expander(f'{underlying} Option Chain'):
    st.dataframe(st.session_state.underlying_option_chain)

# =============================================
# Call and Put Option Pricing 
# =============================================
st.header('Option Pricing under Black and Scholes')

with st.expander('Option Pricing Theory'):
    st.markdown(
    r"""
    European option value (the price we pay for an option) for a non-dividend paying stock is given by solving the black-scholes PDE

    $$
    \frac{1}{2} (\sigma^2 S_t^2) \frac{\partial^2 f(S_t, t)}{\partial S_t^2} + \frac{\partial f(S_t, t)}{\partial t} + r \left(S_t \frac{\partial f(S_t, t)}{\partial S_t} - f(S_t, t)\right) = 0
    $$

    For some option contract the PDE can be solved using the Feynman-kac formulation, that is by solving for the discounted expectation of the option payoff at maturity under the risk neutral measure $\mathbb{Q}$

    $$
    f(S_t, t) = e^{-r(T - t)} \mathbb{E}^{\mathbb{Q}} [g(X_T) | S_t = T], \, dS_t = rS_tdt + \sigma S_t dB_t^{\mathbb{Q}}
    $$

    Where $X_T$ is the terminal option payoff, terminal option payoffs for a call and put respectivley are  
    
    $$
    X_T = (S_T - K)^+ \\
    
    X_T = (K - S_T)^+
    $$
    
    Considering these payoffs, the solution for a call option and put option respectivley are

    $$
    f(S_t, t)_C = S_t \Phi(d_1) - K e^{-r(T - t)} \Phi(d_2) \\
    
    f(S_t, t)_P = K e^{-r(T - t)} \Phi(-d_2) - S_t \Phi(-d_1)
    $$

    Importantly $d_1$ and $d_2 are values from a standard normal.

    $$
    d_1 = \frac{\log(\frac{S_t}{K}) + (r + \frac{1}{2} \sigma^2)}{\sigma \sqrt{(T - t)}}; \, d_2 = d_1 - \sigma \sqrt{(T - t)}
    $$
     
    """
)

# top chain options
o_chain['ts_event'] = pd.to_datetime(o_chain.loc[:, 'ts_event'], format = '%y-%m-%d') # standerdise time
o_chain['expiration'] = pd.to_datetime(o_chain.loc[:, 'expiration'], format = '%y-%m-%d')
top_chain = {}
for i in ['C', "P"]:
    top_chain[i] = o_chain.loc[o_chain['instrument_class'] == 'C', :].iloc[-1, :]

# O(r, S0, t0, T, sigma)
st.subheader('Black Scholes Model Settings')
interest_rate = st.slider('Annualised Interest Rate (%)', min_value = 0.04, max_value = 0.1)

call, put = st.columns([.5, .5])
# call option 
call.subheader('Call Option')
call_t0, call_T = top_chain['C'].ts_event, top_chain['C'].expiration
call_contract_length = int(bs.Time(0, call_t0, call_T).contract_length())
call_option_t0 = call.date_input(r'Call Creation Time ($t_0$)', call_t0)
call_option_T = call.date_input('Call Expiration Time ($T$)', call_T)
call_option_t = call.slider('Call Option Calendar Time ($t$)', min_value = 0, max_value = call_contract_length)
call_strike = call.number_input(r'Call Strike Price ($K$)', top_chain['C'].strike_price)

# put option
put.subheader('Put Option')
put_t0, put_T = top_chain['P'].ts_event, top_chain['P'].expiration
put_contract_length = int(bs.Time(0, put_t0, put_T).contract_length())
put_option_t0 = put.date_input(r'Put Creation Time ($t_0$)', put_t0)
put_option_T = put.date_input(r'Put Expiration Time ($T$)', put_T)
put_option_t = put.slider('Put Option Calendar Time ($t$)', min_value = 0, max_value = put_contract_length)
put_strike = put.number_input(r'Put Strike Price ($K$)', top_chain['P'].strike_price)

# Black-Scholes Model
s0 = prices['close'].iloc[-1]
bs_model = bs.BlackScholesOptionPricing(s0, underlying_returns, risk_free_rate = interest_rate)
bs_model.r = interest_rate

# call and put premiums 
call_value = bs_model.call_option_price(call_strike, call_option_t, call_option_t0, call_option_T)
put_value = bs_model.put_option_price(put_strike, put_option_t, put_option_t0, put_option_T)

# call and put payoffs 
stock_prices = np.linspace(0, 500, 100) # stock price list

call_payoff_T = bs.call_payoff(stock_prices, call_strike, 0)*100
call_pnl = bs.call_payoff(stock_prices, call_strike, call_value)*100
put_payoff_T = bs.put_payoff(stock_prices, put_strike, 0)*100
put_pnl = bs.put_payoff(stock_prices, put_strike, put_value)*100

# call and put option value surface
call_v_t0 = []
put_v_t0 = []
for s_t in stock_prices: 
    opt_model = bs.BlackScholesOptionPricing(s_t, underlying_returns, risk_free_rate = interest_rate)
    call_v_t0.append(opt_model.call_option_price(call_strike, call_option_t, top_chain['C'].ts_event, top_chain['C'].expiration)*100)
    put_v_t0.append(opt_model.put_option_price(put_strike, put_option_t, top_chain['P'].ts_event, top_chain['P'].expiration)*100)

# to dataframe
call_data = pd.DataFrame({'Call Payoff': call_payoff_T, 'Call PnL': call_pnl, 'Call Value': call_v_t0}, index = stock_prices)
put_data = pd.DataFrame({'Put Payoff': put_payoff_T, 'Put PnL': put_pnl, 'Put Value': put_v_t0}, index = stock_prices)

# plot
fig = px.line(call_data, title = 'Call Option Payoff, Profit and Value Curves (OCC)')
fig.update_layout(xaxis_title = r'Stock Price ($S_T$)', yaxis_title = r'$\Pi(S_T, T)$')
fig.add_vline(call_strike, line_dash = 'dash', line_color = 'red', annotation_text = 'Call ATM', annotation_position = 'top')
call.plotly_chart(fig, key = "call_payoff_chart")
fig = px.line(put_data, title = 'Put Option Payoff, Profit and Value Curves (OCC)')
fig.update_layout(xaxis_title = r'Stock Price ($S_T$)', yaxis_title = r'$\Pi(S_T, T)$')
fig.add_vline(put_strike, line_dash = 'dash', line_color = 'orange', annotation_text = 'Put ATM', annotation_position = 'top')
put.plotly_chart(fig, key = "put_payoff_chart")

# call and put value surfaces 
call_times = np.linspace(0, call_contract_length, 100) 
put_times = np.linspace(0, put_contract_length, 100)

call_prem_surface = bs.OptionPremiumSurface(stock_prices, underlying_returns, call_strike, call_times, top_chain['C'], interest_rate)
put_prem_surface = bs.OptionPremiumSurface(stock_prices, underlying_returns, put_strike, put_times, top_chain['P'], interest_rate, 'put')

# plot
opt_lbls = ['Stock Prices', 'Calendar Time', 'Option Payoff']
plot3d(stock_prices, call_times, call_prem_surface['value_axis'], opt_lbls[0], opt_lbls[1], opt_lbls[-1], column = call, key = 'call_value_surface', title = f'{underlying} Call Option Price Surface')
plot3d(stock_prices, put_times, put_prem_surface['value_axis'], opt_lbls[0], opt_lbls[1], opt_lbls[-1], column = put, key = 'put_value_surface', title = f'{underlying} Put Option Price Surface')

# =============================================
# Greeks
# =============================================
delta_tab, gamma_tab, theta_tab = st.tabs(['Delta', 'Gamma', 'Theta'])

# define states 
delta = defaultdict(list)
gamma = defaultdict(list) 
theta = defaultdict(list)

# calculate greeks 
for s_t in stock_prices: 
    bs_model.st = s_t
    # delta
    delta['C'].append(bs_model.delta(call_strike, call_option_t, call_option_t0, call_option_T))
    delta['P'].append(bs_model.delta(put_strike, put_option_t, put_option_t0, put_option_T, 'put'))
    # gamma 
    gamma['C'].append(bs_model.gamma(call_strike, s_t, call_option_t, call_option_t0, call_option_T))
    gamma['P'].append(bs_model.gamma(put_strike, s_t, put_option_t, put_option_t0, put_option_T))
    # theta 
    theta['C'].append(bs_model.theta(call_strike, s_t, call_option_t, call_option_t0, call_option_T))
    theta['P'].append(bs_model.theta(put_strike, s_t, put_option_t, put_option_t0, put_option_T, 'put'))

# dataframes 
delta_df = pd.DataFrame({'Call Delta': delta['C'],  'Put Delta': delta['P']}, index = stock_prices)
gamma_df = pd.DataFrame({'Call Gamma': gamma['C'],  'Put Gamma': gamma['P']}, index = stock_prices)
theta_df = pd.DataFrame({'Call Theta': theta['C'],  'Put Theta': theta['P']}, index = stock_prices)

# axis labels 
greek_lbls = ['Calendar Time', 'Stock Price']

# plot delta
with delta_tab:
    fig = px.line(delta_df, title = f'{underlying} Call/Put Option Delta')
    fig.update_layout(xaxis_title = r'Stock Price $S_t$', yaxis_title = r'$\Delta$')
    fig.add_vline(call_strike, line_dash = 'dash', line_color = 'red', annotation_text = 'Call ATM', annotation_position = 'top')
    fig.add_vline(put_strike, line_dash = 'dash', line_color = 'orange', annotation_text = 'Put ATM', annotation_position = 'top')
    st.plotly_chart(fig, use_container_width = True, key = "delta_line_chart")

    call_delta_surf = bs.GreekSurface(underlying_returns, call_strike, call_option_t0, call_option_T, stock_prices, call_times)
    put_delta_surf = bs.GreekSurface(underlying_returns, put_strike, put_option_t0, put_option_T, stock_prices, put_times, option_type = 'put')
    call_greek, put_greek = st.columns([.5, .5])
    plot3d(call_delta_surf['stock_price_list'], call_delta_surf['times_list'], call_delta_surf['greek_surface'], greek_lbls[0], greek_lbls[1], 'Call Delta', column = call_greek, key = "call_delta_surface", title = f'{underlying} Call Delta Surface')
    plot3d(put_delta_surf['stock_price_list'], put_delta_surf['times_list'], put_delta_surf['greek_surface'], greek_lbls[0], greek_lbls[1], 'Put Delta', column = put_greek, key = "put_delta_surface", title = f'{underlying} Put Delta Surface')

# plot gamma
with gamma_tab:
    fig = px.line(gamma_df, title = f'{underlying} Call/Put Option Gamma')
    fig.update_layout(xaxis_title = r'Stock Price $S_t$', yaxis_title = r'Gamma ($\Gamma$)')
    fig.add_vline(call_strike, line_dash = 'dash', line_color = 'red', annotation_text = 'Call ATM', annotation_position = 'top')
    fig.add_vline(put_strike, line_dash = 'dash', line_color = 'orange', annotation_text = 'Put ATM', annotation_position = 'top')
    st.plotly_chart(fig, use_container_width = True, key = "gamma_line_chart")

    call_gamma_surf = bs.GreekSurface(underlying_returns, call_strike, call_option_t0, call_option_T, stock_prices, call_times, greek_type = 'gamma')
    put_gamma_surf = bs.GreekSurface(underlying_returns, put_strike, put_option_t0, put_option_T, stock_prices, put_times, option_type = 'put', greek_type = 'gamma')
    call_greek, put_greek = st.columns([.5, .5])
    plot3d(call_gamma_surf['stock_price_list'], call_gamma_surf['times_list'], call_gamma_surf['greek_surface'], greek_lbls[0], greek_lbls[1], 'Call Gamma', column = call_greek, key = "call_gamma_surface", title = f'{underlying} Call Gamma Surface')
    plot3d(put_gamma_surf['stock_price_list'], put_gamma_surf['times_list'], put_gamma_surf['greek_surface'], greek_lbls[0], greek_lbls[1], 'Put Gamma', column = put_greek, key = "put_gamma_surface", title = f'{underlying} Put Gamma Surface')

# plot theta
with theta_tab:
    fig = px.line(theta_df, title = f'{underlying} Call/Put Option Theta')
    fig.update_layout(xaxis_title = r'Stock Price $S_t$', yaxis_title = r'Theta ($\Theta$)')
    fig.add_vline(call_strike, line_dash = 'dash', line_color = 'red', annotation_text = 'Call ATM', annotation_position = 'top')
    fig.add_vline(put_strike, line_dash = 'dash', line_color = 'orange', annotation_text = 'Put ATM', annotation_position = 'top')
    st.plotly_chart(fig, use_container_width = True, key = "theta_line_chart")

    call_theta_surf = bs.GreekSurface(underlying_returns, call_strike, call_option_t0, call_option_T, stock_prices, call_times, greek_type = 'theta')
    put_theta_surf = bs.GreekSurface(underlying_returns, put_strike, put_option_t0, put_option_T, stock_prices, put_times, option_type = 'put', greek_type = 'theta')
    call_greek, put_greek = st.columns([.5, .5])
    plot3d(call_theta_surf['stock_price_list'], call_theta_surf['times_list'], call_theta_surf['greek_surface'], greek_lbls[0], greek_lbls[1], 'Call Theta', column = call_greek, key = "call_theta_surface", title = f'{underlying} Call Theta Surface')
    plot3d(put_theta_surf['stock_price_list'], put_theta_surf['times_list'], put_theta_surf['greek_surface'], greek_lbls[0], greek_lbls[1], 'Put Theta', column = put_greek, key = "put_theta_surface", title = f'{underlying} Put Theta Surface')

# =============================================
# Underlying Price Simulation
# =============================================
st.header("Underlying Price Forecast")
tab1, tab2 = st.tabs(['GBM Equity Forecast', 'GBM Simulation Statistics'])

# monte carlo
stoch_sim = StochasticPriceForecast(underlying_returns)

with tab1:
    with st.expander('GBM Theory'):
        st.markdown(
            r"""
            We simulate paths according to the log-normal random walk 

            $$
            S_t = S_0 \exp \left\{ \left( \mu - \frac{1}{2}\sigma^2 \right) + \sigma t B_t \right\}; \, S_t \sim \text{Lognormal} \left( \log(S_0) +\left( \mu - \small \tfrac{1}{2} \sigma^2 \right)t, \sigma^2 t \right)
            $$
            """
        )

    alpha = st.slider(r'Significance Level ($\alpha$)', value = 0.05, min_value = 0.01, max_value = 0.99)
    sims = st.number_input('Number of Simulated Paths', value = 100, min_value = 1, max_value = 1000)
    forecast_period = st.number_input('Forecast Period', value = 100, min_value = 1, max_value = 1000)
    pv_underlying_price = st.number_input('Present Value Underlying Price', value = st.session_state.underlying_prices.loc[:, 'close'].iloc[0])

    gbm_states_types = {
        'strategy_sims': None, 
        'log_normal': None, 
        'equity_vals': [], 
        'statistics': {}
    }
    multi_session_state(gbm_states_types)

    if st.button('Generate Strategy Price Simulations'):
        # gbm sims 
        sims = stoch_sim.gbm_sim(sims, forecast_period, pv_underlying_price)['prices']
        bands = stoch_sim.gbm_confidence_interval(alpha)
        st.session_state.strategy_sims = {'sims': sims, 'bands': bands}
        # gbm sim dist 
        q_bot = np.quantile(sims, alpha/2)
        q_top = np.quantile(sims, 1-(alpha/2))
        st.session_state.equity_vals= np.linspace(q_bot, q_top, 500)
        st.session_state.log_normal = np.array(
        [stoch_sim.price_denisty(s, stoch_sim.dt*stoch_sim.forecast_period) for s in st.session_state.equity_vals]
        )
        # var
        st.session_state.statistics['var'] = stoch_sim.analytical_VaR()
        # probability call in the money
        sims_T = sims[-1, :]
        st.session_state.statistics['call_prob_itm'] = len(sims_T[sims_T > call_strike]) / len(sims_T)
        st.session_state.statistics['put_prob_itm'] = len(sims_T[sims_T < put_strike]) / len(sims_T)
        
    gbm_col, dist_col = st.columns([.5, .5])

    sims = st.session_state.strategy_sims['sims']
    up, low = st.session_state.strategy_sims['bands']
    forecast_dat = pd.DataFrame(sims)
    forecast_dat['upper'] = up
    forecast_dat['lower'] = low

    # gbm plot
    fig = px.line(forecast_dat, title = f'{underlying} Strategy Equity Forecasts')
    fig.update_layout(showlegend = False)
    fig.update_traces(line = dict(color = 'blue', width = 1), opacity = 1)
    fig.update_traces(line = dict(color = 'green', width = 3), selector = dict(name = 'upper'), opacity = 1)
    fig.update_traces(line = dict(color = 'red', width = 3), selector = dict(name = 'lower'), opacity = 1)
    fig.update_xaxes(title = r'Time ($t$)')
    fig.update_yaxes(title = r'Stock Price ($S_t$)')
    fig.add_hline(call_strike, line_dash = 'dash', line_color = 'red', annotation_text = 'Call Strike', annotation_position = 'left', opacity = 1)
    fig.add_hline(put_strike, line_dash = 'dash', line_color = 'orange', annotation_text = 'Put Strike', annotation_position = 'left', opacity = 1)
    gbm_col.plotly_chart(fig, key = "gbm_forecast_chart")

    # log-normal density plot
    fig = px.line(x = st.session_state.equity_vals, y = st.session_state.log_normal, title = f'Terminal {underlying} Log-normal Density')
    fig.update_xaxes(title = f'Stock Price ({1 - alpha}% Confidence Level)')
    fig.update_yaxes(title = 'Probability Density')
    fig.add_vline(call_strike, line_dash = 'dash', line_color = 'red', annotation_text = 'Call ATM', annotation_position = 'top')
    fig.add_vline(put_strike, line_dash = 'dash', line_color = 'orange', annotation_text = 'Put ATM', annotation_position = 'top')
    dist_col.plotly_chart(fig, key = "lognormal_density_chart")

with tab2:
    metric_data = pd.DataFrame({
        f'{alpha}% Value at Risk (VaR)': [st.session_state.statistics.get('var', 0)], 
        'Probability Call ITM at Expiry': [st.session_state.statistics.get('call_prob_itm', 0)], 
        'Probability Put ITM at Expiry': [st.session_state.statistics.get('put_prob_itm', 0)] 
    }, 
    index = ['values']).T
    st.dataframe(metric_data)

# =============================================
# Delta Hedging 
# =============================================
st.header('Long Gamma Delta Hedging')

with st.expander('Delta Hedging Theory'):
    st.markdown(
        r"""
        If the option is miscpriced we can create a delta hedge portfolio and make a riskless profit equal to this mispricing. 

        The option price under black-scholes, $V_{bs}$, is the fair value option price. If the market price of the option, $V_{mrk}$, is less than fair value then we have an underpriced option, $V_{mrk} < V_{bs}$. 

        Assuming all equal, given the option is an increasing function of volatility this is the same as $\sigma_{mrk} < \sigma_{bs}$. Black-scholes uses realised volatility, $\sigma$, (backward looking), while the 
        market price uses implied volatility, $\tilde{\sigma}$, so we can re-write this misspricing as $\tilde{\sigma} < \sigma$. 

        This is a 'long gamma' mispricing and means the market is underpricing future volatility, meaning we can replicate a more expensive option for cheaper, by the replicating portfolio argument, which states that 
        an option payoff can be replicated exactly in the continuous setting by holding an option and a position in the underlying. 
        
        The mark-to-market profit (profit of portfolio for an infinitesimal change in time) of this market portfolio is 

        $$
        d \Pi^i = dV^i - \Delta S - r(V^i - \Delta^i S) dt
        $$

        Similarly for the the fair-value portfolio 

        $$
        d \Pi^a = dV^a - \Delta S - r(V^a - \Delta^a S) dt
        $$

        Note we have used $i$ to denoto implied implied volatility and $a$ to define actual volatility.
        
        We replicate the actual portfolio for cheaper, and the portfolio is self-financing, if we let $\Delta^i = \Delta^a$, then our arbitrage profit is 

        $$
        e^{rt}(d \Pi^i - d \Pi^a) = e^{rt}d\Pi^i = e^{rt}d(e^{-rt}(V^i - V^a))
        $$

        Discounting and suming over all time steps, gives a deterministic total profit

        $$
        \Pi_T = e^{rt} (V_{t_0}^a - V_{t_0}^i) > 0
        $$
        
        Assuming our portfolio does not grow at the risk-free rate we can model the mark-to-market profit directly as a function of time, this gives

        $$
        \Pi_T^i = \frac{1}{2} (\tilde{\sigma}^2 - \sigma^2) \int_0^T S^2 \Gamma^i dt
        $$
        
        Given $S_t$ is stochastic and follows a log-normal random walk, this process is stochastic (no closed for solution exists).

        In the expectation many simulations of this profit will converge to the theoretical arbitrage profit

        $$
        \mathbb{E} [ \Pi_T^i ] = e^{rT} (V_{t_0}^a - V_{t_0}^i)
        $$
        """
    )

hedging_sims = st.number_input('Hedging Simulations', value = 10, step = 1)
hedging_freq = st.number_input('Hedge Rebalance Frequency', value = 100, step = 1)
implied_vol = st.number_input(r'Implied Volatility ($\tilde{\sigma}$)', value = bs_model.std/1.1, step = 0.01)
bs_model.iv = implied_vol
stock_price_s0 = st.number_input(r'Present Value Stock Price ($S_{t_0}$)', value = call_strike)
bs_model.st = stock_price_s0

# choose option
long_put = st.checkbox('Long Gamma Put', value = False)
k = call_strike 
t0 = call_option_t0
t = call_option_t
T = call_option_T
days_to_T = call_contract_length
Va = bs_model.call_option_price(k, t, t0, T)
Vi = bs_model.call_option_price(k, t, t0, T, implied_volatility = True)
if long_put:
    k = put_strike
    t0 = put_option_t0
    t = put_option_t
    T = put_option_T
    days_to_T = put_contract_length
    Va = bs_model.call_option_price(k, t, t0, T)
    Vi = bs_model.put_option_price(k, t, t0, T, implied_volatility = True)

# hedge 
hedge = bs.Hedging(underlying_returns)
delta_hedge = hedge.delta_gamma_hedge(implied_vol, k, stock_price_s0, t0, T, hedging_sims, hedging_freq)
occ_hedge_pnl = delta_hedge['hedge_pnl_paths']

# expected and theoretical profit 
arb_pnl_occ = np.exp(bs_model.r*days_to_T/252)*(Va - Vi)
exp_pnl_occ = np.mean(delta_hedge['final_pnl'], axis = -1)

# data 
t_norm = np.linspace(0, 1, hedging_freq)
hedge_pnl_df = pd.DataFrame(occ_hedge_pnl, index = t_norm)

# plot hegde profit
fig = px.line(hedge_pnl_df, title = 'Long Gamma Implied Volatility Empirical Frictionless Profit')
fig.update_layout(xaxis_title = r'C, alendar Time to Option Expiry ($t$)', yaxis_title = r'Hedging Profit ($\Pi(S_T, T)$)', showlegend = False)
fig.add_hline(exp_pnl_occ, line_dash = 'dash', line_color = 'red', annotation_text = 'Expected Hedging Profit', annotation_position = 'top')
fig.add_hline(arb_pnl_occ, line_dash = 'dash', line_color = 'blue', annotation_text = 'Theoretical Riskless Arbitrage Profit', annotation_position = 'top')
fig.update_traces(line = dict(color = 'green', width = 1))
st.plotly_chart(fig, key = "hedge_pnl_chart")

# =============================================
# Delta Hedging Profit and Loss Surface
# =============================================
st.subheader('Delta Hedging Volatility Surface')

with st.expander('Conditional Expected Total Profit of Implied Volatility Delta Hedge'):
    st.markdown(
        r"""
        To see how hedging profit varies according to changes in input parameterst we compute the conditional expected profit of total hedging profit given the stock growth factor $\mu$ and 
        option strike prices $K$. Mathematicaly this is define as 

        $$
        \mathbb{E} [\Pi_T^i | \mu, K] = \mathbb{E} \left[ \int_{t_0}^T \Pi_t^i dt \bigg | \mu, K \right]
        $$
        """
    )

eval_per_axes = st.number_input('Evalutions per Axis', value = 10)
hedging_paths_per_point = st.number_input('Hedging Simulations per Point', value = 10)
hedging_sims_per_point = st.number_input('Hedging Rebalance Frequency per Point', value = 10)

max_growth = abs(stoch_sim.loc + bs_model.std)
growth_list = np.linspace(-max_growth, max_growth, eval_per_axes)
strike_list = np.linspace(1, call_strike, eval_per_axes)

exp_delta_hedging_pnl = bs.DeltaHedgeProfitSurface(underlying_returns, growth_list, strike_list, stock_price_s0, t0, T, paths = hedging_paths_per_point, paths_per_point = hedging_sims_per_point, implied_volatility = implied_vol)

# plot
fig = go.Figure(data = go.Surface(x = growth_list, y = strike_list, z = exp_delta_hedging_pnl['pnl_surface']))
fig.update_layout(title = dict(text = f'{underlying} Long Gamma Delta Hedging Conditional Expected Profit Surface'), scene = dict(xaxis_title = 'Underlying Growth', yaxis_title = 'Strike Price', zaxis_title = 'Expected Delta Hedging Profit'))
st.plotly_chart(fig, key = "hedge_pnl_surface")
