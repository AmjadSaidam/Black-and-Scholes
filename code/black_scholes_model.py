import pandas as pd
import numpy as np 
import scipy as sp
from dataclasses import dataclass
import code.gbm as gbm

@dataclass
class Time(): 
    """
    time function allows us to define contract duration lengths separately for each option
    """
    current_time: int | float # assumes time in days
    option_creation_time: pd.Timestamp
    expiration_time: pd.Timestamp
    time_frame_in_years: int = 252 

    def contract_length(self):
        return pd.bdate_range(self.option_creation_time, self.expiration_time).shape[0] # option creaton days to expiry

    def time_until_maturity(self):
        """
        returns the time until option expiry 
        """
        contract_duration = self.contract_length()
        self.t = self.current_time/self.time_frame_in_years # current time as fraction of contract duration in years 
        self.T = contract_duration/self.time_frame_in_years # contract duration in years
        return self.T - self.t 
        
class BlackScholesOptionPricing():
    def __init__(self, 
                 current_underlying_price: float, 
                 underlying_returns: np.ndarray, 
                 time_frame_in_years: int = 252, 
                 risk_free_rate: float = 0.04):
        self.st = current_underlying_price
        self.time_frame_in_years = time_frame_in_years
        self.var = np.var(underlying_returns)*self.time_frame_in_years
        self.std = np.sqrt(self.var)
        self.r = risk_free_rate
        self.iv = 0.0

    def call_option_price(self,
                          strike_price, 
                          current_time, 
                          option_creation_time, 
                          expiration_time, 
                          implied_volatility: bool = False):
        """
        call option premium under black and scholes 
        """
        eta = Time(current_time, option_creation_time, expiration_time, self.time_frame_in_years).time_until_maturity()
        
        # return termination option payoff if call time to expiry is zero
        if eta<=0:
            return max(self.st - strike_price, 0)

        d1, d2 = self._d1d2(strike_price, eta, implied_volatility)
        cdf1, cdf2 = self._cdf(d1), self._cdf(d2)
        return self.st*cdf1 - strike_price*np.exp(-self.r*eta)*cdf2
    
    def put_option_price(self, strike_price, 
                         current_time, 
                         option_creation_time, 
                         expiration_time, 
                         implied_volatility: bool = False):
        """
        put option premium under black and scholes 
        """
        eta = Time(current_time, option_creation_time, expiration_time, self.time_frame_in_years).time_until_maturity()
        
        # return termination option payoff if put time to expiry is zero
        if eta<=0:
            return max(strike_price - self.st, 0)

        d1, d2 = self._d1d2(strike_price, eta, implied_volatility)
        cdf1, cdf2 = self._cdf(-d2), self._cdf(-d1)
        return strike_price*np.exp(-self.r*eta)*cdf1 - self.st*cdf2

    # greeks 
    def delta(self, 
              strike_price,
              current_time: float, 
              option_creation_time, 
              expiration_time,
              option_type: bool = 'call'):
        """
        delta of call and put option
        """
        eta = Time(current_time, option_creation_time, expiration_time, self.time_frame_in_years).time_until_maturity()
        d1, _ = self._d1d2(strike_price, eta)
        call_delta = self._cdf(d1)

        if option_type == 'put': 
            return call_delta - 1

        return call_delta

    def gamma(self, 
              strike_price, 
              current_price, 
              current_time, 
              option_creation_time, 
              expiration_time, 
              implied_volatility: bool = False):
        """
        gamma of call and put option
        """
        eta = Time(current_time, option_creation_time, expiration_time).time_until_maturity()
        d1, _ = self._d1d2(strike_price, eta, implied_volatility)
    
        vol = self.iv if implied_volatility else self.std
        return self._pdf(d1)/(current_price*vol*np.sqrt(eta))

    def theta(self, 
              strike_price, 
              current_price, 
              current_time, 
              option_creation_time,
              expiration_time, 
              option_type = 'call'):
        """
        theta of call and put option
        """
        eta = Time(current_time, option_creation_time, expiration_time, self.time_frame_in_years).time_until_maturity()
        d1, d2 = self._d1d2(strike_price, eta)
        
        if option_type == 'put':
            d1, d2 = -d1, -d2

        den_d1 = self._pdf(d1)
        cdf_d2 = self._cdf(d2)

        return -(current_price*den_d1*self.std)/(2*np.sqrt(eta)) - self.r*strike_price*np.exp(-self.r*eta)*cdf_d2 * (-1 if option_type == 'put' else 1)

    # helpers 
    def _pdf(self, x):
        """
        standard normal probability density function 
        """
        return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

    def _cdf(self, x): 
        """
        standard normal cdf 
        """
        return sp.stats.norm.cdf(x, 0, 1)
  
    def _d1d2(self, strike_price, eta, implied: bool = False):
        """
        Black-Scholes d1 and d2 terms
        """
        vol = self.iv if implied else self.std
        num = np.log(self.st/strike_price) + (self.r + .5*vol**2)*eta
        d1 = num / (vol*np.sqrt(eta))
        d2 = d1 - vol*np.sqrt(eta)
        return d1, d2

# implied volatility function 
def implied_volatility():
    """
    calculates implied volatility of option under black scholes pricing using minimising difference approach
    """
    pass
    
# delta hedgind functions 
class Hedging(gbm.StochasticPriceForecast): 
    def __init__(self, returns: np.ndarray): # takes default input of parent class
        super().__init__(returns) # inputs for parent class

    def delta_gamma_hedge(self, 
                      implied_volatility: float, 
                      strike_price: float, 
                      initial_stock_price, 
                      option_creation_time, 
                      option_expiration_time, 
                      paths: int, 
                      hedging_points: int = 1000,
                      hedge_option_realised_volatility: float = 0, 
                      gamma_hedge: bool = False):
        """
        calculates the expectfed profit of a perfectly delta hedged portfolio using implied volatility

        when simulating on synthetic data, make sure initial stock price is approx strike price, so by expiry
        log-normal random walk produces paths approx ATM, otherwise gamma will be zero for far INT and OTM options

        Assumes we are dealing with daily options
        """
        contract_length = int(Time(0, option_creation_time, option_expiration_time).contract_length()) # option TTE in days from t0
        self.T = contract_length/252
        sims_in_time = np.linspace(0, self.T, hedging_points, endpoint = False) # avoid zero time to expiry
        self.dt = sims_in_time[1]

        # mark-to-market profit
        pnl_mtm = np.zeros((hedging_points, paths))

        sims = self.gbm_sim(paths, hedging_points, initial_stock_price)
        prices = sims['prices']

        # simulate deterministic profit 
        for path in range(paths):
            for t in range(hedging_points):
                st = prices[t, path]

                bs = BlackScholesOptionPricing(st, self.rt)
                bs.iv = implied_volatility
                
                avg_vol_diff = .5*(bs.var - implied_volatility**2) if not gamma_hedge else .5*(hedge_option_realised_volatility**2 - implied_volatility**2)
                
                # calculate gamma 
                tau = self.T - sims_in_time[t]
                d1, _ = bs._d1d2(strike_price, tau, implied = True)
                den_d1 = bs._pdf(d1)
                gamma_i = den_d1/(st*bs.iv*np.sqrt(tau))

                # calculate profit and loss 
                pnl_mtm[t, path] = avg_vol_diff*st**2*gamma_i*np.exp(bs.r*tau)*self.dt

                # logs
                # print('gamma', gamma_i, 'stock_price', st, 'average_vol', avg_vol_diff)

        pnl = np.cumsum(pnl_mtm, axis = 0) # discretised total hedged profits integral 
        total_pnls = pnl[-1, :]
    
        return {'hedge_pnl_paths': pnl, 'final_pnl': total_pnls}

def OptionPremiumSurface(stock_prices: np.ndarray, 
                         underlying_returns: np.ndarray, 
                         strike_price: float,
                         times: np.ndarray, 
                         option_info: pd.DataFrame, 
                         interest_rate: float,
                         option_type: str = 'call'):
    """
    alculates option premium at different underlying price and time
    """
    t_list, s_list = np.meshgrid(stock_prices, times)
    n = t_list.shape[0]
    m = s_list.shape[0]
    ov_surf = np.ones((n, m))
    for i, t in enumerate(times):
        for j, st in enumerate(stock_prices): 
            ds_model = BlackScholesOptionPricing(st, underlying_returns, risk_free_rate = interest_rate)
            if option_type == 'call':
                ov_surf[i, j] = ds_model.call_option_price(strike_price, 
                                                                 current_time = t, 
                                                                 option_creation_time = option_info.ts_event, 
                                                                 expiration_time = option_info.expiration)
            elif option_type == 'put':
                ov_surf[i, j] = ds_model.put_option_price(strike_price, 
                                                                current_time = t, 
                                                                option_creation_time = option_info.ts_event, 
                                                                expiration_time = option_info.expiration)  
        
    return {'price_axis': s_list, 'times_axis': t_list, 'value_axis': ov_surf}

def DeltaHedgeProfitSurface(uderlying_returns: np.ndarray,
                            expected_returns: np.ndarray, 
                            strike_prices: np.ndarray,
                            initial_stock_price: np.ndarray, 
                            option_creation_time: pd.Timestamp, 
                            option_expiration_time: pd.Timestamp,
                            paths: int = 10,   
                            paths_per_point: int = 10, 
                            implied_volatility: int = 0.0):
    """
    approximates long-gamma total profit surface
    """
    mu_matrix, k_matrix = np.meshgrid(expected_returns, strike_prices) # symmetrix input grids
    n, m = mu_matrix.shape[0], k_matrix.shape[0]
    pnl_surface = np.ones((n, m))

    for i, mu in enumerate(expected_returns):
        hedge: Hedging = Hedging(uderlying_returns)
        hedge.loc = mu*hedge.days_in_year
        for j, k in enumerate(strike_prices):
            # simulate
            exp_final_pnl = hedge.delta_gamma_hedge(implied_volatility, 
                                                    k, 
                                                    initial_stock_price, 
                                                    option_creation_time, 
                                                    option_expiration_time, 
                                                    paths, 
                                                    paths_per_point)['final_pnl'].mean(axis = -1)
            # fill surface
            pnl_surface[i, j] = exp_final_pnl
    
    return {'growth_rate_axis': mu_matrix, 'strike_price_axis': k_matrix, 'pnl_surface': pnl_surface}

# payoff functions
def call_payoff(stock_prices, strike_price, call_prem):
    """
    call function payoff at expirey
    """
    call_payoff = lambda st,k,delta: max(st - k, 0) - delta
    return np.array([call_payoff(st, strike_price, call_prem) for st in stock_prices])

def put_payoff(stock_prices, strike_price, put_prem):
    """
    put function payoff at expirey
    """
    put_payoff = lambda st,k,delta: max(k - st, 0) - delta
    return np.array([put_payoff(st, strike_price, put_prem) for st in stock_prices])