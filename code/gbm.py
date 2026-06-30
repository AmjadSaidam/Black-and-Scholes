import numpy as np 
import scipy.stats as sp

class StochasticPriceForecast(): 
    def __init__(self, returns: np.ndarray | None = None): # define with defualt class to type errors when class inheriting
        # time 
        self.days_in_year = 252 # days in year (assuming trading equities, and dailt data)
        self.dt = 1/self.days_in_year # default time step frequency (the same as data frequency)
        self.rt = returns 
        self.loc = np.mean(returns, axis = 0)*self.days_in_year
        self.S0 = 0
        self.loc = 0 
        self.scale = 0 
        self.forecast_period = 0
        self.z_alpha = 0

    def gbm_sim(self, n_simulations: int, forecast_period: int, initial_wealth: float = 1) -> dict[np.ndarray]: 
        """
        simulates equaity curves given prior returns, assuming price follows a SDE
        calculautes confidence interval for stock across time, assuming stock price follows log-normal
        """
        self.forecast_period = forecast_period
        self.S0 = initial_wealth
        self.scale = np.var(self.rt, axis = 0)*self.days_in_year

        size = (self.forecast_period, n_simulations)
        sims = np.ones(size)
        sims[0, :] = self.S0
        
        sim_returns = np.zeros(size)
        sim_returns[0, :] = 0

        normal_loc = (self.loc - .5*self.scale)*self.dt
        # simulate paths
        for sim in range(sims.shape[1]):
            S = self.S0
            for t in range(1, self.forecast_period): 
                epsilon = np.random.normal(0, 1)
                normal_scale = np.sqrt(self.scale)*np.sqrt(self.dt)*epsilon
                rt = normal_loc + normal_scale
                S *= np.exp(rt)
                sim_returns[t, sim] = rt
                sims[t, sim] = S

        return {'prices': sims, 'returns': sim_returns}

    def gbm_confidence_interval(self, significance_level: float = 0.05) -> list[np.ndarray]: 
        """
        deterministic confidence intervals
        """
        up_band = np.ones(self.forecast_period)*self.S0
        low_band = np.ones(self.forecast_period)*self.S0

        drift = self.loc - (self.scale/2)

        # estimate standard normal value that corresponds to alpha confidence interval 
        self.z_alpha = sp.norm.ppf(significance_level/2).item()

        # confidence intervals per time step 
        for t in range(1, self.forecast_period):
            del_t = self.dt*t # current calendar time, default daily increment
            loc_term = drift*del_t
            scale_term = abs(self.z_alpha)*np.sqrt(self.scale * del_t)
            up_band[t] = self.S0*np.exp(loc_term + scale_term)
            low_band[t] = self.S0*np.exp(loc_term - scale_term)
        
        return [up_band, low_band]
    
    def price_denisty(self, s: float, t: int) -> np.float32:
        """
        log-normal randonm walk density
        """
        return 1/(np.sqrt(2*np.pi*t*self.scale)*s) * np.exp(
            - (np.log(s/self.S0) - (self.loc - .5*self.scale)*t)**2 / (2*self.scale*t)
        ) 

    def analytical_VaR(self): 
        """
        models VaR assuming log-retruns follow a GBM and price follows a log-normal distribution 
        This is a CDF on the terminal time log-normal stock price density
        """
        tau = self.forecast_period * self.dt # terminal time
        mu_T = self.loc - (self.scale/2)*tau
        scale_T = np.sqrt(self.scale*tau) 

        return self.S0*(1 - np.exp(mu_T + scale_T*self.z_alpha))

    def analytical_CVaR(self): 
        """
        """
        pass
