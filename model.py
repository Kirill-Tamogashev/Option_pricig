import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class OptionsPredictor:
   """
   Class for predicting option prices for classic and compound options

   Parameters
   ----------
   expiration_time : float 
         This parameter specifies expiration time for a given base option

   strike_price : float 
         This parameter specifies strike price for a given base option 

   current_price : float
         This parameter specifies current price for the underlying asset

   discount_rate : float
         Parameter for dicounting that indicates time
         value for money, risk free rate

   expiration_time_compound : float, default=None
         This parameter specifies expiration time for compound option

   strice_price_compound : float, default=None
         This parameter specifies strike price for compound option

   drift : float, default=0
         This parameter specifies the interest rate for a given asset

   volatility : float, default=1
         This parameter specifies volatility of a given asset

   n_estimations : int, default=10000 
         This parameter specifies the number of Monte-Carlo simulations to be made 
         for price estimetion 

   mode : {'discrete', 'continious'}, default='continious'
         This parameter determines the stratagy of price estimetion

   random_state : int, RandomState instance or None, default=None
         Controls the pseudo random number generation for shuffling the data.
         Pass an int for reproducible output across multiple function calls.
   """
   def __init__(self, expiration_time,  strike_price ,
                current_price, discount_rate,
                expiration_time_compound=None, 
                strice_price_compound=None, drift=0, 
                volatility=1, n_estimations=10000, 
                mode='continious', random_state=None):
      self.dt = 1 / 365 
      self.n_iter = expiration_time + 1
      self.T = expiration_time * self.dt
      self.K = strike_price
      self.S0 = current_price
      self.discount_rate = discount_rate
      self.T_tild = expiration_time_compound * self.dt if expiration_time_compound is not None else None
      self.K_tild = strice_price_compound
      self.mu = drift 
      self.sigma = volatility
      self.n_estimations = n_estimations
      self.mode = mode
      self.random_state = random_state
      
      if self.mode != 'continious' and self.mode != 'discrete':
         raise KeyError('Wrong mode selected, you should choose either `continious` or `discrete`')
      if self.T_tild is not None and self.T_tild > self.T:
         raise ValueError('Expiration time for compound option must be less or equal expiration time of  base option')


   
   def generate_discrete_(self):
      """
      This function generated prices by using discrete approach. 
      The underlying formula for computaion is

         dSt / St = mu * dt + sigma * dWt
      
      where `mu` is drift and `sigma` is volatility, 
      dWt ~ N(0, sqrt(dt))

      Returns
      -------
      prices : ndarray of shape (self.n_estimations, self.n_iter)
            Returns the simulated prices
      """
      if self.random_state: np.random.seed(self.random_state)
      prices = np.zeros((self.n_iter, self.n_estimations))
      prices[0] = self.S0
      for i in range(1, self.n_iter):
            dWt = np.random.normal(loc=0, 
                                   scale=np.sqrt(self.dt), size=self.n_estimations)
            dSt = self.mu * self.dt + self.sigma * dWt 
            prices[i]  = prices[i - 1] * (1 + dSt)

      prices = prices.T
      assert prices.shape == (self.n_estimations, self.n_iter)   
      return prices


   def generate_continious_(self):
      """
      This function generated prices by using continious approach. 
      The underlying formula for computaion is

         St = S0 * exp{(mu - sigma ** 2 / 2) * t + sigma * Wt}
      
      where `mu` is drift and `sigma` is volatility
      Wt ~ N(0, sqrt(t))

      Returns
      -------
      prices : ndarray of shape (self.n_estimations, self.n_iter)
            Returns the simulated prices
      """
      if self.random_state: np.random.seed(self.random_state)
      wiener = np.random.normal(loc=0, scale=np.sqrt(self.dt), 
                                size=(self.n_estimations, self.n_iter))
      wiener = np.cumsum(wiener, axis=1) * self.sigma
      time = np.linspace(0, self.T, num=self.n_iter)
      stock_var = (self.mu - (self.sigma**2 / 2)) * time 
      prices = self.S0 * (np.exp(stock_var + wiener))

      assert prices.shape == (self.n_estimations, self.n_iter)   
      return prices

 
   def generate_price(self, out=False, traj_function='mean'):
      """
      This estimates trajectory of a price change

      Parameters
      ----------
      out : bool, defaut=False
            Parameter for choosinf weather to return the array of prices
      
      traj_function : {'mean', 'median'}, default='mean'
            This parameter spesifies a function to be used for computing
            prices trajectory
      
      Returns
      ------
      self.traj : ndarray of shape (self.n_iter, ) if out=True
            Estimated trajectory of prices
      """
      tf = {'mean': np.mean, 'median': np.median}

      if traj_function != 'mean' and traj_function != 'median':
         raise KeyError('Wrong trajectory function, you should choose either `mean` or `median`')    

      if self.mode == 'continious':
         S = self.generate_continious_()
      elif self.mode == 'discrete':
         S = self.generate_discrete_()

      self.traj = tf[traj_function](S, axis=0)
      self.std_traj = np.std(S, axis=0)

      assert self.traj .shape == (self.n_iter, )  
      assert self.std_traj.shape == (self.n_iter, )  

      if out: return self.traj


   def plot_prices(self, confidence_level = 95):
      """
      This function plots the estimeted trajectory 
      and confidence interval for a given confidence level 

      Parameters
      ----------
      confidence_level : {80, 95, 99}, default=95
            Confidence level

      Returns
      -------
      None, funtion prints the plot
      """
      z_crit = {80: 1.28, 95: 1.96, 99: 2.575}
      if confidence_level not in z_crit.keys():
         raise NotImplementedError('This confidence level is not implemented')

      plt.figure(figsize=(12,6))
      plt.plot([self.S0] * self.n_iter, ls='--', c='black', lw=2, label = 'Current price')
      plt.plot(self.traj, 'r', label='Trajectory')
      plt.fill_between(np.arange(self.n_iter), 
                       (self.traj - z_crit[confidence_level] * self.std_traj / np.sqrt(self.n_estimations)), 
                       (self.traj + z_crit[confidence_level] * self.std_traj / np.sqrt(self.n_estimations)), 
                       color='b', alpha = 0.1,
                       label=f'{confidence_level}% confidence interval')
      plt.title(f'Option price estimation with {confidence_level}% confidence interval')
      plt.legend(loc='upper left')
      plt.xlabel('Number of days')
      plt.ylabel('Price of the option')
      plt.xlim(-1, self.n_iter + 1)
      plt.show()
     

   def predict_V(self, f='max', expiraion_time=None, no_discount=False):
      """
      This method predicts the value for Option

      Parameters
      ----------
         f : {'max', 'min'}, default='max'
               Function to be used to computing option value

         expiraion_time : int, default=None 
               expiration time 
         
         no_discount : bool, default=False
               This parameter spesifies weather to return value 
               without dicount or not

      Returns
      -------
         value of option at a given expiration_time
      """
      if expiraion_time is  None:
         T = self.n_iter - 1
      else: 
         T = int(expiraion_time)

      F = {'min': min, 'max': max}
      V = F[f](self.traj[T] - self.K, 0)

      if no_discount: return V

      if self.mode == 'continious':
         PV = V * np.exp(-self.discount_rate * self.T)

      elif self.mode == 'discrete':
         PV = V * (1 + self.discount_rate / 365) ** (-self.T * 365)

      return PV #round(PV, 5)


   def predict_CV(self, f1='max', f2='max'):
      """
      This method predict the value for Compound Option

      Parameters
      ----------
         f1 : {'max', 'min'}, default='max'
               Function to be used to computing compound option value

         f2 : {'max', 'min'}, default='max'
               Function to be used to computing base option value

      Returns
      -------
         value of compound option at a given expiration_time_compound
      """
      if  self.T_tild is None or self.K_tild is None:
         raise ValueError('Parameters for compound option are not specified')

      F = {'min': min, 'max': max}
      CV = F[f1](self.predict_V(f=f2, expiraion_time=self.T_tild * 365) - self.K_tild, 0)

      if self.mode == 'continious':
         PV = CV * np.exp(-self.discount_rate * self.T_tild)

      elif self.mode == 'discrete':
         PV = CV * (1 + self.discount_rate / 365) ** (-self.T_tild * 365)
         
      return PV #round(PV, 5)